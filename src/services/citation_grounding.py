"""
src/services/citation_grounding.py
──────────────────────────────────
Checks every citation in a finished answer against the cited chunk's page
text, and decides what the answer shows. Deterministic, no LLM call.

The rule: a citation is shown only when the page text supports the claim.
The evidence is the chunk's own text (the words on its page, including a
heading printed in that chunk), never its heading path: the model sees the
path above each chunk (prompt_builder), and a resume's name heads every
chunk, so "Rosalind Kettleborne [1]" may point at a hackathons chunk on
another page that never states it.

For each run of markers, the claim it closes (the text back to the start
of its sentence or line, or to the previous markers) is reduced to its
fact words: no stopwords, and no words of the question, which restate
rather than support. A cited chunk supports the claim when its text holds
all of them for a short claim (`SHORT_CLAIM_WORDS`), or at least
`SUPPORT_SHARE` of them for a longer one. Otherwise the citation moves to the
included chunk that does (and is dropped if that chunk is already cited
there), or, when none does, it is dropped.

`finalize_rag_answer` applies the rule to a whole answer: cited claims of
which none is supported become the "not in your documents" reply, and an
answer that cites nothing is general knowledge, shown with the
`NOT_FROM_DOCUMENTS` label and no sources.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.core.logger import get_logger
from src.services.prompt_builder import NO_ANSWER_PHRASE
from src.services.rag_service import CitedSource

log = get_logger(__name__)

# The label of an answer that cites none of the documents.
NOT_FROM_DOCUMENTS = "not_from_documents"

# Share of a claim's fact words the cited chunk's own text must contain.
SUPPORT_SHARE = 0.6
# A claim this short (a name, a role, a date) needs every fact word: "AWS
# Certified Data Engineer" has two of the three words of "Senior Data
# Engineer" but does not state the role.
SHORT_CLAIM_WORDS = 3

_GROUP_RE = re.compile(r"(?:[ \t]*\[\d+\])+")  # a run of markers, e.g. " [1][3]"
_MARKER_RE = re.compile(r"\[(\d+)\]")
_LEADING_SPACE_RE = re.compile(r"^[ \t]*")
# Where a claim starts: after a sentence end followed by a space, or a line break.
_CLAIM_START_RE = re.compile(r"[.!?](?=\s)|\n")
_EDGE_PUNCTUATION = "\"'`*_()[]{}<>,.;:!?…“”‘’|-#"
_POSSESSIVE_RE = re.compile(r"['’]s$")
_STOPWORDS = frozenset(
    """
    a an the and or but if of in on at to for from by with without into onto
    about as is are was were be been being am do does did done has have had
    having it its this that these those he she they them his her their hers
    theirs him we our you your i me my mine not no yes than then there here
    what which who whom whose when where why how all any both each more most
    other some such only own same so too very can will just should would
    could might must shall also
    """.split()
)


@dataclass
class GroundedCitations:
    """The answer with its citations checked against the chunks' own text."""

    answer: str
    moved: dict[int, int] = field(default_factory=dict)  # source n -> source m
    dropped: set[int] = field(default_factory=set)  # cited, never supported


@dataclass(frozen=True)
class FinalAnswer:
    """What a document answer shows: its text, its sources and its label."""

    answer: str
    sources: list[CitedSource]
    label: str | None = None


def ground_citations(
    answer: str, source_texts: list[str], *, question: str
) -> GroundedCitations:
    """
    Re-point each [n] in `answer` to a source whose text states the claim,
    and drop it when none does.

    `source_texts[n - 1]` is the page text of the chunk behind [n]. Markers
    outside that range are left for the answer validator.
    """
    question_words = _words(question)
    chunk_words = [_words(text) for text in source_texts]
    moved: dict[int, int] = {}
    supported: set[int] = set()
    dropped: set[int] = set()
    parts: list[str] = []
    last = 0

    for group in _GROUP_RE.finditer(answer):
        facts = _words(_claim(answer[last : group.start()])) - question_words
        targets: list[int] = []
        for n in (int(m) for m in _MARKER_RE.findall(group.group())):
            target: int | None = n
            if facts and 1 <= n <= len(chunk_words):
                if _supports(chunk_words[n - 1], facts):
                    supported.add(n)
                else:
                    target = _best_support(chunk_words, facts)
                    if target is None:
                        dropped.add(n)
                    else:
                        moved[n] = target
                        supported.add(target)
            if target is not None and target not in targets:
                targets.append(target)

        parts.append(answer[last : group.start()])
        original = [int(m) for m in _MARKER_RE.findall(group.group())]
        if targets == original:
            parts.append(group.group())
        elif targets:
            lead = _LEADING_SPACE_RE.match(group.group()).group()  # type: ignore[union-attr]
            parts.append(lead + "".join(f"[{t}]" for t in targets))
        last = group.end()

    parts.append(answer[last:])
    return GroundedCitations(answer="".join(parts), moved=moved, dropped=dropped)


def finalize_rag_answer(
    answer: str, sources: list[CitedSource], *, question: str
) -> FinalAnswer:
    """
    Apply the citation rule to a validated document answer whose markers
    index the context-ordered `sources` (source n backs [n]).

    - The model's own decline keeps its text and shows no sources.
    - An answer that cites nothing is general knowledge: its text, the
      `NOT_FROM_DOCUMENTS` label, no sources.
    - Otherwise each citation is grounded; when none survives, no claim
      about the documents is supported and the answer is the decline.
    """
    if answer.strip().startswith(NO_ANSWER_PHRASE):
        return FinalAnswer(answer, [])
    if not _MARKER_RE.search(answer):
        log.info("answer_not_from_documents", preview=answer[:120])
        return FinalAnswer(answer, [], NOT_FROM_DOCUMENTS)
    grounded = ground_citations(answer, [s.text for s in sources], question=question)
    if grounded.moved or grounded.dropped:
        log.info(
            "citations_grounded",
            moved=grounded.moved,
            dropped=sorted(grounded.dropped),
        )
    if not _MARKER_RE.search(grounded.answer):
        return FinalAnswer(NO_ANSWER_PHRASE, [])
    return FinalAnswer(grounded.answer, sources)


def _claim(text: str) -> str:
    """The text a run of markers closes: back to its sentence or line start."""
    starts = [m.end() for m in _CLAIM_START_RE.finditer(text)]
    return text[starts[-1] :] if starts else text


def _words(text: str) -> set[str]:
    """
    Normalised content words: lowercased, edge punctuation stripped (so an
    email address stays one word), possessives and plural -s removed.
    """
    words: set[str] = set()
    for raw in text.lower().split():
        word = _POSSESSIVE_RE.sub("", raw.strip(_EDGE_PUNCTUATION))
        word = word.strip(_EDGE_PUNCTUATION)
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        if word and word not in _STOPWORDS:
            words.add(word)
    return words


def _supports(chunk: set[str], facts: set[str]) -> bool:
    if len(facts) <= SHORT_CLAIM_WORDS:
        return facts <= chunk
    return len(facts & chunk) >= SUPPORT_SHARE * len(facts)


def _best_support(chunks: list[set[str]], facts: set[str]) -> int | None:
    """The 1-based source whose text covers most of `facts`, if any covers enough."""
    best, best_share = None, 0.0
    for n, chunk in enumerate(chunks, 1):
        share = len(facts & chunk) / len(facts)
        if share > best_share and _supports(chunk, facts):
            best, best_share = n, share
    return best
