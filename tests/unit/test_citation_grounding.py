"""
tests/unit/test_citation_grounding.py
──────────────────────────────────────
A citation is shown only when the cited chunk's page text states the
claim. The model sees each chunk's heading path above its text, so it can
cite a chunk for a fact only the heading path carries (a resume's name
heads every chunk); a heading path is never evidence. Such a citation
moves to an included chunk whose text has the fact, or is dropped. An
answer whose claims about the documents have no support says the answer
is not in the documents; an answer with no citations is general knowledge
and carries a "Not from your documents" label and no sources.
"""

import pytest

from src.services.citation_grounding import (
    NOT_FROM_DOCUMENTS,
    finalize_rag_answer,
    ground_citations,
)
from src.services.prompt_builder import NO_ANSWER_PHRASE
from src.services.rag_service import CitedSource

pytestmark = pytest.mark.unit

NAME_QUESTION = "What is the name of the candidate?"
SOURCES = [
    "AWS Certified Data Engineer, Associate (2024)",  # [1]
    "I am available to start on May 1, 2026.\n\nSincerely,\n\nTalia Brennerholt",  # [2]
    "Portland, Oregon | talia.brennerholt@example.com | (555) 010-0142\n\nMarch 3, 2026",  # [3]
]


class TestGroundCitations:
    def test_a_citation_whose_text_lacks_the_fact_moves_to_one_that_has_it(self):
        g = ground_citations("Talia Brennerholt [1].", SOURCES, question=NAME_QUESTION)
        assert g.answer == "Talia Brennerholt [2]."
        assert g.moved == {1: 2}
        assert g.dropped == set()

    def test_a_moved_citation_that_duplicates_another_is_dropped(self):
        g = ground_citations("Talia Brennerholt[2][3]", SOURCES, question=NAME_QUESTION)
        assert g.answer == "Talia Brennerholt[2]"

    def test_an_email_address_is_not_the_name(self):
        # [3] holds talia.brennerholt@example.com, not "Talia Brennerholt".
        g = ground_citations("Talia Brennerholt [3]", SOURCES, question=NAME_QUESTION)
        assert g.answer == "Talia Brennerholt [2]"

    def test_question_words_are_not_evidence(self):
        g = ground_citations(
            "The candidate's name is Talia Brennerholt [1].",
            SOURCES,
            question=NAME_QUESTION,
        )
        assert g.answer == "The candidate's name is Talia Brennerholt [2]."

    def test_supported_citations_are_left_alone(self):
        answer = (
            "She can start on May 1, 2026 [2]. The letter is dated March 3, 2026 [3]."
        )
        g = ground_citations(answer, SOURCES, question="When can she start?")
        assert g.answer == answer
        assert g.moved == {}
        assert g.dropped == set()

    def test_each_claim_is_checked_against_its_own_citation(self):
        g = ground_citations(
            "She is AWS certified [1]. She can start on May 1, 2026 [1].",
            SOURCES,
            question="Tell me about her.",
        )
        assert g.answer == "She is AWS certified [1]. She can start on May 1, 2026 [2]."

    def test_a_citation_no_chunk_supports_is_dropped(self):
        g = ground_citations(
            "She speaks fluent Portuguese [1].",
            SOURCES,
            question="Which languages does she speak?",
        )
        assert g.answer == "She speaks fluent Portuguese."
        assert g.dropped == {1}

    def test_only_the_unsupported_claim_loses_its_citation(self):
        g = ground_citations(
            "She is AWS Certified [1]. She speaks Portuguese [1].",
            SOURCES,
            question="Tell me about her.",
        )
        assert g.answer == "She is AWS Certified [1]. She speaks Portuguese."

    def test_a_heading_path_is_never_evidence(self):
        # The chunk text the check reads is the page text; the heading path
        # ("Talia Brennerholt > Certifications") is not part of it.
        sources = ["Certifications\n\nAWS Certified Data Engineer, Associate (2024)"]
        g = ground_citations("Talia Brennerholt [1].", sources, question=NAME_QUESTION)
        assert g.answer == "Talia Brennerholt."
        assert g.dropped == {1}

    def test_bullet_lists_and_markdown_are_handled(self):
        answer = (
            "- **Start date:** May 1, 2026 [1]\n- **Letter date:** March 3, 2026 [3]"
        )
        g = ground_citations(answer, SOURCES, question="Which dates are mentioned?")
        assert (
            g.answer
            == "- **Start date:** May 1, 2026 [2]\n- **Letter date:** March 3, 2026 [3]"
        )

    def test_plural_and_possessive_forms_match(self):
        sources = ["The team cut broken dashboards by 80 percent."]
        g = ground_citations(
            "Broken dashboard counts fell 80 percent [1].",
            sources,
            question="What changed?",
        )
        assert g.dropped == set()

    def test_a_short_claim_needs_every_fact_word(self):
        # "AWS Certified Data Engineer" has 2 of the 3 words, not the role.
        sources = [
            "AWS Certified Data Engineer certificate",
            "the Senior Data Engineer role",
        ]
        g = ground_citations(
            "Senior Data Engineer [1].",
            sources,
            question="Which role is she applying for?",
        )
        assert g.answer == "Senior Data Engineer [2]."

    def test_a_month_named_may_is_a_fact(self):
        sources = ["March 3, 2026", "I am available to start on May 1, 2026."]
        g = ground_citations(
            "May 1, 2026 [1].", sources, question="When can she start?"
        )
        assert g.answer == "May 1, 2026 [2]."

    def test_a_claim_of_only_question_words_is_left_alone(self):
        g = ground_citations("Yes [1].", SOURCES, question="Is she certified?")
        assert g.answer == "Yes [1]."
        assert g.dropped == set()

    def test_unicode_hyphens_match_ascii_ones(self):
        # gpt-oss writes phone numbers with non-breaking hyphens (U+2011).
        sources = [
            "Employees may call the USCIS employee hotline at 1-888-897-7781 "
            "or 1-877-875-6028 (TTY)."
        ]
        answer = (
            "USCIS employee hotline: 1\u2011888\u2011897\u20117781 "
            "(TTY 1\u2011877\u2011875\u20116028) [1]."
        )
        g = ground_citations(
            answer, sources, question="What is the USCIS employee hotline phone number?"
        )
        assert g.answer == answer
        assert g.dropped == set()

    def test_a_run_on_its_own_line_is_checked_against_the_list_above_it(self):
        sources = [
            "Experience\n\nSenior Data Engineer, Pellwick Freight (2022 to present)"
            "\n\nData Engineer, Orrin Labs (2019 to 2022)",
            "AWS Certified Data Engineer, Associate (2024)",
        ]
        g = ground_citations(
            "- Senior Data Engineer at Pellwick Freight\n- Data Engineer at Orrin Labs\n[2]",
            sources,
            question="Tell me about her work history.",
        )
        assert g.answer.endswith("\n[1]")
        assert g.moved == {2: 1}

    def test_short_codes_are_not_facts(self):
        sources = [
            "The CSF Core Functions are Govern, Identify, Protect, Detect, "
            "Respond, and Recover."
        ]
        answer = (
            "Govern (GV), Identify (ID), Protect (PR), Detect (DE), "
            "Respond (RS), Recover (RC) [1]."
        )
        g = ground_citations(answer, sources, question="What are the six Functions?")
        assert g.answer == answer

    def test_initials_do_not_end_a_claim(self):
        # The claim is the whole line, not just "in Computer Science".
        sources = [
            "Education\n\nB.S. in Computer Science, University of Eastmoor, 2018",
            "Volunteering\n\nComputer Science tutor for first-year students",
        ]
        g = ground_citations(
            "University of Eastmoor, B.S. in Computer Science [2].",
            sources,
            question="Where did the candidate go to university?",
        )
        assert g.answer == "University of Eastmoor, B.S. in Computer Science [1]."

    def test_an_answer_without_citations_is_unchanged(self):
        g = ground_citations("No markers here.", SOURCES, question=NAME_QUESTION)
        assert g.answer == "No markers here."


def _source(n: int, text: str) -> CitedSource:
    return CitedSource(
        document_id="doc-1",
        chunk_id=f"chunk-{n}",
        source_filename="resume.pdf",
        page_number=n,
        chunk_index=n,
        relevance_score=0.0,
        text=text,
    )


class TestFinalizeRagAnswer:
    """What the stream endpoint sends: the answer, its sources and its label."""

    @staticmethod
    def _sources() -> list[CitedSource]:
        return [_source(n, text) for n, text in enumerate(SOURCES, 1)]

    def test_a_supported_answer_keeps_its_sources_and_has_no_label(self):
        final = finalize_rag_answer(
            "Talia Brennerholt [1].", self._sources(), question=NAME_QUESTION
        )
        assert final.answer == "Talia Brennerholt [2]."
        assert len(final.sources) == 3
        assert final.label is None

    def test_cited_claims_nothing_supports_say_the_answer_is_not_in_the_documents(
        self,
    ):
        final = finalize_rag_answer(
            "She speaks fluent Portuguese [1].",
            self._sources(),
            question="Which languages does she speak?",
        )
        assert final.answer == NO_ANSWER_PHRASE
        assert final.sources == []
        assert final.label is None

    def test_an_uncited_answer_is_general_knowledge(self):
        final = finalize_rag_answer(
            "Data engineers build and run the pipelines that move data.",
            self._sources(),
            question="What does a data engineer do?",
        )
        assert (
            final.answer == "Data engineers build and run the pipelines that move data."
        )
        assert final.sources == []
        assert final.label == NOT_FROM_DOCUMENTS == "not_from_documents"

    def test_a_model_decline_has_no_sources_and_no_label(self):
        final = finalize_rag_answer(
            NO_ANSWER_PHRASE, self._sources(), question="What is her salary?"
        )
        assert (final.answer, final.sources, final.label) == (
            NO_ANSWER_PHRASE,
            [],
            None,
        )
