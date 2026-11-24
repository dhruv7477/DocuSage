/**
 * Citation markers in an answer. The server's validator writes [n], where
 * source n is `sources[n - 1]`; these helpers also cover answers saved before
 * it rewrote gpt-oss's own form (【3†p.2】).
 */

// The same rules as answer_validator.normalize_citations on the server: a
// marker that is just numbers, or that carries the dagger, is a citation;
// other full-width brackets are ordinary CJK punctuation. Bounded lengths
// keep a stray bracket from causing slow backtracking.
const FULL_WIDTH_CITATION =
  /【\s{0,3}(\d{1,3}(?:\s{0,3}[,，]\s{0,3}\d{1,3}){0,9})\s{0,3}(?:†[^】\n]{0,40})?】|【[^】†\n]{0,40}†[^】\n]{0,40}】/g
const CITATION = /\[(\d+)\]/g

/** Rewrite full-width citation markers as [n], one per number; drop those without one. */
export function normalizeCitations(text: string): string {
  return text.replace(FULL_WIDTH_CITATION, (_marker, numbers?: string) =>
    numbers ? (numbers.match(/\d+/g) ?? []).map((n) => `[${n}]`).join('') : '',
  )
}

export type AnswerPart = { text: string } | { source: number }

/**
 * Split an answer into text and citations of sources 1..sourceCount.
 * Markers outside that range are dropped: they have no source to show.
 */
export function splitCitations(text: string, sourceCount: number): AnswerPart[] {
  const normalized = normalizeCitations(text)
  const parts: AnswerPart[] = []
  let last = 0
  for (const match of normalized.matchAll(CITATION)) {
    const index = match.index ?? 0
    if (index > last) parts.push({ text: normalized.slice(last, index) })
    const n = Number(match[1])
    if (n >= 1 && n <= sourceCount) parts.push({ source: n })
    last = index + match[0].length
  }
  if (last < normalized.length) parts.push({ text: normalized.slice(last) })
  return parts
}

/** The sources an answer cites, each once, in the order first cited. */
export function citedSourceNumbers(text: string, sourceCount: number): number[] {
  const cited = new Set<number>()
  for (const part of splitCitations(text, sourceCount)) {
    if ('source' in part) cited.add(part.source)
  }
  return [...cited]
}
