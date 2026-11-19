# DocuSage: Turning Your Documents Into a Trusted Expert You Can Just *Ask*

## Executive Summary
Every organization sits on a mountain of documents: contracts, policies, manuals, reports. The knowledge inside them is locked away, findable only by whoever remembers which file it's in. **DocuSage** turns that pile into a private, conversational expert: upload your documents, ask questions in plain English, and get clear answers **backed only by your own files**, with the exact source passage shown as proof. It's the convenience of a chatbot without the two things that make generic AI risky for business: making things up, and leaking your data.

## Business Challenges
Before DocuSage, finding answers inside documents meant:
- **Hours of manual hunting.** Staff scrolled through long PDFs to find one clause or figure.
- **Knowledge bottlenecks.** Only a few experts knew where things lived; when they were out, work stalled.
- **Risky shortcuts.** Public AI tools could *invent* confident-sounding but wrong answers, and pasting confidential documents into them exposed sensitive data.
- **No trust, no audit trail.** Even when an answer was found, there was no easy way to prove *where* it came from.

## Solution Approach
DocuSage reads your documents once, indexes them, and then answers questions about them on demand. Behind the simple chat box is a short relay of focused steps, like a well-run research desk: one step works out what you're really asking, another digs up the most relevant evidence, another writes the answer, and a final check confirms it cites real sources. Only the writing (and summarizing) uses a language model; the routing is rules first, and the final check is a program, not another AI call. Every answer is **grounded in your documents and cited**, and if the documents don't contain the answer, DocuSage **says so instead of guessing**. Everything stays in your private, walled-off workspace.

## The Workflow (Simplified)

```
            ┌──────────────────────── DocuSage ────────────────────────┐
  Upload ─▶ │  📥 Librarian: reads · splits · files every document      │
            └───────────────────────────┬───────────────────────────────┘
                                         │  (indexed & ready to search)
                                         ▼
  Question ─▶ 🧭 Router ─┬─ a question  ─▶ 🔎 Researcher ─▶ ✍️ Writer ──┐
                         ├─ a summary   ─▶ 📚 Summarizer ───────────────┤─▶ ✅ Citation check ─▶ 📎 Cited Answer
                         └─ small talk  ─▶ 💬 Quick reply ──────────────┘
```

Think of it as a relay, each step doing one job well:

1. **📥 The Librarian.** When you upload a file (PDF, Word, plain text, Markdown or HTML), it reads it, breaks it into bite-sized passages, and files them so they can be found instantly later. This runs in the background, with live progress in the app.
2. **🧭 The Router.** Decides what you actually want: a specific answer, a whole-document summary, or just a friendly hello. Clear cases (greetings, "summarize ...") are recognized by rules at no cost; everything else goes to the Researcher. An AI classifier can be switched on for harder cases.
3. **🔎 The Researcher.** For a real question, it searches your documents two ways at once (by *meaning* and by *keyword*), then re-ranks the results to surface the best evidence. If nothing relevant turns up, it declines instead of guessing.
4. **✍️ The Writer.** Composes a clear answer using *only* that evidence, attaching citations that link back to the exact page and passage.
5. **📚 The Summarizer.** If you asked for an overview, it reads the whole document in sections and distills an executive summary. (A deployment can switch summaries off to save model capacity; summary requests then get a normal cited answer.)
6. **✅ The Citation check.** Before the answer is saved, a program verifies that every citation points to a real source and that the answer kept the "decline when unsure" promise.

The result streams back live, with a **"Check source" preview** showing a snapshot of the actual document page with the passage highlighted, so you can trust, and verify, in one glance.

## Technical Implementation Plan
- **Architecture:** a FastAPI (Python) backend with a React/TypeScript front end. Business logic is organized as composable services: a `ChatOrchestrator` routes each turn to the right step, following a clean routing pattern rather than one monolithic prompt.
- **Document pipeline:** uploads are parsed, chunked, converted into mathematical "embeddings", and stored in a **Qdrant** vector database; original files live in S3-compatible object storage. Processing runs in a background worker so the app stays responsive.
- **Retrieval (the smart search):** combines dense (semantic) and keyword search, fused with Reciprocal Rank Fusion, then a **cross-encoder re-ranker** sharpens the ordering and enforces a confidence threshold for **abstention**, the anti-hallucination safeguard.
- **Answering:** the chosen language model streams a grounded, cited response; a **programmatic validator** (no extra AI cost) checks citations. Model choice is pluggable: a local model through Ollama, OpenAI or any OpenAI-compatible service such as Groq, or a self-hosted vLLM server.
- **Privacy and multi-tenancy:** each workspace is isolated, with its own vector collection and storage prefix, and every search is additionally filtered by user, so one customer's data cannot surface in another's answers.
- **Operations:** separate models for chat and summarization (to balance speed and cost); rate limits and usage quotas for public deployments; automated tests, dependency and container scanning, static analysis and a web security scan on every change.

---
*One platform, a few focused steps, and a simple promise: answers you can trust, sourced from documents you own.*
