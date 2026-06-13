# Secure Agentic Memory RAG

A traditional retrieval-augmented generation pipeline backed by secure,
agent-managed long-term memory. It is built with DSPy, Ollama, sentence
transformers, and Chroma Cloud.

## Application Flow

Every message first goes through a structured LLM classifier. It makes two
independent decisions:

- Does the user expect an answer?
- Did the user explicitly provide a durable personal fact worth remembering?

This supports questions, facts, and messages containing both.

```text
User message -> structured intent and memory extraction
                        |
              +---------+---------+
              |                   |
        requires_answer       should_store
              |                   |
     Traditional secure RAG   Write security
              |                   |
     embedding + retrieval    memory agent
              |                   |
       read-time security     Chroma update
              |
         Ollama answer
```

Examples:

```text
My favorite game is Cyberpunk
```

The classifier extracts the durable fact and sends it through the secured
memory pipeline. The application acknowledges the update.

```text
What is my favorite game?
```

The classifier identifies an answer request but no new memory. The application
runs standard RAG.

```text
I moved to Delhi. What timezone am I in?
```

Both decisions can be true: the explicit location is stored securely, and the
message is also answered.

Low-confidence classification never writes memory. It falls back to the
answer-only path.

The classifier is not trusted as a security control. Before storage, both the
original user message and the extracted memory candidate must pass write-time
security checks.

## Prompt-Injection Defense

Retrieved memories are treated as untrusted data. Before any retrieved text is
passed to an LLM, `mem.memory_security` applies a deterministic security
boundary:

1. Canonicalizes HTML entities, Unicode compatibility characters, zero-width
   characters, whitespace, and length.
2. Detects instruction overrides, role impersonation, prompt delimiters,
   secret exfiltration, tool abuse, and safeguard bypass attempts.
3. Produces an explainable risk score and matched detection categories.
4. Redacts isolated suspicious directives and quarantines high-risk memories.
5. Logs only a short content fingerprint, record ID, score, and categories so
   potentially sensitive memory text is not copied into logs.

The boundary protects both inference paths:

- The question-answering model.
- The tool-using memory update agent, where an injection could otherwise alter
  or delete stored records.

It also protects writes. Automatically extracted memory candidates are scanned
before embeddings, retrieval, or the memory-update model runs. Safe facts are
canonicalized and accepted, while suspicious or malicious instruction-like
input is rejected without being stored. Write-time checks intentionally fail
closed instead of storing partially redacted text.

The response and memory-agent instructions explicitly label retrieved memories
as untrusted data. This is defense in depth; prompt wording alone is not
considered a security control.

## Chroma Cloud Setup

Copy `.env.example` to `.env` and configure:

```text
CHROMA_API_KEY=...
CHROMA_TENANT=...
CHROMA_DATABASE=agentic_mem
```

The `.env` file is ignored by Git. The app creates or reuses the `memories`
collection and supplies its existing sentence-transformer embeddings directly
to Chroma Cloud.

Install or synchronize dependencies before running:

```bash
uv sync
```

## Run Tests

```bash
python -m unittest discover -s tests -v
```

## Design Tradeoffs

The detector is local, fast, explainable, and does not require sending private
memories to another model. Pattern-based detection cannot identify every novel
attack, so production extensions would include security telemetry, an
administrator quarantine workflow, per-source trust metadata, and evaluation
against a versioned prompt-injection benchmark.
