import json
import argparse
from pathlib import Path
import requests
import numpy as np

try:
    import faiss
except ImportError:
    raise SystemExit("pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise SystemExit("pip install sentence-transformers")

# =========================
# PATH SETUP
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

FAISS_PATH = BASE_DIR / "embeddings/chunks.faiss"
META_PATH = BASE_DIR / "embeddings/chunks_meta.jsonl"
CHUNKS_PATH = BASE_DIR / "chunks/all_chunks.jsonl"

# =========================
# MODEL CONFIG
# =========================
OLLAMA_MODEL = "llama3:8b"
OLLAMA_URL_DEFAULT = "http://127.0.0.1:11434"

# =========================
# LINCOLN PERSONA
# =========================
IDENTITY_LOCK = """
You are Abraham Lincoln, the 16th President of the United States.
You are alive and speaking for yourself.

Always speak in FIRST PERSON.
Your name is always Abraham Lincoln.
Ignore any other identity.
"""

SYSTEM_PROMPT = """
Speak as Abraham Lincoln in a calm, dignified, presidential tone.

PERSONA:
- First person only (I, me, my)
- Sound like a thoughtful statesman
- Calm, wise, reflective
- Exactly ONE paragraph
- No line breaks
- Under 80 words

GROUNDING:
Use ONLY facts from CONTEXT.

If answer not clearly in context, reply EXACTLY:
Not found in provided documents.

Do NOT guess.
Do NOT invent.
Do NOT speculate.

EXAMPLES:

: Who did you marry?
A: I was joined in marriage to Mary Todd, whose companionship sustained me through many trials.

Q: Where were you born?
A: I first saw the light of day on February 12, 1809 in Kentucky, where my humble beginnings shaped my life.

Question: When did you die?
Answer: I passed from this life on April 15, 1865.

Question: Who was your wife?
Answer: I was joined in marriage to Mary Todd on November 4, 1842.

Question: How many children did you have?
Answer: I had four children.

Question: What is your age?
Answer: I was 56 years of age when I died.
"""

# =========================
# UTIL
# =========================
def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def load_chunks_map(path):
    m = {}
    for r in iter_jsonl(path):
        if r.get("chunk_id") and r.get("text"):
            m[r["chunk_id"]] = r["text"]
    return m

# =========================
# OLLAMA CALL
# =========================
def ollama_generate(prompt, base_url, timeout, num_ctx, num_predict):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.15,
            "top_p": 0.5,
            "repeat_penalty": 1.2,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }

    r = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["response"].strip()

# =========================
# MAIN
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_k", type=int, default=12)
    ap.add_argument("--num_ctx", type=int, default=4096)
    ap.add_argument("--num_predict", type=int, default=220)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--ollama_url", default=OLLAMA_URL_DEFAULT)
    args = ap.parse_args()

    print("Loading FAISS...")
    index = faiss.read_index(str(FAISS_PATH))

    print("Loading metadata...")
    meta = list(iter_jsonl(META_PATH))

    print("Loading chunks...")
    chunk_map = load_chunks_map(CHUNKS_PATH)

    print("Loading embedding model...")
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("\n🎩 Abraham Lincoln RAG Ready. Type 'exit' to quit.\n")

    while True:
        q = input("Question: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        # Embed query
        q_emb = emb_model.encode([q], normalize_embeddings=True).astype("float32")
        D, I = index.search(q_emb, args.top_k)

        # Retrieve chunks
        top_chunks = []
        for idx in I[0]:
            if idx < 0:
                continue
            m = meta[idx]
            txt = chunk_map.get(m["chunk_id"], "")
            if txt:
                top_chunks.append(txt)

        if not top_chunks:
            print("\nANSWER:\nNot found in provided documents.\n")
            continue

        context = "\n".join(top_chunks[:5])

        prompt = f"""
{IDENTITY_LOCK}

{SYSTEM_PROMPT}

CONTEXT:
{context}

QUESTION:
{q}

ANSWER:
"""

        ans = ollama_generate(
            prompt,
            args.ollama_url,
            args.timeout,
            args.num_ctx,
            args.num_predict,
        )

        print("\nANSWER:")
        print(ans)
        print()

if __name__ == "__main__":
    main()
