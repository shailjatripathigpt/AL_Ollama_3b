# app.py
"""
Streamlit UI for your RAG (FAISS + Ollama llama3:8b) with:
✅ Ask question -> Top chunks -> Answer (identical to CLI script)
✅ Saves history to PROJECT_ROOT/rag_history/rag_history.jsonl

UI extras added (NO change to RAG functionality):
✅ Fixed-height scrollable chat box
✅ Auto-scroll to latest answer
✅ Typing animation
✅ Light, warm background (matches chat bubble)
✅ Robust portrait display
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Iterable, List, Optional

import numpy as np
import requests
import streamlit as st

try:
    import faiss
except ImportError:
    st.error("faiss not installed. Run: pip install faiss-cpu")
    raise

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    st.error("sentence-transformers not installed. Run: pip install sentence-transformers")
    raise


# =========================
# CONSTANTS – EXACT MATCH TO ORIGINAL SCRIPT
# =========================
OLLAMA_MODEL = "llama3:8b"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT_S = 600

# Original CLI parameters
OLLAMA_NUM_CTX = 4096
OLLAMA_NUM_PREDICT = 220
TOP_K = 12
FINAL_K = 5

PORTRAIT_PATH = "/Users/shailjatripathi/Desktop/Abraham_Lincoln_Final/Abraham_Lincoln.jpg"
CHAT_HEIGHT_PX = 420

# =========================
# ORIGINAL IDENTITY & SYSTEM PROMPT (verbatim)
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

Q: Who did you marry?
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
# ROOT DETECT (unchanged)
# =========================
def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(40):
        if (cur / "embeddings" / "chunks.faiss").exists():
            return cur
        cur = cur.parent
    return start.resolve()


# =========================
# IO HELPERS (unchanged)
# =========================
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        buf = ""
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    yield obj
                buf = ""
                continue
            except Exception:
                pass

            buf += s
            try:
                obj = json.loads(buf)
                if isinstance(obj, dict):
                    yield obj
                    buf = ""
            except Exception:
                if len(buf) > 5_000_000:
                    buf = ""


def write_jsonl_line(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# =========================
# CHUNK TEXT MAP (unchanged)
# =========================
def load_chunks_text_map(chunks_path: Path) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for r in iter_jsonl(chunks_path):
        cid = r.get("chunk_id")
        txt = r.get("text")
        if isinstance(cid, str) and cid and isinstance(txt, str):
            m[cid] = txt
    return m


# =========================
# CONFIDENCE (UI only – COMMENTED OUT, but function kept for history)
# =========================
def compute_confidence(top_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Confidence score is still calculated and saved, but NOT displayed."""
    if not top_chunks:
        return {"confidence": 0, "reason": "no_chunks"}

    scores = np.array([float(c.get("score", 0.0)) for c in top_chunks], dtype=np.float32)
    s1 = float(scores[0])
    s2 = float(scores[1]) if len(scores) > 1 else (s1 - 0.15)

    level = np.clip((s1 - 0.20) / 0.55, 0.0, 1.0)
    margin = np.clip((s1 - s2) / 0.20, 0.0, 1.0)

    good = (scores >= max(0.30, s1 - 0.10)).sum()
    conc = np.clip(good / max(1, len(scores)), 0.0, 1.0)

    srcs = [str(c.get("scan_source") or "") for c in top_chunks]
    srcs = [s for s in srcs if s]
    agree = 0.0
    if srcs:
        most = max(srcs.count(x) for x in set(srcs))
        agree = np.clip(most / len(srcs), 0.0, 1.0)

    conf01 = 0.55 * level + 0.25 * margin + 0.10 * conc + 0.10 * agree
    conf = int(round(float(np.clip(conf01, 0.0, 1.0) * 100)))

    return {
        "confidence": conf,
        "top_score": s1,
        "second_score": s2,
        "level": float(level),
        "margin": float(margin),
        "concentration": float(conc),
        "source_agreement": float(agree),
    }


# =========================
# CONTEXT BUILD – PLAIN CONCATENATION (EXACTLY LIKE CLI)
# =========================
def build_context(top_chunks: List[Dict[str, Any]]) -> str:
    """Join the first 5 chunk texts with newlines – identical to original script."""
    texts = [c["text"] for c in top_chunks if c.get("text")]
    return "\n".join(texts[:5])


# =========================
# OLLAMA – WITH SEED FOR DETERMINISM
# =========================
def ollama_generate(prompt: str) -> str:
    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.15,
            "top_p": 0.5,
            "repeat_penalty": 1.2,
            "num_ctx": int(OLLAMA_NUM_CTX),
            "num_predict": int(OLLAMA_NUM_PREDICT),
            "seed": 42,                     # Makes generation deterministic
        },
    }
    r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT_S)
    r.raise_for_status()
    return (r.json().get("response") or "").strip()


# =========================
# LOAD RESOURCES (CACHED)
# =========================
@st.cache_resource(show_spinner=True)
def load_rag_resources(project_root: Path):
    emb_dir = project_root / "embeddings"
    chunks_dir = project_root / "chunks"

    index_path = emb_dir / "chunks.faiss"
    meta_path = emb_dir / "chunks_meta.jsonl"
    chunks_path = chunks_dir / "all_chunks.jsonl"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta JSONL not found: {meta_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks JSONL not found: {chunks_path}")

    index = faiss.read_index(str(index_path))
    meta_rows = list(iter_jsonl(meta_path))
    chunk_text = load_chunks_text_map(chunks_path)
    emb_model = SentenceTransformer(EMBED_MODEL_NAME)

    return {
        "index": index,
        "meta_rows": meta_rows,
        "chunk_text": chunk_text,
        "emb_model": emb_model,
        "paths": {"index": index_path, "meta": meta_path, "chunks": chunks_path},
    }


# =========================
# SEARCH – TOP_K = 12, FINAL_K = 5 (EXACT CLI)
# =========================
def retrieve_chunks(resources: Dict[str, Any], question: str):
    index = resources["index"]
    meta_rows = resources["meta_rows"]
    chunk_text = resources["chunk_text"]
    emb_model = resources["emb_model"]

    q_emb = emb_model.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(q_emb, TOP_K)

    candidates: List[Dict[str, Any]] = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(meta_rows):
            continue
        m = meta_rows[idx]
        cid = m.get("chunk_id")
        candidates.append(
            {
                "score": float(score),
                "chunk_id": cid,
                "doc_id": m.get("doc_id"),
                "title": m.get("title"),
                "author": m.get("author"),
                "publish_year": m.get("publish_year"),
                "publisher": m.get("publisher"),
                "scan_source": m.get("scan_source"),
                "source_url": m.get("source_url"),
                "page_number": m.get("page_number"),
                "text": chunk_text.get(cid, ""),
            }
        )

    filtered = [c for c in candidates if isinstance(c.get("text"), str) and c["text"].strip()]
    return filtered[:FINAL_K], candidates


# =========================
# UI – LIGHT BACKGROUND, CONFIDENCE REMOVED, PORTRAIT ENHANCED
# =========================
def inject_css():
    st.markdown(
        """
        <style>
          /* ---------- LIGHT, WARM BACKGROUND (matches chat area) ---------- */
          .stApp {
            background: #f9f4e9;  /* soft beige – same family as right column */
          }

          header[data-testid="stHeader"] { background: transparent; }
          footer { visibility: hidden; }

          /* ---------- TOP BANNER (kept dark for contrast) ---------- */
          .lincoln-top {
            background: linear-gradient(180deg, #3b6b8a 0%, #2f556f 100%);
            border: 1px solid rgba(0,0,0,0.35);
            border-radius: 10px;
            padding: 14px 18px;
            margin: 6px 0 12px 0;
            box-shadow: 0 10px 22px rgba(0,0,0,0.25);
            color: #f5e8cf;
            text-align:center;
            font-family: Georgia, 'Times New Roman', serif;
            font-weight: 900;
            letter-spacing: 1px;
            text-transform: uppercase;
            font-size: 26px;
          }

          /* ---------- LEFT PANEL (portrait area) – subtle texture ---------- */
          div[data-testid="stHorizontalBlock"] > div:nth-child(1) div[data-testid="stVerticalBlock"]{
            background: rgba(200,180,150,0.2);  /* light leather tint */
            border: 1px solid rgba(0,0,0,0.2);
            border-radius: 10px;
            padding: 14px;
            box-shadow: 0 10px 22px rgba(0,0,0,0.1);
          }

          /* ---------- RIGHT PANEL (chat area) – creamy paper ---------- */
          div[data-testid="stHorizontalBlock"] > div:nth-child(2) div[data-testid="stVerticalBlock"]{
            background: #fcf8f0;  /* warm off-white */
            border: 1px solid rgba(0,0,0,0.18);
            border-radius: 10px;
            padding: 16px;
            box-shadow: 0 10px 22px rgba(0,0,0,0.12);
          }

          .leftTitle {
            font-family: Georgia, 'Times New Roman', serif;
            font-size: 22px;
            font-weight: 900;
            color: #3e2e1f;  /* dark brown */
            margin-top: 12px;
          }
          .leftSub {
            color: #5e4e3f;
            font-size: 13px;
            margin-top: 2px;
            margin-bottom: 6px;
          }

          /* ---------- CHAT BUBBLES (unchanged) ---------- */
          .row { display:flex; gap:10px; align-items:flex-start; margin: 8px 0; }
          .bubble { color:#000 !important; }

          .avatar {
            width: 34px; height: 34px; border-radius: 50%;
            background: rgba(0,0,0,0.08);
            border: 1px solid rgba(0,0,0,0.18);
            display:flex; align-items:center; justify-content:center;
            font-family: Georgia, serif; font-weight: 900; color: rgba(0,0,0,0.7);
            flex: 0 0 auto;
          }
          .bubble {
            border-radius: 12px;
            padding: 10px 12px;
            border: 1px solid rgba(0,0,0,0.18);
            box-shadow: 0 6px 12px rgba(0,0,0,0.08);
            line-height: 1.35;
            max-width: 92%;
            word-wrap: break-word;
          }
          .assistant { background: rgba(255,255,255,0.9); }  /* slightly more opaque */
          .user { background: rgba(220,235,245,0.85); margin-left:auto; }

          .metaLine { color: rgba(0,0,0,0.6) !important; font-size: 12px; margin-top: 4px; }
          .divider { border-top: 1px solid rgba(0,0,0,0.12); margin: 12px 0; }

          /* ---------- BUTTON ---------- */
          .stButton > button {
            background: #3b6b8a;
            color: #ffffff;
            border: 1px solid rgba(0,0,0,0.35);
            border-radius: 10px;
            padding: 10px 16px;
            font-weight: 800;
          }
          .stButton > button:hover { background: #2f556f; color: #fff; }

          /* ---------- FORM / INPUT – clean, light ---------- */
          div[data-testid="stForm"], div[data-testid="stForm"] > div {
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
            margin: 0 !important;
            box-shadow: none !important;
          }
          div[data-testid="stHorizontalBlock"] div[data-testid="column"] {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
          }

          div[data-testid="stTextInput"] input {
            width: 100% !important;
            background: rgba(255,255,255,0.9) !important;
            border: 1px solid rgba(0,0,0,0.18) !important;
            border-radius: 10px !important;
            padding: 14px 14px !important;
            color: #1e1e1e !important;
            box-shadow: none !important;
          }
          div[data-testid="stTextInput"] input::placeholder { color: rgba(0,0,0,0.45) !important; }

          /* ---------- TYPING ANIMATION ---------- */
          .typing {
            display: inline-flex;
            gap: 6px;
            align-items: center;
            padding: 10px 12px;
            border-radius: 12px;
            border: 1px solid rgba(0,0,0,0.18);
            box-shadow: 0 6px 12px rgba(0,0,0,0.08);
            background: rgba(255,255,255,0.9);
          }
          .dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: rgba(0,0,0,0.45);
            animation: blink 1.2s infinite;
          }
          .dot:nth-child(2){ animation-delay: 0.2s; }
          .dot:nth-child(3){ animation-delay: 0.4s; }

          @keyframes blink {
            0%, 80%, 100% { opacity: 0.25; transform: translateY(0px); }
            40% { opacity: 1; transform: translateY(-2px); }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_topbar():
    st.markdown('<div class="lincoln-top">THE ABRAHAM LINCOLN CHATBOT</div>', unsafe_allow_html=True)


def show_portrait():
    """Display portrait with a fallback if file is missing."""
    p = Path(PORTRAIT_PATH)
    if p.exists():
        st.image(str(p), use_container_width=True)
    else:
        # Show a dignified placeholder instead of a raw warning
        st.markdown(
            """
            <div style="background: rgba(0,0,0,0.04); border-radius: 8px; padding: 20px; text-align: center; color: #5e4e3f; font-family: Georgia, serif;">
                <span style="font-size: 48px;">🎩</span><br>
                <span style="font-size: 16px;"><i>Portrait of Abraham Lincoln</i></span><br>
                <span style="font-size: 12px;">(Placeholder – image not found)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_chat_bubble(role: str, text: str, confidence: Optional[int] = None):
    """Confidence is accepted but NOT displayed – exactly as requested."""
    if role == "user":
        st.markdown(
            f'<div class="row" style="justify-content:flex-end;">'
            f'  <div class="bubble user"><b>You</b>&nbsp;&nbsp; {text}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        # ⚠️ Confidence intentionally omitted from display
        st.markdown(
            f'<div class="row">'
            f'  <div class="avatar">AL</div>'
            f'  <div>'
            f'    <div class="bubble assistant">{text}</div>'
            f'    <!-- Confidence hidden as per request -->'
            f'  </div>'
            f"</div>",
            unsafe_allow_html=True,
        )


def render_typing_indicator():
    st.markdown(
        """
        <div class="row">
          <div class="avatar">AL</div>
          <div class="typing" aria-label="typing">
            <span class="dot"></span><span class="dot"></span><span class="dot"></span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def autoscroll_inside_chat():
    st.markdown(
        """
        <script>
          (function() {
            const root = window.parent.document;
            const el = root.getElementById('chat-scroll-anchor');
            if (el) el.scrollIntoView({behavior:'smooth', block:'end'});
          })();
        </script>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="The Abraham Lincoln Chatbot", layout="wide")
    inject_css()
    render_topbar()

    script_dir = Path(__file__).resolve().parent
    project_root = find_project_root(script_dir)
    history_path = project_root / "rag_history" / "rag_history.jsonl"

    try:
        resources = load_rag_resources(project_root)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # state
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "show_context" not in st.session_state:
        st.session_state.show_context = False
    if "is_typing" not in st.session_state:
        st.session_state.is_typing = False
    if "pending_user_q" not in st.session_state:
        st.session_state.pending_user_q = ""

    left, right = st.columns([1, 2.2], gap="large")

    with left:
        show_portrait()  # now with fallback
        st.markdown('<div class="leftTitle">Chat with Abraham Lincoln</div>', unsafe_allow_html=True)
        st.markdown('<div class="leftSub">Ask me anything about my life and times.</div>', unsafe_allow_html=True)

    with right:
        chat_area = st.container(height=CHAT_HEIGHT_PX, border=False)

        with chat_area:
            if not st.session_state.chat:
                render_chat_bubble("assistant", "Hello there! I am Abraham Lincoln. How can I assist you today?")

            for msg in st.session_state.chat[-200:]:
                if msg["role"] == "user":
                    render_chat_bubble("user", msg["text"])
                else:
                    # Confidence is still stored in msg but not shown
                    render_chat_bubble("assistant", msg["text"])

            if st.session_state.is_typing:
                render_typing_indicator()

            st.markdown('<div id="chat-scroll-anchor"></div>', unsafe_allow_html=True)

        autoscroll_inside_chat()

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        st.session_state.show_context = st.checkbox(
            "Show retrieved context (inside Sources)",
            value=st.session_state.show_context,
        )

        with st.form("chat_form", clear_on_submit=True):
            in_col, send_col = st.columns([5, 1])
            with in_col:
                question = st.text_input(
                    "Type your message...",
                    value="",
                    label_visibility="collapsed",
                    placeholder="Type your message...",
                )
            with send_col:
                send = st.form_submit_button("Send", use_container_width=True)

        if send:
            user_q = (question or "").strip()
            if not user_q:
                st.warning("Please type a message.")
                st.stop()

            st.session_state.chat.append({"role": "user", "text": user_q})
            st.session_state.pending_user_q = user_q
            st.session_state.is_typing = True
            st.rerun()

    # ================== RAG GENERATION (EXACT CLI LOGIC) ==================
    if st.session_state.is_typing:
        user_q = (st.session_state.pending_user_q or "").strip()
        if not user_q:
            st.session_state.is_typing = False
            st.stop()

        with st.spinner("Retrieving relevant chunks..."):
            top_chunks, _ = retrieve_chunks(resources, user_q)

        if not top_chunks:
            answer = "Not found in provided documents."
            st.session_state.chat.append({"role": "assistant", "text": answer, "confidence": 0, "sources": []})
            write_jsonl_line(
                history_path,
                {"ts": utc_now_iso(), "question": user_q, "answer": answer, "confidence": 0, "sources": []},
            )
            st.session_state.is_typing = False
            st.session_state.pending_user_q = ""
            st.rerun()

        conf = compute_confidence(top_chunks)   # still computed, saved, but not shown

        # Build context: plain concatenation of first 5 chunk texts
        context = build_context(top_chunks)

        # Build prompt: exactly like the CLI script
        prompt = f"""
{IDENTITY_LOCK}

{SYSTEM_PROMPT}

CONTEXT:
{context}

QUESTION:
{user_q}

ANSWER:
"""

        with st.spinner("Generating answer with Ollama..."):
            try:
                answer = ollama_generate(prompt)
            except Exception as e:
                answer = f"Ollama error: {type(e).__name__}: {str(e)[:300]}"
                conf = {"confidence": 0}

        st.session_state.chat.append(
            {"role": "assistant", "text": answer, "confidence": conf.get("confidence", 0), "sources": top_chunks}
        )

        write_jsonl_line(
            history_path,
            {
                "ts": utc_now_iso(),
                "question": user_q,
                "answer": answer,
                "confidence": conf.get("confidence", 0),
                "confidence_details": conf,
                "sources": [
                    {
                        "score": c.get("score"),
                        "chunk_id": c.get("chunk_id"),
                        "doc_id": c.get("doc_id"),
                        "title": c.get("title"),
                        "author": c.get("author"),
                        "publish_year": c.get("publish_year"),
                        "publisher": c.get("publisher"),
                        "scan_source": c.get("scan_source"),
                        "source_url": c.get("source_url"),
                        "page_number": c.get("page_number"),
                    }
                    for c in top_chunks
                ],
            },
        )

        st.session_state.is_typing = False
        st.session_state.pending_user_q = ""
        st.rerun()

    # ================== SOURCES DISPLAY (unchanged) ==================
    last = None
    for m in reversed(st.session_state.chat):
        if m.get("role") == "assistant" and m.get("sources"):
            last = m
            break

    if last and last.get("sources"):
        st.markdown("### 📌 Sources (real)")
        for i, c in enumerate(last["sources"], start=1):
            with st.expander(
                f"{i}. score={c['score']:.4f} | {c.get('scan_source')} | page={c.get('page_number')}"
            ):
                st.write(f"**Title:** {c.get('title')}")
                st.write(f"**Author:** {c.get('author')}")
                st.write(f"**Year:** {c.get('publish_year')}")
                st.write(f"**Publisher:** {c.get('publisher')}")
                st.write(f"**Chunk ID:** `{c.get('chunk_id')}`")
                st.write(f"**Source URL:** {c.get('source_url')}")
                if st.session_state.show_context:
                    st.divider()
                    st.write(c.get("text", ""))


if __name__ == "__main__":
    main()