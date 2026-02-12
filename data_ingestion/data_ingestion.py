#!/usr/bin/env python3
"""
Unified Data Ingestion Pipeline for Abraham Lincoln RAG

Accepts:
  --input       : file or folder path (absolute, or relative to project root)
  --run_name    : output subdirectory under PROJECT_ROOT/data/
  --scan_source : source description (e.g., "Project Gutenberg")

Outputs (appended):
  PROJECT_ROOT/data/<run_name>/universal_pages.jsonl
  PROJECT_ROOT/data/<run_name>/universal_metadata.json
"""

import re
import json
import argparse
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import xml.etree.ElementTree as ET
from html.parser import HTMLParser
import html as ihtml

# =========================
# PROJECT ROOT PATHS
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent          # ABRAHAM_LINCOLN_FINAL
BASE_DATA_DIR = PROJECT_ROOT / "data"
BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# CLEANING CONFIG
# =========================
DEFAULT_AUTHOR = "Unknown"
DEFAULT_PUBLISHER = "Unknown"

# ⚠️  Relax this regex if your texts contain many non‑ASCII characters
STRICT_ALLOWED_RE = re.compile(r"[^A-Za-z0-9\s\.\,\;\:\?\!\'\"\-]")

WHITESPACE_RE = re.compile(r"\s+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
CID_RE = re.compile(r"\(cid:\d+\)")
CTRL_RE = re.compile(r"[\x00-\x1f]")
HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\s*(?:\r?\n)+\s*(\w)")
XML_DECL_RE = re.compile(r"^\s*<\?xml[^>]*\?>", flags=re.IGNORECASE)
TEXT_BLOCK_RE = re.compile(r"<text\b[^>]*>.*?</text>", flags=re.IGNORECASE | re.DOTALL)
PAGEINFO_RE = re.compile(r"<pageinfo\b[^>]*>.*?</pageinfo>", flags=re.IGNORECASE | re.DOTALL)
CONTROLPGNO_RE = re.compile(r"<controlpgno\b[^>]*>(.*?)</controlpgno>", flags=re.IGNORECASE | re.DOTALL)
PB_RE = re.compile(r"<pb\b[^>]*/>|<pb\b[^>]*>.*?</pb>", flags=re.IGNORECASE | re.DOTALL)
PAGE_MARK_RE = re.compile(r"<<<PAGE:([^>]*)>>>")

UNICODE_MAP = {
    "\u201c": '"', "\u201d": '"',
    "\u2018": "'", "\u2019": "'",
    "\u2014": "-", "\u2013": "-",
    "\u00a0": " ", "\u200b": "", "\ufeff": ""
}


# =========================
# UTILITIES
# =========================
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def safe_int_year(value: Any) -> Optional[int]:
    if value is None:
        return None
    s = str(value)
    m = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", s)
    return int(m.group(1)) if m else None

def md5_12(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()[:12]

def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag

def file_to_source_url(path: Path) -> str:
    return path.resolve().as_uri()

def read_text_file(path: Path) -> str:
    """Try multiple encodings before giving up."""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'utf-16']
    for enc in encodings:
        try:
            return path.read_text(encoding=enc, errors='replace')
        except UnicodeDecodeError:
            continue
    # Last resort: ignore all errors
    return path.read_text(encoding='latin-1', errors='ignore')

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def first_nonempty(items: List[str]) -> Optional[str]:
    for x in items:
        if isinstance(x, str) and x.strip():
            return x.strip()
    return None

def detect_author_from_from_line(text: str) -> Optional[str]:
    m = re.search(r"\bFrom\s+(.+?)\s+to\s+(.+?)(?:,|\s+)\s", text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else None


# =========================
# STRICT CLEAN TEXT
# =========================
def clean_text_strict(text: str) -> str:
    if not text:
        return ""

    for k, v in UNICODE_MAP.items():
        text = text.replace(k, v)

    text = CID_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = CTRL_RE.sub(" ", text)
    text = HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)

    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = STRICT_ALLOWED_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


# =========================
# HTML -> TEXT
# =========================
class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: List[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in ("script", "style", "noscript"):
            self._skip = True
        if tag in ("p", "br", "hr", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in ("script", "style", "noscript"):
            self._skip = False
        if tag in ("p", "div", "li", "tr"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip and data:
            self.parts.append(data)

def html_to_text(html_str: str) -> str:
    if not html_str:
        return ""
    parser = _HTMLTextExtractor()
    parser.feed(html_str)
    return ihtml.unescape("".join(parser.parts))


# =========================
# TXT SPLIT
# =========================
def split_txt_into_pages(raw: str) -> List[str]:
    if not raw:
        return []
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    if "\f" in raw:
        pages = raw.split("\f")
    else:
        pages = [raw]
    return [p for p in pages if p and p.strip()]


# =========================
# TEI XML SPLIT (pageinfo + pb)
# =========================
def _extract_digits(s: str) -> Optional[int]:
    if not s:
        return None
    m = re.search(r"\b(\d{1,6})\b", s)
    return int(m.group(1)) if m else None

def extract_text_from_xml_fragment(fragment: str) -> str:
    frag = fragment.strip()
    if not frag:
        return ""
    wrapped = f"<root>{frag}</root>"
    try:
        root = ET.fromstring(wrapped)
        return " ".join(x.strip() for x in root.itertext() if x and x.strip())
    except Exception:
        return HTML_TAG_RE.sub(" ", fragment)

def split_tei_xml_to_pages(raw_xml: str) -> List[Tuple[int, str]]:
    if not raw_xml:
        return []

    xml_wo_decl = XML_DECL_RE.sub("", raw_xml).strip()
    m_text = TEXT_BLOCK_RE.search(xml_wo_decl)
    text_block = m_text.group(0) if m_text else xml_wo_decl

    def repl_pageinfo(m):
        block = m.group(0)
        c = CONTROLPGNO_RE.search(block)
        pg = c.group(1).strip() if c else ""
        pg_digits = _extract_digits(pg)
        if pg_digits is None:
            return "\n<<<PAGE:>>>\n"
        return f"\n<<<PAGE:{pg_digits}>>>\n"

    text_block = PAGEINFO_RE.sub(repl_pageinfo, text_block)

    def repl_pb(m):
        tag = m.group(0)
        n = None
        n_m = re.search(r'\bn\s*=\s*"([^"]+)"', tag, flags=re.IGNORECASE)
        if n_m:
            n = _extract_digits(n_m.group(1))
        if n is None:
            facs_m = re.search(r'\bfacs\s*=\s*"([^"]+)"', tag, flags=re.IGNORECASE)
            if facs_m:
                n = _extract_digits(facs_m.group(1))
        if n is None:
            xid_m = re.search(r'\bxml:id\s*=\s*"([^"]+)"', tag, flags=re.IGNORECASE)
            if xid_m:
                n = _extract_digits(xid_m.group(1))
        if n is None:
            return "\n<<<PAGE:>>>\n"
        return f"\n<<<PAGE:{n}>>>\n"

    text_block = PB_RE.sub(repl_pb, text_block)

    parts = PAGE_MARK_RE.split(text_block)
    pages: List[Tuple[int, str]] = []
    current_page = 1

    if parts and parts[0].strip():
        pages.append((current_page, parts[0]))

    i = 1
    while i < len(parts):
        marker = (parts[i] or "").strip()
        chunk = parts[i + 1] if (i + 1) < len(parts) else ""

        if marker:
            try:
                current_page = int(marker)
            except Exception:
                current_page += 1
        else:
            current_page += 1

        if chunk and chunk.strip():
            pages.append((current_page, chunk))

        i += 2

    return pages


# =========================
# METADATA EXTRACTION
# =========================
def meta_from_filename(path: Path) -> Dict[str, Any]:
    return {
        "book_title": path.stem,
        "author_name": DEFAULT_AUTHOR,
        "publish_year": None,
        "publisher": DEFAULT_PUBLISHER,
    }

def meta_from_json_obj(obj: Any, fallback_title: str) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {
            "book_title": fallback_title,
            "author_name": DEFAULT_AUTHOR,
            "publish_year": None,
            "publisher": DEFAULT_PUBLISHER,
        }

    title = obj.get("book_title") or obj.get("title") or obj.get("document_title") or obj.get("heading") or obj.get("name") or fallback_title
    author = obj.get("author_name") or obj.get("author") or obj.get("creator") or obj.get("by") or DEFAULT_AUTHOR
    publisher = obj.get("publisher") or DEFAULT_PUBLISHER
    year = safe_int_year(obj.get("publish_year")) or safe_int_year(obj.get("date")) or safe_int_year(obj.get("year")) or safe_int_year((obj.get("metadata") or {}).get("date"))

    return {
        "book_title": str(title) if title else fallback_title,
        "author_name": str(author) if author else DEFAULT_AUTHOR,
        "publish_year": year,
        "publisher": str(publisher) if publisher else DEFAULT_PUBLISHER,
    }

def meta_from_xml_root(root: ET.Element, fallback_title: str) -> Dict[str, Any]:
    titles, dates, his = [], [], []
    for el in root.iter():
        name = strip_ns(el.tag).lower()
        txt = (el.text or "").strip()
        if not txt:
            continue
        if name in ("title", "head", "doctitle"):
            titles.append(txt)
        if name in ("date", "encodingdate"):
            dates.append(txt)
        if name in ("hi",):
            his.append(txt)

    title = first_nonempty(titles) or fallback_title
    year = None
    for d in dates:
        y = safe_int_year(d)
        if y:
            year = y
            break
    author = DEFAULT_AUTHOR
    for h in his[:30]:
        a = detect_author_from_from_line(h)
        if a:
            author = a
            break

    return {
        "book_title": title,
        "author_name": author,
        "publish_year": year,
        "publisher": DEFAULT_PUBLISHER,
    }


# =========================
# CONTENT EXTRACTION
# =========================
def extract_text_from_json_obj(obj: Any) -> str:
    if isinstance(obj, dict):
        content = obj.get("content")
        if isinstance(content, dict):
            ft = content.get("full_text")
            if isinstance(ft, str) and ft.strip():
                return ft
            paras = content.get("paragraphs")
            if isinstance(paras, list):
                return " ".join(p for p in paras if isinstance(p, str))

        for k in ("full_text", "text", "body", "content", "ocr_text"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v

        v = obj.get("html")
        if isinstance(v, str) and v.strip():
            return html_to_text(v)

    return ""

def guess_url_from_obj(obj: Any, fallback: str) -> str:
    if isinstance(obj, dict):
        for k in ("url", "source_url", "link", "source"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return fallback


# =========================
# INGESTORS (unchanged logic)
# =========================
def ingest_txt(path: Path, scan_source: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw = read_text_file(path)
    pages = split_txt_into_pages(raw)

    identifier = f"txt_{path.stem}_{md5_12(str(path.resolve()))}"
    source_url = file_to_source_url(path)
    meta = meta_from_filename(path)

    meta_entry = {**meta, "identifier": identifier, "source_url": source_url, "scan_source": scan_source}

    records = []
    for pno, p in enumerate(pages, start=1):
        cleaned = clean_text_strict(p)
        if not cleaned:
            continue
        records.append({
            "book_title": meta["book_title"],
            "author_name": meta["author_name"],
            "page_number": pno,
            "ocr_text": cleaned,
            "publish_year": meta["publish_year"],
            "publisher": meta["publisher"],
            "scan_source": scan_source,
            "identifier": identifier,
            "source_url": source_url,
            "original_stream_url": source_url
        })
    return records, meta_entry

def ingest_html(path: Path, scan_source: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw = read_text_file(path)
    text = html_to_text(raw)

    identifier = f"html_{path.stem}_{md5_12(str(path.resolve()))}"
    source_url = file_to_source_url(path)

    title = path.stem
    m = re.search(r"<title[^>]*>(.*?)</title>", raw, flags=re.IGNORECASE | re.DOTALL)
    if m:
        title_candidate = clean_text_strict(ihtml.unescape(m.group(1)))
        if title_candidate:
            title = title_candidate

    meta = {"book_title": title, "author_name": DEFAULT_AUTHOR, "publish_year": None, "publisher": DEFAULT_PUBLISHER}
    meta_entry = {**meta, "identifier": identifier, "source_url": source_url, "scan_source": scan_source}

    cleaned = clean_text_strict(text)
    records = []
    if cleaned:
        records.append({
            "book_title": meta["book_title"],
            "author_name": meta["author_name"],
            "page_number": 1,
            "ocr_text": cleaned,
            "publish_year": meta["publish_year"],
            "publisher": meta["publisher"],
            "scan_source": scan_source,
            "identifier": identifier,
            "source_url": source_url,
            "original_stream_url": source_url
        })
    return records, meta_entry

def ingest_xml(path: Path, scan_source: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw_xml = read_text_file(path)

    identifier = f"xml_{path.stem}_{md5_12(str(path.resolve()))}"
    source_url = file_to_source_url(path)

    try:
        root = ET.parse(path).getroot()
        meta = meta_from_xml_root(root, fallback_title=path.stem)
    except Exception:
        meta = meta_from_filename(path)

    meta_entry = {**meta, "identifier": identifier, "source_url": source_url, "scan_source": scan_source}

    page_chunks = split_tei_xml_to_pages(raw_xml)

    records = []
    for page_number, chunk_xml in page_chunks:
        extracted = extract_text_from_xml_fragment(chunk_xml)
        cleaned = clean_text_strict(extracted)
        if not cleaned:
            continue

        records.append({
            "book_title": meta["book_title"],
            "author_name": meta["author_name"],
            "page_number": page_number,
            "ocr_text": cleaned,
            "publish_year": meta["publish_year"],
            "publisher": meta["publisher"],
            "scan_source": scan_source,
            "identifier": identifier,
            "source_url": source_url,
            "original_stream_url": source_url
        })
    return records, meta_entry

def _iter_json_rows(obj: Any) -> List[Any]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ("documents", "items", "data", "rows"):
            v = obj.get(k)
            if isinstance(v, list):
                return v
        return [obj]
    return []

def ingest_json(path: Path, scan_source: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw = read_text_file(path)
    source_url_file = file_to_source_url(path)

    docs: List[Any] = []
    if path.suffix.lower() == ".jsonl":
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except Exception:
                continue
    else:
        try:
            parsed = json.loads(raw)
            docs = _iter_json_rows(parsed)
        except Exception:
            docs = []

    records: List[Dict[str, Any]] = []
    meta_entries: Dict[str, Any] = {}

    for idx, doc in enumerate(docs, start=1):
        doc_url = guess_url_from_obj(doc, fallback=source_url_file)
        doc_id = None
        if isinstance(doc, dict):
            doc_id = doc.get("doc_id") or doc.get("document_id") or doc.get("id")
        if doc_id:
            identifier = f"json_{doc_id}"
        else:
            identifier = f"json_{path.stem}_{idx}_{md5_12(doc_url)}"

        meta = meta_from_json_obj(doc if isinstance(doc, dict) else {}, fallback_title=path.stem)

        text_raw = extract_text_from_json_obj(doc)
        if (not meta["author_name"] or meta["author_name"] == DEFAULT_AUTHOR) and text_raw:
            a = detect_author_from_from_line(text_raw)
            if a:
                meta["author_name"] = a

        meta_entries[identifier] = {**meta, "identifier": identifier, "source_url": doc_url, "scan_source": scan_source}

        cleaned = clean_text_strict(text_raw)
        if not cleaned:
            continue

        records.append({
            "book_title": meta["book_title"],
            "author_name": meta["author_name"],
            "page_number": 1,
            "ocr_text": cleaned,
            "publish_year": meta["publish_year"],
            "publisher": meta["publisher"],
            "scan_source": scan_source,
            "identifier": identifier,
            "source_url": doc_url,
            "original_stream_url": doc_url
        })

    return records, meta_entries


# =========================
# DRIVER – ROBUST FILE DISCOVERY
# =========================
SUPPORTED_EXTS = {".txt", ".xml", ".html", ".htm", ".json", ".jsonl"}

def ingest_path(input_path: Path, scan_source: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records_all: List[Dict[str, Any]] = []
    meta_index: Dict[str, Any] = {}

    # --- Robust, case‑insensitive file discovery ---
    files: List[Path] = []
    if input_path.is_file():
        files = [input_path]
    else:
        # Recursively get all files, then filter by lowercase suffix
        all_files = list(input_path.rglob("*"))
        files = [f for f in all_files if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS]

    print(f"\n📁 Scanning: {input_path.resolve()}")
    print(f"🔍 Found {len(files)} supported files")
    if not files:
        print("⚠️  No supported files found. Check folder path and extensions.")
        return records_all, meta_index

    # --- Process each file with clear logging ---
    for idx, fp in enumerate(files, 1):
        ext = fp.suffix.lower()
        print(f"\n[{idx}/{len(files)}] Processing: {fp.name}  (ext: {ext})")
        try:
            if ext == ".txt":
                recs, meta_entry = ingest_txt(fp, scan_source)
                if recs:
                    meta_index[meta_entry["identifier"]] = meta_entry
                print(f"   ➕ TXT → {len(recs)} page records")
                records_all.extend(recs)

            elif ext == ".xml":
                recs, meta_entry = ingest_xml(fp, scan_source)
                if recs:
                    meta_index[meta_entry["identifier"]] = meta_entry
                print(f"   ➕ XML → {len(recs)} page records")
                records_all.extend(recs)

            elif ext in (".html", ".htm"):
                recs, meta_entry = ingest_html(fp, scan_source)
                if recs:
                    meta_index[meta_entry["identifier"]] = meta_entry
                print(f"   ➕ HTML → {len(recs)} page records")
                records_all.extend(recs)

            elif ext in (".json", ".jsonl"):
                recs, meta_entries = ingest_json(fp, scan_source)
                if meta_entries:
                    meta_index.update(meta_entries)
                print(f"   ➕ JSON → {len(recs)} document records")
                records_all.extend(recs)

            else:
                print(f"   ⏭️  Skipped (unsupported extension)")

        except Exception as e:
            print(f"   ❌ ERROR: {type(e).__name__}: {str(e)[:200]}")
            continue

    print(f"\n📊 Total records extracted: {len(records_all)}")
    return records_all, meta_index


# =========================
# MAIN – APPEND‑AWARE
# =========================
def main():
    ap = argparse.ArgumentParser(description="Universal data ingestion for RAG corpus")
    ap.add_argument("--input", required=True,
                    help="File or folder path (absolute, or relative to project root)")
    ap.add_argument("--run_name", required=True,
                    help='Output subdirectory under data/ (e.g., "gutenberg") – do NOT use a full path')
    ap.add_argument("--scan_source", default="Local Dump",
                    help="Source description (e.g., 'Project Gutenberg')")
    args = ap.parse_args()

    # --- SMART INPUT PATH RESOLUTION ---
    input_path = Path(args.input)
    if not input_path.exists():
        # Try relative to project root
        alt_path = PROJECT_ROOT / args.input
        if alt_path.exists():
            input_path = alt_path
            print(f"📁 Resolved input to: {input_path.resolve()}")
        else:
            print(f"❌ Input path does not exist: {input_path}")
            print(f"   Also tried: {alt_path.resolve()}")
            return

    # --- OUTPUT DIRECTORY ---
    # run_name must be a simple name; we place output under data/<run_name>
    # If user accidentally passes an absolute path, warn and use only the last part.
    run_name = args.run_name
    run_path = Path(run_name)
    if run_path.is_absolute():
        print(f"⚠️  run_name should be a simple name, not an absolute path.")
        print(f"   Using only the last component: {run_path.name}")
        run_name = run_path.name

    out_dir = BASE_DATA_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / "universal_pages.jsonl"
    out_meta = out_dir / "universal_metadata.json"

    # 1. Ingest the new input
    new_records, new_meta_index = ingest_path(input_path, args.scan_source)

    if not new_records:
        print("\n⚠️  No records extracted, nothing to append.")
        return

    # 2. Append new records to the JSONL file
    with out_jsonl.open("a" if out_jsonl.exists() else "w", encoding="utf-8") as f:
        for record in new_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 3. Count total records in the pages file (existing + new)
    with out_jsonl.open("r", encoding="utf-8") as f:
        total_records = sum(1 for _ in f)

    # 4. Load existing metadata if present
    existing_meta = {}
    if out_meta.exists():
        try:
            with out_meta.open("r", encoding="utf-8") as f:
                existing_meta = json.load(f)
        except Exception as e:
            print(f"⚠️ Could not read existing metadata, will overwrite: {e}")

    # 5. Merge metadata indexes (new entries overwrite old ones if same identifier)
    merged_meta_index = existing_meta.get("metadata_index", {})
    merged_meta_index.update(new_meta_index)

    # 6. Build final metadata object
    final_metadata = {
        "created_at": utc_now_iso(),
        "scan_source": args.scan_source,
        "documents": len(merged_meta_index),
        "records": total_records,
        "metadata_index": merged_meta_index,
        "note": "Universal ingest – appended. Cleaned text: no html, no special symbols, single spaces. XML TEI split by pageinfo/pb."
    }

    # 7. Write merged metadata
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(final_metadata, f, ensure_ascii=False, indent=2)

    print("\n✅ INGESTION COMPLETE (APPENDED)")
    print(f" Saved in: {out_dir.resolve()}")
    print(f" Pages   : {out_jsonl.resolve()} (now contains {total_records} records)")
    print(f" Metadata: {out_meta.resolve()}")
    print(f" Docs    : {len(merged_meta_index)}")
    print(f" Records : {total_records}")


if __name__ == "__main__":
    main()