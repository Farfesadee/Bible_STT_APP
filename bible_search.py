import json
import numpy as np
import re
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------

EMBEDDING_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"
  # 🔒 local-only

model = SentenceTransformer(EMBEDDING_MODEL_PATH)

TOP_K = 5

CONFIDENCE_MIN_SCORE = 0.60
CONFIDENCE_MARGIN = 0.08

MAX_SCORE_CLAMP = 1.25  # safety

# --------------------------------------

print("Loading Bible index...")

embeddings = np.load("bible/kjv_embeddings.npy")

with open("bible/kjv_metadata.json", "r", encoding="utf-8") as f:
    verses = json.load(f)

# 🔒 Load embedding model (OFFLINE ONLY)
model = SentenceTransformer(
    EMBEDDING_MODEL_PATH,
    local_files_only=True
)

# ---------------- NORMALIZATION ----------------

def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 :]", "", text.lower()).strip()

# Pre-normalize verse text ONCE
for v in verses:
    v["_norm"] = normalize(v["text"])

# ---------------- EXPLICIT REFERENCE ----------------

EXPLICIT_REF_RE = re.compile(
    r"(?P<book>[1-3]?\s?[a-z]+)\s+(?P<chapter>\d+)\s*:\s*(?P<verse>\d+)",
    re.IGNORECASE
)

BOOK_ALIASES = {
    "psalm": "Psalms",
    "psalms": "Psalms",
    "john": "John",
}

def parse_explicit_reference(text: str):
    match = EXPLICIT_REF_RE.search(text)
    if not match:
        return None

    book = match.group("book").lower()
    book = BOOK_ALIASES.get(book, book.title())

    return {
        "book": book,
        "chapter": int(match.group("chapter")),
        "verse": int(match.group("verse")),
    }

# ---------------- SEARCH ----------------

def search_bible(query: str, top_k: int = TOP_K):
    query_norm = normalize(query)
    if len(query_norm) < 3:
        return []

    # 🔹 Explicit reference short-circuit
    ref = parse_explicit_reference(query_norm)
    if ref:
        key = (ref["book"], ref["chapter"], ref["verse"])
        for v in verses:
            if (v["book"], v["chapter"], v["verse"]) == key:
                return [{
                    **v,
                    "score": 1.0
                }]

    query_vec = model.encode(
        [query_norm],
        normalize_embeddings=True
    )[0]

    semantic_scores = embeddings @ query_vec
    final_scores = np.zeros(len(verses), dtype=np.float32)

    words = query_norm.split()

    for i, verse in enumerate(verses):
        score = float(semantic_scores[i])
        verse_norm = verse["_norm"]

        # lexical overlap
        overlap = sum(1 for w in words if w in verse_norm)
        score += overlap * 0.03

        # exact phrase
        if query_norm in verse_norm:
            score += 0.15

        # Psalm 23 shepherd contextual bias
        if (
            "shepherd" in query_norm
            and verse["book"] == "Psalms"
            and verse["chapter"] == 23
        ):
            score += 0.10

        final_scores[i] = min(score, MAX_SCORE_CLAMP)

    top_indices = np.argsort(final_scores)[::-1][:top_k]

    return [{
        "book": verses[i]["book"],
        "chapter": verses[i]["chapter"],
        "verse": verses[i]["verse"],
        "text": verses[i]["text"],
        "translation": verses[i]["translation"],
        "score": round(float(final_scores[i]), 3)
    } for i in top_indices]

# ---------------- VERSE CONTINUATION ----------------

VERSE_INDEX = {
    (v["book"], v["chapter"], v["verse"]): v
    for v in verses
}

def get_next_verses(book, chapter, verse, max_verses=5):
    results = []
    current = verse

    for _ in range(max_verses):
        key = (book, chapter, current)
        if key not in VERSE_INDEX:
            break
        results.append(VERSE_INDEX[key])
        current += 1

    return results

# ---------------- CONFIDENCE GATE ----------------

def is_confident_match(results) -> bool:
    if len(results) < 2:
        return False

    top = results[0]["score"]
    second = results[1]["score"]

    if top < CONFIDENCE_MIN_SCORE:
        return False

    if (top - second) < CONFIDENCE_MARGIN:
        return False

    return True

# ---------------- STABILITY HELPER ----------------

def dominant_result(results, min_margin=CONFIDENCE_MARGIN):
    if len(results) < 2:
        return None
    if results[0]["score"] - results[1]["score"] >= min_margin:
        return results[0]
    return None


def get_verse_by_reference(book, chapter, verse, translation):
    for v in verses:
        if (
            v["book"] == book
            and v["chapter"] == chapter
            and v["verse"] == verse
            and v["translation"].lower() == translation.lower()
        ):
            return v
    return None
