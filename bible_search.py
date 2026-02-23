import json
import numpy as np
import re
import os
import sys
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------

EMBEDDING_MODEL_PATH = "models/all-MiniLM-L6-v2"
TOP_K = 5
CONFIDENCE_MIN_SCORE = 0.68
CONFIDENCE_MARGIN = 0.12
MAX_SCORE_CLAMP = 1.25

# --------------------------------------

print("Loading Bible index...")

embeddings = np.load("bible/kjv_embeddings.npy")

with open("bible/kjv_metadata.json", "r", encoding="utf-8") as f:
    verses = json.load(f)

required_files = [
    os.path.join(EMBEDDING_MODEL_PATH, "modules.json"),
    os.path.join(EMBEDDING_MODEL_PATH, "config.json"),
]

if not os.path.isdir(EMBEDDING_MODEL_PATH) or not all(os.path.isfile(p) for p in required_files):
    print(
        f"Missing local embedding model at '{EMBEDDING_MODEL_PATH}'.\n"
        "Download it once with:\n"
        "  from huggingface_hub import snapshot_download\n"
        "  snapshot_download(repo_id='sentence-transformers/all-MiniLM-L6-v2', local_dir='models/all-MiniLM-L6-v2')\n"
    )
    sys.exit(1)

model = SentenceTransformer(EMBEDDING_MODEL_PATH, local_files_only=True)

# ---------------- NORMALIZATION ----------------

def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 :]", "", text.lower()).strip()

for v in verses:
    v["_norm"] = normalize(v["text"])

# ---------------- SPOKEN NUMBER CONVERTER ----------------

SPOKEN_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90,
}

def spoken_to_digits(text: str) -> str:
    words = text.split()
    result = []
    i = 0
    while i < len(words):
        w = words[i]
        if w in SPOKEN_NUMBERS:
            num = SPOKEN_NUMBERS[w]
            # Handle compound like "twenty one" → 21
            if i + 1 < len(words) and words[i + 1] in SPOKEN_NUMBERS and SPOKEN_NUMBERS[words[i + 1]] < 10:
                num += SPOKEN_NUMBERS[words[i + 1]]
                i += 1
            result.append(str(num))
        else:
            result.append(w)
        i += 1
    return " ".join(result)

# ---------------- BOOK ALIASES ----------------

BOOK_ALIASES = {
    "gen": "Genesis", "genesis": "Genesis",
    "ex": "Exodus", "exo": "Exodus", "exodus": "Exodus",
    "lev": "Leviticus", "leviticus": "Leviticus",
    "num": "Numbers", "numbers": "Numbers",
    "deut": "Deuteronomy", "deuteronomy": "Deuteronomy",
    "josh": "Joshua", "joshua": "Joshua",
    "judg": "Judges", "judges": "Judges",
    "ruth": "Ruth",
    "1sam": "1 Samuel", "1 samuel": "1 Samuel",
    "2sam": "2 Samuel", "2 samuel": "2 Samuel",
    "1ki": "1 Kings", "1 kings": "1 Kings",
    "2ki": "2 Kings", "2 kings": "2 Kings",
    "1chr": "1 Chronicles", "1 chronicles": "1 Chronicles",
    "2chr": "2 Chronicles", "2 chronicles": "2 Chronicles",
    "ezra": "Ezra", "neh": "Nehemiah", "nehemiah": "Nehemiah",
    "esth": "Esther", "esther": "Esther",
    "job": "Job",
    "ps": "Psalms", "psalm": "Psalms", "psalms": "Psalms",
    "prov": "Proverbs", "proverbs": "Proverbs",
    "ecc": "Ecclesiastes", "eccl": "Ecclesiastes", "ecclesiastes": "Ecclesiastes",
    "song": "Song of Solomon", "sos": "Song of Solomon",
    "isa": "Isaiah", "isaiah": "Isaiah",
    "jer": "Jeremiah", "jeremiah": "Jeremiah",
    "lam": "Lamentations", "lamentations": "Lamentations",
    "ezek": "Ezekiel", "ezekiel": "Ezekiel",
    "dan": "Daniel", "daniel": "Daniel",
    "hos": "Hosea", "hosea": "Hosea",
    "joel": "Joel", "amos": "Amos", "obad": "Obadiah",
    "jon": "Jonah", "jonah": "Jonah",
    "mic": "Micah", "micah": "Micah",
    "nah": "Nahum", "hab": "Habakkuk", "zeph": "Zephaniah",
    "hag": "Haggai", "zech": "Zechariah", "mal": "Malachi",
    "matt": "Matthew", "matthew": "Matthew",
    "mk": "Mark", "mark": "Mark",
    "lk": "Luke", "luke": "Luke",
    "jn": "John", "john": "John",
    "acts": "Acts",
    "rom": "Romans", "romans": "Romans",
    "1cor": "1 Corinthians", "1 corinthians": "1 Corinthians",
    "2cor": "2 Corinthians", "2 corinthians": "2 Corinthians",
    "gal": "Galatians", "galatians": "Galatians",
    "eph": "Ephesians", "ephesians": "Ephesians",
    "phil": "Philippians", "philippians": "Philippians",
    "col": "Colossians", "colossians": "Colossians",
    "1thess": "1 Thessalonians", "1 thessalonians": "1 Thessalonians",
    "2thess": "2 Thessalonians", "2 thessalonians": "2 Thessalonians",
    "1tim": "1 Timothy", "1 timothy": "1 Timothy",
    "2tim": "2 Timothy", "2 timothy": "2 Timothy",
    "titus": "Titus", "tit": "Titus",
    "phlm": "Philemon", "philemon": "Philemon",
    "heb": "Hebrews", "hebrews": "Hebrews",
    "jas": "James", "james": "James",
    "1pet": "1 Peter", "1 peter": "1 Peter",
    "2pet": "2 Peter", "2 peter": "2 Peter",
    "1jn": "1 John", "1 john": "1 John",
    "2jn": "2 John", "2 john": "2 John",
    "3jn": "3 John", "3 john": "3 John",
    "jude": "Jude",
    "rev": "Revelation", "revelation": "Revelation",
}

# ---------------- EXPLICIT REFERENCE ----------------

EXPLICIT_REF_RE = re.compile(
    r"(?P<book>[1-3]?\s?[a-z]+)\s+(?P<chapter>\d+)\s*:?\s*(?P<verse>\d+)",
    re.IGNORECASE
)

def parse_explicit_reference(text: str):
    text = spoken_to_digits(text)
    match = EXPLICIT_REF_RE.search(text)
    if not match:
        return None

    raw_book = match.group("book").strip().lower()
    book = BOOK_ALIASES.get(raw_book, raw_book.title())

    return {
        "book": book,
        "chapter": int(match.group("chapter")),
        "verse": int(match.group("verse")),
    }

# ---------------- SEARCH ----------------

def search_bible(query: str, top_k: int = TOP_K, book_hint: str | None = None):
    query_norm = normalize(query)
    if len(query_norm) < 3:
        return []

    # ---------- FAMOUS PHRASES SHORTCUT ----------
    FAMOUS_PHRASES = {
        "in the beginning": ("Genesis", 1, 1),
        "for god so loved": ("John", 3, 16),
        "the lord is my shepherd": ("Psalms", 23, 1),
        "love is patient": ("1 Corinthians", 13, 4),
        "i can do all things": ("Philippians", 4, 13),
        "faith without works": ("James", 2, 26),
        "the truth shall make you free": ("John", 8, 32),
        "the wages of sin": ("Romans", 6, 23),
        "be still and know": ("Psalms", 46, 10),
        "fear not": ("Isaiah", 41, 10),
        "ask and it shall be given": ("Matthew", 7, 7),
        "knock and the door": ("Matthew", 7, 7),
        "blessed are the poor in spirit": ("Matthew", 5, 3),
        "the fruit of the spirit": ("Galatians", 5, 22),
        "greater is he": ("1 John", 4, 4),
        "no weapon formed": ("Isaiah", 54, 17),
        "i am the way": ("John", 14, 6),
        "the lord is my light": ("Psalms", 27, 1),
    }

    for phrase, (book, chapter, verse_num) in FAMOUS_PHRASES.items():
        if phrase in query_norm:
            for v in verses:
                if v["book"] == book and v["chapter"] == chapter and v["verse"] == verse_num:
                    return [{**v, "score": 1.0}]

    # ... rest of the function unchanged

    # Explicit reference short-circuit
    ref = parse_explicit_reference(query_norm)
    if ref:
        key = (ref["book"], ref["chapter"], ref["verse"])
        for v in verses:
            if (v["book"], v["chapter"], v["verse"]) == key:
                return [{**v, "score": 1.0}]

    query_vec = model.encode([query_norm], normalize_embeddings=True)[0]
    semantic_scores = embeddings @ query_vec
    final_scores = np.zeros(len(verses), dtype=np.float32)

    words = query_norm.split()

    for i, verse in enumerate(verses):
        score = float(semantic_scores[i])
        verse_norm = verse["_norm"]

        overlap = sum(1 for w in words if w in verse_norm)
        score += overlap * 0.03

        if query_norm in verse_norm:
            score += 0.15

        if (
            "shepherd" in query_norm
            and verse["book"] == "Psalms"
            and verse["chapter"] == 23
        ):
            score += 0.10

        # Book hint boost
        if book_hint and verse["book"] == book_hint:
            score += 0.15

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

def is_confident_match(results, threshold=CONFIDENCE_MIN_SCORE) -> bool:
    if not results:
        return False

    top = results[0]["score"]

    if top < threshold:
        return False

    if len(results) == 1:
        return True

    second = results[1]["score"]

    # High confidence — don't require margin
    if top >= 0.80:
        return True

    if (top - second) < CONFIDENCE_MARGIN:
        return False

    return True

# ---------------- STABILITY HELPERS ----------------

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