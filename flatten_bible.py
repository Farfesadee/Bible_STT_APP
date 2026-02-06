import json

INPUT_FILE = "bible/kjv.json"
OUTPUT_FILE = "bible/kjv_flat.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    bible = json.load(f)

flat_verses = []

for book in bible:
    book_name = book["name"]
    for chapter_idx, chapter in enumerate(book["chapters"], start=1):
        for verse_idx, verse_text in enumerate(chapter, start=1):
            flat_verses.append({
                "book": book_name,
                "chapter": chapter_idx,
                "verse": verse_idx,
                "text": verse_text,
                "translation": "KJV"
            })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(flat_verses, f, ensure_ascii=False, indent=2)

print(f"Flattened verses saved to {OUTPUT_FILE}")
print(f"Total verses: {len(flat_verses)}")
