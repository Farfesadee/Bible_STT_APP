import json

with open("bible/kjv.json", "r", encoding="utf-8") as f:
    bible = json.load(f)

verse_count = 0

for book in bible:
    for chapter in book["chapters"]:
        verse_count += len(chapter)

print("Total verses:", verse_count)
