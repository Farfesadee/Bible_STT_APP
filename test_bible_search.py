from bible_search import search_bible

query = "the lord is my shepherd"
results = search_bible(query)

for r in results:
    print(
        f"{r['book']} {r['chapter']}:{r['verse']} "
        f"({r['score']:.3f}) — {r['text']}"
    )
