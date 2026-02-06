import json
import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_FILE = "bible/kjv_flat.json"
OUTPUT_FILE = "bible/kjv_embeddings.npy"
META_FILE = "bible/kjv_metadata.json"

print("Loading Bible verses...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    verses = json.load(f)

texts = [v["text"] for v in verses]

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding verses (this will take a few minutes)...")
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

np.save(OUTPUT_FILE, embeddings)

with open(META_FILE, "w", encoding="utf-8") as f:
    json.dump(verses, f, ensure_ascii=False, indent=2)

print("✅ Embeddings saved")
print("Total verses:", len(embeddings))
