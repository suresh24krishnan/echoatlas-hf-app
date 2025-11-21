from sentence_transformers import SentenceTransformer, util
import json

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load region data
with open("region_data.json", "r", encoding="utf-8") as f:
    REGION_DATA = json.load(f)

def semantic_translate(location, user_input):
    """
    Returns the most semantically appropriate phrase for the given location and user input.
    Uses cosine similarity to match meaning.
    """
    if location not in REGION_DATA:
        return "Location not supported"

    base_phrase = REGION_DATA[location]["phrase"]
    alt_phrases = [
        base_phrase,
        "Can I get some noodles?",
        "Where can I find a local dish?",
        "Is there a food stall nearby?",
        "I'd love to try something traditional"
    ]

    # Embed all phrases + user input
    embeddings = model.encode(alt_phrases + [user_input], convert_to_tensor=True)
    scores = util.cos_sim(embeddings[-1], embeddings[:-1])
    best_match = alt_phrases[scores.argmax()]
    return best_match
