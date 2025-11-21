import os
import datetime
import uuid
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import streamlit as st

# ---------------------------------
# Environment & Chroma setup
# ---------------------------------
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment variables.")

CHROMA_PATH = "memory_store"
COLLECTION_NAME = "echoatlas_memory"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

# Flag file used for restart-safe factory reset
RESET_FLAG_PATH = "reset_memory_store.flag"

# --- Restart-safe reset: if flag exists, delete memory_store BEFORE opening Chroma ---
if os.path.exists(RESET_FLAG_PATH):
    try:
        if os.path.isdir(CHROMA_PATH):
            import shutil
            shutil.rmtree(CHROMA_PATH)
        os.remove(RESET_FLAG_PATH)
        print("🔥 Factory reset applied: memory_store folder deleted on startup.")
    except Exception as e:
        # Don't crash the app; just log the issue
        print(f"⚠️ Failed to apply factory reset on startup: {e}")

# One global client/collection for the app
_embedding_fn = OpenAIEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=_embedding_fn,
)


# ---------------------------------
# Helpers
# ---------------------------------
def _clean(text: str) -> str:
    """Normalize region/location keys (strip emojis, weird chars, extra spaces)."""
    if not text:
        return ""
    return "".join(ch for ch in text if ch.isalnum() or ch in " ()-,").strip()


def _build_where(
    region: str,
    location: str | None = None,
    mode: str | None = None,
    context: str | None = None,
) -> dict:
    """
    Build a D-level filter:
    Region + Location + (optional) Mode + (optional) Context.

    Chroma requires:
    - a single dict like {"region": {"$eq": "USA"}}  OR
    - {"$and": [cond1, cond2, ...]} with at least 2 items.

    So:
    - 0 clauses -> return {}  (no filter)
    - 1 clause  -> return that clause directly
    - 2+        -> wrap in {"$and": [...]}
    """
    clauses: list[dict] = []

    if region:
        clauses.append({"region": {"$eq": _clean(region)}})

    if location is not None and location != "":
        clauses.append({"location": {"$eq": _clean(location)}})

    if mode:
        clauses.append({"mode": {"$eq": mode}})

    if context:
        clauses.append({"context": {"$eq": context}})

    if not clauses:
        return {}
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}



def _normalize_metadatas(raw):
    """
    Chroma sometimes returns metadatas as [ [ {...}, {...} ] ] or [ {...}, {...} ].
    Normalize to a flat list of dicts.
    """
    if not raw:
        return []
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        # [[meta, meta, ...]]
        return raw[0]
    return raw


# ---------------------------------
# Public API
# ---------------------------------
def setup_memory_schema():
    """
    Kept for compatibility with app.py.
    Ensures the collection is initialized.
    """
    _ = _collection.name
    return


def store_interaction(
    region: str,
    location: str,
    phrase: str,          # user question
    tone: str,
    gesture: str,
    custom: str,
    mode: str = "Text",
    context: str | None = "default",
    answer: str | None = None,  # agent answer
):
    """
    Store a new interaction in memory, fully scoped by:
    Region + Location + Mode + Context  (D-level isolation).

    We store:
    - phrase: user input / question
    - answer: agent's response
    - tone / gesture / custom: cultural metadata
    """

    clean_region = _clean(region)
    clean_location = _clean(location)
    context = context or "default"

    print(
        f"📝 Storing interaction -> "
        f"region='{clean_region}', location='{clean_location}', "
        f"mode='{mode}', context='{context}', phrase='{phrase}'"
    )

    uid = str(uuid.uuid4())
    timestamp = datetime.datetime.utcnow().isoformat()

    _collection.add(
        documents=[phrase],
        metadatas=[
            {
                "region": clean_region,
                "location": clean_location,
                "mode": mode,
                "context": context,
                "field": "phrase",
                "phrase": phrase,
                "answer": answer or "",
                "tone": tone,
                "gesture": gesture,
                "custom": custom,
                "timestamp": timestamp,
            }
        ],
        ids=[uid],
    )

    print(
        f"✅ Stored memory for region='{clean_region}', "
        f"location='{clean_location}', mode='{mode}', context='{context}'"
    )


def recall_similar(
    region: str,
    location: str,
    user_input: str,
    mode: str | None = None,
    context: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    Recall memories with strict D-level isolation.

    - Always filters by *region*.
    - Filters by *location* if provided.
    - Optionally filters by *mode* (Mic/Text) and *context*.
    - If user_input is empty/whitespace, returns ALL memories for that scope.
    """

    clean_region = _clean(region)
    clean_location = _clean(location)

    print(
        f"🔍 recall_similar -> "
        f"region='{clean_region}', location='{clean_location}', "
        f"mode='{mode}', context='{context}', user_input='{user_input}'"
    )

    where = _build_where(clean_region, clean_location, mode, context)

    # Case 1: no input → fetch all memories for this scope
    if not user_input or not user_input.strip():
        raw = _collection.get(where=where)
        metas = _normalize_metadatas(raw.get("metadatas", []))
        memories = [
            {
                "phrase": m.get("phrase", ""),
                "answer": m.get("answer", ""),
                "gesture": m.get("gesture", "🤷"),
                "custom": m.get("custom", "No cultural insight available."),
                "tone": m.get("tone", "Neutral"),
                "mode": m.get("mode", "Unknown"),
                "region": m.get("region", clean_region),
                "location": m.get("location", clean_location),
                "context": m.get("context", "default"),
                "timestamp": m.get("timestamp", ""),
            }
            for m in metas
        ]
        memories.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return memories

    # Case 2: semantic similarity query
    raw = _collection.query(
        query_texts=[user_input],
        n_results=top_k,
        where=where,
    )

    docs = raw.get("documents", [[]])[0] if raw.get("documents") else []
    metas = raw.get("metadatas", [[]])[0] if raw.get("metadatas") else []

    memories: list[dict] = []
    for doc, meta in zip(docs, metas):
        print(
            f"   ➡️ Returned meta.region='{meta.get('region')}', "
            f"location='{meta.get('location')}', mode='{meta.get('mode')}', "
            f"context='{meta.get('context')}'"
        )
        memories.append(
            {
                "phrase": meta.get("phrase", doc),
                "answer": meta.get("answer", ""),
                "gesture": meta.get("gesture", "🤷"),
                "custom": meta.get("custom", "No cultural insight available."),
                "tone": meta.get("tone", "Neutral"),
                "mode": meta.get("mode", "Unknown"),
                "region": meta.get("region", clean_region),
                "location": meta.get("location", clean_location),
                "context": meta.get("context", "default"),
                "timestamp": meta.get("timestamp", ""),
            }
        )

    memories.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return memories


def display_memory(memory: dict):
    """Render a memory: show both the user question and the agent answer."""
    question = memory.get("phrase", "")
    answer = memory.get("answer", "")

    st.markdown("### 💬 Conversation Memory")

    st.markdown(f"**🧑‍💻 User asked:** {question}")
    if answer:
        st.success(f"✅ Agent answered:\n\n{answer}")
    else:
        st.info("ℹ️ No saved answer for this memory (older entry).")

    st.info(f"🙇 Gesture tip: {memory.get('gesture', '🤷')}")
    st.warning(
        f"📚 Cultural insight: "
        f"{memory.get('custom', 'No cultural insight available.')}"
    )
    st.markdown(f"🎭 Tone: {memory.get('tone', 'Neutral')}")

    st.caption(
        f"🕒 {memory.get('timestamp', '')} | "
        f"🏙️ {memory.get('region', '')} → {memory.get('location', '')} | "
        f"🎛️ {memory.get('mode', '')} | 🎯 {memory.get('context', '')}"
    )


def delete_memories_for_region(
    region: str,
    location: str,
    mode: str = None,
    context: str = None,
) -> str:
    """
    Delete all memories for a specific region/location.
    If mode/context are provided, filter by them; otherwise delete ALL modes/contexts
    for that region + location.

    Returns a human-readable message for the UI.
    """
    clean_region = "".join(ch for ch in region if ch.isalnum() or ch in " ()-,").strip()
    clean_location = "".join(ch for ch in location if ch.isalnum() or ch in " ()-,").strip()

    client = chromadb.PersistentClient(path="memory_store")
    collection = client.get_or_create_collection(
        name="echoatlas_memory",
        embedding_function=OpenAIEmbeddingFunction(
            model_name="text-embedding-3-small"
        ),
    )

    # Build the filter (region + location always; mode/context optional)
    where_clauses = [
        {"region": {"$eq": clean_region}},
        {"location": {"$eq": clean_location}},
    ]
    if mode is not None:
        where_clauses.append({"mode": {"$eq": mode}})
    if context is not None:
        where_clauses.append({"context": {"$eq": context}})

    where = {"$and": where_clauses}

    # Fetch IDs to delete
    raw = collection.get(where=where)
    ids_to_delete = raw.get("ids", [])

    # Chroma often returns nested lists: [[id1, id2, ...]]
    if ids_to_delete and isinstance(ids_to_delete[0], list):
        ids_to_delete = ids_to_delete[0]

    if not ids_to_delete:
        return (
            f"ℹ️ No memories found for {clean_region} / {clean_location} "
            f"(mode={mode or 'ALL'}, context={context or 'ALL'})."
        )

    collection.delete(ids=ids_to_delete)
    return (
        f"🧹 Deleted {len(ids_to_delete)} memories for {clean_region} / {clean_location} "
        f"(mode={mode or 'ALL'}, context={context or 'ALL'})."
    )

def list_all_regions() -> list[str]:
    """
    Return all distinct regions currently stored.
    Used by the Memory Management section in app.py.
    """
    raw = _collection.get()
    metas = _normalize_metadatas(raw.get("metadatas", []))
    regions = {_clean(m.get("region", "Unknown")) for m in metas}
    return sorted(r for r in regions if r)
