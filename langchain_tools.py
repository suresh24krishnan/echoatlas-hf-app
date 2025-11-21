from langchain.tools import Tool
from agents.culture_agent import suggest_phrase
from agents.tone_agent import adjust_tone
from agents.customs_agent import get_customs
from agents.memory_agent import recall_similar
from agents.semantic_phrase_agent import semantic_translate

# Utility for safe query parsing
def parse_query(query):
    try:
        location, user_input = query.split("|", 1)
        return location.strip(), user_input.strip()
    except ValueError:
        return "", ""

# Error wrapper
def safe_tool(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return f"Error: {str(e)}"
    return wrapper

# Culture Tool (static phrasing)
culture_tool = Tool(
    name="CultureAgent",
    func=safe_tool(lambda query: suggest_phrase(*parse_query(query))),
    description="Suggests culturally appropriate phrasing for a given location and user input"
)

# Tone Tool
tone_tool = Tool(
    name="ToneAgent",
    func=safe_tool(lambda location: adjust_tone(location)),
    description="Returns appropriate emotional tone for a location",
    return_direct=True
)

# Customs Tool
customs_tool = Tool(
    name="CustomsAgent",
    func=safe_tool(lambda location: str(get_customs(location))),
    description="Returns gesture and etiquette tips for a location",
    return_direct=True
)

# Memory Tool
memory_tool = Tool(
    name="MemoryAgent",
    func=safe_tool(lambda query: str(recall_similar(*parse_query(query)))),
    description="Recalls similar past interactions for a location and phrase"
)

# Semantic Phrase Tool (meaning-aware phrasing)
semantic_tool = Tool(
    name="SemanticPhraseAgent",
    func=safe_tool(lambda query: semantic_translate(*parse_query(query))),
    description="Suggests culturally appropriate phrasing using semantic similarity for a given location and user input"
)

# Final tool list
tools = [culture_tool, tone_tool, customs_tool, memory_tool, semantic_tool]
