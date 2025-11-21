from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from agents.memory_agent import recall_similar


def run_agent(
    user_input: str,
    region: str,
    location: str,
    mode: str = "Text",
    context: str | None = None,
) -> dict:
    """
    Run EchoAtlas agent with semantic memory recall and OpenAI response.

    Location-aware behaviour (D-level):
    - Always answer from the perspective of the given region + location.
    - If the user is ambiguous (e.g. "best tourist destinations?"),
      interpret the question as being about THIS region/location,
      not the whole world.
    """

    if not user_input or not user_input.strip():
        user_input = "Tell me something interesting about this place."

    context = context or "casual"
    print("Agent input:", repr(user_input))

    # 1) Recall similar interactions from memory for this region/location/mode/context
    recalled = recall_similar(
        region=region,
        location=location,
        user_input=user_input,
        mode=mode,
        context=context,
    )

    memory_context = (
        "\n\n".join(
            f"- Phrase: {r['phrase']}\n"
            f"  Gesture: {r['gesture']}\n"
            f"  Custom: {r['custom']}\n"
            f"  Tone: {r['tone']}"
            for r in recalled
        )
        if recalled
        else "No prior interactions found for this region/location."
    )

    # 2) Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 3) Strongly location-anchored system prompt
    system_prompt = (
        "You are EchoAtlas, a culturally-aware assistant bound to the "
        f"current region '{region}' and city/location '{location}'.\n\n"
        "GENERAL RULE:\n"
        "- Always answer from the perspective of THIS region/location.\n"
        "- If the user asks about 'tourist destinations', 'places to visit', "
        "or similar, interpret it as destinations IN or AROUND this city/region "
        "or at least within this country, not worldwide.\n"
        "- Use a friendly, concise tone.\n"
        "- When helpful, include short cultural or etiquette tips.\n\n"
        f"Current context tag: {context}\n"
        f"Input mode: {mode}\n\n"
        "Relevant past interactions for this region/location:\n"
        f"{memory_context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # 4) Create agent (no external tools yet, but ready for future ones)
    agent = create_tool_calling_agent(llm=llm, tools=[], prompt=prompt)

    # 5) Run agent
    result = agent.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "intermediate_steps": [],
        }
    )

    return {"phrase": result.return_values["output"]}
