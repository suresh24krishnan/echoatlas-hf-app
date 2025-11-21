import streamlit as st
from typing import Dict, Any, List
from pathlib import Path
import json
import shutil

# =========================================
# IMPORT YOUR REAL ECHOATLAS LOGIC
# =========================================
from agents.memory_agent import (
    setup_memory_schema,
    store_interaction,
    recall_similar,
    display_memory,
    delete_memories_for_region,
)
from langchain_runner import run_agent

from openai import OpenAI

client = OpenAI()

# --------- Factory reset paths ----------
RESET_FLAG_PATH = Path("reset_memory_store.flag")
MEMORY_STORE_PATH = Path("memory_store")


def apply_scheduled_factory_reset() -> None:
    """If a reset flag exists, wipe memory_store once on startup."""
    if not RESET_FLAG_PATH.exists():
        return
    try:
        if MEMORY_STORE_PATH.exists():
            shutil.rmtree(MEMORY_STORE_PATH)
        RESET_FLAG_PATH.unlink(missing_ok=True)
        try:
            st.toast("✅ Memory store factory-reset on startup.", icon="🧨")
        except Exception:
            st.info("✅ Memory store factory-reset on startup.")
    except Exception as e:
        st.error(f"❌ Could not complete scheduled memory reset: {e}")


def generate_dynamic_culture_profile(region: str, city: str) -> Dict[str, str]:
    """
    Use the LLM to generate a small culture profile for (region, city).
    Cached in session_state to avoid repeated calls.
    """
    if "dynamic_culture_cache" not in st.session_state:
        st.session_state.dynamic_culture_cache = {}
    cache = st.session_state.dynamic_culture_cache
    key = f"{region.strip().lower()}|{city.strip().lower()}"

    # 1) Return from cache if already computed
    if key in cache:
        return cache[key]

    prompt = f"""
You are a cultural communication expert.

For the following place:
- Country / Region: {region}
- City / Area: {city}

Generate a short practical profile with JSON ONLY in this exact shape:
{{
  "phrase": "Example polite phrase for asking something...",
  "gesture": "Short description of appropriate gesture/body language.",
  "tone": "2–5 words describing the recommended tone of voice.",
  "custom": "1–2 sentences with a key cultural tip for this place."
}}

Return ONLY valid JSON, with no extra commentary, markdown, or explanation.
    """.strip()

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a cultural communication expert. "
                        "You MUST return ONLY valid JSON for the user."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            response_format={"type": "json_object"},
        )

        raw_content = completion.choices[0].message.content or "{}"

        try:
            data = json.loads(raw_content)
        except json.JSONDecodeError:
            data = {}

        result = {
            "phrase": (data.get("phrase") or "").strip()
            or f"Hello, could you please help me here in {city}?",
            "gesture": (data.get("gesture") or "Smile gently and be respectful.").strip(),
            "tone": (data.get("tone") or "Polite and friendly").strip(),
            "custom": (
                data.get("custom")
                or f"Be respectful and observe how locals behave in {city}."
            ).strip(),
        }

    except Exception:
        # Quiet fallback if anything fails
        result = {
            "phrase": f"Hello, could you please help me here in {city}?",
            "gesture": "Smile gently and be respectful.",
            "tone": "Polite and friendly",
            "custom": f"Be respectful and observe how locals behave in {city}.",
        }

    cache[key] = result
    return result


def generate_cultural_playbook(region: str, city: str) -> dict:
    """
    Build a structured cultural playbook for (region, city) using:
    - dynamic culture profile
    - past memories
    - an LLM synthesis pass
    """
    base = generate_dynamic_culture_profile(region, city)

    mems = recall_similar(
        region=region,
        location=city,
        user_input="",
        mode=None,
        context=None,
        top_k=50,
    )

    memory_lines = []
    for m in mems:
        memory_lines.append(
            f"- Q: {m.get('phrase','')} | Tone: {m.get('tone','')} | "
            f"Gesture: {m.get('gesture','')} | Tip: {m.get('custom','')}"
        )
    memory_text = "\n".join(memory_lines) if memory_lines else "No prior interactions recorded."

    prompt = f"""
You are building a CULTURAL PLAYBOOK for:

Region / Country: {region}
City / Area: {city}

Base:
- Suggested tone: {base.get('tone','')}
- Gesture/body language: {base.get('gesture','')}
- Cultural tip: {base.get('custom','')}

Observed past interactions:
{memory_text}

Return ONLY JSON with this structure:

{{
  "communication_style": {{

    "tone_overview": "...",
    "body_language_overview": "...",
    "phrasing_examples": [
      "Example polite request...",
      "Example asking for help...",
      "Example declining politely..."
    ],
    "taboo_topics_or_phrases": [
      "...", "...", "..."
    ],
    "formal_vs_informal": "..."
  }},
  "etiquette": {{
    "greetings": "...",
    "public_behavior": "...",
    "restaurant_etiquette": "...",
    "business_etiquette": "...",
    "gift_giving": "..."
  }},
  "do_and_donts": {{
    "do": [
      "Do this...",
      "Do that..."
    ],
    "dont": [
      "Don't do this...",
      "Don't do that..."
    ]
  }},
  "emerging_patterns_from_memory": {{
    "common_questions": [
      "People often ask about...",
      "They frequently wonder about..."
    ],
    "common_mistakes": [
      "Visitors sometimes make this mistake...",
      "Another recurring mistake is..."
    ],
    "recommendations": [
      "My main advice would be...",
      "Another key recommendation is..."
    ]
  }},
  "examples": [
    {{
      "scenario": "Ordering food at a restaurant",
      "what_to_say": "...",
      "how_to_act": "..."
    }},
    {{
      "scenario": "Asking for directions",
      "what_to_say": "...",
      "how_to_act": "..."
    }},
    {{
      "scenario": "Meeting someone for the first time",
      "what_to_say": "...",
      "how_to_act": "..."
    }}
  ]
}}

Return ONLY valid JSON, nothing else.
    """.strip()

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You output only well-structured JSON cultural playbooks.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        content = completion.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        st.warning(f"Cultural playbook generation failed: {e}")
        return {
            "communication_style": {
                "tone_overview": base.get("tone", "Polite and friendly"),
                "body_language_overview": base.get("gesture", "Smile gently."),
                "phrasing_examples": [base.get("phrase", "Hello, could you please help me?")],
                "taboo_topics_or_phrases": [],
                "formal_vs_informal": "Use polite language with a friendly tone.",
            },
            "etiquette": {
                "greetings": base.get("custom", ""),
                "public_behavior": "",
                "restaurant_etiquette": "",
                "business_etiquette": "",
                "gift_giving": "",
            },
            "do_and_donts": {"do": [], "dont": []},
            "emerging_patterns_from_memory": {
                "common_questions": [],
                "common_mistakes": [],
                "recommendations": [],
            },
            "examples": [],
        }


def playbook_to_markdown(playbook: dict, region: str, city: str) -> str:
    """Convert playbook JSON to a readable Markdown document."""
    lines: List[str] = []
    lines.append(f"# Cultural Playbook — {city}, {region}")
    lines.append("")

    cs = playbook.get("communication_style", {}) or {}
    lines.append("## Communication Style")
    lines.append("")
    for key, val in cs.items():
        lines.append(f"**{key.replace('_', ' ').title()}**")
        lines.append("")
        if isinstance(val, list):
            for item in val:
                lines.append(f"- {item}")
        else:
            lines.append(str(val))
        lines.append("")

    et = playbook.get("etiquette", {}) or {}
    lines.append("## Etiquette")
    lines.append("")
    for key, val in et.items():
        lines.append(f"**{key.replace('_', ' ').title()}**")
        lines.append("")
        lines.append(str(val))
        lines.append("")

    dd = playbook.get("do_and_donts", {}) or {}
    lines.append("## Do & Don't")
    lines.append("")
    do_list = dd.get("do", []) or []
    dont_list = dd.get("dont", []) or []

    lines.append("### Do")
    for item in do_list:
        lines.append(f"- {item}")
    if not do_list:
        lines.append("- (No items)")
    lines.append("")

    lines.append("### Don't")
    for item in dont_list:
        lines.append(f"- {item}")
    if not dont_list:
        lines.append("- (No items)")
    lines.append("")

    ep = playbook.get("emerging_patterns_from_memory", {}) or {}
    lines.append("## Observed Patterns from Your Questions")
    lines.append("")
    for key, val in ep.items():
        lines.append(f"**{key.replace('_', ' ').title()}**")
        lines.append("")
        if isinstance(val, list):
            for item in val:
                lines.append(f"- {item}")
        else:
            lines.append(str(val))
        lines.append("")

    examples = playbook.get("examples", []) or []
    lines.append("## Practical Examples")
    lines.append("")
    if examples:
        for ex in examples:
            scenario = ex.get("scenario", "Scenario")
            what = ex.get("what_to_say", "")
            how = ex.get("how_to_act", "")
            lines.append(f"### {scenario}")
            lines.append("")
            lines.append(f"**What to say:** {what}")
            lines.append("")
            lines.append(f"**How to act:** {how}")
            lines.append("")
    else:
        lines.append("_No examples generated yet._")
        lines.append("")

    return "\n".join(lines)


# ------------------------- Initialize --------------------------
apply_scheduled_factory_reset()
setup_memory_schema()

st.set_page_config(
    page_title="EchoAtlas",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Basic styling (simple, HF-safe) ---
st.markdown(
    """
    <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    .ea-card {
        background: rgba(15,23,42,0.95);
        border-radius: 18px;
        padding: 18px 22px;
        border: 1px solid rgba(148,163,184,0.4);
        box-shadow: 0 16px 40px rgba(0,0,0,0.65);
    }
    .ea-card-soft {
        background: rgba(15,23,42,0.85);
        border-radius: 14px;
        padding: 14px 18px;
        border: 1px solid rgba(148,163,184,0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------- Sidebar nav ----------------------
with st.sidebar:
    st.markdown("## 🌍 EchoAtlas")
    st.caption("Cultural Intelligence Assistant")

    page = st.radio(
        "Navigate",
        [
            "Ask EchoAtlas",
            "Conversation Memory",
            "Cultural Playbook",
            "FAQ & Sample Prompts",
            "Settings",
        ],
        index=0,
        key="nav_page",
    )

    st.markdown("---")
    st.caption("© EchoAtlas · 2025")

# ---------------------- Session State ----------------------
if "selected_region" not in st.session_state:
    st.session_state.selected_region = "United States"
if "selected_city" not in st.session_state:
    st.session_state.selected_city = "New York"

if "last_user_input" not in st.session_state:
    st.session_state.last_user_input = ""
if "last_agent_output" not in st.session_state:
    st.session_state.last_agent_output = {}

if "prefill_text" not in st.session_state:
    st.session_state.prefill_text = ""
if "prefill_just_set" not in st.session_state:
    st.session_state.prefill_just_set = False

if "custom_city" not in st.session_state:
    st.session_state.custom_city = ""


# ========================== PAGE: ASK ECHOATLAS ==========================
if page == "Ask EchoAtlas":
    st.markdown(
        """
        <div class="ea-card">
            <div style="font-size:1.8rem; font-weight:700; color:#e5e7eb;">
                🌐 EchoAtlas — Cultural Intelligence Assistant
            </div>
            <div style="margin-top:4px; color:#cbd5e1;">
                Ask region-aware cultural questions and get phrasing, tone, gestures, and tips.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ----- Location selector -----
    st.markdown("### 🌏 Choose Your Region & Location")

    col1, col2 = st.columns(2)

    region_options = [
        "United States",
        "Canada",
        "United Kingdom",
        "Australia",
        "Singapore",
        "Germany",
        "France",
        "India",
        "Other",
    ]
    stored_region = st.session_state.get("selected_region", "United States")
    if stored_region not in region_options:
        stored_region = "Other"
    region_index = region_options.index(stored_region)

    with col1:
        region = st.selectbox(
            "Region / Country",
            region_options,
            index=region_index,
        )

    with col2:
        # Region-specific city menus, with persistence
        stored_city = st.session_state.get("selected_city", "New York")

        def city_select(label: str, cities: List[str], fallback: str) -> str:
            base = (
                stored_city
                if st.session_state.get("selected_region") == region
                else fallback
            )
            # 🔧 Fix: if stored_city is not in the list (custom city), default to "Other" if present
            if base not in cities:
                if "Other" in cities:
                    base = "Other"
                else:
                    base = fallback
            idx = cities.index(base)
            return st.selectbox(label, cities, index=idx)

        if region == "United States":
            city = city_select(
                "City",
                ["New York", "Seattle", "Chicago", "San Francisco", "Los Angeles", "Other"],
                "New York",
            )

        elif region == "Canada":
            city = city_select(
                "City",
                ["Toronto", "Vancouver", "Montreal", "Calgary", "Ottawa", "Other"],
                "Toronto",
            )

        elif region == "United Kingdom":
            city = city_select(
                "City",
                ["London", "Manchester", "Birmingham", "Liverpool", "Other"],
                "London",
            )

        elif region == "Australia":
            city = city_select(
                "City",
                ["Sydney", "Melbourne", "Brisbane", "Perth", "Other"],
                "Sydney",
            )

        elif region == "Singapore":
            city = city_select(
                "City",
                ["Singapore City", "Other"],
                "Singapore City",
            )

        elif region == "Germany":
            city = city_select(
                "City",
                ["Berlin", "Munich", "Frankfurt", "Hamburg", "Other"],
                "Berlin",
            )

        elif region == "France":
            city = city_select(
                "City",
                ["Paris", "Lyon", "Marseille", "Nice", "Other"],
                "Paris",
            )

        elif region == "India":
            city = city_select(
                "City",
                ["Chennai", "Bengaluru", "Hyderabad", "Mumbai", "Delhi", "Other"],
                "Chennai",
            )

        else:
            # Completely custom region & city
            region = st.text_input(
                "Custom Region / Country",
                value=stored_region if stored_region != "Other" else "",
            )
            city = st.text_input("City", value=stored_city)

    # If user picked a known region but "Other" city, let them type a custom city
    if region != "Other" and city == "Other":
        custom_city = st.text_input(
            "Custom City",
            value=st.session_state.get("custom_city", ""),
        )
        if custom_city.strip():
            city = custom_city.strip()
        st.session_state.custom_city = custom_city

    # ----- Track previous vs new location, clear prefill/response when user changes -----
    prev_region = st.session_state.get("selected_region", region)
    prev_city = st.session_state.get("selected_city", city)

    final_region = region
    final_city = city

    location_changed = (final_region != prev_region) or (final_city != prev_city)

    if location_changed:
        # Keep compatibility keys even in text-only version
        st.session_state.transcript_region = final_region
        st.session_state.transcript_city = final_city

        # Clear prefill only if it's not just set by FAQ prompt
        if not st.session_state.get("prefill_just_set", False):
            st.session_state.prefill_text = ""

        # ❗ Also clear last response so old city answer doesn't show under new city
        st.session_state.last_agent_output = {}
        st.session_state.last_user_input = ""

    st.session_state.selected_region = final_region
    st.session_state.selected_city = final_city

    st.markdown(
        f"""
        <div class="ea-card-soft" style="margin-top:8px;">
          📌 <b>Selected:</b> {final_city}, {final_region}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ----- TEXT INPUT -----
    st.markdown("### 🎤 Ask EchoAtlas (Text Only)")

    default_text = st.session_state.get("prefill_text", "")

    typed = st.text_area(
        "✍️ Type your question here",
        value=default_text,
        placeholder="E.g., What is a polite way to ask for directions in this city?",
        height=140,
    )

    # Update prefill text and consume FAQ flag
    st.session_state.prefill_text = typed
    st.session_state.prefill_just_set = False

    user_input = typed.strip()

    submit_query = st.button("🚀 Ask EchoAtlas", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 💬 EchoAtlas Response")

    agent_output: Dict[str, Any] | None = None

    if submit_query and user_input:
        agent_output = run_agent(
            user_input=user_input,
            region=final_region,
            location=final_city,
            mode="Text",
            context="default",
        )

        # Enrich with dynamic culture profile if missing keys
        dyn = generate_dynamic_culture_profile(final_region, final_city)
        agent_output["gesture"] = agent_output.get("gesture") or dyn.get("gesture")
        agent_output["tone"] = agent_output.get("tone") or dyn.get("tone")
        agent_output["custom"] = agent_output.get("custom") or dyn.get("custom")
        if not agent_output.get("phrase"):
            agent_output["phrase"] = dyn.get("phrase")

        store_interaction(
            region=final_region,
            location=final_city,
            phrase=user_input,
            tone=agent_output.get("tone", "Neutral"),
            gesture=agent_output.get("gesture", "🤝"),
            custom=agent_output.get(
                "custom", "Be respectful and observe local behavior."
            ),
            mode="Text",
            context="default",
            answer=agent_output.get("phrase", ""),
        )

        st.session_state.last_user_input = user_input
        st.session_state.last_agent_output = agent_output
    else:
        last = st.session_state.get("last_agent_output")
        if last and st.session_state.get("last_user_input"):
            agent_output = last
            user_input = st.session_state.last_user_input

    if agent_output:
        phrase = agent_output.get("phrase", "")
        gesture = agent_output.get("gesture", "Smile and be respectful.")
        tone = agent_output.get("tone", "Neutral and polite")
        custom_tip = agent_output.get(
            "custom", "Be respectful and observe how locals behave."
        )

        st.markdown(
            f"""
            <div class="ea-card">
              <div style="font-size:0.95rem; color:#e5e7eb;"><b>You asked:</b></div>
              <div style="margin-top:6px; color:#cbd5e1;">“{user_input}”</div>
              <hr style="border-color:rgba(148,163,184,0.4); margin:12px 0;">
              <div style="font-size:0.95rem; color:#e5e7eb;"><b>EchoAtlas suggests:</b></div>
              <div style="margin-top:6px; color:#e5e7eb; font-size:1.02rem; line-height:1.5;">
                {phrase}
              </div>
              <div style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;">
                <span style="
                    background:#1e3a8a; color:#bfdbfe; padding:4px 10px;
                    border-radius:999px; font-size:0.8rem; border:1px solid #3b82f6;">
                    🌍 {final_city}, {final_region}
                </span>
                <span style="
                    background:#4b5563; color:#e5e7eb; padding:4px 10px;
                    border-radius:999px; font-size:0.8rem; border:1px solid #9ca3af;">
                    🎧 Mode: Text
                </span>
                <span style="
                    background:#14532d; color:#bbf7d0; padding:4px 10px;
                    border-radius:999px; font-size:0.8rem; border:1px solid #22c55e;">
                    🎵 Tone: {tone}
                </span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🧭 Cultural Insights")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"""
                <div class="ea-card-soft">
                  <div style="font-size:0.9rem; color:#93c5fd; font-weight:600;">Gesture</div>
                  <div style="margin-top:4px; color:#e5e7eb;">{gesture}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="ea-card-soft">
                  <div style="font-size:0.9rem; color:#93c5fd; font-weight:600;">Tone</div>
                  <div style="margin-top:4px; color:#e5e7eb;">{tone}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""
                <div class="ea-card-soft">
                  <div style="font-size:0.9rem; color:#93c5fd; font-weight:600;">Cultural Tip</div>
                  <div style="margin-top:4px; color:#e5e7eb;">{custom_tip}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🧠 Related Memories for this City")

        related = recall_similar(
            region=final_region,
            location=final_city,
            user_input=user_input,
            mode=None,
            context=None,
            top_k=5,
        )
        if related:
            for idx, m in enumerate(related[:5], start=1):
                preview = m.get("phrase", "") or ""
                if len(preview) > 80:
                    preview = preview[:77] + "..."
                label = f"💬 Memory {idx}: {preview}"
                with st.expander(label):
                    display_memory(m)
        else:
            st.caption(
                "No related memories yet. As you keep asking questions, "
                "EchoAtlas will build a cultural trail here."
            )
    else:
        st.info("Ask a question above to see a region-aware EchoAtlas response here.")

# ========================== PAGE: CONVERSATION MEMORY ==========================
elif page == "Conversation Memory":
    st.markdown("## 🧠 Conversation Memory")

    region = st.session_state.get("selected_region", "United States")
    city = st.session_state.get("selected_city", "New York")

    col1, col2 = st.columns([1.3, 0.7])
    with col1:
        st.markdown(
            f"<div class='ea-card-soft'>Current scope: <b>{city}, {region}</b></div>",
            unsafe_allow_html=True,
        )
    with col2:
        if st.button("🧹 Clear memories for this city", use_container_width=True):
            msg = delete_memories_for_region(
                region=region, location=city, mode=None, context=None
            )
            st.success(msg)

    mems = recall_similar(
        region=region,
        location=city,
        user_input="",
        mode=None,
        context=None,
        top_k=50,
    )

    if mems:
        st.write(f"Found **{len(mems)}** memories.")
        for idx, m in enumerate(mems, start=1):
            preview = m.get("phrase", "")
            if len(preview) > 80:
                preview = preview[:77] + "..."
            label = f"💬 Turn {idx}: {preview}"
            with st.expander(label):
                display_memory(m)
    else:
        st.info("No memories stored yet for this city. Ask EchoAtlas something first.")

# ========================== PAGE: CULTURAL PLAYBOOK ==========================
elif page == "Cultural Playbook":
    st.markdown("## 📘 Cultural Playbook")

    region = st.session_state.get("selected_region", "United States")
    city = st.session_state.get("selected_city", "New York")

    st.markdown(
        f"""
        <div class="ea-card-soft">
          Generating a cultural playbook for: <b>{city}, {region}</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "cached_playbook" not in st.session_state:
        st.session_state.cached_playbook = {}

    cache_key = f"{region}|{city}"
    cache = st.session_state.cached_playbook

    if st.button("🔁 Regenerate Playbook"):
        if cache_key in cache:
            del cache[cache_key]

    playbook = cache.get(cache_key)
    if playbook is None:
        with st.spinner("Synthesizing cultural playbook from EchoAtlas memory..."):
            playbook = generate_cultural_playbook(region, city)
        cache[cache_key] = playbook
        st.session_state.cached_playbook = cache

    json_str = json.dumps(playbook, indent=2, ensure_ascii=False)
    md_str = playbook_to_markdown(playbook, region, city)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Export as JSON",
            data=json_str,
            file_name=f"echoatlas_playbook_{city}_{region}.json",
            mime="application/json",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "⬇️ Export as Markdown",
            data=md_str,
            file_name=f"echoatlas_playbook_{city}_{region}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.json(playbook)

# ========================== PAGE: FAQ & SAMPLE PROMPTS ==========================
elif page == "FAQ & Sample Prompts":
    st.markdown("## ❓ FAQ & Sample Prompts")
    st.markdown(
        """
        <div class="ea-card" style="margin-bottom:1rem;">
          <div style="font-size:1rem; font-weight:600; color:#e5e7eb;">
            Want ideas on what to ask EchoAtlas?
          </div>
          <div style="font-size:0.9rem; color:#cbd5e1; margin-top:4px;">
            Tap any interesting scenario below — it will auto-load into the Ask EchoAtlas page.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    def prompt_button(
        label: str,
        prompt: str,
        key: str,
        region: str | None = None,
        city: str | None = None,
    ):
        if st.button(f"💬 {label}", key=key, use_container_width=True):
            st.session_state.prefill_text = prompt
            st.session_state.prefill_just_set = True
            if region and city:
                st.session_state.selected_region = region
                st.session_state.selected_city = city
            st.success("Loaded into Ask EchoAtlas — switch to that tab to send it!")

    # -------------- UNITED STATES --------------
    with st.expander("🌍 United States (New York, Seattle, Chicago)", expanded=False):
        prompt_button(
            "Ask a stranger for help without sounding intrusive",
            "How do I politely ask a stranger for help in New York without sounding intrusive?",
            "us_ask_help",
            region="United States",
            city="New York",
        )
        prompt_button(
            "Small talk at a US tech company",
            "What is a natural way to start small talk with coworkers at a US tech company?",
            "us_smalltalk",
            region="United States",
            city="Seattle",
        )
        prompt_button(
            "Declining an invitation politely",
            "How do I politely decline a social invitation in the US without sounding rude?",
            "us_decline",
            region="United States",
            city="Chicago",
        )
        prompt_button(
            "Handling direct feedback",
            "How should I respond when an American coworker gives very direct feedback?",
            "us_direct",
            region="United States",
            city="San Francisco",
        )

    # -------------- CANADA --------------
    with st.expander("🇨🇦 Canada (Toronto, Vancouver)", expanded=False):
        prompt_button(
            "Apologizing in a Canadian way",
            "What is the most Canadian way to apologize after bumping into someone?",
            "ca_apology",
            region="Canada",
            city="Toronto",
        )
        prompt_button(
            "Starting small talk in an elevator",
            "How can I start small talk with a stranger in an elevator in Canada?",
            "ca_elevator",
            region="Canada",
            city="Vancouver",
        )
        prompt_button(
            "Polite disagreement",
            "How do I disagree politely in a meeting with Canadian colleagues?",
            "ca_disagree",
            region="Canada",
            city="Toronto",
        )

    # -------------- UNITED KINGDOM --------------
    with st.expander("🇬🇧 United Kingdom (London, Manchester)", expanded=False):
        prompt_button(
            "Asking for directions in London",
            "How do I ask for directions in London without sounding abrupt?",
            "uk_directions",
            region="United Kingdom",
            city="London",
        )
        prompt_button(
            "Joining a queue properly",
            "What should I know about queue etiquette in the UK?",
            "uk_queue",
            region="United Kingdom",
            city="London",
        )
        prompt_button(
            "Polite disagreement in a meeting",
            "How should I express disagreement politely in a UK meeting?",
            "uk_meeting",
            region="United Kingdom",
            city="Manchester",
        )

    # -------------- AUSTRALIA --------------
    with st.expander("🇦🇺 Australia (Sydney, Melbourne)", expanded=False):
        prompt_button(
            "Casual greetings at work",
            "How do Australians usually greet coworkers at the office?",
            "au_greetings",
            region="Australia",
            city="Sydney",
        )
        prompt_button(
            "Understanding Aussie humor",
            "How do I understand and react to Australian humor without misunderstanding it?",
            "au_humor",
            region="Australia",
            city="Melbourne",
        )
        prompt_button(
            "Saying no without sounding rude",
            "How can I say ‘no’ politely in Australian social situations?",
            "au_no",
            region="Australia",
            city="Sydney",
        )

    # -------------- SINGAPORE --------------
    with st.expander("🇸🇬 Singapore (Singapore City)", expanded=False):
        prompt_button(
            "Respecting elders and hierarchy",
            "How do I show respect to elders and seniors in Singaporean culture?",
            "sg_elders",
            region="Singapore",
            city="Singapore City",
        )
        prompt_button(
            "Asking for help in public",
            "What is a polite way to ask for help on the MRT or in malls in Singapore?",
            "sg_help",
            region="Singapore",
            city="Singapore City",
        )
        prompt_button(
            "Office etiquette in Singapore",
            "What office etiquette should I be aware of when working in Singapore?",
            "sg_office",
            region="Singapore",
            city="Singapore City",
        )

    # -------------- GERMANY --------------
    with st.expander("🇩🇪 Germany (Berlin, Munich)", expanded=False):
        prompt_button(
            "Being on time",
            "How strict is punctuality in Germany, and how should I handle being late?",
            "de_time",
            region="Germany",
            city="Berlin",
        )
        prompt_button(
            "Giving feedback at work",
            "What is an appropriate way to give direct feedback in German workplaces?",
            "de_feedback",
            region="Germany",
            city="Munich",
        )
        prompt_button(
            "Public behavior rules",
            "What should I keep in mind about public behavior and noise levels in Germany?",
            "de_behavior",
            region="Germany",
            city="Berlin",
        )

    # -------------- FRANCE --------------
    with st.expander("🇫🇷 France (Paris, Lyon)", expanded=False):
        prompt_button(
            "Start with ‘bonjour’ — always",
            "Why is saying ‘bonjour’ before any question so important in France?",
            "fr_bonjour",
            region="France",
            city="Paris",
        )
        prompt_button(
            "Polite complaints",
            "What is a polite way to raise a complaint at a hotel or restaurant in France?",
            "fr_complaint",
            region="France",
            city="Paris",
        )
        prompt_button(
            "Body language in Paris",
            "What kind of body language should I avoid in Paris to not appear rude?",
            "fr_body",
            region="France",
            city="Paris",
        )

    # ===================== INDIA =====================
    st.markdown("### 🇮🇳 India – State-Specific Real Questions")

    # ---------- Tamil Nadu ----------
    with st.expander("🇮🇳 Tamil Nadu (Chennai, Coimbatore)", expanded=False):
        prompt_button(
            "Respecting elders in Chennai",
            "How should I address an elder respectfully in Chennai?",
            "tn_elder",
            region="India",
            city="Chennai",
        )
        prompt_button(
            "Temple etiquette in Tamil Nadu",
            "What should I know about temple etiquette and dress code in Tamil Nadu?",
            "tn_temple",
            region="India",
            city="Chennai",
        )
        prompt_button(
            "Auto-rickshaw negotiation",
            "How can I ask an auto driver in Chennai to go by meter politely?",
            "tn_auto",
            region="India",
            city="Chennai",
        )

    # ---------- Karnataka ----------
    with st.expander("🇮🇳 Karnataka (Bengaluru, Mysuru)", expanded=False):
        prompt_button(
            "Office small talk in Bengaluru",
            "What is a natural way to start small talk with IT coworkers in Bengaluru?",
            "ka_office",
            region="India",
            city="Bengaluru",
        )
        prompt_button(
            "Addressing strangers respectfully",
            "How should I address unknown people respectfully in Bengaluru?",
            "ka_strangers",
            region="India",
            city="Bengaluru",
        )
        prompt_button(
            "Dealing with traffic situations",
            "How can I ask someone politely to move their vehicle in a crowded Bengaluru street?",
            "ka_traffic",
            region="India",
            city="Bengaluru",
        )

    # ---------- Telangana ----------
    with st.expander("🇮🇳 Telangana (Hyderabad)", expanded=False):
        prompt_button(
            "Hyderabadi polite phrases",
            "What are some polite ways to ask for help or directions in Hyderabad?",
            "ts_help",
            region="India",
            city="Hyderabad",
        )
        prompt_button(
            "Restaurant etiquette with spicy food",
            "How do I ask for ‘less spicy’ politely at a restaurant in Hyderabad?",
            "ts_spice",
            region="India",
            city="Hyderabad",
        )
        prompt_button(
            "Talking to auto drivers",
            "How can I politely negotiate fare with auto drivers in Hyderabad?",
            "ts_auto",
            region="India",
            city="Hyderabad",
        )

    # ---------- Maharashtra ----------
    with st.expander("🇮🇳 Maharashtra (Mumbai, Pune)", expanded=False):
        prompt_button(
            "Local train etiquette in Mumbai",
            "What should I know about behavior and etiquette on Mumbai local trains?",
            "mh_trains",
            region="India",
            city="Mumbai",
        )
        prompt_button(
            "Addressing neighbors in apartments",
            "How do I introduce myself and talk politely to neighbors in a Mumbai apartment?",
            "mh_neighbors",
            region="India",
            city="Mumbai",
        )
        prompt_button(
            "Talking to service staff respectfully",
            "How do I politely talk to security guards and drivers in Mumbai?",
            "mh_staff",
            region="India",
            city="Mumbai",
        )

    # ---------- Kerala ----------
    with st.expander("🇮🇳 Kerala (Kochi, Thiruvananthapuram)", expanded=False):
        prompt_button(
            "Talking about food preferences",
            "How can I politely explain my dietary preferences when invited for meals in Kerala?",
            "kl_food",
            region="India",
            city="Kochi",
        )
        prompt_button(
            "Temple etiquette",
            "What should I know before visiting temples in Kerala?",
            "kl_temple",
            region="India",
            city="Kochi",
        )
        prompt_button(
            "Seafood preferences",
            "How do I ask for mild-spice seafood dishes in Kerala?",
            "kl_seafood",
            region="India",
            city="Kochi",
        )
        prompt_button(
            "Public behavior norms",
            "What are general social behavior expectations in Kerala?",
            "kl_behavior",
            region="India",
            city="Kochi",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Tip: Click a prompt, then switch to Ask EchoAtlas — it will be pre-filled!")

# ========================== PAGE: SETTINGS ==========================
else:
    st.markdown("## ⚙️ Settings")
    st.markdown(
        """
        <div class="ea-card">
          Configure EchoAtlas behavior and reset memory if needed.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.checkbox("Show debug info (placeholder)", value=False)

    st.markdown("### Memory Controls")

    if "show_factory_reset_confirm" not in st.session_state:
        st.session_state.show_factory_reset_confirm = False

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("🧨 Factory reset all memories", use_container_width=True):
            st.session_state.show_factory_reset_confirm = True
    with col2:
        st.caption(
            "Schedules a complete wipe of the EchoAtlas memory store. "
            "The deletion happens on the next app restart."
        )

    if st.session_state.show_factory_reset_confirm:
        st.warning(
            "⚠️ You are about to schedule a Factory Reset of the memory store.\n\n"
            "This will delete ALL saved memories. The deletion happens on next restart."
        )
        choice = st.radio(
            "Are you sure?",
            ["No", "Yes"],
            index=0,
            key="factory_reset_choice",
            horizontal=True,
        )
        colA, colB = st.columns(2)
        with colA:
            if st.button("✅ Confirm reset", use_container_width=True):
                if choice == "Yes":
                    try:
                        RESET_FLAG_PATH.write_text("reset", encoding="utf-8")
                        st.success(
                            "🧨 Factory reset scheduled.\n\n"
                            "Please restart the app. On startup, memory_store will be deleted."
                        )
                    except Exception as e:
                        st.error(f"❌ Could not create reset flag: {e}")
                else:
                    st.info("Factory reset cancelled (you selected 'No').")
                st.session_state.show_factory_reset_confirm = False
        with colB:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.show_factory_reset_confirm = False
                st.info("Factory reset cancelled.")
