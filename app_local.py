import streamlit as st
from typing import Dict, Any, List

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

# =========================================
# EXTRA IMPORTS (MIC + LLM FOR CULTURE)
# =========================================
import queue
import json
import time
import shutil
from pathlib import Path

import sounddevice as sd
from vosk import Model, KaldiRecognizer

from openai import OpenAI

client = OpenAI()

# ------------ Vosk setup (adjust model path if needed) ------------
VOSK_MODEL_PATH = "models/vosk-model-small-en-us-0.15"

q: queue.Queue[bytes] = queue.Queue()
vosk_model = Model(VOSK_MODEL_PATH)
rec = KaldiRecognizer(vosk_model, 16000)

# ------------ Factory reset paths ------------
RESET_FLAG_PATH = Path("reset_memory_store.flag")
MEMORY_STORE_PATH = Path("memory_store")


def audio_callback(indata, frames, time_info, status):
    """Vosk audio callback."""
    if status:
        print(status, flush=True)
    q.put(bytes(indata))


def apply_scheduled_factory_reset() -> None:
    """
    If a reset flag exists, wipe the memory_store folder once on startup.

    This runs before setup_memory_schema(), so Chroma will rebuild cleanly.
    """
    if not RESET_FLAG_PATH.exists():
        return

    try:
        if MEMORY_STORE_PATH.exists():
            shutil.rmtree(MEMORY_STORE_PATH)
        RESET_FLAG_PATH.unlink(missing_ok=True)
        # toast is nice, but falls back gracefully if not available
        try:
            st.toast("✅ Memory store factory-reset on startup.", icon="🧨")
        except Exception:
            st.info("✅ Memory store factory-reset on startup.")
    except Exception as e:
        st.error(f"❌ Could not complete scheduled memory reset: {e}")


def generate_dynamic_culture_profile(region: str, location: str) -> dict:
    """
    Use LLM to dynamically generate a culture profile for (region, location).

    Returns a dict with keys: phrase, gesture, tone, custom.
    Uses a small cache in st.session_state to avoid repeating calls.
    """
    if "dynamic_culture_cache" not in st.session_state:
        st.session_state.dynamic_culture_cache = {}

    cache = st.session_state.dynamic_culture_cache
    cache_key = f"{region.strip().lower()}|{location.strip().lower()}"

    # 1) Check cache first
    if cache_key in cache:
        return cache[cache_key]

    # 2) Call LLM
    try:
        prompt = f"""
You are a cultural communication expert.

For the following place:
- Country or State/Region: {region}
- City/Area: {location}

Generate a short, practical profile for how a visitor should speak and behave.
Return ONLY valid JSON with these keys:
- "phrase": a short example phrase for politely asking for something (in English or local language).
- "gesture": a one-sentence description of an appropriate gesture/body language.
- "tone": 2–5 words describing the recommended tone of voice.
- "custom": 1–2 sentences of a key cultural tip for everyday interactions.

Example output:
{{
  "phrase": "Can I get a coffee, please?",
  "gesture": "Smile and make brief eye contact.",
  "tone": "Friendly and polite",
  "custom": "Start with a short greeting before making your request."
}}
        """.strip()

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a cultural communication expert. Return ONLY compact JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )

        content = completion.choices[0].message.content
        data = json.loads(content)

        result = {
            "phrase": (data.get("phrase") or "").strip()
            or f"Hello, could you please help me here in {location}?",
            "gesture": (data.get("gesture") or "Smile gently and be respectful.").strip(),
            "tone": (data.get("tone") or "Polite and friendly").strip(),
            "custom": (
                data.get("custom")
                or f"Be respectful and observe how locals behave in {location}."
            ).strip(),
        }

    except Exception as e:
        st.warning(f"Dynamic culture profile failed: {e}")
        result = {
            "phrase": f"Hello, could you please help me here in {location}?",
            "gesture": "Smile gently and be respectful.",
            "tone": "Polite and friendly",
            "custom": f"Be respectful and observe how locals behave in {location}.",
        }

    cache[cache_key] = result
    return result


# --------- Apply any scheduled factory reset, then init schema ---------
apply_scheduled_factory_reset()
setup_memory_schema()


def generate_cultural_playbook(region: str, city: str) -> dict:
    """
    Build a structured cultural playbook for (region, city) by combining:
    - the dynamic culture profile (tone, gesture, tip)
    - past memories from the memory store
    - an LLM synthesis pass

    Returns a dict ready for UI rendering.
    """
    base = generate_dynamic_culture_profile(region=region, location=city)

    mems = recall_similar(
        region=region,
        location=city,
        user_input="",  # empty → fetch ALL in this scope
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
You are building a CULTURAL PLAYBOOK for visitors to the following location:

Region/Country: {region}
City/Area: {city}

Base cultural profile:
- Suggested tone: {base.get('tone','')}
- Suggested gesture: {base.get('gesture','')}
- Cultural tip: {base.get('custom','')}

Observed past interactions (questions and responses):
{memory_text}

Synthesize a practical CULTURAL PLAYBOOK with this JSON structure ONLY:

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

Return ONLY valid JSON. Do not include any commentary outside the JSON.
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
        data = json.loads(content)
        return data
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
            "do_and_donts": {
                "do": [],
                "dont": [],
            },
            "emerging_patterns_from_memory": {
                "common_questions": [],
                "common_mistakes": [],
                "recommendations": [],
            },
            "examples": [],
        }


def render_cultural_playbook(playbook: dict, region: str, city: str) -> None:
    """Pretty glass UI for the cultural playbook."""
    st.markdown(
        f"""
        <div class="ea-card" style="margin-bottom:1rem;">
          <div style="font-size:1.2rem; font-weight:600; color:#e5e7eb;">
            📘 Cultural Playbook — {city}, {region}
          </div>
          <div style="font-size:0.9rem; color:#cbd5e1; margin-top:4px;">
            A synthesized guide combining EchoAtlas cultural DNA and your past conversations.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cs = playbook.get("communication_style", {}) or {}
    st.markdown("### 🗣 Communication Style")
    st.markdown('<div class="ea-card-soft">', unsafe_allow_html=True)
    st.write(cs)
    st.markdown("</div>", unsafe_allow_html=True)

    et = playbook.get("etiquette", {}) or {}
    st.markdown("### 🤝 Etiquette Guidelines")
    st.markdown('<div class="ea-card-soft">', unsafe_allow_html=True)
    st.write(et)
    st.markdown("</div>", unsafe_allow_html=True)

    dd = playbook.get("do_and_donts", {}) or {}
    do_list = dd.get("do", []) or []
    dont_list = dd.get("dont", []) or []

    st.markdown("### ✔️ Do & ❌ Don’t")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### ✔️ Do")
        if do_list:
            for item in do_list:
                st.markdown(f"- {item}")
        else:
            st.caption("No specific 'Do' items generated yet.")
    with c2:
        st.markdown("#### ❌ Don’t")
        if dont_list:
            for item in dont_list:
                st.markdown(f"- {item}")
        else:
            st.caption("No specific 'Don't' items generated yet.")

    ep = playbook.get("emerging_patterns_from_memory", {}) or {}
    st.markdown("### 🔍 Observed Patterns from Your Questions")
    st.markdown('<div class="ea-card-soft">', unsafe_allow_html=True)
    st.write(ep)
    st.markdown("</div>", unsafe_allow_html=True)

    examples = playbook.get("examples", []) or []
    if examples:
        st.markdown("### 💬 Practical Examples")
        for ex in examples:
            scenario = ex.get("scenario", "Scenario")
            what = ex.get("what_to_say", "")
            how = ex.get("how_to_act", "")
            st.markdown(
                f"""
                <div class="ea-card-soft" style="margin-top:0.4rem;">
                  <div style="font-weight:600; color:#e5e7eb;">🎯 {scenario}</div>
                  <div style="margin-top:4px;"><b>What to say:</b> {what}</div>
                  <div style="margin-top:2px;"><b>How to act:</b> {how}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown("### 💬 Practical Examples")
        st.caption("No concrete examples generated yet for this location.")


def playbook_to_markdown(playbook: dict, region: str, city: str) -> str:
    """Convert a cultural playbook dict to a readable Markdown document."""
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


# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="EchoAtlas 2.0",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================
# GLOBAL CSS – Glass + Illustration + Anim
# =========================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 45%, #020617 100%) fixed;
    color: #e5e7eb;
}

.block-container {
    max-width: 1500px;
    margin: auto;
    padding-top: 1.1rem;
    padding-bottom: 1.4rem;
}

.ea-hero-illustration {
    background-image: url('https://images.unsplash.com/photo-1520975693419-6229dce8f1ed?auto=format&fit=crop&w=1950&q=80');
    background-size: cover;
    background-position: center;
    border-radius: 18px;
    padding: 38px 50px;
    backdrop-filter: blur(4px);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 12px 38px rgba(0,0,0,0.45);
    position: relative;
    overflow: hidden;
}

.ea-hero-illustration::before {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at top, rgba(15,23,42,0.5), rgba(15,23,42,0.8));
    z-index: 0;
}
.ea-hero-inner {
    position: relative;
    z-index: 1;
}
.ea-title {
    font-size: 1.9rem;
    font-weight: 700;
    color: #f9fafb;
}
.ea-sub {
    font-size: 1rem;
    color: #e5e7eb;
    margin-top: 6px;
}

.ea-card {
    background: rgba(15,23,42,0.86);
    border: 1px solid rgba(148,163,184,0.35);
    padding: 20px 24px;
    border-radius: 18px;
    backdrop-filter: blur(16px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.45);
    animation: fadeIn 0.6s ease;
}
.ea-card-soft {
    background: rgba(15,23,42,0.78);
    border: 1px solid rgba(148,163,184,0.25);
    padding: 16px 18px;
    border-radius: 14px;
    backdrop-filter: blur(14px);
    box-shadow: 0 8px 22px rgba(0,0,0,0.4);
}
.ea-label {
    font-weight: 600;
    font-size: 0.85rem;
    color: #a7b5c8;
    margin-bottom: 4px;
}

@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(6px); }
    100% { opacity: 1; transform: translateY(0); }
}

section[data-testid="stSidebar"] {
    background: rgba(15,23,42,0.92);
    border-right: 1px solid rgba(148,163,184,0.35);
    backdrop-filter: blur(18px);
}
.sidebar-title {
    font-size: 1.35rem;
    font-weight: 700;
    padding-bottom: 0.4rem;
}

.ea-transcript-box {
    background: #020617;
    border-radius: 10px;
    padding: 12px 14px;
    border: 1px solid #1e293b;
    font-size: 0.95rem;
    min-height: 56px;
    color: #e5e7eb;
    box-shadow: 0 4px 12px rgba(15,23,42,0.6);
}
.ea-transcript-empty {
    color: #64748b;
    font-style: italic;
}

.ea-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 500;
}
.ea-status-running {
    background: #064e3b;
    color: #bbf7d0;
    border: 1px solid #22c55e;
}
.ea-status-stopped {
    background: #450a0a;
    color: #fecaca;
    border: 1px solid #ef4444;
}
.ea-dot {
    width: 8px;
    height: 8px;
    border-radius: 999px;
}
.ea-dot-running { background: #22c55e; }
.ea-dot-stopped { background: #ef4444; }

@keyframes eaPulse {
    0%   { transform: scaleY(0.6); }
    50%  { transform: scaleY(1.2); }
    100% { transform: scaleY(0.6); }
}
.ea-wave span {
    display: inline-block;
    width: 4px;
    margin: 0 1px;
    border-radius: 999px;
    background: #22c55e;
    animation: eaPulse 1s ease-in-out infinite;
}
.ea-wave span:nth-child(2) { animation-delay: 0.1s; }
.ea-wave span:nth-child(3) { animation-delay: 0.2s; }
.ea-wave span:nth-child(4) { animation-delay: 0.3s; }
.ea-wave span:nth-child(5) { animation-delay: 0.4s; }

div.streamlit-expander {
    border-radius: 14px !important;
    border: 1px solid #1f2937 !important;
    margin-bottom: 0.4rem;
    background: rgba(15,23,42,0.9) !important;
}
div.streamlit-expanderHeader {
    font-weight: 500 !important;
    color: #e5e7eb !important;
}
.ea-mem-meta {
    font-size: 0.75rem;
    color: #9ca3af;
    margin-bottom: 0.35rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================
# SIDEBAR NAV
# =========================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">🌍 EchoAtlas</div>', unsafe_allow_html=True)
    st.caption("Cultural Intelligence Assistant")

    page = st.radio(
        "Navigate",
        [
            "Ask EchoAtlas",
            "Conversation Memory",
            "Cultural Playbook",
            "FAQ & Sample Prompts",   # 👈 NEW
            "Settings",
        ],
        index=0,
    )

    st.markdown("---")
    st.caption("© EchoAtlas · 2025")

# =========================================
# BASIC SESSION STATE
# =========================================
if "selected_region_group" not in st.session_state:
    st.session_state.selected_region_group = "International"
if "selected_region" not in st.session_state:
    st.session_state.selected_region = "United States"
if "selected_city" not in st.session_state:
    st.session_state.selected_city = "New York"

if "last_user_input" not in st.session_state:
    st.session_state.last_user_input = ""
if "last_agent_output" not in st.session_state:
    st.session_state.last_agent_output = {}
if "last_region" not in st.session_state:
    st.session_state.last_region = st.session_state.selected_region
if "last_city" not in st.session_state:
    st.session_state.last_city = st.session_state.selected_city
if "recent_locations" not in st.session_state:
    st.session_state.recent_locations = []

if "recording" not in st.session_state:
    st.session_state.recording = False
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "run_from_mic" not in st.session_state:
    st.session_state.run_from_mic = False
if "prefill_text" not in st.session_state:
    st.session_state.prefill_text = ""
if "prefill_just_set" not in st.session_state:
    st.session_state.prefill_just_set = False

# ============================================================
# PAGE: ASK ECHOATLAS
# ============================================================
if page == "Ask EchoAtlas":
    st.markdown(
        """
        <div class="ea-hero-illustration">
          <div class="ea-hero-inner">
            <div class="ea-title">🌐 EchoAtlas — Cultural Intelligence Assistant</div>
            <div class="ea-sub">
              Speak, explore, and understand cultures worldwide.<br>
              Get real-time cultural cues, tone guidance, and region-aware phrasing.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- LOCATION PICKER ----------------
    st.markdown("### 🌏 Choose Your Region & Location")

    if "selected_region_group" not in st.session_state:
        st.session_state.selected_region_group = "International"
    if "selected_region" not in st.session_state:
        st.session_state.selected_region = "United States"
    if "selected_city" not in st.session_state:
        st.session_state.selected_city = "New York"
    if "region_is_custom" not in st.session_state:
        st.session_state.region_is_custom = False
    if "city_is_custom" not in st.session_state:
        st.session_state.city_is_custom = False
    if "transcript_region" not in st.session_state:
        st.session_state.transcript_region = st.session_state.selected_region
    if "transcript_city" not in st.session_state:
        st.session_state.transcript_city = st.session_state.selected_city

    prev_region = st.session_state.selected_region
    prev_city = st.session_state.selected_city

    region_groups = ["International", "India"]
    current_group = st.session_state.selected_region_group
    if current_group not in region_groups:
        current_group = "International"

    region_group = st.selectbox(
        "🌎 Step 1 — Select Region Group",
        region_groups,
        index=region_groups.index(current_group),
        help="Choose whether you're exploring countries or Indian states.",
    )

    if region_group == "International":
        level2_options = [
            "United States",
            "Canada",
            "United Kingdom",
            "Australia",
            "Singapore",
            "Germany",
            "France",
            "Other (Specify…) 🤔",
        ]
        level2_label = "🌍 Step 2 — Select Country"
    else:
        level2_options = [
            "Tamil Nadu",
            "Karnataka",
            "Kerala",
            "Telangana",
            "Maharashtra",
            "Other (Specify…) 🤔",
        ]
        level2_label = "🇮🇳 Step 2 — Select Indian State"

    if st.session_state.region_is_custom:
        default_level2 = next((opt for opt in level2_options if "Other" in opt), level2_options[0])
    else:
        default_level2 = prev_region if prev_region in level2_options else level2_options[0]

    selected_level2 = st.selectbox(
        level2_label,
        level2_options,
        index=level2_options.index(default_level2),
    )

    if "Other" in selected_level2:
        st.session_state.region_is_custom = True
        level2_custom = st.text_input(
            "Enter Country/State Name",
            value=prev_region if st.session_state.region_is_custom else "",
        )
        final_region = level2_custom.strip() if level2_custom.strip() else "Custom Region"
    else:
        st.session_state.region_is_custom = False
        final_region = selected_level2

    city_options_default = {
        "United States": ["New York", "Los Angeles", "Chicago", "Seattle", "San Francisco", "Other (Specify…)"],
        "Canada": ["Toronto", "Vancouver", "Montreal", "Calgary", "Other (Specify…)"],
        "United Kingdom": ["London", "Manchester", "Birmingham", "Edinburgh", "Other (Specify…)"],
        "Australia": ["Sydney", "Melbourne", "Brisbane", "Perth", "Other (Specify…)"],
        "Singapore": ["Singapore City"],
        "Germany": ["Berlin", "Munich", "Frankfurt", "Hamburg", "Other (Specify…)"],
        "France": ["Paris", "Lyon", "Nice", "Marseille", "Other (Specify…)"],
        "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Other (Specify…)"],
        "Karnataka": ["Bengaluru", "Mysuru", "Mangalore", "Other (Specify…)"],
        "Kerala": ["Kochi", "Thiruvananthapuram", "Kozhikode", "Other (Specify…)"],
        "Telangana": ["Hyderabad", "Warangal", "Other (Specify…)"],
        "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Other (Specify…)"],
    }

    if st.session_state.region_is_custom:
        city_custom = st.text_input(
            "Enter City Name",
            value=prev_city if st.session_state.city_is_custom and prev_region == final_region else "",
        )
        if city_custom.strip():
            final_city = city_custom.strip()
            st.session_state.city_is_custom = True
        else:
            final_city = "Custom City"
            st.session_state.city_is_custom = False
    else:
        city_options = city_options_default.get(final_region, ["Other (Specify…)"])

        if prev_region == final_region:
            if st.session_state.city_is_custom and any("Other" in c for c in city_options):
                default_city_option = next(c for c in city_options if "Other" in c)
            elif prev_city in city_options:
                default_city_option = prev_city
            else:
                default_city_option = city_options[0]
        else:
            default_city_option = city_options[0]

        selected_city = st.selectbox(
            "🏙️ Step 3 — Select City",
            city_options,
            index=city_options.index(default_city_option),
        )

        if "Other" in selected_city:
            city_custom = st.text_input(
                "Enter City Name",
                value=prev_city if (st.session_state.city_is_custom and prev_region == final_region) else "",
            )
            if city_custom.strip():
                final_city = city_custom.strip()
                st.session_state.city_is_custom = True
            else:
                final_city = selected_city
                st.session_state.city_is_custom = False
        else:
            final_city = selected_city
            st.session_state.city_is_custom = False

    if final_region != prev_region or final_city != prev_city:
        st.session_state.transcript = ""
        st.session_state.recording = False
        st.session_state.run_from_mic = False
        st.session_state.transcript_region = final_region
        st.session_state.transcript_city = final_city
        
        # Only clear prefill when user actually changed location,
        # not right after FAQ set a prompt.
        if not st.session_state.get("prefill_just_set", False):
            st.session_state.prefill_text = ""

    st.session_state.selected_region_group = region_group
    st.session_state.selected_region = final_region
    st.session_state.selected_city = final_city

    st.markdown(
        f"""
        <div class="ea-card" style="margin-top:12px; text-align:center;">
            <span style="color:#f8fafc; font-size:1.05rem;">
                📌 <b>Selected:</b> {final_city}, {final_region} ({region_group})
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- INPUT PANEL (MIC / TEXT) ----------------
    st.markdown("## 🎤 Ask EchoAtlas")

    input_mode = st.radio(
        "Choose Input Method",
        ["🎙 Microphone", "⌨️ Text Input"],
        horizontal=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    user_input = ""
    submit_query = False

    with st.container():
        st.markdown('<div class="ea-card">', unsafe_allow_html=True)

        if input_mode == "🎙 Microphone":
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("🎙 Start Listening", use_container_width=True):
                    st.session_state.recording = True
                    st.session_state.transcript = ""
                    st.session_state.run_from_mic = False

            with col2:
                if st.button("⏹ Stop", use_container_width=True):
                    st.session_state.recording = False

            with col3:
                if st.button("🧹 Clear", use_container_width=True):
                    st.session_state.transcript = ""
                    st.session_state.run_from_mic = False

            if st.session_state.recording:
                st.markdown(
                    """
                    <div class='ea-status-pill ea-status-running' style='margin-top:10px;'>
                        <span class='ea-dot ea-dot-running'></span> Listening...
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class='ea-status-pill ea-status-stopped' style='margin-top:10px;'>
                        <span class='ea-dot ea-dot-stopped'></span> Mic Stopped
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            placeholder = st.empty()

            if st.session_state.recording:
                with sd.RawInputStream(
                    samplerate=16000,
                    blocksize=8000,
                    dtype="int16",
                    channels=1,
                    callback=audio_callback,
                ):
                    st.info("🎧 Speak now… press Stop when you are done.")
                    while st.session_state.recording:
                        if not q.empty():
                            data = q.get()
                            if rec.AcceptWaveform(data):
                                result = json.loads(rec.Result())
                                txt = result.get("text", "")
                                if txt:
                                    st.session_state.transcript += " " + txt
                            else:
                                partial = json.loads(rec.PartialResult())
                                if partial.get("partial"):
                                    placeholder.write(
                                        "🗣 " + st.session_state.transcript + " " + partial["partial"]
                                    )
                                    continue
                        placeholder.write("🗣 " + st.session_state.transcript)
                        time.sleep(0.1)

            st.markdown("#### 📝 Captured Transcript")
            transcript = st.session_state.transcript.strip()

            if transcript:
                st.markdown(
                    f"""
                    <div class='ea-transcript-box'>
                        {transcript}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class='ea-transcript-box ea-transcript-empty'>
                        No transcript yet. Press <b>Start Listening</b> and speak.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if transcript and st.button("🚀 Ask EchoAtlas with transcript", use_container_width=True):
                user_input = transcript
                submit_query = True
            else:
                user_input = ""
        else:
            # ----- TEXT MODE -----

                # Prefill text if coming from FAQ
                default_text = st.session_state.get("prefill_text", "")

                typed = st.text_area(
                    "✍️ Type your question here",
                    value=default_text,
                    placeholder="E.g., What is a polite way to ask for directions in this city?",
                    height=140,
                )

                # Keep prefill text updated ONLY in this branch
                st.session_state.prefill_text = typed
                st.session_state.prefill_just_set = False   # 👈 we've now "consumed" the FAQ prefill
                user_input = typed.strip()
                submit_query = st.button("🚀 Ask EchoAtlas", use_container_width=True)


    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## 💬 EchoAtlas Response")

    agent_output: Dict[str, Any] | None = None

    region = st.session_state.selected_region
    city = st.session_state.selected_city

    if submit_query and user_input:
        mode_clean = "Mic" if input_mode.startswith("🎙") else "Text"

        agent_output = run_agent(
            user_input=user_input,
            region=region,
            location=city,
            mode=mode_clean,
            context="default",
        )

        region_is_custom = st.session_state.get("region_is_custom", False)
        city_is_custom = st.session_state.get("city_is_custom", False)

        if region_is_custom or city_is_custom:
            dyn = generate_dynamic_culture_profile(
                region=region,
                location=city,
            )
            agent_output["gesture"] = agent_output.get("gesture") or dyn.get("gesture")
            agent_output["tone"] = agent_output.get("tone") or dyn.get("tone")
            agent_output["custom"] = agent_output.get("custom") or dyn.get("custom")
            if not agent_output.get("phrase"):
                agent_output["phrase"] = dyn.get("phrase")

        store_interaction(
            region=region,
            location=city,
            phrase=user_input,
            tone=agent_output.get("tone", "Neutral"),
            gesture=agent_output.get("gesture", "🤝"),
            custom=agent_output.get("custom", "Be respectful and observe local behavior."),
            mode=mode_clean,
            context="default",
            answer=agent_output.get("phrase", ""),
        )

        st.session_state.last_region = region
        st.session_state.last_city = city
        st.session_state.last_user_input = user_input
        st.session_state.last_agent_output = agent_output

        loc = {"region": region, "city": city}
        recent = st.session_state.get("recent_locations", [])
        recent = [r for r in recent if not (r.get("region") == region and r.get("city") == city)]
        recent.insert(0, loc)
        if len(recent) > 10:
            recent = recent[:10]
        st.session_state.recent_locations = recent

    else:
        last_output = st.session_state.get("last_agent_output")
        if (
            last_output
            and st.session_state.get("last_region") == region
            and st.session_state.get("last_city") == city
        ):
            agent_output = last_output
            user_input = st.session_state.last_user_input
        else:
            agent_output = None
            user_input = ""

    if agent_output:
        phrase = agent_output.get("phrase", "")
        gesture = agent_output.get("gesture", "Smile and be respectful.")
        tone = agent_output.get("tone", "Neutral and polite")
        custom = agent_output.get("custom", "Be respectful and observe how locals behave.")
        mode_clean = "Mic" if input_mode.startswith("🎙") else "Text"

        st.markdown('<div class="ea-card">', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div style="padding:6px 0 10px 0;">
                <div style="font-size:1rem; color:#f1f5f9;">🗣 <b>You said:</b></div>
                <div style="margin-top:4px; color:#cbd5e1; font-size:1.05rem;">
                    “{user_input}”
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<hr style='border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div style="margin-top:8px;">
                <div style="font-size:1rem; color:#f1f5f9;">
                    🤖 <b>EchoAtlas suggests:</b>
                </div>
                <div style="margin-top:6px; font-size:1.0rem; line-height:1.5; color:#e5e7eb;">
                    {phrase}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="display:flex; gap:10px; margin-top:12px; flex-wrap:wrap;">
                <div style="
                    background:#1e3a8a;
                    padding:4px 12px;
                    border-radius:999px;
                    color:#bfdbfe;
                    font-size:0.8rem;
                    border:1px solid #3b82f6;">
                    🌍 {city}, {region}
                </div>
                <div style="
                    background:#4b5563;
                    padding:4px 12px;
                    border-radius:999px;
                    color:#e5e7eb;
                    font-size:0.8rem;
                    border:1px solid #9ca3af;">
                    🎧 Mode: {mode_clean}
                </div>
                <div style="
                    background:#14532d;
                    padding:4px 12px;
                    border-radius:999px;
                    color:#bbf7d0;
                    font-size:0.8rem;
                    border:1px solid #22c55e;">
                    🎵 Tone: {tone}
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
                    <div style="margin-top:6px; font-size:0.95rem; color:#e5e7eb;">
                        {gesture}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="ea-card-soft">
                    <div style="font-size:0.9rem; color:#93c5fd; font-weight:600;">Tone</div>
                    <div style="margin-top:6px; font-size:0.95rem; color:#e5e7eb;">
                        {tone}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""
                <div class="ea-card-soft">
                    <div style="font-size:0.9rem; color:#93c5fd; font-weight:600;">Cultural Tip</div>
                    <div style="margin-top:6px; font-size:0.95rem; color:#e5e7eb;">
                        {custom}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🧠 Related Memories for this City")

        related = recall_similar(
            region=region,
            location=city,
            user_input=user_input,
            mode=None,
            context=None,
            top_k=5,
        )

        if related:
            st.caption(f"Showing up to {min(5, len(related))} related memories for {city}.")
            for idx, m in enumerate(related[:5], start=1):
                preview = m.get("phrase", "") or ""
                if len(preview) > 80:
                    preview = preview[:77] + "..."
                label = f"💬 Memory {idx}: {preview}"
                with st.expander(label):
                    st.markdown(
                        f"<div class='ea-mem-meta'>Region: {m.get('region','')} · Location: {m.get('location','')} · Mode: {m.get('mode','')}</div>",
                        unsafe_allow_html=True,
                    )
                    display_memory(m)
        else:
            st.caption(
                "No related memories yet for this city. As you keep asking questions, "
                "EchoAtlas will build a cultural trail here."
            )
    else:
        st.info("Ask a question above to see a region-aware EchoAtlas response here.")

# ============================================================
# PAGE: CONVERSATION MEMORY
# ============================================================
elif page == "Conversation Memory":
    st.markdown("## 🧠 Conversation Memory")
    st.markdown(
        """
        <div class="ea-card" style="margin-bottom:1rem;">
            Browse and manage stored interactions across regions and cities.
        </div>
        """,
        unsafe_allow_html=True,
    )

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
            msg = delete_memories_for_region(region=region, location=city, mode=None, context=None)
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
                st.markdown(
                    f"<div class='ea-mem-meta'>Region: {m.get('region','')} · Location: {m.get('location','')} · Mode: {m.get('mode','')}</div>",
                    unsafe_allow_html=True,
                )
                display_memory(m)
    else:
        st.info("No memories stored yet for this city. Ask EchoAtlas something first.")

# ============================================================
# PAGE: CULTURAL PLAYBOOK
# ============================================================
elif page == "Cultural Playbook":
    st.markdown("## 📘 Cultural Playbook")

    region = st.session_state.get("selected_region", "United States")
    city = st.session_state.get("selected_city", "New York")

    st.markdown(
        f"""
        <div class="ea-card" style="margin-bottom:1rem;">
            <div style="font-size:0.95rem; color:#cbd5e1;">
                This playbook is generated for:
                <b>{city}, {region}</b>. It blends EchoAtlas' cultural profile with
                patterns learned from your past questions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🔁 Regenerate Playbook for this City"):
        cache = st.session_state.get("cached_playbook", {})
        cache_key = f"{region}|{city}"
        if cache_key in cache:
            del cache[cache_key]
        st.session_state.cached_playbook = cache

    cache_key = f"{region}|{city}"
    cache = st.session_state.get("cached_playbook", {})
    playbook = cache.get(cache_key)

    if playbook is None:
        with st.spinner("Synthesizing cultural playbook from EchoAtlas memory..."):
            playbook = generate_cultural_playbook(region=region, city=city)
        cache[cache_key] = playbook
        st.session_state.cached_playbook = cache

    if playbook:
        json_str = json.dumps(playbook, indent=2, ensure_ascii=False)
        md_str = playbook_to_markdown(playbook, region, city)

        col_export_1, col_export_2 = st.columns(2)
        with col_export_1:
            st.download_button(
                "⬇️ Export as JSON",
                data=json_str,
                file_name=f"echoatlas_playbook_{city}_{region}.json",
                mime="application/json",
                use_container_width=True,
            )
        with col_export_2:
            st.download_button(
                "⬇️ Export as Markdown",
                data=md_str,
                file_name=f"echoatlas_playbook_{city}_{region}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        render_cultural_playbook(playbook, region, city)
    else:
        st.info("No playbook available yet. Try asking a few questions in this city first.")

# ============================================================
# PAGE: FAQ & SAMPLE PROMPTS (UPGRADED ENGAGING VERSION)
# ============================================================
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

    def prompt_button(label: str, prompt: str, key: str):
        if st.button(f"💬 {label}", key=key, use_container_width=True):
            st.session_state.prefill_text = prompt
            st.session_state.prefill_just_set = True
            st.success("Loaded into Ask EchoAtlas — switch to that tab to send it!")

    # -------------- UNITED STATES --------------
    with st.expander("🌍 United States (New York, Seattle, Chicago)", expanded=False):
        prompt_button(
            "Ask a stranger for help without sounding intrusive",
            "How do I politely ask a stranger for help in New York without sounding intrusive?",
            "us_ask_help"
        )
        prompt_button(
            "Small talk at a US tech company",
            "What is a natural way to start small talk with coworkers at a US tech company?",
            "us_smalltalk"
        )
        prompt_button(
            "Talking to someone who seems busy",
            "How do I approach someone who looks busy without appearing rude?",
            "us_busy"
        )
        prompt_button(
            "Declining an invite politely",
            "How do I politely decline a social invitation in the US without offending the person?",
            "us_decline"
        )
        prompt_button(
            "Handling direct feedback",
            "How should I respond when an American coworker gives very direct feedback?",
            "us_direct"
        )

    # -------------- CANADA --------------
    with st.expander("🇨🇦 Canada (Toronto, Vancouver)", expanded=False):
        prompt_button(
            "Apologizing in a Canadian way",
            "What is the most Canadian way to apologize after bumping into someone?",
            "ca_apology"
        )
        prompt_button(
            "Being friendly but not intrusive",
            "How do I engage in friendly small talk with Canadians without crossing boundaries?",
            "ca_friendly"
        )
        prompt_button(
            "Asking someone to repeat themselves",
            "What’s a polite way to ask someone to repeat themselves in Canada?",
            "ca_repeat"
        )
        prompt_button(
            "Making dinner plans politely",
            "How do Canadians usually phrase dinner invitations politely?",
            "ca_dinner"
        )

    # -------------- UNITED KINGDOM --------------
    with st.expander("🇬🇧 United Kingdom (London, Manchester)", expanded=False):
        prompt_button(
            "Asking for directions in London",
            "How do I ask for directions in London without sounding abrupt?",
            "uk_directions"
        )
        prompt_button(
            "Joining a queue properly",
            "What should I know about queue etiquette in the UK?",
            "uk_queue"
        )
        prompt_button(
            "Polite disagreement in a meeting",
            "How should I express disagreement politely in a UK business meeting?",
            "uk_disagree"
        )
        prompt_button(
            "Small talk that British people enjoy",
            "What are some safe and enjoyable small talk topics for people in the UK?",
            "uk_smalltalk"
        )

    # -------------- AUSTRALIA --------------
    with st.expander("🇦🇺 Australia (Sydney, Melbourne)", expanded=False):
        prompt_button(
            "Friendly teasing (“banter”)",
            "How should I respond to friendly teasing or ‘banter’ in Australian culture?",
            "au_banter"
        )
        prompt_button(
            "Ordering coffee in a busy Aussie café",
            "What’s the polite and quick way to order coffee in a busy Melbourne café?",
            "au_coffee"
        )
        prompt_button(
            "Understanding Aussie humor",
            "How do I understand and react to Australian humor without misunderstanding it?",
            "au_humor"
        )
        prompt_button(
            "Saying no without sounding rude",
            "How can I say ‘no’ politely in Australian social situations?",
            "au_no"
        )

    # -------------- SINGAPORE --------------
    with st.expander("🇸🇬 Singapore (Singapore City)", expanded=False):
        prompt_button(
            "Hawker centre etiquette",
            "What’s the polite way to ask for customization at a hawker centre in Singapore?",
            "sg_hawker"
        )
        prompt_button(
            "Business greeting etiquette",
            "How formal should I be when greeting someone in a Singapore business meeting?",
            "sg_greeting"
        )
        prompt_button(
            "When not to joke",
            "What kind of jokes or comments should I avoid in Singapore?",
            "sg_jokes"
        )
        prompt_button(
            "Talking to older Singaporeans respectfully",
            "How do I address and speak to an older person respectfully in Singapore?",
            "sg_elder"
        )

    # -------------- GERMANY --------------
    with st.expander("🇩🇪 Germany (Berlin, Munich)", expanded=False):
        prompt_button(
            "Being on time (very important!)",
            "Why is punctuality so important in Germany and how do I show respect?",
            "de_punctual"
        )
        prompt_button(
            "Direct but respectful communication",
            "How do I communicate in a direct but respectful manner in Germany?",
            "de_direct"
        )
        prompt_button(
            "Splitting the bill",
            "How do I politely ask to split the bill in Germany?",
            "de_split"
        )
        prompt_button(
            "Discussing work-life boundaries",
            "How do Germans perceive work-life boundaries and how should I respect them?",
            "de_boundaries"
        )

    # -------------- FRANCE --------------
    with st.expander("🇫🇷 France (Paris, Lyon)", expanded=False):
        prompt_button(
            "Start with ‘bonjour’ — always!",
            "Why is saying ‘bonjour’ before any question so important in France?",
            "fr_bonjour"
        )
        prompt_button(
            "Restaurant etiquette",
            "How do I politely call a server in a French restaurant?",
            "fr_server"
        )
        prompt_button(
            "Polite complaints",
            "What is a polite way to raise a complaint at a hotel or restaurant in France?",
            "fr_complaint"
        )
        prompt_button(
            "Talking to Parisians politely",
            "What tone do Parisians appreciate in short interactions?",
            "fr_tone"
        )

    # ===================== INDIA =====================
    st.markdown("### 🇮🇳 India – State-Specific Real Questions")

    # ---------- Tamil Nadu ----------
    with st.expander("🇮🇳 Tamil Nadu (Chennai, Coimbatore)", expanded=False):
        prompt_button(
            "Respecting elders in Chennai",
            "How should I address an elder respectfully in Chennai?",
            "tn_elder"
        )
        prompt_button(
            "Temple etiquette in Tamil Nadu",
            "What should I know about temple etiquette and dress code in Tamil Nadu?",
            "tn_temple"
        )
        prompt_button(
            "Auto-rickshaw negotiation",
            "How can I ask an auto driver in Chennai to go by meter politely?",
            "tn_auto"
        )
        prompt_button(
            "Ordering food politely",
            "How do I ask for less spicy food in Tamil Nadu without sounding rude?",
            "tn_spice"
        )

    # ---------- Karnataka ----------
    with st.expander("🇮🇳 Karnataka (Bengaluru, Mysuru)", expanded=False):
        prompt_button(
            "Talking to IT coworkers in Bengaluru",
            "How should I greet or start conversations in a Bengaluru IT company?",
            "ka_it"
        )
        prompt_button(
            "Meter request for auto",
            "What is the polite way to ask a Bengaluru auto driver to use the meter?",
            "ka_meter"
        )
        prompt_button(
            "Breaking the ice in Bengaluru",
            "What are natural ice-breakers when talking to locals in Bengaluru?",
            "ka_ice"
        )
        prompt_button(
            "Respectful tone with elders in Karnataka",
            "How should I speak to an elder respectfully in Karnataka?",
            "ka_elder"
        )

    # ---------- Kerala ----------
    with st.expander("🇮🇳 Kerala (Kochi, Thiruvananthapuram)", expanded=False):
        prompt_button(
            "Backwater recommendations",
            "How do I politely ask a local in Kochi for backwater tourism suggestions?",
            "kl_backwater"
        )
        prompt_button(
            "Temple etiquette",
            "What should I know before visiting temples in Kerala?",
            "kl_temple"
        )
        prompt_button(
            "Seafood preferences",
            "How do I ask for mild-spice seafood dishes in Kerala?",
            "kl_seafood"
        )
        prompt_button(
            "Public behavior norms",
            "What are general social behavior expectations in Kerala?",
            "kl_behavior"
        )

    # ---------- Telangana ----------
    with st.expander("🇮🇳 Telangana (Hyderabad, Warangal)", expanded=False):
        prompt_button(
            "Talking to elders in Hyderabad",
            "How do I speak respectfully to elders in Hyderabad?",
            "ts_elder"
        )
        prompt_button(
            "Asking a vendor for lower price",
            "What is a polite way to ask a street vendor for a lower price in Hyderabad?",
            "ts_vendor"
        )
        prompt_button(
            "Office behavior in Hyderabad",
            "How formal or informal should I be in Hyderabad workplaces?",
            "ts_office"
        )
        prompt_button(
            "Handling spicy dishes politely",
            "What is a polite way to request less spicy food in Telangana?",
            "ts_spice"
        )

    # ---------- Maharashtra ----------
    with st.expander("🇮🇳 Maharashtra (Mumbai, Pune)", expanded=False):
        prompt_button(
            "Crowded train etiquette",
            "How do I ask for help inside a crowded Mumbai local train?",
            "mh_train"
        )
        prompt_button(
            "Small talk in Mumbai offices",
            "How do I start casual conversations with coworkers in Mumbai?",
            "mh_smalltalk"
        )
        prompt_button(
            "Housing society manners",
            "What’s a polite way to complain about noise to a neighbor in Maharashtra?",
            "mh_noise"
        )
        prompt_button(
            "Talking to service staff respectfully",
            "How do I politely talk to security guards and drivers in Mumbai?",
            "mh_staff"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Tip: Click a prompt, then switch to Ask EchoAtlas — it will be pre-filled!")

# ============================================================
# PAGE: SETTINGS
# ============================================================
else:
    st.markdown("## ⚙️ Settings")
    st.markdown(
        """
        <div class="ea-card" style="margin-bottom:1rem;">
            Configure EchoAtlas behavior, appearance, and data preferences.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.selectbox("Theme", ["Glassmorphism Dark (current)", "Light (future)", "High Contrast (future)"])
    st.checkbox("Enable microphone features", value=True)
    st.checkbox("Show developer debug info", value=False)

    st.markdown("### Memory Controls")

    if "show_factory_reset_confirm" not in st.session_state:
        st.session_state.show_factory_reset_confirm = False

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("🧨 Factory reset all memories", use_container_width=True):
            st.session_state.show_factory_reset_confirm = True
    with col2:
        st.caption(
            "Schedule a complete wipe of the EchoAtlas memory store. "
            "The actual deletion happens the next time you restart the app."
        )

    if st.session_state.show_factory_reset_confirm:
        st.warning(
            "⚠️ You are about to schedule a **Factory Reset** of the memory store.\n\n"
            "This will delete **ALL saved memories** for every region, city, mode, and context.\n"
            "The deletion will occur on the next app restart."
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
                            "Please **stop & restart** the Streamlit app.\n"
                            "On next startup, the `memory_store` folder will be deleted "
                            "and a fresh memory database will be created."
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
