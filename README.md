
---
title: EchoAtlas
emoji: 🌍
colorFrom: indigo
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
license: mit
---

# 🌍 EchoAtlas – Cultural Intelligence Assistant

EchoAtlas is a next-generation **cultural intelligence assistant** that helps travelers, immigrants, professionals, and global citizens communicate appropriately across different cities and cultures around the world.

This Hugging Face Space runs the **text-only version** of EchoAtlas (microphone removed), fully optimized for CPU spaces.  
HF CPU does **not** support real-time WebRTC or Vosk ASR — therefore, voice features are excluded in this version.

---

# 🚀 What EchoAtlas Can Do

### 🌏 1. Region-Aware Cultural Phrasing
Ask questions about any:
- Country  
- State or province  
- City  

EchoAtlas returns:
- Polite phrasing  
- Tone guidance  
- Gesture/body-language tips  
- Etiquette cues  
- Local do’s & don’ts  
- Cultural insights  

---

### 🧠 2. Conversation Memory (ChromaDB)

EchoAtlas remembers your past questions and responses, grouped by **region + city**, including:

- What you asked  
- How EchoAtlas responded  
- Tone, gesture, tips  
- Mode (text)  

You can:
- Browse memories  
- Delete memories per location  
- Reset the entire memory database  

---

### 📘 3. Dynamic Cultural Profiles

If you choose a **custom country or city**, EchoAtlas automatically generates a tailored culture profile:

- Suggested tone  
- Suggested gesture  
- Cultural tip  
- Example phrasing  

This enables guidance even for uncommon or remote cities.

---

### 📙 4. Cultural Playbook Generator

EchoAtlas synthesizes a full cultural playbook for any region/city, combining:

- Communication style  
- Etiquette rules  
- Do’s & don’ts  
- Patterns from your own previous interactions  
- Practical examples with “what to say” and “how to act”

You can export the playbook as:
- 📄 JSON  
- 📝 Markdown  

---

### 💬 5. Sample Prompt Library

EchoAtlas includes curated prompts for:

- USA  
- Canada  
- UK  
- Australia  
- Singapore  
- Germany  
- France  
- India (state-specific)

Click a prompt → auto-loads into Ask EchoAtlas.

---

# 🧱 Architecture Overview



app.py ← Main Streamlit interface (formerly app_glass_ultra.py)
agents/
memory_agent.py ← ChromaDB memory logic
langchain_runner.py ← LLM orchestration
langchain_tools.py ← (Optional future tools)
memory_store/ ← Auto-created persistent vector DB


### Technologies Used
- **Streamlit** – UI  
- **OpenAI GPT-4o-mini** – Cultural reasoning  
- **LangChain** – Prompt runner  
- **ChromaDB** – Long-term memory  
- **JSON-driven modeling** – Cultural profiles  

---

# 🧩 Requirements (Install Automatically on HF)

Your repo must include this `requirements.txt`:



streamlit
openai>=1.0.0
langchain
langchain-openai
chromadb
tiktoken
python-dotenv
pydantic


No microphone/audio libraries are needed.

---

# 🔧 Environment Variables (HF Spaces)

Go to:

### **Settings → Variables and Secrets**

Add:



OPENAI_API_KEY = your_openai_key_here


This is required for all phrasing, culture generation, and playbook synthesis.

---

# 🚀 Deployment Instructions (Hugging Face)

This README header tells HF to:

- Use the **Streamlit SDK**
- Run **app.py**
- Map `$PORT` automatically

HF internally runs:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT


You do not need to configure anything else.

🏃 Running EchoAtlas Locally

You can also run this on your laptop:

streamlit run app.py


Set your OpenAI key:

Windows:
set OPENAI_API_KEY=your_key

Mac/Linux:
export OPENAI_API_KEY=your_key

🧪 Memory Storage

EchoAtlas automatically creates:

/memory_store/


On Hugging Face:

Memory persists across container restarts

Only a factory-reset deletes it

❓ FAQ
❓ Does this version support microphone/voice?

No — Hugging Face CPU Spaces do not support:

WebRTC

Vosk ASR

Real-time audio streaming

A Deepgram/Whisper version can be built, but needs Streamlit Cloud / Railway / Render.

❓ Does memory persist after restart?

Yes — Hugging Face preserves workspace files.

❓ Does EchoAtlas work offline?

No — it requires OpenAI GPT-4o-mini.

❓ Can you help deploy a voice-enabled version?

Yes — I can generate:

Deepgram streaming version

Whisper Realtime version

Gradio version

Streamlit Cloud version

Just ask.

💙 Credits

Built using:

Streamlit

OpenAI GPT-4o

LangChain

ChromaDB

Python

Created by Suresh Krishnan
Assisted by ChatGPT
