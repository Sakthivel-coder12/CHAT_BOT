# 🤖 NaviBot — GenAI Fintech FAQ Chatbot

A **prompt-engineered LLM chatbot** built for fintech customer support, powered by **Ollama (local LLM — phi3)**. Runs fully offline — no API key, no quota limits. Demonstrates advanced AI engineering: Chain-of-Thought reasoning, few-shot prompting, structured output parsing, and real-time **AI quality metrics monitoring**.

---

## 🎯 Key Features

| Feature | Description |
|---|---|
| **Local LLM** | Runs on Ollama (phi3) — offline, no API key needed |
| **Prompt Engineering** | Chain-of-Thought (CoT) + Few-Shot + Domain constraints |
| **Structured Output** | Model responds in JSON for reliable parsing |
| **Quality Metrics** | Tracks relevance, completeness, confidence, latency per response |
| **Hallucination Control** | Domain enforcement + confidence scoring |
| **Conversation Memory** | Maintains last 6 turns of context |
| **Session Reports** | Aggregated quality benchmarks after each session |

---

## 📁 Project Structure

```
genai-faq-chatbot/
├── main.py                  # Entry point — run this to start chatbot
├── requirements.txt         # Python dependencies
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── chatbot.py           # Core chatbot engine (Ollama backend)
│   ├── prompt_engine.py     # Prompt engineering (CoT + few-shot)
│   └── metrics.py           # AI quality metrics tracker
│
├── data/
│   └── fintech_faqs.json    # Sample FAQ dataset
│
├── logs/
│   └── chat_metrics.jsonl   # Auto-generated interaction logs
│
└── tests/
    └── test_chatbot.py      # Unit tests (pytest)
```

---

## ⚡ Quick Start

### 1. Install Ollama
Download from: https://ollama.com/download

### 2. Pull the phi3 model
```bash
ollama pull phi3
```

### 3. Start Ollama server (keep this terminal open)
```bash
ollama serve
```

### 4. Clone and install dependencies
```bash
git clone https://github.com/YOUR_USERNAME/genai-faq-chatbot.git
cd genai-faq-chatbot
pip install -r requirements.txt
```

### 5. Run the chatbot (in a new terminal)
```bash
python main.py
```

### 6. Run with a different model
```bash
python main.py --model llama3.2
```

### 7. Run tests
```bash
python -m pytest tests/ -v
```

---

## 💬 Usage

```
╔══════════════════════════════════════════════════════════════════╗
║          NaviBot — GenAI Fintech FAQ Chatbot                    ║
╚══════════════════════════════════════════════════════════════════╝

  You: What is a mutual fund?

  NaviBot: A mutual fund pools money from investors to invest in
  stocks, bonds, or securities managed by professionals...

  📊 Quality Metrics:
     Relevance     : ██████████ 1.00
     Completeness  : █████████░ 0.90
     Confidence    : █████████░ 0.95
     Overall Score : 0.948 / 1.000
     Latency       : 3243 ms  [GOOD]
     Halluc. Risk  : 0.05  (lower is better)
```

**Special Commands:**
- `metrics` — View full session quality report
- `reset` — Start a new conversation
- `toggle` — Show/hide metrics per response
- `quit` — Exit and view session report

---

## 📊 Quality Metrics Explained

| Metric | Description |
|---|---|
| **Relevance Score** | Is the response relevant to fintech domain? (0.0–1.0) |
| **Completeness Score** | Does it fully address the question? (0.0–1.0) |
| **Confidence Score** | Model's self-reported confidence via CoT (0.0–1.0) |
| **Hallucination Risk** | Inverse confidence proxy — lower is better |
| **Latency** | Response time in milliseconds |
| **Overall Quality** | Weighted: 35% relevance + 35% completeness + 30% confidence |

---

## 🧠 Prompt Engineering Techniques Used

1. **System Persona** — Domain-specific role with strict constraints
2. **Few-Shot Examples** — 3 curated examples guide model behavior
3. **Chain-of-Thought (CoT)** — Forces step-by-step reasoning before answering
4. **Structured Output** — JSON format ensures reliable parsing
5. **Conversation Memory** — Last 6 turns injected for context continuity
6. **Temperature Control** — Set to 0.3 for factual, consistent responses

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Ollama** — Local LLM runtime (phi3 / llama3.2 / gemma)
- **requests** — HTTP calls to Ollama API
- **pytest** — Unit testing

---

## 👨‍💻 Author

**Sakthivel M** — AI/ML Engineer | NLP & GenAI Specialist  
VIT Vellore | CGPA: 9.17 / 10  
[LinkedIn](https://linkedin.com) • [GitHub](https://github.com) • [LeetCode](https://leetcode.com)

---

## 📄 License

MIT License — free to use, modify, and distribute.
