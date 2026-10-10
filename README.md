# 🔮 ORACLE — Predictive Competitive Intelligence

> **Built for HackIndia AgentVerse 2026**

An autonomous multi-agent AI engine that predicts what a competitor will do in the next 90 days using **5 SerpApi engines** and an **adversarial multi-agent debate**.

![SerpApi](https://img.shields.io/badge/SerpApi-5%20Engines-FF5A1F)
![Groq](https://img.shields.io/badge/Groq-gpt--oss--120b-0A0A0A)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![anime.js](https://img.shields.io/badge/anime.js-Animated-yellow)

---

## 🎯 What It Does

ORACLE replaces weeks of manual competitive research with a **60-second AI analysis**:

1. **Scans 5 Signal Sources** — Hiring, Patents, Pricing, News, Position
2. **Scores Signal Strength** — Each category gets a 1-10 confidence rating
3. **Runs Adversarial Analysis** — A Threat Analyst predicts incoming moves, an Opportunity Analyst finds counter-plays
4. **Delivers a 90-Day Forecast** — ALERT, MONITOR, STABLE, or OPPORTUNITY verdict
5. **Exports a Briefing** — Download the full intelligence report as PDF

---

## 🧠 The 5 AI Agents

| Agent | Job |
|-------|-----|
| 🔍 **Signal Scanner** | Queries 5 SerpApi engines in parallel |
| 📊 **Pattern Analyst** | Scores each finding for signal strength |
| 🚨 **Threat Analyst** | Predicts what the competitor will do |
| 🎯 **Opportunity Analyst** | Finds where to counter-strike |
| 🔮 **Forecaster** | Delivers the final 90-day verdict |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Uvicorn (Python 3.13) |
| AI Models | Groq (`openai/gpt-oss-120b`) |
| Search | SerpApi (5 engines) |
| Frontend | HTML + Tailwind + anime.js |
| Streaming | Server-Sent Events (SSE) |

---

## 🚀 How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/Mayank-Kadam/ORACLE-Competitive-Intelligence.git
cd ORACLE-Competitive-Intelligence
```

### 2. Install dependencies
```bash
pip install fastapi uvicorn langchain langchain-groq requests python-dotenv
```

### 3. Get your API keys (both are FREE)
- **Groq:** [console.groq.com/keys](https://console.groq.com/keys)
- **SerpApi:** [serpapi.com/users/sign_up](https://serpapi.com/users/sign_up)

### 4. Create `.env` in the root folder
```
GROQ_API_KEY=your_groq_key_here
SERPAPI_API_KEY=your_serpapi_key_here
```

### 5. Start the backend
```bash
python -m uvicorn backend.app:app --reload --port 8001
```

### 6. Open `frontend/index.html` in your browser

### 7. Enter a competitor (e.g., "Tesla") and click Run

---

## 📊 Sample Analysis

Try ORACLE on:
- **Tesla** — "An electric vehicle and clean energy company"
- **Nvidia** — "A semiconductor company designing GPUs for AI"
- **OpenAI** — "An AI research and deployment company"

Each analysis returns a 90-day verdict with full source citations and a downloadable PDF briefing.

---

## 📁 Project Structure

```
ORACLE-Competitive-Intelligence/
├── backend/
│   └── app.py                # FastAPI server + 5 SerpApi engines + 5 AI agents
├── frontend/
│   └── index.html            # Payway-inspired UI + real-time SSE streaming
├── requirements.txt
├── Procfile
└── README.md
```

---

## 🔑 Environment Variables

| Variable | Description | Where to Get It |
|----------|-------------|-----------------|
| `GROQ_API_KEY` | Groq LLM inference key | https://console.groq.com |
| `SERPAPI_API_KEY` | SerpApi search engine key | https://serpapi.com |

---

## 🎥 Demo Video

[Link to 3-minute demo]

---

## 📜 License

MIT — Free to use, modify, and distribute.

---

## 🙏 Acknowledgments

- **SerpApi** for the multi-engine search API
- **Groq** for blazing-fast free LLM inference
- **anime.js** for the smooth frontend animations

---

**Built for HackIndia AgentVerse 2026.**
