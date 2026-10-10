# ==========================================================
# ORACLE BACKEND — Predictive Competitive Intelligence (v4)
# Fixed: jobs/shopping parsing + robust verdict detection
# ==========================================================
import os
import json
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not GROQ_API_KEY or not SERPAPI_API_KEY:
    raise ValueError("Missing API keys!")

llm = ChatGroq(api_key=GROQ_API_KEY, model="openai/gpt-oss-120b", temperature=0.1)
creative_llm = ChatGroq(api_key=GROQ_API_KEY, model="openai/gpt-oss-120b", temperature=0.4)

app = FastAPI(title="Oracle API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class AnalyzeRequest(BaseModel):
    company: str
    description: str

# ==== SERPAPI WITH ROBUST PARSING ====
def serpapi_call(query, engine):
    """Returns list of dicts with title, link, snippet, thumbnail, source."""
    url = "https://serpapi.com/search"
    params = {"q": query, "api_key": SERPAPI_API_KEY, "engine": engine}

    # Location helps jobs and shopping return real results
    if engine in ["google_jobs", "google_shopping"]:
        params["location"] = "United States"

    try:
        response = requests.get(url, params=params, timeout=20)
        results = response.json()

        # Try each result key in order
        search_items = (
            results.get("organic_results") or
            results.get("news_results") or
            results.get("jobs_results") or
            results.get("shopping_results") or
            []
        )

        items = []
        for r in search_items[:3]:
            # --- Link extraction ---
            link = r.get("link", "")
            if not link and isinstance(r.get("source"), dict):
                link = r["source"].get("link", "")
            # Jobs: link lives inside apply_options
            if not link and isinstance(r.get("apply_options"), list) and r["apply_options"]:
                try:
                    link = r["apply_options"][0].get("link", "")
                except Exception:
                    pass
            if not link:
                link = r.get("share_link", "")

            # --- Snippet extraction ---
            snippet = r.get("snippet") or r.get("description") or ""
            if not snippet and r.get("price"):
                snippet = f"Price: {r['price']}"
            if not snippet and r.get("extensions"):
                snippet = " · ".join(r["extensions"][:3])
            # Strip HTML and truncate
            snippet = snippet.replace("<b>", "").replace("</b>", "").replace("<br>", " ")
            snippet = snippet[:250].strip()

            # --- Thumbnail extraction ---
            thumbnail = r.get("thumbnail", "")
            if not thumbnail and isinstance(r.get("source"), dict):
                thumbnail = r["source"].get("logo", "")

            # --- Source name extraction ---
            source_name = ""
            if isinstance(r.get("source"), dict):
                source_name = r["source"].get("name", "")
            elif isinstance(r.get("source"), str):
                source_name = r["source"]
            elif r.get("company_name"):
                source_name = r["company_name"]
            elif r.get("via"):
                source_name = r["via"]

            items.append({
                "title": r.get("title", "No title"),
                "link": link,
                "snippet": snippet,
                "thumbnail": thumbnail,
                "source": source_name,
            })
        return items
    except Exception as e:
        return [{"title": "Search error", "snippet": str(e), "link": "", "thumbnail": "", "source": ""}]


def format_for_llm(items, index_offset=0):
    output = ""
    for i, item in enumerate(items):
        idx = index_offset + i + 1
        output += f"[{idx}] {item['title']}\n"
        if item.get('source'):
            output += f"    Source: {item['source']}\n"
        if item.get('snippet'):
            output += f"    Snippet: {item['snippet']}\n"
        output += "\n"
    return output if output else "No results found."


# ==== EVENT STREAM ====
def event_stream(company, description):
    def emit(event_type, data):
        return f"data: {json.dumps({'type': event_type, 'data': data})}\n\n"

    yield emit("start", {"company": company, "description": description})

    signals = [
        ("hiring",   "google_jobs",     f"{company} engineer",                               "Hiring"),
        ("patents",  "google_scholar",  f"{company} patents research innovation",            "Patents"),
        ("pricing",  "google_shopping", f"{company} product",                                "Pricing"),
        ("news",     "google_news",     f"{company} strategy announcement",                  "News"),
        ("position", "google",          f"{company} market expansion partnership strategy",  "Position"),
    ]

    all_sources = []
    research_doc = ""
    source_counter = 0

    for key, engine, query, label in signals:
        yield emit("signal_start", {"signal": key})
        items = serpapi_call(query, engine)
        research_doc += f"=== {label.upper()} SIGNALS ===\n"
        research_doc += format_for_llm(items, source_counter)
        research_doc += "\n"
        for item in items:
            source_counter += 1
            item_with_idx = dict(item)
            item_with_idx["id"] = source_counter
            item_with_idx["category"] = label
            all_sources.append(item_with_idx)
        yield emit("signal_complete", {"signal": key, "count": len(items), "sources": items})

    yield emit("sources_collected", {"sources": all_sources})

    # ---- Pattern Analysis ----
    yield emit("pattern_start", {})
    pattern_prompt = (
        f"You are a competitive intelligence analyst. Analyze these signals about {company} "
        f"(which is: {description}).\n\n"
        f"Signals:\n{research_doc}\n\n"
        f"Output EXACTLY this structure — no preamble, no closing summary:\n\n"
        f"## Signal Strength Summary\n\n"
        f"| Category | Strength | Key Finding | Sources |\n"
        f"|----------|----------|-------------|---------|\n"
        f"| Hiring | N/10 | one-line finding | [1] |\n"
        f"| Patents | N/10 | one-line finding | [2] |\n"
        f"| Pricing | N/10 | one-line finding | [3] |\n"
        f"| News | N/10 | one-line finding | [4] |\n"
        f"| Position | N/10 | one-line finding | [5] |\n\n"
        f"Replace N with the actual strength number (1-10). "
        f"## Detailed Findings\n\n"
        f"### Hiring — Strength: N/10\n"
        f"- **What we found:** one plain-English sentence\n"
        f"- **Why it matters:** one sentence on strategic intent\n"
        f"- **Sources:** [1]\n\n"
        f"### Patents — Strength: N/10\n"
        f"- **What we found:** one sentence\n"
        f"- **Why it matters:** one sentence\n"
        f"- **Sources:** [2]\n\n"
        f"### Pricing — Strength: N/10\n"
        f"- **What we found:** one sentence\n"
        f"- **Why it matters:** one sentence\n"
        f"- **Sources:** [3]\n\n"
        f"### News — Strength: N/10\n"
        f"- **What we found:** one sentence\n"
        f"- **Why it matters:** one sentence\n"
        f"- **Sources:** [4]\n\n"
        f"### Position — Strength: N/10\n"
        f"- **What we found:** one sentence\n"
        f"- **Why it matters:** one sentence\n"
        f"- **Sources:** [5]\n\n"
        f"RULES:\n"
        f"- Simple language, no jargon.\n"
        f"- One sentence per bullet.\n"
        f"- Cite sources ONLY with [N] markers.\n"
        f"- No URLs in the text.\n"
        f"- No preamble, no conclusion."
    )
    pattern = llm.invoke([HumanMessage(content=pattern_prompt)]).content
    yield emit("pattern_complete", {"pattern": pattern})

    # ---- Threat Analysis ----
    yield emit("threat_start", {})
    threat_prompt = (
        f"You are a Threat Analyst. Based on these signals about {company}, predict what they will do "
        f"in the next 90 days that could harm competitors.\n\n"
        f"Signals:\n{pattern}\n\n"
        f"Output EXACTLY this structure:\n\n"
        f"**Prediction 1:** <one clear sentence describing the move>\n"
        f"**Confidence:** XX% · **Sources:** [N]\n\n"
        f"**Prediction 2:** <one clear sentence>\n"
        f"**Confidence:** XX% · **Sources:** [N]\n\n"
        f"**Prediction 3:** <one clear sentence>\n"
        f"**Confidence:** XX% · **Sources:** [N]\n\n"
        f"RULES:\n"
        f"- Each prediction is ONE sentence.\n"
        f"- Cite sources with [N] markers.\n"
        f"- No preamble, no conclusion."
    )
    threat = creative_llm.invoke([HumanMessage(content=threat_prompt)]).content
    yield emit("threat_complete", {"threat": threat})

    # ---- Opportunity Analysis ----
    yield emit("opportunity_start", {})
    opportunity_prompt = (
        f"You are an Opportunity Analyst. Based on these signals about {company}, identify where "
        f"competitors can gain advantage.\n\n"
        f"Signals:\n{pattern}\n\n"
        f"Output EXACTLY this structure:\n\n"
        f"**The Opening:** <one sentence describing the strategic window>\n\n"
        f"**Counter-Play 1:** <one sentence with sources [N]>\n\n"
        f"**Counter-Play 2:** <one sentence with sources [N]>\n\n"
        f"**Counter-Play 3:** <one sentence with sources [N]>\n\n"
        f"RULES:\n"
        f"- One sentence per item.\n"
        f"- Cite sources with [N] markers.\n"
        f"- No preamble, no conclusion."
    )
    opportunity = creative_llm.invoke([HumanMessage(content=opportunity_prompt)]).content
    yield emit("opportunity_complete", {"opportunity": opportunity})

    # ---- Final Forecast ----
    yield emit("forecast_start", {})
    forecast_prompt = (
        f"You are the Chief Strategy Officer. Based on this intelligence on {company}:\n\n"
        f"THREAT:\n{threat}\n\nOPPORTUNITY:\n{opportunity}\n\n"
        f"First line: EXACTLY ONE WORD — ALERT, MONITOR, STABLE, or OPPORTUNITY. Do not use any markdown on the first line.\n"
        f"Then a blank line. Then EXACTLY this structure:\n\n"
        f"## 90-Day Prediction\n"
        f"- **Most likely move:** <one sentence with confidence %>\n"
        f"- **Secondary signal:** <one sentence with confidence %>\n"
        f"- **Wildcard:** <one sentence>\n\n"
        f"## Recommended Actions\n"
        f"1. **<Short action title>** — one sentence\n"
        f"2. **<Short action title>** — one sentence\n"
        f"3. **<Short action title>** — one sentence\n\n"
        f"## Confidence Level\n"
        f"**<LOW / MEDIUM / HIGH>** — one sentence explaining why\n\n"
        f"RULES: Short and punchy. No long paragraphs."
    )
    forecast_raw = creative_llm.invoke([HumanMessage(content=forecast_prompt)]).content

    # ---- Robust decision parsing (strip markdown) ----
    forecast_lines = forecast_raw.strip().split('\n')
    first_line_clean = forecast_lines[0].replace("*", "").replace("#", "").strip().upper()

    if "ALERT" in first_line_clean: decision = "ALERT"
    elif "OPPORTUNITY" in first_line_clean: decision = "OPPORTUNITY"
    elif "MONITOR" in first_line_clean: decision = "MONITOR"
    elif "STABLE" in first_line_clean: decision = "STABLE"
    else:
        # Fallback: search entire response
        fu = forecast_raw.upper()
        if "OPPORTUNITY" in fu: decision = "OPPORTUNITY"
        elif "ALERT" in fu: decision = "ALERT"
        elif "MONITOR" in fu: decision = "MONITOR"
        else: decision = "STABLE"

    # Clean forecast body — strip the first line if it was the verdict
    if first_line_clean in ["ALERT", "MONITOR", "STABLE", "OPPORTUNITY"]:
        forecast = '\n'.join(forecast_lines[1:]).strip()
    else:
        forecast = forecast_raw

    yield emit("forecast_complete", {"forecast": forecast, "decision": decision})
    yield emit("complete", {})


@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    return StreamingResponse(event_stream(request.company, request.description), media_type="text/event-stream")


@app.get("/")
async def root():
    return {"status": "Oracle API is running", "version": "4.0"}
