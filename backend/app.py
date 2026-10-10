# ==========================================================
# ORACLE BACKEND — Predictive Competitive Intelligence (v6)
# Two-layer verdict: AI reasoning + code-level sanity check
# ==========================================================
import os
import json
import re
import time
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


# ==========================================================
# SERPAPI — RETRY + QUOTA DETECTION + ROBUST PARSING
# ==========================================================
def serpapi_call(query, engine, retries=2):
    """Returns list of dicts. Retries on failure. Detects quota issues."""
    url = "https://serpapi.com/search"
    params = {"q": query, "api_key": SERPAPI_API_KEY, "engine": engine}

    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=20)
            results = response.json()

            # Quota error detection
            if "error" in results:
                err_msg = str(results["error"])
                print(f"[SERPAPI {engine}] Attempt {attempt+1} ERROR: {err_msg}")
                if any(kw in err_msg.lower() for kw in ["exhausted", "quota", "limit", "plan"]):
                    return [{
                        "title": "⚠️ SerpApi quota exhausted",
                        "snippet": "Free tier allows 250 searches/month. Resets on the 1st.",
                        "link": "https://serpapi.com/dashboard",
                        "thumbnail": "",
                        "source": "System"
                    }]
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                return [{"title": f"API error: {err_msg}", "snippet": "", "link": "", "thumbnail": "", "source": ""}]

            search_items = (
                results.get("organic_results") or
                results.get("news_results") or
                results.get("jobs_results") or
                results.get("shopping_results") or
                []
            )
            print(f"[SERPAPI {engine}] Attempt {attempt+1}: query='{query}' → {len(search_items)} items")

            if not search_items and attempt < retries - 1:
                time.sleep(1)
                continue

            items = []
            for r in search_items[:3]:
                # --- Link ---
                link = r.get("link", "")
                if not link and isinstance(r.get("source"), dict):
                    link = r["source"].get("link", "")
                if not link and isinstance(r.get("apply_options"), list) and r["apply_options"]:
                    try:
                        link = r["apply_options"][0].get("link", "")
                    except Exception:
                        pass
                if not link:
                    link = r.get("share_link", "")

                # --- Snippet ---
                snippet = r.get("snippet") or r.get("description") or ""
                if not snippet and r.get("price"):
                    snippet = f"Price: {r['price']}"
                if not snippet and r.get("extensions"):
                    snippet = " · ".join(r["extensions"][:3])
                snippet = str(snippet).replace("<b>", "").replace("</b>", "").replace("<br>", " ")
                snippet = snippet[:250].strip()

                # --- Thumbnail ---
                thumbnail = r.get("thumbnail", "")
                if not thumbnail and isinstance(r.get("source"), dict):
                    thumbnail = r["source"].get("logo", "")

                # --- Source name ---
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
            print(f"[SERPAPI {engine}] Attempt {attempt+1} EXCEPTION: {str(e)}")
            if attempt == retries - 1:
                return [{"title": f"Search failed after {retries} attempts", "snippet": str(e), "link": "", "thumbnail": "", "source": ""}]
            time.sleep(1)

    return [{"title": "No results", "snippet": "Empty response", "link": "", "thumbnail": "", "source": ""}]


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


# ==========================================================
# VERDICT LOGIC — Pure function, easy to test
# ==========================================================
def compute_verdict_from_strengths(strengths):
    """
    Given a list of integer strengths (1-10), return the mathematically
    correct verdict. This overrides any AI hallucination.
    """
    if not strengths:
        return "MONITOR", 5.0

    avg = sum(strengths) / len(strengths)

    # Count strong and weak signals
    strong_count = sum(1 for s in strengths if s >= 7)
    weak_count = sum(1 for s in strengths if s <= 3)

    # ALERT: Many strong signals OR high average
    if avg >= 6.0 or strong_count >= 3:
        return "ALERT", avg

    # STABLE: Everything is weak
    if avg < 2.5 and strong_count == 0:
        return "STABLE", avg

    # OPPORTUNITY: Competitor is genuinely weak
    if avg < 4.0 and strong_count <= 1:
        return "OPPORTUNITY", avg

    # MONITOR: Everything else (mixed signals)
    return "MONITOR", avg


# ==========================================================
# EVENT STREAM
# ==========================================================
def event_stream(company, description):
    def emit(event_type, data):
        return f"data: {json.dumps({'type': event_type, 'data': data})}\n\n"

    yield emit("start", {"company": company, "description": description})

    # ---- Quota pre-check ----
    try:
        acct = requests.get(
            "https://serpapi.com/account",
            params={"api_key": SERPAPI_API_KEY},
            timeout=10
        ).json()
        left = acct.get("total_searches_left", None)
        print(f"[SERPAPI] Account status: {left} searches left this month")
        if left is not None and left <= 0:
            yield emit("error", {"message": "SerpApi quota exhausted for this month."})
            yield emit("complete", {})
            return
    except Exception as e:
        print(f"[SERPAPI] Pre-check failed: {e}")

    # ---- 5 signal scans ----
    signals = [
        ("hiring",   "google_jobs",     f"{company} engineer",              "Hiring"),
        ("patents",  "google_scholar",  f"{company} patents research",      "Patents"),
        ("pricing",  "google_shopping", f"{company} product",               "Pricing"),
        ("news",     "google_news",     f"{company} strategy",              "News"),
        ("position", "google_news",     f"{company} partnership expansion", "Position"),
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
        time.sleep(0.3)

    yield emit("sources_collected", {"sources": all_sources})

    # ==========================================================
    # PATTERN ANALYSIS
    # ==========================================================
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
        f"STRENGTH SCORING GUIDE (be consistent):\n"
        f"- 8-10: Strong signal, clear strategic intent, active moves\n"
        f"- 6-7: Moderate signal, meaningful activity\n"
        f"- 4-5: Weak signal, some activity\n"
        f"- 1-3: No signal detected or minimal activity\n\n"
        f"Replace N with the actual strength number (1-10).\n\n"
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

    # ==========================================================
    # EXTRACT STRENGTHS (from the summary table, precisely)
    # ==========================================================
    # Match rows like: | Hiring | 8/10 | ...
    strength_matches = re.findall(
        r'\|\s*(?:Hiring|Patents|Pricing|News|Position)\s*\|\s*\*{0,2}(\d+)\s*/\s*10',
        pattern
    )
    strengths = [int(s) for s in strength_matches if 1 <= int(s) <= 10]
    print(f"[ORACLE] Extracted strengths: {strengths}")

    # Compute verdict upfront — this is the ground truth
    computed_decision, avg_strength = compute_verdict_from_strengths(strengths)
    print(f"[ORACLE] Computed verdict: {computed_decision} (avg {avg_strength:.1f})")

    # ==========================================================
    # THREAT ANALYSIS
    # ==========================================================
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

    # ==========================================================
    # OPPORTUNITY ANALYSIS
    # ==========================================================
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

    # ==========================================================
    # FINAL FORECAST (verdict is FIXED — not left to the AI)
    # ==========================================================
    yield emit("forecast_start", {})

    # Build the verdict explanation
    verdict_explanations = {
        "ALERT": "The competitor is aggressively moving. Multiple strong signals detected. Competitors must respond within 7 days or risk losing market position.",
        "MONITOR": "Signals are mixed. Some movement detected but not conclusive. Re-check in 30 days to see if the pattern strengthens.",
        "OPPORTUNITY": "The competitor is weak. Low activity across most signals. This is a window to strike and gain market share.",
        "STABLE": "The competitor is dormant. No meaningful activity detected. No action required — focus on your own roadmap."
    }

    forecast_prompt = (
        f"You are the Chief Strategy Officer. Based on this intelligence on {company}:\n\n"
        f"THREAT ANALYSIS:\n{threat}\n\n"
        f"OPPORTUNITY ANALYSIS:\n{opportunity}\n\n"
        f"THE FINAL VERDICT IS: **{computed_decision}**\n"
        f"(Based on average signal strength of {avg_strength:.1f}/10 across 5 categories)\n\n"
        f"Explain WHY this verdict is correct given the signals. "
        f"Then output EXACTLY this structure (no first-line verdict — that is already decided):\n\n"
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
        f"RULES:\n"
        f"- Short and punchy. No long paragraphs.\n"
        f"- Match the tone to the verdict: "
        f"if ALERT, urgency; if OPPORTUNITY, aggressive; if STABLE, calm; if MONITOR, watchful."
    )
    forecast = creative_llm.invoke([HumanMessage(content=forecast_prompt)]).content

    # The verdict is code-controlled — never AI-generated
    decision = computed_decision

    print(f"[ORACLE] Final verdict: {decision}")
    yield emit("forecast_complete", {"forecast": forecast, "decision": decision, "avg_strength": round(avg_strength, 1)})
    yield emit("complete", {})


# ==========================================================
# ROUTES
# ==========================================================
@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    return StreamingResponse(event_stream(request.company, request.description), media_type="text/event-stream")


@app.get("/")
async def root():
    return {"status": "Oracle API is running", "version": "6.0"}
