"""
Feature B — Scout Copilot
Institutional memory for talent scouts / SAI coaches.
Log observations from the field; query across districts/seasons.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from lib.memory import remember, recall, forget
from lib.groq_agent import chat

router = APIRouter()

SYSTEM = """You are Udaan Scout Copilot, an AI assistant for Indian talent scouts and Khelo India coaches.
You hold a persistent field-observation log across all scouting sessions.
When answering queries, reference specific athletes, districts, dates, and ratings from the log.
Help scouts: compare athletes, recall who they flagged in past sessions, spot district-level
patterns, and shortlist talent for trials. Be direct and data-focused.
If the observation log is empty, prompt the scout to log their first athlete."""

DATASET = "scout_sessions"


class ObservationLog(BaseModel):
    athlete_name: str
    age: int
    gender: str
    district: str
    sport: str
    observed_on: str       # ISO date
    notes: str
    rating: int            # 1–5 talent rating
    tags: list[str] = []   # e.g. ["explosive", "coachable", "flags-nutrition"]


class ScoutQuery(BaseModel):
    question: str
    history: list[dict] = []


@router.post("/log")
async def log_observation(obs: ObservationLog):
    """remember() — store a field observation into the scout's knowledge graph."""
    text = (
        f"Scout Observation — {obs.observed_on}\n"
        f"Athlete: {obs.athlete_name} | Age: {obs.age} | {obs.gender} | District: {obs.district}\n"
        f"Sport: {obs.sport} | Talent rating: {obs.rating}/5\n"
        f"Tags: {', '.join(obs.tags) or 'none'}\n"
        f"Field notes: {obs.notes}"
    )
    await remember(text, DATASET)
    return {"ok": True, "logged": obs.athlete_name}


@router.post("/search")
async def scout_search(q: ScoutQuery):
    """recall() — semantic + graph search across all observations."""
    context = await recall(q.question, DATASET)
    reply = await chat(q.question, context, SYSTEM, history=q.history or None)
    return {"reply": reply, "context_chunks": len(context)}


@router.delete("/clear")
async def scout_clear():
    """forget() — wipe all scout session memory."""
    await forget(DATASET)
    return {"ok": True, "message": "All scout session memory cleared."}
