"""
Feature D — Training Memory Loop
Continual-learning AI that remembers every session, adapts recommendations
based on explicit drill ratings (improve()), and avoids flagged harmful drills.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from lib.memory import remember, recall, improve, forget
from lib.groq_agent import chat

router = APIRouter()

SYSTEM = """You are Udaan Train Coach, a continual-learning AI that tracks an athlete's full training history.
You remember every session: drills, duration, fatigue level, injuries, and what the athlete rated as helpful.
Adapt recommendations session-by-session: increase load when athlete consistently rates low fatigue,
reduce intensity after injury notes, and avoid drills explicitly flagged as ineffective or harmful.
When answering, reference specific past sessions and explain the reasoning (e.g. "last Tuesday you did X,
fatigue was 4/5, so this week let's reduce sets"). Be concise and practical (max 150 words)."""


def _ds(uid: str) -> str:
    return f"train_{uid}"


class TrainingSession(BaseModel):
    uid: str
    date: str
    sport: str
    drills: list[str]
    duration_min: int
    fatigue_level: int   # 1 (fresh) – 5 (exhausted)
    notes: str = ""
    injuries: str = ""


class TrainQuery(BaseModel):
    uid: str
    question: str
    history: list[dict] = []


class DrillRating(BaseModel):
    uid: str
    drill: str
    session_date: str
    helpful: bool
    reason: str = ""


@router.post("/log")
async def log_session(s: TrainingSession):
    """remember() — store a completed training session."""
    text = (
        f"Training Session — {s.date}\n"
        f"Athlete UID: {s.uid} | Sport: {s.sport} | Duration: {s.duration_min} min\n"
        f"Fatigue level: {s.fatigue_level}/5\n"
        f"Drills: {', '.join(s.drills)}\n"
        f"Notes: {s.notes or 'none'}\n"
        f"Injury/pain report: {s.injuries or 'none'}"
    )
    await remember(text, _ds(s.uid))
    return {"ok": True}


@router.post("/ask")
async def train_ask(q: TrainQuery):
    """recall() → Groq — answers training questions using full session history."""
    context = await recall(q.question, _ds(q.uid))
    reply = await chat(q.question, context, SYSTEM, history=q.history or None)
    return {"reply": reply, "context_chunks": len(context)}


@router.post("/rate")
async def rate_drill(r: DrillRating):
    """
    improve() — athlete explicitly rates a drill.
    This reweights the knowledge graph so future recommendations adapt.
    """
    label = "EFFECTIVE ✓" if r.helpful else "INEFFECTIVE / HARMFUL ✗"
    instruction = (
        "Continue recommending this drill; increase frequency if fatigue allows."
        if r.helpful
        else "Do NOT recommend this drill again without explicit medical/coach clearance."
    )
    note = (
        f"Drill Rating ({r.session_date}) — [{label}]\n"
        f"Drill: \"{r.drill}\"\n"
        f"Athlete reason: {r.reason or 'none given'}\n"
        f"Future instruction: {instruction}"
    )
    await improve(note, _ds(r.uid))
    return {"ok": True}


@router.delete("/reset/{uid}")
async def reset_training(uid: str):
    """forget() — erase all training memory for an athlete."""
    await forget(_ds(uid))
    return {"ok": True, "erased": _ds(uid)}
