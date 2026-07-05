"""
Feature A — Udaan Coach
A memory-backed AI career coach.  Athlete syncs their passport, then chats;
every conversation is stored so context survives across sessions.
forget() = consent / right-to-erasure demo.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from lib.memory import remember, recall, improve, forget
from lib.groq_agent import chat

router = APIRouter()

SYSTEM = """You are Udaan Coach, a personal AI sports career advisor for young Indian athletes.
You have full access to the athlete's career memory — fitness test scores, competition results,
self-rated strengths, past coaching conversations, and training history.
Always personalise your advice using facts from memory (name specific events, scores, dates).
Be encouraging, culturally aware, and concise (max 180 words per reply).
If memory is empty, ask the athlete to tap "Sync Passport" first."""


def _ds(uid: str) -> str:
    return f"coach_{uid}"


# ── Request models ─────────────────────────────────────────────────────────────

class AthletePayload(BaseModel):
    uid: str
    name: str
    age: int
    gender: str
    district: str
    scores: dict[str, float]         # speed/agility/endurance/technique (0-100)
    power: float | None = None        # talent percentile
    archetype: str = ""
    records: list[dict] = []
    tests: list[dict] = []


class ChatMsg(BaseModel):
    uid: str
    message: str
    history: list[dict] = []          # [{role, content}] optional short-term turns


class FeedbackMsg(BaseModel):
    uid: str
    advice: str
    helpful: bool
    reason: str = ""


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/remember")
async def coach_remember(data: AthletePayload):
    """Ingest (or refresh) an athlete's passport data into Cognee memory."""
    lines = [
        f"Athlete Profile — {data.name}",
        f"Age: {data.age} | Gender: {data.gender} | District: {data.district}",
        f"Self-rated scores — Speed: {data.scores.get('speed','?')}/100, "
        f"Agility: {data.scores.get('agility','?')}/100, "
        f"Endurance: {data.scores.get('endurance','?')}/100, "
        f"Technique: {data.scores.get('technique','?')}/100",
        f"Talent percentile (Power score): {data.power}th percentile",
        f"Archetype: {data.archetype}",
        f"Verified competition records on passport: {len(data.records)}",
        f"Talent tests taken: {len(data.tests)}",
    ]

    if data.records:
        lines.append("\nRecent Competition Results:")
        for r in data.records[-5:]:
            lines.append(
                f"  • {r.get('event','')} — {r.get('result','')} "
                f"({r.get('place','')}) at {r.get('venue','')} on {r.get('date','')}"
                + (f" | Verified by {r.get('org','')}" if r.get("org") else "")
            )

    if data.tests:
        lines.append("\nTalent Test History:")
        for t in data.tests[-5:]:
            lines.append(
                f"  • {t.get('label','')}: {t.get('value','')} {t.get('unit','')} "
                f"(percentile: {t.get('percentile','')})"
            )

    await remember("\n".join(lines), _ds(data.uid))
    return {"ok": True, "dataset": _ds(data.uid), "ingested_chars": sum(len(l) for l in lines)}


@router.post("/chat")
async def coach_chat(msg: ChatMsg):
    """Q&A with the AI coach; context is pulled from Cognee memory."""
    context = await recall(msg.message, _ds(msg.uid))
    reply = await chat(msg.message, context, SYSTEM, history=msg.history or None)
    # Store this exchange so future sessions remember it
    await remember(
        f"Coaching conversation:\nAthlete: {msg.message}\nCoach: {reply}",
        _ds(msg.uid),
    )
    return {"reply": reply, "context_chunks": len(context)}


@router.post("/feedback")
async def coach_feedback(fb: FeedbackMsg):
    """
    Feature: improve() — athlete rates advice; Cognee folds the signal into
    the knowledge graph so future answers are reweighted.
    """
    label = "HELPFUL ✓" if fb.helpful else "NOT HELPFUL ✗"
    note = (
        f"Athlete feedback on coaching advice ({label}):\n"
        f"Advice: \"{fb.advice}\"\n"
        f"Reason: {fb.reason or 'none given'}\n"
        f"Instruction: {'Reinforce this approach.' if fb.helpful else 'Avoid this approach in future.'}"
    )
    await improve(note, _ds(fb.uid))
    return {"ok": True}


@router.delete("/forget/{uid}")
async def coach_forget(uid: str):
    """
    Feature: forget() — guardian exercises right to erasure.
    Surgically deletes the athlete's coaching memory dataset.
    """
    await forget(_ds(uid))
    return {
        "ok": True,
        "erased": _ds(uid),
        "message": "Athlete coaching memory fully erased (right to erasure honoured).",
    }
