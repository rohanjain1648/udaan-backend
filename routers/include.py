"""
Feature C — Include Knowledge Copilot
Para-sport classification + equipment knowledge base.
Seeded from structured docs; scouts/coaches can ingest more via the API.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from lib.memory import remember, recall, forget
from lib.groq_agent import chat

router = APIRouter()

SYSTEM = """You are Udaan Include Advisor, an expert on para-sports for Indian youth with disabilities.
You have access to a knowledge base covering IPC/Paralympic classification systems, sport eligibility rules,
equipment specifications, and India-specific para-sport programs.
When advising: match the athlete's functional classification to appropriate sports and equipment.
Always remind users that official classification must be done by a certified IPC classifier.
Be empathetic, encouraging, and precise. Cite classification codes (e.g. T54, B3) where relevant."""

DATASET = "include_knowledge"

_SEED = """
## IPC Functional Classification — Overview (India / Paralympic)

### Physical Impairment Classes
- **T/F 11–13**: Visual impairment (T11 = totally blind, T12 = light perception, T13 = low vision)
- **T/F 20**: Intellectual impairment
- **T/F 31–38**: Hypertonia / athetosis / ataxia (cerebral palsy spectrum)
- **T/F 40–41**: Short stature
- **T/F 42–47**: Limb deficiency / leg-length difference / impaired muscle power (standing)
- **T/F 51–57**: Impaired muscle power / range of motion (wheelchair / seated)
- **T/F 61–64**: Prosthesis users

### Sport Recommendations by Class
| Class | Sports |
|-------|--------|
| T/F 51–54 (wheelchair) | Athletics wheelchair racing, Para swimming, Basketball (WBFI), Tennis, Badminton |
| T/F 55–57 (partial trunk) | Athletics throws + racing, Powerlifting, Table tennis |
| T/F 11–13 (visually impaired) | Athletics, Swimming, Judo (B1 only), Tandem cycling, Goalball |
| T/F 31–38 (CP) | Athletics, Swimming, Boccia (BC1/BC2 CP-specific), Cycling CP |
| T/F 61–64 (prosthesis) | Athletics, Swimming, Amputee football, Cycling, Handball |
| T/F 20 (intellectual) | Athletics, Swimming, Cycling, Gymnastics |
| T/F 40–41 (short stature) | Athletics, Swimming, Powerlifting |

### Equipment Guide
- **Racing wheelchair (T51–54)**: Lightweight titanium/aluminium frame, bucket seat, push rims, gloves
- **Throwing frame**: Secured frame for seated throws (F51–57 / F32–34)
- **Guide runner**: Sighted guide + tether (10 cm rope) for T11/T12 races
- **Goalball**: Audible ball with bells; 9×18m court, blindfolds mandatory
- **Boccia**: 6 leather balls per player/pair, polyester ramp for BC3 class
- **Blind cricket (BCAI)**: Audible ball, tactile boundary markers, 6-player teams
- **Tandem cycle**: Pilot (sighted) + stoker (VI) on a two-person bicycle
- **Sitting volleyball**: Standard ball, net lowered to 1.15 m (men) / 1.05 m (women)

### India-specific Programs
- **Paralympic Committee of India (PCI)**: National governing body for Paralympic sports
- **Khelo India Para Games**: Annual national event, open to all 26 para disciplines
- **SAI Centres**: Special Centres for Excellence for Para Sports (SCESP) at Pune, Lucknow, etc.
- **Blind Cricket Association of India (BCAI)**: Runs state + national leagues
- **Wheelchair Basketball Federation of India (WBFI)**: Affiliated to IWBF

### Udaan Integration Principle
One passport — same Udaan identity for able-bodied AND para classification.
Udaan records the class; only a certified IPC classifier assigns the class.
The Include module matches capability → sport; it does NOT replace classification.
"""


async def seed_knowledge() -> None:
    """Called from main.py lifespan to pre-load the knowledge base."""
    await remember(_SEED, DATASET)
    print("[include] Para-sport knowledge base seeded into Cognee")


class DocInput(BaseModel):
    title: str
    content: str


class AdvisorQuery(BaseModel):
    question: str
    athlete_context: str = ""   # e.g. "age 14, cerebral palsy type 3, uses wheelchair"
    history: list[dict] = []


@router.post("/ingest")
async def ingest_doc(doc: DocInput):
    """remember() — add a custom doc (classification guide, equipment catalog, etc.)."""
    await remember(f"Document: {doc.title}\n{doc.content}", DATASET)
    return {"ok": True, "title": doc.title}


@router.post("/advise")
async def advise(q: AdvisorQuery):
    """recall() + Groq → sport/equipment recommendation with graph context."""
    query = q.question
    if q.athlete_context:
        query = f"Athlete context: {q.athlete_context}\nQuestion: {q.question}"
    context = await recall(query, DATASET)
    reply = await chat(query, context, SYSTEM, history=q.history or None)
    return {"reply": reply, "context_chunks": len(context)}


@router.delete("/forget")
async def forget_knowledge():
    """forget() — clear the knowledge base (e.g. to reload with updated docs)."""
    await forget(DATASET)
    return {"ok": True, "message": "Para-sport knowledge base cleared."}
