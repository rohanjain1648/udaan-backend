"""
Udaan × Cognee API
FastAPI server wiring all four memory-backed features together.
Start: uvicorn main:app --reload --port 8000
"""
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lib.cognee_setup import setup_cognee
from routers import coach, scout, include, train
from routers.include import seed_knowledge


@asynccontextmanager
async def lifespan(app: FastAPI):
    await setup_cognee()
    await seed_knowledge()   # pre-load para-sport knowledge base
    yield


app = FastAPI(
    title="Udaan × Cognee API",
    description="Memory-backed AI features for the Udaan Verified Athlete Passport",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        "*",   # narrow this in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(coach.router,   prefix="/coach",   tags=["A · Athlete Coach"])
app.include_router(scout.router,   prefix="/scout",   tags=["B · Scout Copilot"])
app.include_router(include.router, prefix="/include", tags=["C · Include Advisor"])
app.include_router(train.router,   prefix="/train",   tags=["D · Training Loop"])


@app.get("/health", tags=["meta"])
async def health():
    return {
        "status": "ok",
        "llm": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "provider": "Groq via OpenAI-compatible API",
        "embeddings": "fastembed (local)",
        "features": ["coach", "scout", "include", "train"],
    }
