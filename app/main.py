"""
LangChain + Jev Web Application

A web application demonstrating the integration of:
- Jev: TypeSafe AI's fast decision model for routing and guardrails
- LangChain: For LLM-powered text generation

Jev handles fast, structured decisions while LangChain handles
open-ended reasoning and text generation.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.config import get_settings
from app.routers import tickets_router, chat_router
from app.models import HealthResponse
from app.services import JevService


# Global services for cleanup
_jev_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global _jev_service
    _jev_service = JevService()
    yield
    # Cleanup
    if _jev_service:
        await _jev_service.close()


# Create FastAPI app
app = FastAPI(
    title="LangChain + Jev Web App",
    description="""
    A demonstration of combining LangChain with Jev (TypeSafe AI's System One model).
    
    ## What is Jev?
    
    Jev is a fast decision model that outputs structured probabilities and confidence scores.
    Unlike traditional LLMs, Jev doesn't generate text - it makes classification decisions.
    
    ## Architecture
    
    - **Jev**: Fast routing, classification, and guardrail decisions
    - **LangChain**: Text generation and complex reasoning
    
    ## Endpoints
    
    - `/api/tickets/classify`: Classify support tickets with Jev + generate responses with LangChain
    - `/api/chat/`: Chat with Jev-based routing to different handling strategies
    - `/api/chat/with-guardrails`: Chat with Jev safety checks on input/output
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Include routers
app.include_router(tickets_router)
app.include_router(chat_router)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main web application"""
    settings = get_settings()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "app_name": settings.app_name,
            "has_jev_key": bool(settings.typesafe_api_key or settings.openrouter_api_key),
            "has_openai_key": bool(settings.openai_api_key)
        }
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check application health and service availability"""
    settings = get_settings()
    
    return HealthResponse(
        status="healthy",
        jev_available=bool(settings.typesafe_api_key or settings.openrouter_api_key),
        llm_available=bool(settings.openai_api_key),
        version="1.0.0"
    )


@app.get("/api/info")
async def api_info():
    """Get information about Jev and the API"""
    return {
        "what_is_jev": {
            "summary": "Jev is TypeSafe AI's System One decision model",
            "key_features": [
                "Fast, structured decisions with confidence scores",
                "Up to 200x faster than LLMs for classification tasks",
                "Three decision types: Choice, Score, Noul (yes/no)",
                "Designed for routing, guardrails, and tool selection"
            ],
            "not_for": [
                "Text generation",
                "Open-ended reasoning",
                "Creative writing"
            ],
            "integration_pattern": "Use Jev for decisions, LLMs for generation"
        },
        "api_endpoints": {
            "tickets": {
                "POST /api/tickets/classify": "Classify a support ticket",
                "POST /api/tickets/batch-classify": "Batch classify tickets"
            },
            "chat": {
                "POST /api/chat/": "Chat with Jev-based routing",
                "POST /api/chat/with-guardrails": "Chat with safety guardrails",
                "GET /api/chat/new-conversation": "Start new conversation"
            }
        },
        "demo_mode": "Running in demo mode - configure API keys for full functionality"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
