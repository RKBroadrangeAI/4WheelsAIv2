"""
Chat API Router with Jev-based Routing

Uses Jev for intelligent query routing and LangChain for response generation.
"""

import uuid
from fastapi import APIRouter, HTTPException, Depends

from app.models import ChatRequest, ChatResponse
from app.services import JevService, LangChainService

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Service instances
_jev_service = None
_langchain_service = None


def get_jev_service() -> JevService:
    global _jev_service
    if _jev_service is None:
        _jev_service = JevService()
    return _jev_service


def get_langchain_service() -> LangChainService:
    global _langchain_service
    if _langchain_service is None:
        _langchain_service = LangChainService(get_jev_service())
    return _langchain_service


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    langchain: LangChainService = Depends(get_langchain_service)
):
    """
    Process a chat message with Jev-based routing.
    
    The flow:
    1. Jev classifies the query type (simple, complex, action, escalation)
    2. Based on the route, the appropriate handling strategy is selected
    3. LangChain generates the response using the selected strategy
    """
    try:
        result = await langchain.chat_with_routing(
            message=request.message,
            conversation_id=request.conversation_id,
            context=request.context
        )
        
        return ChatResponse(
            response=result["response"],
            route=result["route"],
            confidence=result["confidence"],
            model_used=result["model_used"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.post("/with-guardrails")
async def chat_with_guardrails(
    request: ChatRequest,
    langchain: LangChainService = Depends(get_langchain_service)
):
    """
    Process a chat message with Jev safety guardrails.
    
    Demonstrates using Jev as input/output guardrails:
    1. Check input safety with Jev
    2. Generate response with LangChain
    3. Check output safety with Jev
    """
    try:
        result = await langchain.process_with_guardrails(
            user_input=request.message,
            task="respond"
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("/new-conversation")
async def new_conversation():
    """Generate a new conversation ID"""
    return {"conversation_id": str(uuid.uuid4())}
