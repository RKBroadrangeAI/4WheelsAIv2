"""
Ticket Classification API Router

Uses Jev for fast classification and LangChain for response generation.
"""

import uuid
from fastapi import APIRouter, HTTPException, Depends

from app.models import TicketRequest, TicketClassification, JevDecision
from app.services import JevService, LangChainService

router = APIRouter(prefix="/api/tickets", tags=["tickets"])

# Service instances (in production, use proper dependency injection)
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


@router.post("/classify", response_model=TicketClassification)
async def classify_ticket(
    ticket: TicketRequest,
    jev: JevService = Depends(get_jev_service),
    langchain: LangChainService = Depends(get_langchain_service)
):
    """
    Classify a support ticket using Jev and generate a response with LangChain.
    
    This endpoint demonstrates the Jev + LangChain pattern:
    1. Jev makes fast classification decisions (department, urgency, human review)
    2. LangChain generates a contextual response based on Jev's decisions
    """
    try:
        # Step 1: Use Jev for fast classification
        classification = await jev.classify_ticket(
            subject=ticket.subject,
            message=ticket.message,
            customer_id=ticket.customer_id
        )
        
        answers = classification.get("answers", {})
        
        # Extract Jev decisions
        dept_answer = answers.get("department", {})
        urgency_answer = answers.get("urgency", {})
        human_answer = answers.get("requires_human", {})
        
        department = JevDecision(
            choice=dept_answer.get("choice", "general"),
            confidence=dept_answer.get("confidence", 0.5),
            probabilities=dept_answer.get("probabilities", {})
        )
        
        urgency = JevDecision(
            choice=urgency_answer.get("choice", "medium"),
            confidence=urgency_answer.get("confidence", 0.5),
            probabilities=urgency_answer.get("probabilities", {})
        )
        
        requires_human = JevDecision(
            choice="yes" if human_answer.get("decision", False) else "no",
            confidence=human_answer.get("confidence", 0.5),
            probabilities={}
        )
        
        # Step 2: Use LangChain to generate a suggested response
        suggested_response = await langchain.generate_ticket_response(
            subject=ticket.subject,
            message=ticket.message,
            department=department.choice,
            urgency=urgency.choice
        )
        
        return TicketClassification(
            ticket_id=str(uuid.uuid4())[:8],
            department=department,
            urgency=urgency,
            requires_human=requires_human,
            suggested_response=suggested_response,
            processing_time_ms=classification.get("processing_time_ms", 0)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@router.post("/batch-classify")
async def batch_classify_tickets(
    tickets: list[TicketRequest],
    jev: JevService = Depends(get_jev_service)
):
    """
    Classify multiple tickets in batch.
    
    Demonstrates Jev's efficiency for high-volume classification tasks.
    """
    results = []
    
    for ticket in tickets:
        try:
            classification = await jev.classify_ticket(
                subject=ticket.subject,
                message=ticket.message,
                customer_id=ticket.customer_id
            )
            results.append({
                "status": "success",
                "ticket_subject": ticket.subject,
                "classification": classification.get("answers", {})
            })
        except Exception as e:
            results.append({
                "status": "error",
                "ticket_subject": ticket.subject,
                "error": str(e)
            })
    
    return {"results": results, "total": len(results)}
