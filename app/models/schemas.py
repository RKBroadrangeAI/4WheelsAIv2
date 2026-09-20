from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum


class Department(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    SALES = "sales"
    GENERAL = "general"


class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketRequest(BaseModel):
    """Input ticket to be classified"""
    subject: str = Field(..., description="Ticket subject line")
    message: str = Field(..., description="Ticket message body")
    customer_id: Optional[str] = Field(None, description="Optional customer ID")


class JevDecision(BaseModel):
    """A single Jev decision result"""
    choice: str = Field(..., description="The selected choice")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    probabilities: Dict[str, float] = Field(default_factory=dict, description="Probability for each option")


class TicketClassification(BaseModel):
    """Classification result from Jev"""
    ticket_id: str = Field(..., description="Generated ticket ID")
    department: JevDecision = Field(..., description="Department routing decision")
    urgency: JevDecision = Field(..., description="Urgency level decision")
    requires_human: JevDecision = Field(..., description="Whether human review is needed")
    suggested_response: Optional[str] = Field(None, description="LLM-generated response suggestion")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class ChatRequest(BaseModel):
    """Chat request for the conversational interface"""
    message: str = Field(..., description="User message")
    context: Optional[str] = Field(None, description="Additional context")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for continuity")


class ChatResponse(BaseModel):
    """Chat response with routing info"""
    response: str = Field(..., description="Generated response")
    route: str = Field(..., description="Route taken (jev decision)")
    confidence: float = Field(..., description="Routing confidence")
    model_used: str = Field(..., description="Model used for response")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    jev_available: bool
    llm_available: bool
    version: str = "1.0.0"
