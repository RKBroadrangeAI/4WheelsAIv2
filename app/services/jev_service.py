"""
Jev Service - TypeSafe AI's System One Decision Model Integration

Jev is a fast, structured decision model that outputs typed results with
confidence scores. It's designed for classification tasks like routing,
guardrails, and tool selection - not text generation.

Supports three decision types:
- Choice: Select from predefined options
- Score: Return a numerical score  
- Noul: Boolean yes/no decisions

The official Python SDK is `typesafe-sdk` (pip install typesafe-sdk).
"""

import httpx
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from app.config import get_settings

try:
    from typesafe_sdk import TypeSafeClient, AsyncTypeSafeClient, Choice, Noul, Score
    TYPESAFE_SDK_AVAILABLE = True
except ImportError:
    TYPESAFE_SDK_AVAILABLE = False


@dataclass
class JevQuestion:
    """A question for Jev to evaluate"""
    id: str
    type: str  # "choice", "score", or "noul"
    criteria: Dict[str, str]  # For choice: option -> description
    description: Optional[str] = None


class JevService:
    """
    Service for interacting with TypeSafe AI's Jev model.
    
    Jev is a System One model that makes fast, structured decisions
    with calibrated confidence scores. It evaluates state against
    typed questions and returns decisions your code can branch on.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def _get_headers(self) -> Dict[str, str]:
        """Get authorization headers for API calls"""
        if self.settings.openrouter_api_key:
            return {
                "Authorization": f"Bearer {self.settings.openrouter_api_key}",
                "Content-Type": "application/json",
            }
        elif self.settings.typesafe_api_key:
            return {
                "Authorization": f"Bearer {self.settings.typesafe_api_key}",
                "Content-Type": "application/json",
            }
        return {"Content-Type": "application/json"}
    
    def _get_base_url(self) -> str:
        """Get the appropriate base URL"""
        if self.settings.openrouter_api_key:
            return self.settings.openrouter_base_url
        return self.settings.typesafe_base_url
    
    async def classify_ticket(
        self,
        subject: str,
        message: str,
        customer_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Classify a support ticket using Jev's fast decision model.
        
        Uses three Jev questions:
        - department: Which team should handle this? (Choice)
        - urgency: How urgent is this? (Choice mapped from Score)
        - requires_human: Does this need human review? (Noul)
        """
        start_time = time.time()
        
        state = {
            "ticket": {
                "subject": subject,
                "message": message,
                "customer_id": customer_id,
            }
        }
        
        # Check if we have API credentials and SDK
        if not self.settings.typesafe_api_key and not self.settings.openrouter_api_key:
            return self._mock_classification(state, start_time)
        
        # Try using the official TypeSafe SDK first
        if TYPESAFE_SDK_AVAILABLE and self.settings.typesafe_api_key:
            try:
                import os
                os.environ["TYPESAFE_API_KEY"] = self.settings.typesafe_api_key
                
                async with AsyncTypeSafeClient(model=self.settings.jev_model) as client:
                    response = await client.system_one(
                        state=state,
                        questions={
                            "department": Choice(
                                instructions="Which department should handle this ticket?",
                                criteria={
                                    "billing": "Payment issues, invoices, subscriptions, refunds, charges",
                                    "technical": "Bugs, errors, technical problems, integrations, API issues",
                                    "sales": "Pricing questions, upgrades, enterprise plans, demos",
                                    "general": "General inquiries, feedback, feature requests, other"
                                }
                            ),
                            "urgency": Choice(
                                instructions="How urgent is this ticket?",
                                criteria={
                                    "critical": "Production down, data loss, security breach, business blocked",
                                    "high": "Significant impact, workaround difficult, time-sensitive",
                                    "medium": "Moderate impact, workaround available, not urgent",
                                    "low": "Minor issue, question, or feedback with no immediate impact"
                                }
                            ),
                            "requires_human": Noul(
                                instructions="Does this ticket require human review due to complexity, sensitivity, or explicit request?"
                            )
                        }
                    )
                    
                    processing_time = (time.time() - start_time) * 1000
                    
                    return {
                        "answers": {
                            "department": {
                                "choice": response.choices["department"].choice,
                                "confidence": response.choices["department"].confidence,
                                "probabilities": response.choices["department"].probabilities or {}
                            },
                            "urgency": {
                                "choice": response.choices["urgency"].choice,
                                "confidence": response.choices["urgency"].confidence,
                                "probabilities": response.choices["urgency"].probabilities or {}
                            },
                            "requires_human": {
                                "decision": response.nouls["requires_human"].noul,
                                "confidence": response.nouls["requires_human"].confidence
                            }
                        },
                        "processing_time_ms": processing_time,
                        "model": response.model or self.settings.jev_model
                    }
            except Exception as e:
                print(f"TypeSafe SDK error: {e}, falling back to HTTP API")
        
        # Fallback to direct HTTP API (for OpenRouter or if SDK fails)
        questions = {
            "department": {
                "type": "choice",
                "criteria": {
                    "billing": "Payment issues, invoices, subscriptions, refunds, charges",
                    "technical": "Bugs, errors, technical problems, integrations, API issues",
                    "sales": "Pricing questions, upgrades, enterprise plans, demos",
                    "general": "General inquiries, feedback, feature requests, other"
                }
            },
            "urgency": {
                "type": "choice",
                "criteria": {
                    "critical": "Production down, data loss, security breach, business blocked",
                    "high": "Significant impact, workaround difficult, time-sensitive",
                    "medium": "Moderate impact, workaround available, not urgent",
                    "low": "Minor issue, question, or feedback with no immediate impact"
                }
            },
            "requires_human": {
                "type": "noul",
                "description": "Does this ticket require human review due to complexity, sensitivity, or explicit request?"
            }
        }
        
        try:
            payload = {
                "model": f"typesafe/{self.settings.jev_model}",
                "state": state,
                "questions": questions
            }
            
            base_url = self._get_base_url()
            response = await self.client.post(
                f"{base_url}/decisions",
                headers=self._get_headers(),
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "answers": result.get("answers", {}),
                "processing_time_ms": processing_time,
                "model": result.get("model", self.settings.jev_model)
            }
            
        except Exception as e:
            print(f"Jev API error: {e}, using mock response")
            return self._mock_classification(state, start_time)
    
    def _mock_classification(self, state: Dict, start_time: float) -> Dict[str, Any]:
        """
        Generate a mock classification for demo purposes.
        Simulates Jev's response structure.
        """
        message = state.get("ticket", {}).get("message", "").lower()
        subject = state.get("ticket", {}).get("subject", "").lower()
        combined = f"{subject} {message}"
        
        # Simple keyword-based classification for demo
        if any(w in combined for w in ["bill", "charge", "invoice", "payment", "refund", "subscription"]):
            dept = "billing"
            dept_conf = 0.92
        elif any(w in combined for w in ["error", "bug", "crash", "not working", "broken", "api", "integration"]):
            dept = "technical"
            dept_conf = 0.88
        elif any(w in combined for w in ["pricing", "upgrade", "enterprise", "demo", "trial"]):
            dept = "sales"
            dept_conf = 0.85
        else:
            dept = "general"
            dept_conf = 0.75
        
        # Urgency detection
        if any(w in combined for w in ["urgent", "asap", "critical", "down", "emergency", "blocked"]):
            urgency = "critical"
            urg_conf = 0.90
        elif any(w in combined for w in ["important", "soon", "need help"]):
            urgency = "high"
            urg_conf = 0.82
        elif any(w in combined for w in ["when you can", "question", "wondering"]):
            urgency = "low"
            urg_conf = 0.78
        else:
            urgency = "medium"
            urg_conf = 0.70
        
        # Human review detection
        needs_human = any(w in combined for w in ["speak to", "manager", "escalate", "legal", "lawyer", "cancel account"])
        human_conf = 0.95 if needs_human else 0.85
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "answers": {
                "department": {
                    "choice": dept,
                    "confidence": dept_conf,
                    "probabilities": {
                        "billing": 0.92 if dept == "billing" else 0.05,
                        "technical": 0.88 if dept == "technical" else 0.05,
                        "sales": 0.85 if dept == "sales" else 0.03,
                        "general": 0.75 if dept == "general" else 0.02,
                    }
                },
                "urgency": {
                    "choice": urgency,
                    "confidence": urg_conf,
                    "probabilities": {
                        "critical": 0.90 if urgency == "critical" else 0.02,
                        "high": 0.82 if urgency == "high" else 0.08,
                        "medium": 0.70 if urgency == "medium" else 0.15,
                        "low": 0.78 if urgency == "low" else 0.10,
                    }
                },
                "requires_human": {
                    "decision": needs_human,
                    "confidence": human_conf
                }
            },
            "processing_time_ms": processing_time,
            "model": "jev-mock (demo mode)"
        }
    
    async def route_query(
        self,
        query: str,
        available_routes: List[str],
        route_descriptions: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Use Jev to route a query to the appropriate handler.
        
        This is the core pattern for using Jev in an agent loop:
        fast routing decisions with confidence scores.
        """
        start_time = time.time()
        
        if not self.settings.openrouter_api_key and not self.settings.typesafe_api_key:
            # Demo mode
            return self._mock_route(query, available_routes, start_time)
        
        try:
            payload = {
                "model": f"typesafe/{self.settings.jev_model}",
                "state": {"query": query},
                "questions": {
                    "route": {
                        "type": "choice",
                        "criteria": route_descriptions
                    }
                }
            }
            
            response = await self.client.post(
                f"{self._get_base_url()}/decisions",
                headers=self._get_headers(),
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "route": result["answers"]["route"]["choice"],
                "confidence": result["answers"]["route"]["confidence"],
                "probabilities": result["answers"]["route"].get("probabilities", {}),
                "processing_time_ms": processing_time
            }
            
        except Exception as e:
            print(f"Jev routing error: {e}")
            return self._mock_route(query, available_routes, start_time)
    
    def _mock_route(
        self,
        query: str,
        available_routes: List[str],
        start_time: float
    ) -> Dict[str, Any]:
        """Mock routing for demo mode"""
        query_lower = query.lower()
        
        # Simple keyword matching for demo
        route = available_routes[0]  # Default to first
        confidence = 0.65
        
        for r in available_routes:
            if r.lower() in query_lower:
                route = r
                confidence = 0.88
                break
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "route": route,
            "confidence": confidence,
            "probabilities": {r: 0.88 if r == route else 0.04 for r in available_routes},
            "processing_time_ms": processing_time
        }
    
    async def check_safety(self, text: str) -> Dict[str, Any]:
        """
        Use Jev as a guardrail to check content safety.
        
        Jev can classify whether content violates policies
        before it's processed or sent to users.
        """
        start_time = time.time()
        
        if not self.settings.openrouter_api_key and not self.settings.typesafe_api_key:
            return {
                "is_safe": True,
                "confidence": 0.95,
                "flags": [],
                "processing_time_ms": (time.time() - start_time) * 1000
            }
        
        try:
            payload = {
                "model": f"typesafe/{self.settings.jev_model}",
                "state": {"content": text},
                "questions": {
                    "is_safe": {
                        "type": "noul",
                        "description": "Is this content safe and appropriate for a professional support context?"
                    },
                    "category": {
                        "type": "choice",
                        "criteria": {
                            "safe": "Normal, professional content",
                            "potentially_sensitive": "May contain sensitive topics but not harmful",
                            "needs_review": "Contains content that should be reviewed by a human"
                        }
                    }
                }
            }
            
            response = await self.client.post(
                f"{self._get_base_url()}/decisions",
                headers=self._get_headers(),
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "is_safe": result["answers"]["is_safe"]["decision"],
                "confidence": result["answers"]["is_safe"]["confidence"],
                "category": result["answers"]["category"]["choice"],
                "processing_time_ms": processing_time
            }
            
        except Exception as e:
            print(f"Safety check error: {e}")
            return {
                "is_safe": True,
                "confidence": 0.5,
                "flags": ["check_failed"],
                "processing_time_ms": (time.time() - start_time) * 1000
            }
