"""
LangChain Service with Jev Integration

This service combines LangChain's text generation capabilities with
Jev's fast decision-making for intelligent routing and guardrails.

Pattern:
- Use Jev for fast, structured decisions (routing, safety, tool selection)
- Use LangChain LLM for open-ended reasoning and text generation
"""

from typing import Dict, Any, Optional, List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.config import get_settings
from app.services.jev_service import JevService


class LangChainService:
    """
    LangChain service that uses Jev for routing decisions
    and LLMs for text generation.
    """
    
    def __init__(self, jev_service: JevService):
        self.settings = get_settings()
        self.jev = jev_service
        self._llm = None
        self._conversations: Dict[str, List] = {}
    
    @property
    def llm(self):
        """Lazy-load the LLM"""
        if self._llm is None:
            if self.settings.openai_api_key:
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    model=self.settings.openai_model,
                    api_key=self.settings.openai_api_key,
                    temperature=0.7
                )
            else:
                # Use a mock LLM for demo mode
                self._llm = MockLLM()
        return self._llm
    
    async def generate_ticket_response(
        self,
        subject: str,
        message: str,
        department: str,
        urgency: str
    ) -> str:
        """
        Generate a response for a support ticket using LangChain.
        
        The department and urgency are determined by Jev before this
        function is called, demonstrating the Jev + LLM pattern.
        """
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""You are a helpful customer support agent for the {department} department.
The ticket has been classified as {urgency} urgency.

Guidelines:
- Be professional and empathetic
- For {department} issues, provide relevant expertise
- If urgency is critical/high, acknowledge the urgency
- Keep responses concise but helpful
- If you need more information, ask specific questions"""),
            HumanMessage(content=f"Subject: {subject}\n\nMessage: {message}")
        ])
        
        try:
            if isinstance(self.llm, MockLLM):
                return self.llm.generate(subject, message, department, urgency)
            
            chain = prompt | self.llm
            response = await chain.ainvoke({})
            return response.content
        except Exception as e:
            return f"Thank you for contacting {department} support. Your ticket has been received and will be addressed shortly. (Error generating detailed response: {e})"
    
    async def chat_with_routing(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a chat message with Jev-based routing.
        
        This demonstrates using Jev to route queries to different
        handling strategies, then using LangChain for the response.
        """
        routes = {
            "simple_query": "Simple questions that can be answered directly",
            "complex_query": "Complex questions requiring detailed explanation",
            "action_request": "Requests to perform an action or make changes",
            "escalation": "Requests for human support or complaints"
        }
        
        # Use Jev to route the query
        routing = await self.jev.route_query(
            query=message,
            available_routes=list(routes.keys()),
            route_descriptions=routes
        )
        
        route = routing["route"]
        confidence = routing["confidence"]
        
        # Select prompt based on route
        prompts = {
            "simple_query": "Provide a concise, direct answer.",
            "complex_query": "Provide a thorough explanation with examples if helpful.",
            "action_request": "Acknowledge the request and explain what steps will be taken.",
            "escalation": "Express empathy and explain how we'll connect them with a human agent."
        }
        
        system_prompt = f"""You are a helpful AI assistant.
Route selected: {route} (confidence: {confidence:.2f})
Instruction: {prompts.get(route, prompts['simple_query'])}

{f'Additional context: {context}' if context else ''}"""

        # Get conversation history
        history = self._conversations.get(conversation_id, []) if conversation_id else []
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="history"),
            HumanMessage(content=message)
        ])
        
        try:
            if isinstance(self.llm, MockLLM):
                response_text = self.llm.chat_response(message, route)
            else:
                chain = prompt | self.llm
                response = await chain.ainvoke({"history": history})
                response_text = response.content
            
            # Update conversation history
            if conversation_id:
                if conversation_id not in self._conversations:
                    self._conversations[conversation_id] = []
                self._conversations[conversation_id].append(HumanMessage(content=message))
                self._conversations[conversation_id].append(AIMessage(content=response_text))
                # Keep only last 10 messages
                self._conversations[conversation_id] = self._conversations[conversation_id][-10:]
            
            return {
                "response": response_text,
                "route": route,
                "confidence": confidence,
                "model_used": self.settings.openai_model if self.settings.openai_api_key else "mock-llm"
            }
            
        except Exception as e:
            return {
                "response": f"I apologize, but I encountered an error processing your request. Please try again. (Error: {e})",
                "route": route,
                "confidence": confidence,
                "model_used": "error"
            }
    
    async def process_with_guardrails(
        self,
        user_input: str,
        task: str = "respond"
    ) -> Dict[str, Any]:
        """
        Process input with Jev safety guardrails.
        
        This pattern uses Jev to check input/output safety before
        and after LLM processing.
        """
        # Input safety check with Jev
        input_check = await self.jev.check_safety(user_input)
        
        if not input_check.get("is_safe", True):
            return {
                "response": "I'm sorry, but I can't process that request.",
                "blocked": True,
                "reason": "input_safety",
                "safety_check": input_check
            }
        
        # Generate response with LangChain
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful, professional assistant."),
            HumanMessage(content=user_input)
        ])
        
        try:
            if isinstance(self.llm, MockLLM):
                response_text = self.llm.simple_response(user_input)
            else:
                chain = prompt | self.llm
                response = await chain.ainvoke({})
                response_text = response.content
            
            # Output safety check with Jev
            output_check = await self.jev.check_safety(response_text)
            
            if not output_check.get("is_safe", True):
                return {
                    "response": "I generated a response but it was flagged for review.",
                    "blocked": True,
                    "reason": "output_safety",
                    "safety_check": output_check
                }
            
            return {
                "response": response_text,
                "blocked": False,
                "input_safety": input_check,
                "output_safety": output_check
            }
            
        except Exception as e:
            return {
                "response": f"Error processing request: {e}",
                "blocked": False,
                "error": str(e)
            }


class MockLLM:
    """Mock LLM for demo mode when no API key is configured"""
    
    def generate(self, subject: str, message: str, department: str, urgency: str) -> str:
        """Generate a mock ticket response"""
        urgency_prefix = {
            "critical": "We understand this is critical and are prioritizing your ticket. ",
            "high": "We've marked this as high priority. ",
            "medium": "",
            "low": ""
        }
        
        department_responses = {
            "billing": f"Thank you for contacting billing support regarding '{subject}'. We've received your inquiry and a billing specialist will review your account shortly. {urgency_prefix.get(urgency, '')}If you need immediate assistance, please have your account number ready.",
            "technical": f"Thank you for reporting this technical issue: '{subject}'. Our engineering team has been notified. {urgency_prefix.get(urgency, '')}Could you please provide any error messages or screenshots that might help us diagnose the problem?",
            "sales": f"Thank you for your interest! Regarding '{subject}', I'd be happy to discuss our options with you. {urgency_prefix.get(urgency, '')}A sales representative will reach out within 24 hours, or you can schedule a demo at your convenience.",
            "general": f"Thank you for reaching out about '{subject}'. We've received your message and will respond as soon as possible. {urgency_prefix.get(urgency, '')}Is there anything specific you'd like us to address?"
        }
        
        return department_responses.get(department, department_responses["general"])
    
    def chat_response(self, message: str, route: str) -> str:
        """Generate a mock chat response based on route"""
        responses = {
            "simple_query": f"To answer your question: This is a demo response. In a production environment with API keys configured, you would receive an AI-generated response here.",
            "complex_query": f"Let me explain in detail: This is a demo mode response. The system has routed your query as complex, which would normally trigger a more thorough explanation from the LLM.",
            "action_request": f"I understand you'd like to take an action. In demo mode, I can't perform actual actions, but with proper configuration, this system would handle your request appropriately.",
            "escalation": f"I hear that you'd like to speak with someone. In a production system, this would trigger an escalation workflow to connect you with a human agent."
        }
        return responses.get(route, responses["simple_query"])
    
    def simple_response(self, message: str) -> str:
        """Generate a simple mock response"""
        return f"Thank you for your message. This is a demo response - configure API keys for full functionality."
