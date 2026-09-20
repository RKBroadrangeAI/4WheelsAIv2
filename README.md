# LangChain + Jev Web Application

A modern web application demonstrating the integration of **LangChain** with **Jev** (TypeSafe AI's System One decision model).

## What is Jev?

**Jev** is TypeSafe AI's System One decision model. Unlike traditional LLMs that generate text, Jev is designed for:

- **Fast, structured decisions** with calibrated confidence scores
- **Up to 200x faster** inference than LLMs for classification tasks  
- **Up to 400x lower cost** for decision-making operations
- **Three decision types**: Choice, Score, and Noul (yes/no)

### When to Use Jev vs LLM

| Use Jev For | Use LLM For |
|-------------|-------------|
| Routing queries | Generating responses |
| Classification | Open-ended reasoning |
| Guardrails/safety | Creative writing |
| Tool selection | Explaining concepts |
| Quick decisions | Complex analysis |

## Architecture

```
Input → [Jev Decision] → [LangChain Response] → Output
         (fast routing)    (text generation)
```

This app demonstrates the pattern of using Jev for quick, structured decisions (routing, classification, guardrails) while using LangChain for text generation.

## Features

- **Ticket Classifier**: Submit support tickets and watch Jev classify them by department and urgency, then LangChain generates a suggested response
- **Smart Chat**: Chat with an AI that uses Jev to route queries to different handling strategies
- **Guardrails Demo**: See how Jev can check input/output safety
- **Demo Mode**: Works without API keys using mock responses

## Quick Start

### Prerequisites

- Python 3.10+
- pip or uv

### Installation

1. Clone and install dependencies:

```bash
pip install -r requirements.txt
```

2. (Optional) Configure API keys:

```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Run the application:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4. Open http://localhost:8000 in your browser

## API Endpoints

### Ticket Classification

```bash
POST /api/tickets/classify
{
  "subject": "Payment failed",
  "message": "I can't complete my subscription payment"
}
```

### Chat with Routing

```bash
POST /api/chat/
{
  "message": "How do I reset my password?",
  "conversation_id": "optional-uuid"
}
```

### Chat with Guardrails

```bash
POST /api/chat/with-guardrails
{
  "message": "Your message here"
}
```

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `TYPESAFE_API_KEY` | TypeSafe AI API key for Jev | - |
| `OPENROUTER_API_KEY` | OpenRouter API key (alternative Jev access) | - |
| `OPENAI_API_KEY` | OpenAI API key for LangChain | - |
| `OPENAI_MODEL` | OpenAI model to use | gpt-4o-mini |
| `DEBUG` | Enable debug mode | true |

## Project Structure

```
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── models/              # Pydantic schemas
│   │   └── schemas.py
│   ├── routers/             # API endpoints
│   │   ├── tickets.py       # Ticket classification
│   │   └── chat.py          # Chat endpoints
│   └── services/            # Business logic
│       ├── jev_service.py   # Jev integration
│       └── langchain_service.py  # LangChain integration
├── templates/
│   └── index.html           # Frontend template
├── static/
│   ├── css/styles.css       # Styles
│   └── js/app.js            # Frontend JavaScript
├── requirements.txt
└── .env.example
```

## Learn More

- [Jev Documentation](https://typesafe.ai/docs)
- [LangChain Documentation](https://python.langchain.com/)
- [Building a Harness with Jev](https://www.langchain.com/blog/building-a-harness-with-jev)
