import os
import json
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
import chromadb
from dotenv import load_dotenv
# --- DATABASE SIDE QUEST IMPORTS ---
from sqlalchemy.orm import Session
from fastapi import Depends
from database import SessionLocal
import models
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Load local .env configurations from this app folder, regardless of where uvicorn is run.
load_dotenv(BASE_DIR / ".env")

app = FastAPI()
# Database Session Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 1. Initialize Clients securely via Environment Variables
chroma_client = chromadb.Client()

def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to ai-ticket-api/.env or set it in your terminal before starting uvicorn."
        )
    return Groq(api_key=api_key)

# 2. Setup Vector Database (RAG Layer)
collection = chroma_client.get_or_create_collection(name="ticket_knowledge")

# Seed the vector database with company policy on startup
with open(BASE_DIR / "company_policy.txt", "r") as f:
    policy_text = f.read()

chunks = [chunk.strip() for chunk in policy_text.split("\n\n") if chunk.strip()]
collection.add(
    documents=chunks,
    ids=[f"policy_chunk_{i}" for i in range(len(chunks))]
)

# 3. Define the Incoming Request Schema
class Ticket(BaseModel):
    sender: str
    subject: str
    message: str

# 4. Define Agentic Execution Tools (The Python "Hands")
def lookup_refund_eligibility(user_email: str) -> str:
    """Looks up if a user is eligible for a refund based on internal corporate tracking logs."""
    if "btech" in user_email.lower() or "vikas" in user_email.lower():
        return "DENIED: System logs verify that B.Tech training keys have already been activated for this account."
    return "APPROVED: Account within the standard 14-day window. No keys activated."

def trigger_account_audit(user_email: str, issue_description: str) -> str:
    """Flags a user account for a manual backend system technical or security audit."""
    return f"SUCCESS: Technical incident token generated for {user_email}. Issue registered: '{issue_description}'."

# 5. Define the Groq Tool Schemas (The Blueprint for Llama 3)
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "lookup_refund_eligibility",
            "description": "Use this tool when a customer explicitly requests a refund or money back for a purchase.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_email": {
                        "type": "string",
                        "description": "The email address of the customer making the request."
                    }
                },
                "required": ["user_email"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "trigger_account_audit",
            "description": "Use this tool ONLY when a customer reports server crashes, database errors, timeouts, or potential security vulnerabilities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_email": {
                        "type": "string",
                        "description": "The customer email address."
                    },
                    "issue_description": {
                        "type": "string",
                        "description": "A brief technical summary of the system failure or crash."
                    }
                },
                "required": ["user_email", "issue_description"]
            }
        }
    }
]

@app.get("/")
def home():
    return {"message": "AI Agentic RAG Ticket API Running"}

@app.post("/analyze-ticket")
def analyze_ticket(ticket: Ticket):
    try:
        incoming_query = f"Subject: {ticket.subject}. Message: {ticket.message}"
        
        # Step A: Perform Vector Search to find matching organizational policy
        results = collection.query(
            query_texts=[incoming_query],
            n_results=1
        )
        retrieved_context = results['documents'][0][0] if results['documents'] else "No specific policy found."

        # Step B: Instruct the model to analyze context and select appropriate tools
# Step B: Instruct the model explicitly on tool execution standards
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an advanced enterprise AI Agent with access to tool-calling functions.\n"
                    "You MUST use the internal company context provided below to dictate your tool calling logic.\n"
                    "When calling a function/tool, you MUST strictly output the parameters in proper JSON format matching the schema rules provided.\n"
                    "Do not invent functions outside the provided tools_schema list.\n\n"
                    f"INTERNAL COMPANY CONTEXT:\n{retrieved_context}"
                )
            },
            {"role": "user", "content": incoming_query}
        ]

        # Step C: Send the payload to Groq with full tool features enabled
        client = get_groq_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=tools_schema,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # Step D: The Execution Layer (If the Agent wants to fire a function)
        if tool_calls:
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name == "lookup_refund_eligibility":
                    tool_output = lookup_refund_eligibility(user_email=ticket.sender)
                    summary_text = "Automated refund check executed."
                    urgency_level = "Medium"
                    dept = "Billing"
                    
                elif function_name == "trigger_account_audit":
                    tool_output = trigger_account_audit(
                        user_email=ticket.sender, 
                        issue_description=function_args.get("issue_description", ticket.subject)
                    )
                    summary_text = "System audit triggered for critical technical failure."
                    urgency_level = "High"
                    dept = "Technical Support"
                else:
                    tool_output = "Unknown action requested."
                    summary_text = "Agent attempted invalid tool selection."
                    urgency_level = "Medium"
                    dept = "Technical Support"
                
                # Clean structural return object sent down the pipeline to n8n
                return {
                    "summary": summary_text,
                    "urgency": urgency_level,
                    "department": dept,
                    "sentiment": "Anxious",
                    "action_taken": tool_output
                }

        # Step E: Fallback if the ticket is simple conversation requiring no tools
        return {
            "summary": "General inquiry handled without automation tools.",
            "urgency": "Low",
            "department": "Sales",
            "sentiment": "Neutral",
            "action_taken": "No automated action required."
        }

    except Exception as e:
        return {
            "summary": "Agentic Loop Error",
            "urgency": "High",
            "department": "Technical Support",
            "sentiment": "Stressed",
            "action_taken": f"Error: {str(e)}"
        }

@app.get("/tickets")
def get_tickets(db: Session = Depends(get_db)):
    """Fetches every single support ticket record resting inside the cloud database."""
    tickets = db.query(models.Ticket).all()
    return tickets


@app.get("/tickets/high")
def get_high_priority_tickets(db: Session = Depends(get_db)):
    """Fetches ONLY the tickets categorized under 'High' urgency status."""
    tickets = db.query(models.Ticket).filter(models.Ticket.urgency == "High").all()
    return tickets

@app.get("/tickets/billing")
def get_billing_tickets(db: Session = Depends(get_db)):
    """Fetches ONLY the tickets routed to the Billing department."""
    tickets = db.query(models.Ticket).filter(models.Ticket.department == "Billing").all()
    return tickets

@app.get("/analytics")
def get_analytics(db: Session = Depends(get_db)):
    """Computes live aggregated system KPIs across the cloud database cluster."""
    total_count = db.query(models.Ticket).count()
    
    high_priority_count = db.query(models.Ticket).filter(
        models.Ticket.urgency == "High"
    ).count()

    return {
        "total_tickets": total_count,
        "high_priority_tickets": high_priority_count,
        "system_status": "Healthy" if total_count > 0 else "Empty"
    }
