from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
import chromadb

app = FastAPI()

# 1. Initialize Groq and ChromaDB clients
client = Groq(api_key="YOUR_REAL_GROQ_API_KEY")
chroma_client = chromadb.Client()

# 2. Create an in-memory vector database collection
collection = chroma_client.get_or_create_collection(name="ticket_knowledge")

# 3. Seed the vector database with our operational rules on startup
with open("company_policy.txt", "r") as f:
    policy_text = f.read()

# Split policy into logical chunks and index them
chunks = [chunk.strip() for chunk in policy_text.split("\n\n") if chunk.strip()]
collection.add(
    documents=chunks,
    ids=[f"policy_chunk_{i}" for i in range(len(chunks))]
)

class Ticket(BaseModel):
    subject: str
    message: str

@app.get("/")
def home():
    return {"message": "AI RAG Ticket API Running"}

@app.post("/analyze-ticket")
def analyze_ticket(ticket: Ticket):
    try:
        incoming_query = f"Subject: {ticket.subject}. Message: {ticket.message}"
        
        # 4. Vector Search: Query the local database for the closest matching policy chunks
        results = collection.query(
            query_texts=[incoming_query],
            n_results=1
        )
        retrieved_context = results['documents'][0][0] if results['documents'] else "No specific policy found."

        # 5. Context-Injected Prompting
        prompt = (
            "You are an enterprise triage backend routing system.\n"
            "You MUST use the internal company context provided below to guide your classification logic.\n"
            "Return ONLY a valid, minified JSON object without markdown formatting.\n\n"
            "Expected JSON Format:\n"
            "{\n"
            '    "summary": "one short sentence summarizing the issue",\n'
            '    "urgency": "Low, Medium, or High",\n'
            '    "department": "Technical Support, Billing, or Sales"\n'
            "}\n\n"
            f"INTERNAL COMPANY CONTEXT:\n{retrieved_context}\n\n"
            f"CUSTOMER TICKET:\n{incoming_query}"
        )

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}]
        )

        return {"analysis": response.choices[0].message.content}

    except Exception as e:
        return {
            "summary": "RAG API Error Occurred",
            "urgency": "High",
            "department": "Technical Support",
            "error": str(e)
        }