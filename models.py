#to be used with updated_main.py and database.py

from sqlalchemy import Column, Integer, String, Text, TIMESTAMP
from database import Base
from sqlalchemy.sql import func

class Ticket(Base):
    __tablename__ = "tickets"

    # Match this exactly to your live Supabase table structure
    id = Column(Integer, primary_key=True, index=True)
    sender = Column(String)
    subject = Column(String)
    summary = Column(Text)
    urgency = Column(String)
    department = Column(String)
    sentiment = Column(String)
    action_taken = Column(Text)  # Our Day 6 agentic addition!
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
