"""
ai_consultant.py
NagarMitra Phase 8 — AI Research Consultant (Groq)
Provides real-time academic summaries and implementation plans based on environmental research.
"""

import os
import logging
from groq import Groq

logger = logging.getLogger(__name__)

# Initialize Groq Client
# By default, it expects the GROQ_API_KEY environment variable.
groq_client = None
try:
    groq_client = Groq()
except Exception as e:
    logger.warning("Groq API Key not found. AI Consultant features will be unavailable.")


def get_research_consultation(query: str, ward_data: dict = None) -> str:
    """
    Pings Groq's LLaMa-3 model to act as an environmental scientist, retrieving cited 
    research summaries to answer the administrator's query.
    """
    if not groq_client:
        return "Error: GROQ_API_KEY is not configured in the environment variables."

    # Construct the foundational system prompt
    system_prompt = (
        "You are NagarMitra AI, an expert Environmental Scientist & Civic Consultant for the Government of Delhi. "
        "Your role is to answer questions from civic administrators about urban pollution, mitigation strategies, and health impacts. "
        "You MUST base your answers on published scientific research, citing organizations like the WHO, TERI, or CPCB when applicable. "
        "Keep your response structured, actionable, and strictly focused on environmental science. Do not hallucinate data."
    )

    if ward_data:
        system_prompt += f"\n\nContext: The administrator is asking this question in the context of the `{ward_data.get('ward')}` ward, which currently has an AQI of {ward_data.get('aqi')}."

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            model="llama-3.3-70b-versatile", # Updated to currently supported model
            temperature=0.3, # Keep it scientific and deterministic
            max_tokens=1024,
        )
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Groq Inference Error: {e}")
        return f"Warning: The AI Consultation engine encountered an error. Details: {str(e)}"
