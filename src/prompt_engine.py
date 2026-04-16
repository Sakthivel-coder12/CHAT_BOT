"""
Prompt Engineering Module
==========================
Implements advanced prompt engineering techniques:
- Chain-of-Thought (CoT) reasoning
- Few-Shot examples
- Domain constraints (fintech-specific)
- Conversation memory injection
- Structured output formatting
"""

import json
import re


FEW_SHOT_EXAMPLES = [
    {
        "user": "What is a personal loan?",
        "answer": "A personal loan is an unsecured loan provided by financial institutions for personal use — such as medical emergencies, travel, or home improvement. At sakthivel, personal loans are processed digitally with quick disbursal directly to your bank account. You repay in fixed monthly EMIs over a chosen tenure.",
        "confidence": 0.95,
        "is_relevant": True,
        "disclaimer": "Loan approval is subject to eligibility and credit assessment."
    },
    {
        "user": "How do I claim insurance?",
        "answer": "To file an insurance claim: (1) Report the incident immediately via the app or helpline. (2) Submit required documents such as policy number, ID proof, and incident report. (3) The insurer reviews and processes the claim within the stipulated timeframe. Always keep your policy documents handy for faster processing.",
        "confidence": 0.92,
        "is_relevant": True,
        "disclaimer": "Claim approval depends on policy terms and conditions."
    },
    {
        "user": "Tell me a joke",
        "answer": "I'm specialized in fintech topics like loans, insurance, investments, and payments. For jokes, I'd recommend a comedy app! Is there anything finance-related I can help you with today?",
        "confidence": 1.0,
        "is_relevant": False,
        "disclaimer": ""
    },
]


SYSTEM_PERSONA = """You are sakthivelBot, an expert AI assistant for sakthivel — a leading Indian fintech company.
Your expertise covers: Personal Loans, Home Loans, Insurance (health & motor), Mutual Funds, Digital Gold, UPI Payments, and general financial literacy.

Your personality: Professional, clear, empathetic, and concise. You speak like a knowledgeable friend, not a robot.
Your goal: Help users understand financial products, resolve queries, and make informed decisions.

STRICT RULES:
1. Only answer questions related to finance, fintech, banking, insurance, loans, investments, or sakthivel products.
2. For off-topic questions, politely redirect to finance topics.
3. Always add a brief disclaimer for regulatory topics (loans, investments, insurance).
4. Never give definitive legal or tax advice — always recommend consulting a certified professional.
5. Be concise — answers should be under 150 words unless complexity demands more.
6. Use numbered steps for processes, plain language for concepts."""


class PromptEngine:
    """
    Builds engineered prompts using CoT, few-shot, and structured output techniques.
    """

    def __init__(self, domain: str = "fintech"):
        self.domain = domain
        self.system_persona = SYSTEM_PERSONA
        self.few_shot_examples = FEW_SHOT_EXAMPLES

    def build_prompt(self, user_query: str, history: list) -> str:
        """
        Constructs the full engineered prompt with:
        - System persona
        - Few-shot examples
        - Conversation history
        - Chain-of-thought instruction
        - Structured output format
        """

        # Build few-shot block
        few_shot_block = "\n\n--- FEW-SHOT EXAMPLES ---\n"
        for ex in self.few_shot_examples:
            few_shot_block += f"""
User: {ex['user']}
Response JSON:
{{
  "answer": "{ex['answer']}",
  "confidence": {ex['confidence']},
  "is_relevant": {str(ex['is_relevant']).lower()},
  "disclaimer": "{ex['disclaimer']}"
}}
"""

        # Build conversation history block
        history_block = ""
        if history:
            history_block = "\n\n--- CONVERSATION HISTORY ---\n"
            for turn in history[-6:]:  # Last 3 turns
                role = "User" if turn["role"] == "user" else "sakthivelBot"
                history_block += f"{role}: {turn['content']}\n"

        # Chain-of-thought instruction
        cot_instruction = """
--- YOUR TASK ---
Think step-by-step before answering:
1. Is this question related to finance, banking, insurance, loans, or investments?
2. What is the user really asking? (intent recognition)
3. What is the most accurate, helpful answer?
4. Does this need a disclaimer?
5. Rate your confidence (0.0 to 1.0) in your answer.

Then respond ONLY with a valid JSON object in this exact format:
{
  "answer": "<your helpful answer here>",
  "confidence": <float between 0.0 and 1.0>,
  "is_relevant": <true or false>,
  "disclaimer": "<regulatory disclaimer or empty string>"
}

Do NOT include any text outside the JSON. Do NOT use markdown code blocks."""

        full_prompt = (
            f"{self.system_persona}"
            f"{few_shot_block}"
            f"{history_block}"
            f"\n\n--- CURRENT USER QUERY ---\n"
            f"User: {user_query}"
            f"\n{cot_instruction}"
        )

        return full_prompt

    def parse_response(self, raw_response: str) -> dict:
        """
        Parses the structured JSON response from the model.
        Falls back gracefully if JSON is malformed.
        """
        try:
            # Strip markdown code blocks if present
            clean = re.sub(r"```json|```", "", raw_response).strip()
            parsed = json.loads(clean)
            return {
                "answer": parsed.get("answer", raw_response),
                "confidence": float(parsed.get("confidence", 0.8)),
                "is_relevant": bool(parsed.get("is_relevant", True)),
                "disclaimer": parsed.get("disclaimer", ""),
            }
        except (json.JSONDecodeError, ValueError):
            # Graceful fallback — use raw response as answer
            return {
                "answer": raw_response.strip(),
                "confidence": 0.7,
                "is_relevant": True,
                "disclaimer": "",
            }
