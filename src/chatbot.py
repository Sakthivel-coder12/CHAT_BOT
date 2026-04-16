"""
GenAI FAQ Chatbot - Core Engine
================================
Prompt-engineered LLM chatbot for fintech customer support.
Uses Ollama (local LLM - phi3) — no API key, no quota, fully offline.
Tracks quality metrics: relevance, accuracy, completeness, latency.
"""

import os
import time
import json
import re
import requests
from datetime import datetime
from src.metrics import MetricsTracker
from src.prompt_engine import PromptEngine

OLLAMA_URL = "http://localhost:11434/api/generate"


class FAQChatbot:
    """
    Fintech FAQ Chatbot powered by Ollama (phi3) with advanced prompt engineering.
    Monitors AI output quality metrics on every response.
    """

    def __init__(self, model: str = "phi3", domain: str = "fintech"):
        self.model = model
        self.domain = domain
        self.conversation_history = []
        self.metrics_tracker = MetricsTracker()
        self.prompt_engine = PromptEngine(domain=domain)

        # Verify Ollama is running
        try:
            r = requests.get("http://localhost:11434", timeout=3)
            print(f"[Chatbot] Ollama is running ✓")
        except Exception:
            print("[Chatbot] ⚠️  Ollama not detected! Make sure Ollama is running.")
            print("[Chatbot]    Run: ollama serve  (in a separate terminal)")

        print(f"[Chatbot] Model: {self.model} | Prompt Engineering: ON | Metrics: ON\n")

    def chat(self, user_input: str) -> dict:
        """
        Process user query using chain-of-thought + few-shot prompt engineering.
        Returns response with full quality metrics.
        """
        start_time = time.time()

        # Build engineered prompt
        engineered_prompt = self.prompt_engine.build_prompt(
            user_query=user_input,
            history=self.conversation_history
        )

        # Call Ollama API
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": self.model,
                    "prompt": engineered_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.85,
                        "top_k": 40,
                        "num_predict": 512,
                    }
                },
                timeout=120
            )
            response.raise_for_status()
            raw_response = response.json().get("response", "")
        except requests.exceptions.ConnectionError:
            return {"error": "connection", "response": "⚠️  Cannot connect to Ollama. Run 'ollama serve' in a separate terminal."}
        except Exception as e:
            print(f"[ERROR] Ollama call failed: {e}")
            return {"error": str(e), "response": f"Error: {e}"}

        latency_ms = round((time.time() - start_time) * 1000, 2)

        # Parse structured response
        parsed = self.prompt_engine.parse_response(raw_response)

        # Calculate quality metrics
        metrics = self.metrics_tracker.evaluate(
            query=user_input,
            response=parsed["answer"],
            confidence=parsed.get("confidence", 0.8),
            latency_ms=latency_ms,
            is_domain_relevant=parsed.get("is_relevant", True),
        )

        # Update conversation memory (last 6 turns)
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": parsed["answer"],
            "timestamp": datetime.now().isoformat()
        })
        if len(self.conversation_history) > 12:
            self.conversation_history = self.conversation_history[-12:]

        # Log interaction
        self.metrics_tracker.log_interaction(
            query=user_input,
            response=parsed["answer"],
            metrics=metrics
        )

        return {
            "response": parsed["answer"],
            "disclaimer": parsed.get("disclaimer", ""),
            "metrics": metrics,
            "latency_ms": latency_ms,
        }

    def get_session_report(self) -> dict:
        return self.metrics_tracker.session_summary()

    def reset_conversation(self):
        self.conversation_history = []
        print("[Chatbot] Conversation reset.\n")
