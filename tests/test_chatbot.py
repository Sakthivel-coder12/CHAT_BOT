"""
Unit Tests — GenAI FAQ Chatbot
================================
Tests prompt engineering, response parsing, and metrics calculation.
Run: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.prompt_engine import PromptEngine
from src.metrics import MetricsTracker


class TestPromptEngine:
    """Test prompt engineering logic."""

    def setup_method(self):
        self.engine = PromptEngine(domain="fintech")

    def test_prompt_contains_system_persona(self):
        prompt = self.engine.build_prompt("What is a loan?", history=[])
        assert "NaviBot" in prompt
        assert "fintech" in prompt.lower() or "finance" in prompt.lower()

    def test_prompt_contains_few_shot_examples(self):
        prompt = self.engine.build_prompt("What is a loan?", history=[])
        assert "FEW-SHOT EXAMPLES" in prompt
        assert "personal loan" in prompt.lower()

    def test_prompt_contains_chain_of_thought(self):
        prompt = self.engine.build_prompt("What is insurance?", history=[])
        assert "step-by-step" in prompt or "Think" in prompt

    def test_prompt_includes_history(self):
        history = [
            {"role": "user", "content": "Hello", "timestamp": ""},
            {"role": "assistant", "content": "Hi there!", "timestamp": ""},
        ]
        prompt = self.engine.build_prompt("Tell me more", history=history)
        assert "CONVERSATION HISTORY" in prompt
        assert "Hello" in prompt

    def test_parse_valid_json_response(self):
        raw = '{"answer": "A loan is borrowed money.", "confidence": 0.95, "is_relevant": true, "disclaimer": "Subject to approval."}'
        parsed = self.engine.parse_response(raw)
        assert parsed["answer"] == "A loan is borrowed money."
        assert parsed["confidence"] == 0.95
        assert parsed["is_relevant"] is True
        assert parsed["disclaimer"] == "Subject to approval."

    def test_parse_response_with_markdown_fences(self):
        raw = '```json\n{"answer": "Test answer.", "confidence": 0.8, "is_relevant": true, "disclaimer": ""}\n```'
        parsed = self.engine.parse_response(raw)
        assert parsed["answer"] == "Test answer."

    def test_parse_malformed_json_falls_back(self):
        raw = "This is not JSON but a plain text answer."
        parsed = self.engine.parse_response(raw)
        assert parsed["answer"] == raw.strip()
        assert parsed["confidence"] == 0.7  # default fallback


class TestMetricsTracker:
    """Test quality metrics calculation."""

    def setup_method(self):
        self.tracker = MetricsTracker()

    def test_high_quality_response_scores_well(self):
        metrics = self.tracker.evaluate(
            query="What is a mutual fund?",
            response="A mutual fund pools money from many investors to invest in stocks, bonds, or other securities. Managed by professionals, it offers diversification and is suitable for long-term wealth building.",
            confidence=0.95,
            latency_ms=1200,
            is_domain_relevant=True,
        )
        assert metrics["relevance_score"] == 1.0
        assert metrics["overall_quality_score"] >= 0.85
        assert metrics["latency_quality"] == "EXCELLENT"

    def test_off_topic_response_penalized(self):
        metrics = self.tracker.evaluate(
            query="Tell me a joke",
            response="I only help with finance.",
            confidence=1.0,
            latency_ms=500,
            is_domain_relevant=False,
        )
        assert metrics["relevance_score"] == 0.2
        assert metrics["overall_quality_score"] < 0.7

    def test_short_response_penalized(self):
        metrics = self.tracker.evaluate(
            query="Explain loans",
            response="Loans are money.",
            confidence=0.9,
            latency_ms=800,
            is_domain_relevant=True,
        )
        assert metrics["completeness_score"] < 0.8

    def test_slow_latency_flagged(self):
        metrics = self.tracker.evaluate(
            query="What is UPI?",
            response="UPI stands for Unified Payments Interface. It is a real-time payment system developed by NPCI that facilitates inter-bank peer-to-peer and person-to-merchant transactions.",
            confidence=0.9,
            latency_ms=8000,
            is_domain_relevant=True,
        )
        assert metrics["latency_quality"] == "NEEDS IMPROVEMENT"

    def test_session_summary_after_interactions(self):
        for i in range(3):
            self.tracker.evaluate(
                query=f"Question {i}",
                response="This is a detailed and helpful response about fintech topics.",
                confidence=0.9,
                latency_ms=1000 + i * 200,
                is_domain_relevant=True,
            )
        summary = self.tracker.session_summary()
        assert summary["total_interactions"] == 3
        assert "avg_overall_quality" in summary
        assert summary["quality_benchmark"] in ["PASS", "NEEDS IMPROVEMENT"]

    def test_hallucination_risk_is_inverse_confidence(self):
        metrics = self.tracker.evaluate(
            query="What is a ULIP?",
            response="A ULIP combines insurance and investment in a single product.",
            confidence=0.85,
            latency_ms=900,
            is_domain_relevant=True,
        )
        assert abs(metrics["hallucination_risk"] - (1.0 - 0.85)) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
