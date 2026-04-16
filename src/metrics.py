"""
AI Quality Metrics Tracker
============================
Tracks and evaluates AI output quality metrics per Sakthivel JD requirements:
- Relevance Score
- Completeness Score
- Confidence Score
- Latency (ms)
- Hallucination Risk Score
- Session-level aggregated metrics
"""

import json
import os
from datetime import datetime
from collections import defaultdict


LOG_FILE = "logs/chat_metrics.jsonl"


class MetricsTracker:
    """
    Evaluates and tracks quality metrics for every AI response.
    Generates session-level reports to identify improvement areas.
    """

    def __init__(self):
        self.session_metrics = []
        os.makedirs("logs", exist_ok=True)

    def evaluate(
        self,
        query: str,
        response: str,
        confidence: float,
        latency_ms: float,
        is_domain_relevant: bool,
    ) -> dict:
        """
        Compute quality metrics for a single AI response.

        Metrics:
        - relevance_score: Is the response relevant to the domain & query?
        - completeness_score: Does the response adequately address the query?
        - confidence_score: Model's self-reported confidence (from CoT)
        - latency_ms: Response time in milliseconds
        - hallucination_risk: Inverse of confidence (proxy metric)
        - overall_quality: Weighted composite score
        """

        # Relevance Score (0.0 - 1.0)
        relevance_score = 1.0 if is_domain_relevant else 0.2

        # Completeness Score — heuristic based on response length & structure
        word_count = len(response.split())
        if word_count < 10:
            completeness_score = 0.4
        elif word_count < 30:
            completeness_score = 0.7
        elif word_count <= 150:
            completeness_score = 1.0
        else:
            completeness_score = 0.85  # Too verbose — slight penalty

        # Hallucination Risk (inverse confidence proxy)
        hallucination_risk = round(1.0 - confidence, 2)

        # Latency quality (under 2000ms = good, under 5000ms = acceptable)
        if latency_ms <= 2000:
            latency_quality = "EXCELLENT"
        elif latency_ms <= 5000:
            latency_quality = "GOOD"
        else:
            latency_quality = "NEEDS IMPROVEMENT"

        # Weighted Overall Quality Score
        overall_quality = round(
            (relevance_score * 0.35) +
            (completeness_score * 0.35) +
            (confidence * 0.30),
            3
        )

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "relevance_score": round(relevance_score, 2),
            "completeness_score": round(completeness_score, 2),
            "confidence_score": round(confidence, 2),
            "hallucination_risk": hallucination_risk,
            "latency_ms": latency_ms,
            "latency_quality": latency_quality,
            "overall_quality_score": overall_quality,
            "query_word_count": len(query.split()),
            "response_word_count": word_count,
        }

        self.session_metrics.append(metrics)
        return metrics

    def log_interaction(self, query: str, response: str, metrics: dict):
        """Append interaction + metrics to JSONL log file."""
        log_entry = {
            "query": query,
            "response": response,
            "metrics": metrics,
        }
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    def session_summary(self) -> dict:
        """Aggregate metrics across the session for performance report."""
        if not self.session_metrics:
            return {"message": "No interactions recorded yet."}

        n = len(self.session_metrics)
        avg = lambda key: round(sum(m[key] for m in self.session_metrics) / n, 3)

        latency_values = [m["latency_ms"] for m in self.session_metrics]
        quality_values = [m["overall_quality_score"] for m in self.session_metrics]

        return {
            "total_interactions": n,
            "avg_relevance_score": avg("relevance_score"),
            "avg_completeness_score": avg("completeness_score"),
            "avg_confidence_score": avg("confidence_score"),
            "avg_hallucination_risk": avg("hallucination_risk"),
            "avg_latency_ms": avg("latency_ms"),
            "min_latency_ms": min(latency_values),
            "max_latency_ms": max(latency_values),
            "avg_overall_quality": avg("overall_quality_score"),
            "best_response_quality": max(quality_values),
            "worst_response_quality": min(quality_values),
            "quality_benchmark": "PASS" if avg("overall_quality_score") >= 0.75 else "NEEDS IMPROVEMENT",
        }
