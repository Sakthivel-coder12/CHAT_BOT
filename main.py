"""
GenAI FAQ Chatbot - Main Entry Point
======================================
Runs the chatbot using Ollama (local LLM, no API key needed).

Usage:
    1. Make sure Ollama is running: ollama serve
    2. Run: python main.py
    3. Optional - specify model: python main.py --model phi3
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.chatbot import FAQChatbot

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║          sakthivelBot — GenAI Fintech FAQ Chatbot                    ║
║          Powered by Ollama (Local LLM) + Prompt Engineering     ║
║          Quality Metrics: Relevance | Accuracy | Latency        ║
╚══════════════════════════════════════════════════════════════════╝

  Ask me about: Loans • Insurance • Mutual Funds • UPI • Savings
  Commands: 'metrics' → session report | 'reset' → new session | 'quit' → exit
"""

DIVIDER = "─" * 66


def print_metrics(metrics: dict):
    print(f"\n  📊 Quality Metrics:")
    print(f"     Relevance     : {'█' * int(metrics['relevance_score'] * 10)}{'░' * (10 - int(metrics['relevance_score'] * 10))} {metrics['relevance_score']:.2f}")
    print(f"     Completeness  : {'█' * int(metrics['completeness_score'] * 10)}{'░' * (10 - int(metrics['completeness_score'] * 10))} {metrics['completeness_score']:.2f}")
    print(f"     Confidence    : {'█' * int(metrics['confidence_score'] * 10)}{'░' * (10 - int(metrics['confidence_score'] * 10))} {metrics['confidence_score']:.2f}")
    print(f"     Overall Score : {metrics['overall_quality_score']:.3f} / 1.000")
    print(f"     Latency       : {metrics['latency_ms']} ms  [{metrics['latency_quality']}]")
    print(f"     Halluc. Risk  : {metrics['hallucination_risk']:.2f}  (lower is better)")


def print_session_report(report: dict):
    print(f"\n{'═' * 66}")
    print(f"  📈 SESSION QUALITY REPORT")
    print(f"{'═' * 66}")
    print(f"  Total Interactions    : {report.get('total_interactions', 0)}")
    print(f"  Avg Relevance Score   : {report.get('avg_relevance_score', 0):.3f}")
    print(f"  Avg Completeness      : {report.get('avg_completeness_score', 0):.3f}")
    print(f"  Avg Confidence        : {report.get('avg_confidence_score', 0):.3f}")
    print(f"  Avg Hallucination Risk: {report.get('avg_hallucination_risk', 0):.3f}")
    print(f"  Avg Latency           : {report.get('avg_latency_ms', 0):.1f} ms")
    print(f"  Best Response Quality : {report.get('best_response_quality', 0):.3f}")
    print(f"  Overall Quality Bench : {report.get('quality_benchmark', 'N/A')}")
    print(f"{'═' * 66}\n")


def main():
    parser = argparse.ArgumentParser(description="sakthivelBot - GenAI Fintech Chatbot")
    parser.add_argument("--model", default="phi3", help="Ollama model to use (default: phi3)")
    args = parser.parse_args()

    print(BANNER)

    bot = FAQChatbot(model=args.model, domain="fintech")
    show_metrics = True

    print(DIVIDER)

    while True:
        try:
            user_input = input("\n  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  👋 Goodbye! Session ended.\n")
            report = bot.get_session_report()
            if report.get("total_interactions", 0) > 0:
                print_session_report(report)
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("\n  👋 Goodbye! Here's your session report:")
            print_session_report(bot.get_session_report())
            break
        elif user_input.lower() == "reset":
            bot.reset_conversation()
            print("  ✅ Conversation reset. Starting fresh!\n")
            continue
        elif user_input.lower() == "metrics":
            print_session_report(bot.get_session_report())
            continue
        elif user_input.lower() == "toggle":
            show_metrics = not show_metrics
            print(f"  📊 Metrics display: {'ON' if show_metrics else 'OFF'}\n")
            continue

        print(f"\n  sakthivelBot: ", end="", flush=True)
        result = bot.chat(user_input)

        if "error" in result:
            print(result["response"])
            continue

        print(result["response"])

        if result.get("disclaimer"):
            print(f"\n  ⚠️  {result['disclaimer']}")

        if show_metrics:
            print_metrics(result["metrics"])

        print(f"\n{DIVIDER}")


if __name__ == "__main__":
    main()
