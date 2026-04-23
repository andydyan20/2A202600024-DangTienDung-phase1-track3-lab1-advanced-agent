from __future__ import annotations
import time
import json
import ollama
import re
import os
from typing import Any
from .schemas import QAExample, JudgeResult, ReflectionEntry
from .utils import normalize_answer
from dotenv import load_dotenv

load_dotenv()

# Initialize Ollama client
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:1b")


def _call_ollama(prompt: str, system: str = None, json_mode: bool = False) -> dict[str, Any]:
    """Call Ollama API and return response."""
    try:
        start_time = time.time()
        response = ollama.generate(
            model=MODEL_NAME, prompt=prompt, system=system, stream=False, format="json" if json_mode else None
        )
        end_time = time.time()

        # Extract token count from Ollama response
        # Ollama provides prompt_eval_count and eval_count
        prompt_tokens = response.get("prompt_eval_count", 0)
        completion_tokens = response.get("eval_count", 0)
        total_tokens = prompt_tokens + completion_tokens

        latency_ms = int((end_time - start_time) * 1000)

        return {
            "response": response["response"],
            "tokens": total_tokens,
            "latency_ms": latency_ms,
        }
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        # Fallback values
        return {
            "response": "Error: Could not generate response",
            "tokens": 0,
            "latency_ms": 0,
        }


def actor_answer(
    example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]
) -> tuple[str, int, int]:
    """Generate answer using actual LLM (Ollama).

    Returns:
        tuple: (answer, tokens_used, latency_ms)
    """
    # Prepare context from examples
    context_text = "\n\n".join(
        [f"Title: {ctx.title}\nText: {ctx.text}" for ctx in example.context]
    )

    # Build prompt for actor
    system_prompt = """You are a helpful AI agent that answers questions based on provided context. 
    Read the context carefully and answer the question concisely and accurately.
    Only use information from the provided context to form your answer.
    If the context does not contain sufficient information to answer the question, state that you cannot answer based on the given information."""

    user_prompt = f"""Context:
    {context_text}

    Question: {example.question}

    Answer the question based only on the provided context. Be concise and direct."""

    # Add reflection memory if available (for reflexion agent)
    if agent_type == "reflexion" and reflection_memory:
        user_prompt += f"\n\nReflections from previous attempts:\n" + "\n".join(
            [f"- {mem}" for mem in reflection_memory]
        )

    result = _call_ollama(user_prompt, system_prompt, json_mode=False)
    return result["response"].strip(), result["tokens"], result["latency_ms"]


def evaluator(example: QAExample, answer: str) -> JudgeResult:
    """Evaluate answer using actual LLM (Ollama) as judge."""
    # Normalize answers for comparison
    normalized_gold = normalize_answer(example.gold_answer)
    normalized_answer = normalize_answer(answer)

    # Simple exact match check first
    if normalized_gold == normalized_answer:
        return JudgeResult(
            score=1, reason="Final answer matches the gold answer after normalization."
        )

    # Use LLM as judge for more nuanced evaluation
    system_prompt = """You are an expert evaluator that determines if a predicted answer is correct 
    compared to a gold answer. You must respond with valid JSON only.
    
    Consider semantic equivalence, not just exact string matching.
    Respond with:
    {
        "score": 0 or 1,
        "reason": "explanation for the score",
        "missing_evidence": ["list of missing evidence if score is 0"],
        "spurious_claims": ["list of spurious claims if score is 0"]
    }"""

    user_prompt = f"""Question: {example.question}
    Gold Answer: {example.gold_answer}
    Predicted Answer: {answer}

    Is the predicted answer correct? Respond with JSON only."""

    result = _call_ollama(user_prompt, system_prompt, json_mode=True)

    try:
        # Regex extraction for safety
        json_match = re.search(r"\{.*\}", result["response"], re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found")
            
        judge_dict = json.loads(json_match.group(0))

        # SANITIZATION: Ensure these are lists to satisfy Pydantic
        def ensure_list(val):
            
            return val if isinstance(val, list) else []

        return JudgeResult(
            score=int(judge_dict.get("score", 0)),
            reason=str(judge_dict.get("reason", "Evaluation completed")),
            missing_evidence=ensure_list(judge_dict.get("missing_evidence")),
            spurious_claims=ensure_list(judge_dict.get("spurious_claims")),
        )
    except Exception as e:
        print(f"Warning: LLM judge failed ({e}), falling back to exact match")
        return JudgeResult(
            score=1 if normalized_gold == normalized_answer else 0,
            reason="Fallback evaluation: exact match",
        )


def reflector(
    example: QAExample, attempt_id: int, judge: JudgeResult
) -> ReflectionEntry:
    """Generate reflection using actual LLM (Ollama)."""
    system_prompt = """You are an expert at analyzing failures and generating insights for improvement.
    Given a question, the predicted answer, and evaluation feedback, generate a concise reflection
    that includes:
    1. The failure reason
    2. A lesson learned from the failure
    3. A strategy for the next attempt

    CRITICAL: You MUST respond with valid JSON only, in the exact format:
    {
        "failure_reason": "explanation of why the answer was wrong",
        "lesson": "what was learned from this failure",
        "next_strategy": "specific strategy to try next time"
    }
    Do not include any text before or after the JSON. Do not use markdown formatting."""

    user_prompt = f"""Question: {example.question}
    Gold Answer: {example.gold_answer}
    Predicted Answer: {attempt_id} (this is attempt number)
    Evaluation Feedback: {judge.reason}

    Generate a reflection to improve future attempts. Respond with ONLY the JSON object."""

    result = _call_ollama(user_prompt, system_prompt, json_mode=True)

    try:
        # Parse JSON response
        print(f"LLM reflection response: {result}")
        reflection_dict = json.loads(result["response"])
        return ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=reflection_dict.get("failure_reason") or judge.reason,
            lesson=reflection_dict.get("lesson") or "Need to improve answer accuracy",
            next_strategy=reflection_dict.get("next_strategy") or "Try a different approach",
        )
    except (json.JSONDecodeError, KeyError) as e:
        # Try to extract JSON from the response if there's extra text
        try:
            import re

            # Look for JSON-like pattern in the response
            json_match = re.search(r"\{.*\}", result["response"], re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                reflection_dict = json.loads(json_str)
                return ReflectionEntry(
                    attempt_id=attempt_id,
                    failure_reason=reflection_dict.get("failure_reason") or judge.reason,
                    lesson=reflection_dict.get(
                        "lesson", "Need to improve answer accuracy"
                    ),
                    next_strategy=reflection_dict.get(
                        "next_strategy", "Try a different approach"
                    ),
                )
        except Exception:
            pass

        # Fallback reflection if LLM reflector fails
        print(f"Warning: LLM reflector failed ({e}), using fallback")
        return ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="Need to improve answer accuracy based on feedback",
            next_strategy="Focus on addressing the specific failure reason mentioned",
        )

# def clean_json_string(raw_str):
#     # Loại bỏ markdown code blocks nếu AI lỡ tay thêm vào
#     raw_str = re.sub(r"```json|```", "", raw_str)
#     # Loại bỏ các ký tự điều khiển không hợp lệ
#     raw_str = raw_str.strip()
#     return raw_str