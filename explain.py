# explain.py
import os
from dotenv import load_dotenv
load_dotenv()

# new v1 client
from openai import OpenAI
client = OpenAI()

LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o")

def _extract_message_content(resp):
    """
    Safe extraction for different response shapes from openai.chat.completions.create(...)
    Returns string content or None.
    """
    try:
        # 1) dict-like path (older or dict responses)
        if isinstance(resp, dict):
            return resp["choices"][0]["message"]["content"]
    except Exception:
        pass

    try:
        # 2) object-like path: resp.choices[0].message -> ChatCompletionMessage
        choice = resp.choices[0]
        # message may be an object with .content attribute or .content list/dict
        msg = getattr(choice, "message", None)
        if msg is None:
            # some shapes have .message as a dict-like
            msg = choice.get("message") if hasattr(choice, "get") else None

        if msg is None:
            # as fallback, try choice["message"]
            try:
                return choice["message"]["content"]
            except Exception:
                pass

        # msg could be dict-like
        if isinstance(msg, dict):
            return msg.get("content")

        # msg could be an object with attribute .content (string)
        content = getattr(msg, "content", None)
        if content:
            # sometimes content itself is a dict/list, handle common cases:
            if isinstance(content, (list, tuple)) and len(content) > 0:
                # e.g. [{"type":"output_text","text":"..."}] style
                first = content[0]
                if isinstance(first, dict) and "text" in first:
                    return first["text"]
                # otherwise join
                return " ".join(str(x) for x in content)
            if isinstance(content, dict) and "text" in content:
                return content["text"]
            return str(content)

    except Exception:
        pass

    # final fallback: try stringifying resp
    try:
        return str(resp)
    except Exception:
        return None

def generate_rationale_v1(title: str, score: int, top_snippet: str = "") -> str:
    system_prompt = (
        "You are a concise evaluator. Based on a video's TITLE and its computed relevance SCORE, "
        "provide a short verdict (Strong match / Partial match / Low relevance) and one sentence justification. "
        "Do NOT invent facts. Use only the TITLE and SCORE."
    )

    user_prompt = (
        f"TITLE: {title}\nSCORE: {score}\nTOP_SNIPPET: {top_snippet}\n"
                    "Provide: <VERDICT> — one-sentence justification referring only to the snippet and score."
    )

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=80
        )

        content = _extract_message_content(resp)
        if not content:
            return "Rationale generation error: could not parse model response."
        return content.strip()

    except Exception as e:
        # include short message but do not leak keys
        return f"Rationale generation error: {e}"