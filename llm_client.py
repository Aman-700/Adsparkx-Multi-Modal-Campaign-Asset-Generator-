# llm_client.py
import os
import json
from typing import List, Dict, Any

# Try to use legacy openai package for chat calls (most common local setups)
def _has_openai_key():
    return bool(os.getenv("OPENAI_API_KEY"))

class LLMClient:
    def __init__(self, api_key: str = None):
        # optional: set API key in environment or here
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

    def generate_copy_variations(self, brief: Dict[str, Any], num_headlines: int = 2, num_long: int = 1):
        """
        Returns tuple (headlines:list, long:list)
        Uses OpenAI chat completion (legacy openai) if OPENAI_API_KEY exists, otherwise deterministic fallback.
        """
        product = brief.get("product", "Product")
        audience = brief.get("audience", "everyone")
        tone = brief.get("tone", "neutral")
        goal = brief.get("goal", "awareness")

        # if we have an API key, attempt real generation
        if _has_openai_key():
            try:
                import openai
                # ensure API key is set
                if os.getenv("OPENAI_API_KEY"):
                    openai.api_key = os.getenv("OPENAI_API_KEY")

                system = (
                    "You are a creative marketing copywriter. "
                    "Given a short brief, produce the requested number of short headline variations "
                    "and long copy variations. Output only valid JSON with keys 'headlines' and 'long'."
                )
                user_msg = (
                    f"Brief:\nProduct: {product}\nAudience: {audience}\nTone: {tone}\nGoal: {goal}\n\n"
                    f"Instructions: Return a JSON object like {{\"headlines\": [..], \"long\": [..]}} with "
                    f"{num_headlines} short headline variations (keep them <=8 words) and {num_long} long copy variations (1-2 sentences)."
                )

                resp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=400,
                    n=1,
                    temperature=0.9,
                )

                text = resp["choices"][0]["message"]["content"]
                # Try to extract JSON from text (in case assistant wraps in markdown)
                try:
                    j = json.loads(text.strip())
                except Exception:
                    # attempt to find first JSON object in the response
                    import re
                    m = re.search(r"(\{[\s\S]*\})", text)
                    if m:
                        j = json.loads(m.group(1))
                    else:
                        raise

                headlines = j.get("headlines", [])[:num_headlines]
                long = j.get("long", [])[:num_long]
                # if not enough results, pad deterministically
                while len(headlines) < num_headlines:
                    headlines.append(f"{product} - Unleash Energy #{len(headlines)+1}")
                while len(long) < num_long:
                    long.append(f"{product} long copy version {len(long)+1}")
                return headlines, long

            except Exception as e:
                # print helpful debug and fall back
                print("LLM generation (openai) failed:", type(e).__name__, str(e)[:300])

        # Deterministic fallback (safe offline)
        headlines = [f"{product} - Unleash Energy #{i+1}" for i in range(num_headlines)]
        long = [f"{product} long copy version {i+1}" for i in range(num_long)]
        return headlines, long

    def generate_image_prompts(self, brief: Dict[str, Any], copies: List[str]):
        prompts = []
        for c in copies:
            p = f"Ad image for {brief.get('product')} targeting {brief.get('audience')} — style: vibrant, dynamic, in-frame product shot, bold colors, youthful energy. Include branding and clear product focus."
            prompts.append(p)
        return prompts
