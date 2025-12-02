"""
LangChain-based OrchestratorAgent

This module uses LangChain to construct two chains:
 - copy_chain: generates headlines and long copies
 - prompt_chain: converts a given copy into an image prompt

The OrchestratorAgent.run method executes:
 1. copy_chain -> headlines + long copies
 2. prompt_chain (for each copy) -> image prompts
 3. uses image_client to generate images (async)
 4. scores copy-image pairs via scorer

Notes:
 - Requires environment OPENAI_API_KEY for LangChain's OpenAI LLM wrapper.
 - If LangChain is not installed or fails, falls back to the simple orchestrator implementation.
"""

import os, asyncio, logging
from typing import List, Dict, Any

logger = logging.getLogger("orchestrator.langchain")

try:
    # LangChain imports
    from langchain import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain, SimpleSequentialChain
    LANGCHAIN_AVAILABLE = True
except Exception as e:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available: %s", e)

class OrchestratorAgent:
    def __init__(self, llm, image_client, scorer, cache):
        """
        llm: existing LLMClient (fallback) but we'll also create LangChain LLM if available
        image_client: async image generator
        scorer: CoherenceScorer instance
        cache: PromptCache instance
        """
        self.llm = llm
        self.image_client = image_client
        self.scorer = scorer
        self.cache = cache

        if LANGCHAIN_AVAILABLE:
            # Create a LangChain OpenAI LLM wrapper using environment key
            openai_api_key = os.getenv("OPENAI_API_KEY")
            self.lc_llm = OpenAI(model_name=os.getenv("OPENAI_CHAT_MODEL","gpt-4o-mini"), openai_api_key=openai_api_key, temperature=0.8)
            # Define prompt templates
            self.headline_prompt = PromptTemplate(
                input_variables=["product","audience","tone","num_headlines"],
                template=("You are a creative ad copywriter.\n"
                          "Generate {num_headlines} energetic, short ad headlines (<=6 words) for product: {product} targeting {audience}.\n"
                          "Tone: {tone}.\n"
                          "Return as a JSON array of strings with no extra commentary.")
            )
            self.long_prompt = PromptTemplate(
                input_variables=["product","audience","tone","num_long"],
                template=("You are a creative ad copywriter.\n"
                          "Generate {num_long} different long-form ad variations (2-3 sentences) for {product} aimed at {audience}.\n"
                          "Tone: {tone}.\n"
                          "Return as a JSON array of strings.")
            )
            self.prompt_from_copy_template = PromptTemplate(
                input_variables=["product","audience","tone","copy","platform"],
                template=("Create a high-quality image generation prompt for an advertising creative.\n"
                          "Product: {product}\nAudience: {audience}\nTone: {tone}\nPlatform: {platform}\nCopy: {copy}\n\n"
                          "Describe the scene, visual style, colors, mood, camera angle, and any props. Keep under 2 sentences.")
            )
            # Chains
            self.headline_chain = LLMChain(llm=self.lc_llm, prompt=self.headline_prompt, verbose=False, output_key="headlines_raw")
            self.long_chain = LLMChain(llm=self.lc_llm, prompt=self.long_prompt, verbose=False, output_key="long_raw")
            self.prompt_chain = LLMChain(llm=self.lc_llm, prompt=self.prompt_from_copy_template, verbose=False, output_key="image_prompt")
        else:
            logger.info("LangChain not available; OrchestratorAgent will use fallback llm client.")

    async def run(self, brief: Dict[str,Any]) -> Dict[str,Any]:
        """
        Orchestrates the full pipeline and returns top assets.
        """
        if LANGCHAIN_AVAILABLE:
            # Run headline and long copy generation using LangChain in thread
            product = brief.get("product")
            audience = brief.get("audience")
            tone = brief.get("tone")
            num_headlines = int(brief.get("num_headlines",5))
            num_long = int(brief.get("num_long",3))

            # LangChain LLMChain runs synchronously; run in thread to avoid blocking event loop
            def run_headlines():
                out = self.headline_chain.run({"product":product,"audience":audience,"tone":tone,"num_headlines":num_headlines})
                return out
            def run_long():
                out = self.long_chain.run({"product":product,"audience":audience,"tone":tone,"num_long":num_long})
                return out

            raw_headlines = await asyncio.to_thread(run_headlines)
            raw_long = await asyncio.to_thread(run_long)

            # Attempt to eval JSON arrays; fallback to line splitting
            try:
                import json
                headlines = json.loads(raw_headlines)
                if not isinstance(headlines, list):
                    raise ValueError("headlines not list")
            except Exception:
                headlines = [line.strip("- ").strip() for line in raw_headlines.splitlines() if line.strip()]

            try:
                long_copies = json.loads(raw_long)
                if not isinstance(long_copies, list):
                    raise ValueError("long not list")
            except Exception:
                long_copies = [line.strip() for line in raw_long.splitlines() if line.strip()]

            copies = headlines + long_copies

            # Generate prompts for each copy using prompt_chain
            prompts = []
            for c in copies:
                def run_prompt():
                    return self.prompt_chain.run({"product":product,"audience":audience,"tone":tone,"copy":c,"platform":",".join(brief.get("platform",[]))})
                p = await asyncio.to_thread(run_prompt)
                prompts.append(p)

        else:
            # Fallback to existing llm client methods
            headlines, long_copies = self.llm.generate_copy_variations(brief)
            copies = headlines + long_copies
            prompts = self.llm.generate_image_prompts(brief, copies)

        # 3. Generate images (async)
        images = []
        async def gen_prompt(p):
            cached = self.cache.get(p)
            if cached:
                return cached
            img = await self.image_client.generate_image(p)
            self.cache.set(p, img)
            return img

        tasks = [gen_prompt(p) for p in prompts]
        images = await asyncio.gather(*tasks)

        # 4. Score copy-image pairs
        results = []
        for copy in copies:
            for img in images:
                s = self.scorer.score(copy, img["path"])
                results.append({"copy": copy, "image_url": img["url"], "local_path": img["path"], "score": s})
        top = sorted(results, key=lambda x: x["score"], reverse=True)[:6]
        return {"top_assets": top}
