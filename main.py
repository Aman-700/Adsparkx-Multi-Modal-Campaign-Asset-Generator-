import os
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from llm_client import LLMClient
from image_client import ImageClient
from scorer import CoherenceScorer
from storage import SanityStorage
from cache import PromptCache
from report import ReportGenerator
from langchain_agent import OrchestratorAgent

# logging / observability
logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"))
logger = logging.getLogger("campaign-generator")

app = FastAPI(title="Multi-Modal Campaign Asset Generator")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set; LLM calls will fail in runtime if not provided")

llm = LLMClient(api_key=OPENAI_API_KEY)
image_client = ImageClient()
scorer = CoherenceScorer(openai_api_key=OPENAI_API_KEY)
cache = PromptCache()
reporter = ReportGenerator()
agent = OrchestratorAgent(llm=llm, image_client=image_client, scorer=scorer, cache=cache)

sanity = None
if os.getenv("SANITY_PROJECT_ID") and os.getenv("SANITY_TOKEN"):
    sanity = SanityStorage(project_id=os.getenv("SANITY_PROJECT_ID"),
                           dataset=os.getenv("SANITY_DATASET","production"),
                           token=os.getenv("SANITY_TOKEN"))

class Brief(BaseModel):
    product: str
    audience: str
    tone: str
    goal: str
    platform: List[str]
    extra: Dict[str, Any] = {}
    num_headlines: int = 5
    num_long: int = 3

@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/metrics")
def metrics():
    # Very simple metrics example
    return {"requests_handled": 1, "version": "1.0"}

@app.post("/generate")
async def generate(brief: Brief):
    try:
        logger.info("Received brief for product=%s", brief.product)
        result = await agent.run(brief.dict())
        top = result["top_assets"]

        # optional store
        if sanity:
            for item in top:
                sanity.create_asset({
                    "product": brief.product,
                    "copy": item["copy"],
                    "image_url": item["image_url"],
                    "score": item["score"],
                    "platform": ",".join(brief.platform)
                })

        # produce report CSV/PDF
        report_path = reporter.generate_report(brief.dict(), top)
        logger.info("Report generated at %s", report_path)

        return {"top_assets": top, "report": report_path}
    except Exception as e:
        logger.exception("Error in generate endpoint")
        raise HTTPException(status_code=500, detail=str(e))
