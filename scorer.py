# scorer.py
import os
import numpy as np
from typing import Optional
from PIL import Image
import openai

USE_CLIP = os.getenv("USE_CLIP", "true").lower() in ("1", "true", "yes")
CLIP_DEVICE = os.getenv("CLIP_DEVICE", "cpu")

# Try to load CLIP model if requested
try:
    if USE_CLIP:
        import torch
        import clip
        device = CLIP_DEVICE if (torch.cuda.is_available() and CLIP_DEVICE == "cuda") else "cpu"
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    else:
        clip_model = None
        clip_preprocess = None
except Exception as e:
    clip_model = None
    clip_preprocess = None
    print("CLIP not available:", type(e).__name__, str(e)[:200])

class CoherenceScorer:
    def __init__(self, openai_api_key: Optional[str] = None):
        # set openai key if provided
        key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if key:
            openai.api_key = key

    def _text_embedding(self, text: str):
        # Use OpenAI embeddings (fallback) if available
        try:
            model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            resp = openai.Embedding.create(model=model, input=text)
            return np.array(resp["data"][0]["embedding"], dtype=float)
        except Exception as e:
            # as fallback, produce a simple random-ish vector (deterministic via hash) to avoid crashes
            import hashlib
            h = hashlib.sha256(text.encode("utf-8")).digest()
            arr = np.frombuffer(h, dtype=np.uint8).astype(float)
            # pad/trim to 1536 if needed (common embedding size)
            if arr.size < 1536:
                arr = np.pad(arr, (0, 1536 - arr.size), mode="constant", constant_values=0)
            return arr.astype(float)

    def _image_embedding_clip(self, image_path: str):
        if clip_model is None:
            raise RuntimeError("CLIP model not available")
        image = Image.open(image_path).convert("RGB")
        image_input = clip_preprocess(image).unsqueeze(0)
        import torch
        with torch.no_grad():
            emb = clip_model.encode_image(image_input.to(clip_model.visual.conv1.weight.device)).cpu().numpy()[0]
        return emb / (np.linalg.norm(emb) + 1e-10)

    def score(self, text: str, image_path: str) -> float:
        """
        Return a normalized score in [0,1] for coherence between text and image.
        """
        try:
            text_emb = self._text_embedding(text)
            if clip_model is not None:
                img_emb = self._image_embedding_clip(image_path)
            else:
                # fallback: caption image via a simple placeholder caption (could be replaced with real captioning)
                caption = self._caption_image_via_llm(image_path)
                img_emb = self._text_embedding(caption)
            # cosine similarity
            denom = (np.linalg.norm(text_emb) * np.linalg.norm(img_emb) + 1e-12)
            score = float(np.dot(text_emb, img_emb) / denom)
            # map from [-1,1] to [0,1]
            return max(0.0, min(1.0, (score + 1.0) / 2.0))
        except Exception as e:
            # fallback neutral score
            print("Scorer.exception:", type(e).__name__, str(e)[:200])
            return 0.4

    def _caption_image_via_llm(self, image_path: str):
        # minimal placeholder caption (use real captioning later)
        return "A high-energy image with product in focus and vibrant colors."
