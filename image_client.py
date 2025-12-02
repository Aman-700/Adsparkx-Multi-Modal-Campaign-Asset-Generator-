# image_client.py
import os
import uuid
import base64
from pathlib import Path
from PIL import Image

OUT = os.getenv("OUT_DIR", "./outputs")
Path(OUT).mkdir(parents=True, exist_ok=True)

def _write_bytes_to_png(b: bytes):
    filename = f"{uuid.uuid4().hex[:12]}.png"
    path = os.path.join(OUT, filename)
    with open(path, "wb") as f:
        f.write(b)
    return path

class ImageClient:
    def __init__(self):
        # provider string: "openai" or "placeholder" (default)
        self.provider = os.getenv("IMAGE_PROVIDER", "placeholder").lower()
        # optional: you may have OPENAI_API_KEY in env
        self.openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

    async def generate_image(self, prompt: str):
        """
        Return dict {path, url, prompt}
        Tries OpenAI (new client), falls back to legacy openai.Image.create,
        otherwise returns a placeholder image.
        """
        if self.provider == "openai":
            # Try new OpenAI client first (OpenAI())
            try:
                # try new client
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_key) if self.openai_key else OpenAI()
                # new client: images.generate
                resp = client.images.generate(model="gpt-image-1", prompt=prompt, n=1, size="1024x1024")
                # resp.data[0].b64_json
                try:
                    b64 = resp.data[0].b64_json
                except Exception:
                    # some versions may expose differently
                    b64 = getattr(resp.data[0], "b64_json", None)
                if not b64:
                    raise RuntimeError("OpenAI response missing base64 image data (new client).")
                img_bytes = base64.b64decode(b64)
                path = _write_bytes_to_png(img_bytes)
                return {"path": path, "url": f"file://{os.path.abspath(path)}", "prompt": prompt}
            except Exception as e_new:
                # print helpful error and fall back to legacy client
                print("OpenAI (new client) image generation failed:", type(e_new).__name__, str(e_new)[:300])

            # Try legacy openai package
            try:
                import openai
                if self.openai_key:
                    openai.api_key = self.openai_key
                resp = openai.Image.create(prompt=prompt, n=1, size="1024x1024")
                # older SDKs put base64 in resp['data'][0]['b64_json'] or ['b64']
                data0 = resp['data'][0]
                b64 = data0.get('b64_json') or data0.get('b64') or data0.get('b64_url')
                if not b64:
                    # if provider returned a url instead of base64, attempt to use that url
                    # but for now treat as error
                    raise RuntimeError("OpenAI (legacy) returned no base64 image data.")
                img_bytes = base64.b64decode(b64)
                path = _write_bytes_to_png(img_bytes)
                return {"path": path, "url": f"file://{os.path.abspath(path)}", "prompt": prompt}
            except Exception as e_legacy:
                print("OpenAI (legacy) image generation failed:", type(e_legacy).__name__, str(e_legacy)[:300])
                # fall through to placeholder

        # Placeholder fallback (safe offline demo)
        try:
            fname = f"{uuid.uuid4().hex[:12]}.png"
            path = os.path.join(OUT, fname)
            img = Image.new("RGB", (1024, 1024), (255, 80, 80))
            img.save(path)
            return {"path": path, "url": f"file://{os.path.abspath(path)}", "prompt": prompt}
        except Exception as e:
            # Last-ditch fallback: return a path-like string but don't crash
            print("Placeholder image generation failed:", type(e).__name__, str(e)[:200])
            return {"path": "", "url": "", "prompt": prompt}
