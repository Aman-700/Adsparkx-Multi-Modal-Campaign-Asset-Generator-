import os, json, hashlib
CACHE_DIR = os.getenv("CACHE_DIR","./cache")
os.makedirs(CACHE_DIR, exist_ok=True)

class PromptCache:
    def __init__(self):
        self.dir = CACHE_DIR

    def _key(self, prompt: str):
        h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return os.path.join(self.dir, h + ".json")

    def get(self, prompt: str):
        path = self._key(prompt)
        if os.path.exists(path):
            with open(path,"r",encoding="utf-8") as f:
                return json.load(f)
        return None

    def set(self, prompt: str, data):
        path = self._key(prompt)
        with open(path,"w",encoding="utf-8") as f:
            json.dump(data,f)
        return path
