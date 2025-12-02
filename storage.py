import os, requests
class SanityStorage:
    def __init__(self, project_id: str, dataset: str, token: str):
        self.project_id = project_id
        self.dataset = dataset
        self.token = token
        self.base = f"https://{project_id}.api.sanity.io/v1/data/mutate/{dataset}"

    def create_asset(self, doc: dict):
        mutations = [{"create": {"_type":"creativeAsset", **doc}}]
        url = self.base
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type":"application/json"}
        resp = requests.post(url, json={"mutations": mutations}, headers=headers)
        return resp.json()
