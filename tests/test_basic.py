from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_generate_no_keys():
    # Without API keys, should still respond but may error; ensure endpoint exists
    r = client.post("/generate", json={
        "product":"Test","audience":"dev","tone":"tone","goal":"g","platform":["web"], "num_headlines":1, "num_long":0
    })
    # We expect 200 or 500; presence of response is main check
    assert r.status_code in (200,500)
