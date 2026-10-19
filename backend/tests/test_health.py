def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["engine_mode"] == "mock"
    assert "workers" in data
