from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_enqueue_generate(client):
    with patch("app.api.generate.create_job", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = ("job-123", "draft", None)
        resp = client.post(
            "/api/generate",
            json={
                "prompt": "Фон для слайда про AI",
                "preset_id": "slide-background",
                "quality": "draft",
                "num_images": 2,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["job_id"] == "job-123"
        mock_create.assert_awaited_once()


def test_generate_validation(client):
    resp = client.post("/api/generate", json={"prompt": ""})
    assert resp.status_code == 422
