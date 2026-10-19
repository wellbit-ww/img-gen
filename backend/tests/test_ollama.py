import httpx
import pytest
import respx

from app.config import get_settings
from app.core.schemas import EnhancePromptRequest, TextMode
from app.services.ollama import OllamaClient


@pytest.mark.asyncio
@respx.mock
async def test_ollama_enhance_fallback():
    get_settings.cache_clear()
    settings = get_settings()
    route = respx.post(f"{settings.ollama_url}/api/chat").mock(
        return_value=httpx.Response(500)
    )
    client = OllamaClient(settings)
    result = await client.enhance(
        EnhancePromptRequest(prompt='Заголовок "Демо"', text_mode=TextMode.NATIVE)
    )
    assert result.prompt == 'Заголовок "Демо"'
    assert route.called


@pytest.mark.asyncio
@respx.mock
async def test_ollama_enhance_preserves_quoted_text():
    get_settings.cache_clear()
    settings = get_settings()
    quoted = 'Презентация с текстом "Демо 2026"'
    body = {
        "message": {
            "content": (
                '{"prompt": "Corporate slide with text \\"Демо 2026\\" in quotes", '
                '"negative_prompt": "watermark"}'
            )
        }
    }
    respx.post(f"{settings.ollama_url}/api/chat").mock(
        return_value=httpx.Response(200, json=body)
    )
    client = OllamaClient(settings)
    result = await client.enhance(EnhancePromptRequest(prompt=quoted, text_mode=TextMode.NATIVE))
    assert "Демо 2026" in result.prompt
    assert result.negative_prompt == "watermark"
