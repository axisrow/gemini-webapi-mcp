# Gemini WebAPI MCP — заметки для разработки

## Обновление MCP-сервера

При любых изменениях в коде сервера — **обязательно** очищать кэш `uv` перед перезапуском Claude Desktop:

```bash
uv cache clean --force
rm -rf ~/.cache/uv/archive-v0/
```

`uv cache clean <pkg>` **недостаточно** — он не чистит `archive-v0` (полные виртуальные окружения). Без полной очистки Claude Desktop будет использовать старую кэшированную копию.

После очистки — перезапустить Claude Desktop.

## Ключевые файлы

- Исходник сервера: `src/gemini_webapi_mcp/server.py`
- Конфиг Claude Desktop: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Логи MCP: `~/Library/Logs/Claude/mcp-server-gemini.log`

## API-заметки

- `chat_id` принимает как raw ID (`c_abc123`), так и полный URL (`https://gemini.google.com/app/abc123`) — парсится через `_resolve_chat_id()` (~строка 245)
- Дефолтная модель для image gen: `gemini-3.0-flash-thinking`
- Дефолтная модель для остальных инструментов: `gemini-3.0-flash`

## Ротация Model ID (image generation)

Google **периодически ротирует model ID** для генерации изображений. Когда image gen перестаёт работать (Gemini отвечает "не могу создавать изображения" или ошибка 1052):

1. Открыть Gemini чат в Chrome с режимом "Создание изображений"
2. Перехватить POST к `StreamGenerate` через DevTools (Network tab)
3. Из заголовка `x-goog-ext-525001261-jspb` взять актуальный model ID (16-символьный hex в кавычках)
4. Обновить `_MODEL_ID_MAP` в `server.py` (~строка 269) — все значения на новый ID
5. Также проверить `_BROWSER_PARAMS` (~строка 277) — сравнить с body запроса из браузера (индексы `inner_req_list`)

Текущий актуальный model ID (февраль 2026): `56fdd199312815e2`

## Архитектура image generation

- `_image_mode` (глобальный флаг) — включается в `gemini_generate_image`, выключается после генерации
- `patched_request` (~строка 294) — при `_image_mode=True` инжектирует:
  - Заголовки: `x-goog-ext-73010989-jspb`, `x-goog-ext-73010990-jspb`, `x-goog-ext-525005358-jspb`
  - Remapping model ID через `_MODEL_ID_MAP`
  - `_BROWSER_PARAMS` в body (`inner_req_list`)
- `_do_generate` (~строка 420) — устанавливает `chat.model` через `kwargs.pop("model")`, fallback Pro→Flash
- `send_message` в gemini_webapi передаёт `model=self.model` в `generate_content` — поэтому model нельзя передавать через kwargs (будет duplicate)
