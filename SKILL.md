---
name: gemini-mcp
description: Google Gemini via MCP for image generation, image editing, text chat, and file analysis. Use when the user asks to (1) generate or create images with Gemini, (2) edit, modify, or transform existing images with Gemini, (3) chat or ask questions to Gemini, (4) analyze or describe files — video, images, PDF, documents — with Gemini, (5) use Gemini models (flash, pro, flash-thinking). Triggers on keywords like "Gemini", "сгенерируй картинку", "нарисуй", "отредактируй изображение", "проанализируй видео/файл", "спроси у Gemini".
---

# Gemini MCP

Invoke tools via `mcp-cli call gemini <tool> '<json>'`.

## Tools

### `gemini_generate_image`
Generate new images or edit existing ones. Default model: `gemini-3.0-pro`.

```
mcp-cli call gemini gemini_generate_image '{"prompt": "a cat in watercolor style"}'
mcp-cli call gemini gemini_generate_image '{"prompt": "make the cat gray", "files": ["/path/to/cat.png"]}'
```

- Without `files` — generate from scratch
- With `files` — edit/transform the provided image(s)
- Pro supports aspect ratios (specify in prompt, e.g. "wide 16:9"), Flash always 1024x1024
- Images saved to `~/Pictures/gemini/` as full-resolution PNGs

### `gemini_upload_file`
Upload a file (video, image, PDF, document) and ask about it.

```
mcp-cli call gemini gemini_upload_file '{"file_path": "/path/to/video.mp4", "prompt": "О чем это видео?"}'
```

### `gemini_chat`
Text chat. Optional `session_id` for multi-turn, optional `model`.

```
mcp-cli call gemini gemini_chat '{"prompt": "Explain X"}'
mcp-cli call gemini gemini_chat '{"prompt": "Continue", "session_id": "abc123"}'
```

### `gemini_start_chat`
Start a multi-turn session. Returns `session_id` for `gemini_chat`.

```
mcp-cli call gemini gemini_start_chat '{}'
```

### `gemini_reset`
Re-initialize client when auth fails.

## Models

| Model | Default for | Notes |
|-------|-------------|-------|
| `gemini-3.0-flash` | chat, file analysis | Fast |
| `gemini-3.0-pro` | image generation | Supports aspect ratios |
| `gemini-3.0-flash-thinking` | — | Shows reasoning process |
