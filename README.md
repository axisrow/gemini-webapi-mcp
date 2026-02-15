# gemini-webapi-mcp

MCP server for Google Gemini — free image generation/editing, chat and file analysis via browser cookies, no API key required.

Uses the [gemini-webapi](https://github.com/HanaokaYuzu/Gemini-API) library to access Google Gemini through the same interface as the web app.

## Features

- **Image generation** from text descriptions (Pro model with aspect ratios)
- **Image editing** — send an existing image + prompt to modify it
- **Text chat** with Gemini (Flash, Pro, Flash-Thinking models)
- **File analysis** — upload images, video, PDF and ask questions
- **Auto-authentication** via Chrome browser cookies
- **Browser-compatible requests** for full image generation capabilities

## Installation

```bash
# Using uv (recommended)
uv run --with gemini-webapi-mcp gemini-webapi-mcp

# Using pip
pip install gemini-webapi-mcp
```

## MCP Client Configuration

### Claude Code

Add to `~/.config/mcp/mcp_servers.json`:

```json
{
  "mcpServers": {
    "gemini": {
      "command": "uv",
      "args": ["run", "--with", "gemini-webapi-mcp", "gemini-webapi-mcp"]
    }
  }
}
```

### Claude Desktop

Add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "gemini": {
      "command": "uvx",
      "args": ["gemini-webapi-mcp"]
    }
  }
}
```

## Authentication

The server authenticates using Google cookies from your browser. Two options:

### Option 1: Automatic (recommended)

1. Log into [gemini.google.com](https://gemini.google.com) in Chrome
2. Install `browser-cookie3` (included as dependency)
3. The server reads cookies automatically

### Option 2: Environment variables

Set `GEMINI_PSID` (and optionally `GEMINI_PSIDTS`):

```json
{
  "mcpServers": {
    "gemini": {
      "command": "uv",
      "args": ["run", "--with", "gemini-webapi-mcp", "gemini-webapi-mcp"],
      "env": {
        "GEMINI_PSID": "your_cookie_value",
        "GEMINI_PSIDTS": "your_cookie_value"
      }
    }
  }
}
```

To get cookie values: open Chrome DevTools on gemini.google.com → Application → Cookies → copy `__Secure-1PSID` and `__Secure-1PSIDTS`.

## Available Tools

| Tool | Description |
|------|-------------|
| `gemini_chat` | Send a text prompt, get a text response |
| `gemini_start_chat` | Start a multi-turn chat session |
| `gemini_generate_image` | Generate or edit images |
| `gemini_upload_file` | Upload a file (video, image, PDF) and ask questions |
| `gemini_reset` | Re-initialize client (refresh cookies) |

## Models

| Model | Best for |
|-------|----------|
| `gemini-3.0-flash` | Fast text responses (default for chat) |
| `gemini-3.0-pro` | Image generation/editing with aspect ratios (default for images) |
| `gemini-3.0-flash-thinking` | Complex reasoning tasks |

## Image Generation & Editing

- **Generate**: describe what you want — style, composition, colors, mood.
- **Edit**: pass `files` with path(s) to source image(s) and describe changes in `prompt`.
- **Pro model** supports non-square aspect ratios (e.g. 16:9, 9:16).
- **Flash model** always generates 1024x1024 images.
- Generated images are saved to `~/Pictures/gemini/` as full-resolution PNGs.

## Development

```bash
git clone https://github.com/mac-andy/gemini-webapi-mcp.git
cd gemini-webapi-mcp
uv sync
uv run gemini-webapi-mcp
```

## License

MIT
