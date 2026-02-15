# gemini-webapi-mcp

MCP server for Google Gemini — image generation/editing, chat and file analysis via browser cookies. No API key required.

## Quick Start

### 1. Log into Gemini

Open Chrome, go to [gemini.google.com](https://gemini.google.com) and log in with your Google account. The server reads cookies from Chrome automatically.

### 2. Install the MCP server

**From GitHub:**

```bash
uv run --with "gemini-webapi-mcp @ git+https://github.com/AndyShaman/gemini-webapi-mcp.git" gemini-webapi-mcp
```

**From local clone:**

```bash
git clone https://github.com/AndyShaman/gemini-webapi-mcp.git
cd gemini-webapi-mcp
uv sync
uv run gemini-webapi-mcp
```

### 3. Add MCP config

Add to `~/.config/mcp/mcp_servers.json`:

**From GitHub (no clone needed):**

```json
{
  "mcpServers": {
    "gemini": {
      "command": "uv",
      "args": ["run", "--with", "gemini-webapi-mcp @ git+https://github.com/AndyShaman/gemini-webapi-mcp.git", "gemini-webapi-mcp"]
    }
  }
}
```

**From local clone:**

```json
{
  "mcpServers": {
    "gemini": {
      "command": "uv",
      "args": ["run", "--with", "/path/to/gemini-webapi-mcp", "gemini-webapi-mcp"]
    }
  }
}
```

Replace `/path/to/gemini-webapi-mcp` with the actual path to the cloned repository.

### 4. Install the skill (optional)

Copy `SKILL.md` to your Claude Code commands directory so Claude knows when and how to use the tools:

```bash
mkdir -p ~/.claude/commands
cp SKILL.md ~/.claude/commands/gemini-mcp.md
```

### 5. Verify

```bash
mcp-cli call gemini gemini_chat '{"prompt": "Hello!"}'
```

If you get a response from Gemini, everything works.

## Authentication

The server reads cookies from Chrome automatically via `browser-cookie3`.

If auto-detection fails, set environment variables manually:

1. Open Chrome DevTools on gemini.google.com → Application → Cookies
2. Copy values of `__Secure-1PSID` and `__Secure-1PSIDTS`
3. Add to your MCP config:

```json
{
  "mcpServers": {
    "gemini": {
      "command": "uv",
      "args": ["run", "--with", "gemini-webapi-mcp @ git+https://github.com/AndyShaman/gemini-webapi-mcp.git", "gemini-webapi-mcp"],
      "env": {
        "GEMINI_PSID": "your__Secure-1PSID_value",
        "GEMINI_PSIDTS": "your__Secure-1PSIDTS_value"
      }
    }
  }
}
```

## Available Tools

| Tool | Description |
|------|-------------|
| `gemini_generate_image` | Generate new images or edit existing ones |
| `gemini_upload_file` | Analyze files — video, images, PDF, documents |
| `gemini_chat` | Text chat (single or multi-turn) |
| `gemini_start_chat` | Start a multi-turn session |
| `gemini_reset` | Re-initialize client when auth fails |

## Models

| Model | Default for | Notes |
|-------|-------------|-------|
| `gemini-3.0-flash` | chat, file analysis | Fast |
| `gemini-3.0-pro` | image generation | Supports aspect ratios |
| `gemini-3.0-flash-thinking` | — | Shows reasoning process |

## Usage Examples

Generate an image:
```bash
mcp-cli call gemini gemini_generate_image '{"prompt": "a cat in watercolor style"}'
```

Edit an image:
```bash
mcp-cli call gemini gemini_generate_image '{"prompt": "make it gray", "files": ["/path/to/cat.png"]}'
```

Analyze a video:
```bash
mcp-cli call gemini gemini_upload_file '{"file_path": "/path/to/video.mp4", "prompt": "What happens here?"}'
```

## License

MIT
