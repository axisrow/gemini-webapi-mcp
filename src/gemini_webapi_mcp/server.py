"""
MCP Server for Google Gemini via browser cookies.

Uses gemini_webapi library to access Gemini Web App for free,
without requiring paid API keys. Authentication is done through
browser cookies (__Secure-1PSID and __Secure-1PSIDTS).
"""

import asyncio
import json
import logging
import os
import random
import re
import string
import sys
import urllib.request
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP, Context

# ---------------------------------------------------------------------------
# Logging (stderr only — stdout reserved for MCP stdio transport)
# ---------------------------------------------------------------------------
logger = logging.getLogger("gemini_mcp")
logger.addHandler(logging.StreamHandler(sys.stderr))
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGES_DIR = Path.home() / "Pictures" / "gemini"
DEFAULT_MODEL = "gemini-3.0-flash"

# ---------------------------------------------------------------------------
# LaMa watermark removal (optional — requires onnxruntime)
# ---------------------------------------------------------------------------
_LAMA_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
_LAMA_CACHE = Path.home() / ".cache" / "gemini-mcp" / "lama_fp32.onnx"
_lama_session = None

# Watermark is a 4-point sparkle (superellipse shape), fixed position:
# center at (w-57, h-57). Measured across 1024x1024, 2048x1117,
# 1536x2752, 2816x1536 images — position is stable.
_WM_OFFSET = 57       # center offset from bottom-right corner


def _get_lama_session():
    """Download LaMa model on first use, return ONNX InferenceSession or None."""
    global _lama_session
    if _lama_session is not None:
        return _lama_session
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    if not _LAMA_CACHE.exists():
        _LAMA_CACHE.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading LaMa model (208 MB)...")
        urllib.request.urlretrieve(_LAMA_URL, _LAMA_CACHE)
        logger.info("LaMa model downloaded to %s", _LAMA_CACHE)
    _lama_session = ort.InferenceSession(
        str(_LAMA_CACHE), providers=["CPUExecutionProvider"]
    )
    return _lama_session


def _make_sparkle_mask(size: int, center: tuple[int, int], scale: int = 1) -> "np.ndarray":
    """Create a 4-point star mask matching Gemini's sparkle watermark.

    The sparkle follows a superellipse (Lamé curve): |x/R|^p + |y/R|^p <= 1
    with p≈0.85, R=24px at native resolution. Measured from actual Gemini
    output across multiple aspect ratios (1024x1024, 2048x1117, 1536x2752).

    Returns a binary numpy uint8 array (0 or 255) for LaMa inpainting.

    Args:
        scale: 1 for native resolution, 2 for 2x upscaled images.
    """
    import numpy as np
    from PIL import Image, ImageFilter

    cx, cy = center
    R = 24 * scale   # sparkle radius: 24px at 1x, 48px at 2x
    p = 0.85          # superellipse exponent (4-pointed star shape)

    y_grid, x_grid = np.ogrid[:size, :size]
    dx = np.abs(x_grid - cx).astype(float)
    dy = np.abs(y_grid - cy).astype(float)
    val = (dx / R) ** p + (dy / R) ** p
    mask_arr = np.where(val <= 1.0, 255, 0).astype(np.uint8)

    mask_img = Image.fromarray(mask_arr)
    mf = 7 * scale + (0 if (7 * scale) % 2 == 1 else 1)  # 7 at 1x, 15 at 2x
    dilated = mask_img.filter(ImageFilter.MaxFilter(mf))
    return np.array(dilated)


def _remove_watermark(image_path: str, scale: int = 1) -> bool:
    """Remove Gemini sparkle watermark from bottom-right corner using LaMa.

    The watermark is always at a fixed position: center at (w-57, h-57) for
    native (1x) images. For 2x upscaled images, position and size double.
    Returns True if removed, False if onnxruntime is not available.

    Args:
        scale: 1 for native resolution, 2 for 2x upscaled images.
    """
    session = _get_lama_session()
    if session is None:
        return False

    import numpy as np
    from PIL import Image

    img = Image.open(image_path)
    w, h = img.size
    if w < 512 or h < 512:
        return False

    offset = _WM_OFFSET * scale  # 57 at 1x, 114 at 2x

    # --- Crop 512x512 from bottom-right for LaMa ---
    crop_x0 = w - 512
    crop_y0 = h - 512
    crop = img.crop((crop_x0, crop_y0, w, h))

    # Watermark center within the 512x512 crop
    local_cx = 512 - offset
    local_cy = 512 - offset

    # --- Build mask ---
    mask_arr = _make_sparkle_mask(512, (local_cx, local_cy), scale=scale)

    # --- LaMa inference ---
    crop_rgb = np.array(crop).astype(np.float32) / 255.0       # (512,512,3)
    img_input = crop_rgb.transpose(2, 0, 1)[None]               # (1,3,512,512)
    mask_input = (mask_arr.astype(np.float32) / 255.0)[None, None]  # (1,1,512,512)

    output = session.run(None, {"image": img_input, "mask": mask_input})[0]
    # LaMa output is already in 0-255 range (not 0-1)
    result = output[0].transpose(1, 2, 0).clip(0, 255).astype(np.uint8)

    # --- Paste back only masked pixels ---
    result_img = Image.fromarray(result)
    mask_pil = Image.fromarray(mask_arr)
    crop.paste(result_img, mask=mask_pil)
    img.paste(crop, (crop_x0, crop_y0))

    img.save(image_path)
    return True


# ---------------------------------------------------------------------------
# Cookie resolution: env vars → browser-cookie3 → error
# ---------------------------------------------------------------------------

def _resolve_cookies() -> tuple[str, str]:
    """Resolve Gemini auth cookies with clear priority chain.

    1. Environment variables GEMINI_PSID / GEMINI_PSIDTS (explicit override).
    2. Chrome browser cookies via browser-cookie3 (automatic).
    3. RuntimeError with actionable instructions.

    Returns (psid, psidts). psidts may be empty.
    Cookie values are never logged.
    """
    # --- Priority 1: explicit env vars ---
    psid = os.environ.get("GEMINI_PSID", "")
    psidts = os.environ.get("GEMINI_PSIDTS", "")
    if psid:
        logger.info("Using Gemini cookies from environment variables")
        return psid, psidts

    # --- Priority 2: Chrome browser cookies ---
    try:
        import browser_cookie3

        cj = browser_cookie3.chrome(domain_name=".google.com")
        for cookie in cj:
            if cookie.name == "__Secure-1PSID" and cookie.value:
                psid = cookie.value
            elif cookie.name == "__Secure-1PSIDTS" and cookie.value:
                psidts = cookie.value
        if psid:
            logger.info("Using Gemini cookies from Chrome browser")
            return psid, psidts
        logger.warning("browser-cookie3: no __Secure-1PSID cookie found in Chrome")
    except ImportError:
        logger.warning("browser-cookie3 not installed — cannot read cookies from Chrome")
    except Exception as exc:
        logger.warning("browser-cookie3: failed to read Chrome cookies — %s", type(exc).__name__)

    # --- Nothing worked ---
    raise RuntimeError(
        "Gemini cookies not found. Options:\n"
        "  1. Log into gemini.google.com in Chrome and install browser-cookie3, or\n"
        "  2. Set GEMINI_PSID (and optionally GEMINI_PSIDTS) environment variables."
    )


def _make_gen_id() -> str:
    """Generate a client-side gen_id for c8o8Fe RPC (16-char repeating pattern)."""
    base = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return (base * 3)[:16]


# ---------------------------------------------------------------------------
# Lifespan: initialise GeminiClient once, reuse across all tool calls
# ---------------------------------------------------------------------------

@asynccontextmanager
async def app_lifespan(server):
    from gemini_webapi import GeminiClient

    psid, psidts = _resolve_cookies()

    client = GeminiClient(secure_1psid=psid, secure_1psidts=psidts or None)
    await client.init(timeout=600, watchdog_timeout=120, auto_close=False, auto_refresh=True)
    _patch_client(client)

    yield {"gemini_client": client, "chat_sessions": {}}

    await client.close()


mcp = FastMCP("gemini-webapi-mcp", lifespan=app_lifespan)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_client(ctx: Context):
    return ctx.request_context.lifespan_context["gemini_client"]


def _get_sessions(ctx: Context) -> dict:
    return ctx.request_context.lifespan_context["chat_sessions"]


def _resolve_chat_id(raw: str) -> str:
    """Extract cid from a Gemini URL or return raw chat_id as-is."""
    m = re.match(r"https?://gemini\.google\.com/app/(?:c_)?([a-f0-9]+)", raw)
    if m:
        return f"c_{m.group(1)}"
    return raw


_image_mode = False
_image_lock = asyncio.Lock()

# Populated by _patched_parse hook during StreamGenerate response parsing.
_image_tokens: dict[str, str] = {}   # preview_url -> download_token
_last_metadata: list = []             # [cid, rid, rcid, ...] from last response


def _patch_client(gemini_client):
    """Patch GeminiClient for image generation and 2x download support.

    1. Override model ID in header (Google rotates IDs periodically).
    2. Add browser-compatible body params and extra headers during image generation.
    3. Intercept response parsing to capture image download tokens for c8o8Fe RPC.
    """
    # Google rotates model IDs periodically. Update these when generation fails (error 1052).
    _MODEL_ID_MAP = {
        "5bf011840784117a": "56fdd199312815e2",  # flash-thinking (Nano Banana 2)
        "9d8ca3786ebdfbea": "56fdd199312815e2",  # pro -> same current ID
        "fbb127bbb056c959": "56fdd199312815e2",  # flash -> same current ID
        "e051ce1aa80aa576": "56fdd199312815e2",  # previous rotation -> current
        "e6fa609c3fa255c0": "56fdd199312815e2",  # feb 2026 rotation -> current
    }

    # Browser-compatible body params (indices in inner_req_list).
    # Without these, certain operations (image editing with files) may fail.
    _BROWSER_PARAMS = {
        1: [os.environ.get("GEMINI_LANGUAGE", "ru")],
        6: [1],
        10: 1,
        11: 0,
        17: [[2]],
        18: 0,
        27: 1,
        30: [4],
        41: [1],
        53: 0,
        68: 2,
    }

    http = gemini_client.client  # curl_cffi.AsyncSession
    _orig_request = http.request

    async def patched_request(method, url, **kwargs):
        global _image_mode
        is_generate = method == "POST" and "StreamGenerate" in str(url)
        if is_generate:
            logger.info("patched_request: _image_mode=%s, method=%s, url_contains_StreamGenerate=True",
                        _image_mode, method)
            logger.info("patched_request: headers_keys=%s",
                        list((kwargs.get("headers") or {}).keys()))
        if is_generate and _image_mode:
            headers = kwargs.get("headers") or {}

            # Remap model ID in the model header
            model_hdr = headers.get("x-goog-ext-525001261-jspb", "")
            if model_hdr:
                for old_id, new_id in _MODEL_ID_MAP.items():
                    if old_id in model_hdr:
                        model_hdr = model_hdr.replace(old_id, new_id)
                        break
                # Update trailing version flag: ,1] -> ,2]
                if model_hdr.endswith(",1]"):
                    model_hdr = model_hdr[:-2] + "2]"
                headers["x-goog-ext-525001261-jspb"] = model_hdr

            headers["x-goog-ext-73010989-jspb"] = "[0]"
            headers["x-goog-ext-73010990-jspb"] = "[0]"
            headers["x-goog-ext-525005358-jspb"] = json.dumps(
                [str(uuid.uuid4()), 1]
            )

            kwargs["headers"] = headers
            logger.info("patched_request: headers after patch: %s", list(headers.keys()))

            # Inject browser-compatible body params into f.req
            data = kwargs.get("data")
            if isinstance(data, dict) and "f.req" in data:
                try:
                    outer = json.loads(data["f.req"])
                    inner = json.loads(outer[1])

                    # Log chat metadata (cid) for debugging
                    metadata = inner[2] if len(inner) > 2 else None
                    cid = metadata[0] if isinstance(metadata, list) and metadata else None
                    logger.info("patched_request: cid=%s, model_hdr=%s",
                                cid, headers.get("x-goog-ext-525001261-jspb", "")[:40])

                    for idx, val in _BROWSER_PARAMS.items():
                        if inner[idx] is None:
                            inner[idx] = val

                    # Log which browser params were injected
                    injected = [idx for idx, val in _BROWSER_PARAMS.items() if inner[idx] == val]
                    logger.info("patched_request: injected browser params at indices %s", injected)

                    # Fix file_data format:
                    # Library:  [[[url], "name"]]
                    # Browser:  [[[url, 1, null, "mime"], "name", null*6, [0]]]
                    file_data = inner[0][3] if isinstance(inner[0], list) and len(inner[0]) > 3 else None
                    if file_data and isinstance(file_data, list):
                        _MIME_MAP = {
                            ".png": "image/png", ".jpg": "image/jpeg",
                            ".jpeg": "image/jpeg", ".webp": "image/webp",
                            ".gif": "image/gif", ".bmp": "image/bmp",
                        }
                        for fd in file_data:
                            if isinstance(fd, list) and len(fd) == 2:
                                url_arr, filename = fd[0], fd[1]
                                if isinstance(url_arr, list) and len(url_arr) == 1:
                                    ext = Path(filename).suffix.lower() if isinstance(filename, str) else ""
                                    mime = _MIME_MAP.get(ext, "image/png")
                                    fd[0] = [url_arr[0], 1, None, mime]
                                    fd.extend([None, None, None, None, None, None, [0]])

                    outer[1] = json.dumps(inner)
                    data["f.req"] = json.dumps(outer)
                    kwargs["data"] = data
                except Exception as exc:
                    logger.warning("patched_request: body injection failed: %s", exc)
            else:
                logger.warning("patched_request: no f.req in data, data type=%s", type(data))

        return await _orig_request(method, url, **kwargs)

    http.request = patched_request

    # --- Wrap parse_response_by_frame to capture image download tokens ---
    import gemini_webapi.client as _gwc
    from gemini_webapi.utils import (
        get_nested_value,
        parse_response_by_frame as _orig_parse,
    )
    import orjson as _json

    def _patched_parse(buffer):
        parts, remaining = _orig_parse(buffer)
        for part in parts:
            inner_json_str = get_nested_value(part, [2])
            if not inner_json_str:
                continue
            try:
                part_json = _json.loads(inner_json_str)
                # Capture conversation metadata (cid/rid)
                m_data = get_nested_value(part_json, [1])
                if isinstance(m_data, list) and len(m_data) >= 2 and m_data[0]:
                    if len(_last_metadata) >= 3:
                        # Update cid/rid but preserve captured rcid
                        _last_metadata[0] = m_data[0]
                        _last_metadata[1] = m_data[1]
                    else:
                        _last_metadata.clear()
                        _last_metadata.extend(m_data)
                # Capture image download tokens + rcid from candidates
                candidates = get_nested_value(part_json, [4], [])
                for cand in candidates:
                    rcid = get_nested_value(cand, [0])
                    if rcid and isinstance(rcid, str) and rcid.startswith("rc_"):
                        if len(_last_metadata) >= 2:
                            # Ensure rcid is at index 2
                            if len(_last_metadata) == 2:
                                _last_metadata.append(rcid)
                            else:
                                _last_metadata[2] = rcid
                    for gid in get_nested_value(cand, [12, 7, 0], []):
                        url = get_nested_value(gid, [0, 3, 3])
                        token = get_nested_value(gid, [0, 3, 5])
                        if url and token:
                            _image_tokens[url] = token
            except Exception:
                pass
        return parts, remaining

    _gwc.parse_response_by_frame = _patched_parse
    logger.info("Patched GeminiClient with browser-compatible parameters")

    # --- Increase retry tolerance for long image generation ---
    import gemini_webapi.utils.decorators as _dmod
    from gemini_webapi.utils.decorators import running

    _dmod.DELAY_FACTOR = 2  # Was 5 — reconnect faster after stream interrupts

    _orig_gen = type(gemini_client)._generate
    while hasattr(_orig_gen, '__wrapped__'):
        _orig_gen = _orig_gen.__wrapped__
    type(gemini_client)._generate = running(retry=12)(_orig_gen)
    logger.info("Patched _generate retry: 12 retries, DELAY_FACTOR=2")


async def _do_generate(client, chat, prompt, **kwargs):
    """Run image generation, with Pro→Flash fallback."""
    from gemini_webapi import ChatSession

    # Extract model from kwargs — send_message passes model=self.model
    # to generate_content, so we must set it on ChatSession, not in kwargs.
    model = kwargs.pop("model", "gemini-3.0-flash-thinking")

    if not chat:
        chat = ChatSession(geminiclient=client, model=model)
    else:
        chat.model = model
    try:
        return await chat.send_message(prompt, **kwargs)
    except Exception as err:
        logger.warning("Pro model failed (%s), falling back to flash", err)
        flash_chat = ChatSession(geminiclient=client, model="gemini-3.0-flash")
        return await flash_chat.send_message(prompt, **kwargs)


def _handle_error(e: Exception) -> str:
    from gemini_webapi import AuthError, APIError, RequestTimeoutError

    if isinstance(e, AuthError):
        return (
            "Error: Authentication failed. Cookies may have expired. "
            "Re-login to gemini.google.com in Chrome, then call gemini_reset."
        )
    if isinstance(e, RequestTimeoutError):
        return "Error: Request timed out. Try again or use a lighter model."
    if isinstance(e, APIError):
        return f"Error: Gemini API error — {e}"
    return f"Error: {type(e).__name__} — {e}"


async def _fetch_download_url(client, token: str, prompt: str, metadata: list, image_index: int = 0) -> str | None:
    """Call c8o8Fe RPC to get a high-resolution (2x) download URL for a generated image.

    Google stores a 2x upscaled version accessible only through this RPC endpoint.
    Returns the download URL or None on failure.
    """
    import orjson as _json
    from gemini_webapi.constants import Endpoint
    from gemini_webapi.utils import parse_response_by_frame, get_nested_value

    cid = metadata[0] if metadata else None
    rid = metadata[1] if len(metadata) > 1 else None
    rcid = metadata[2] if len(metadata) > 2 else None
    if not (rid and rcid and cid):
        logger.warning("c8o8Fe skipped: missing metadata (cid/rid/rcid)")
        return None

    gen_id = _make_gen_id()
    inner_payload = _json.dumps([
        [
            [None, None, None, [None, None, None, None, None, token]],
            [f"http://googleusercontent.com/image_generation_content/{image_index}", image_index],
            None,
            [19, prompt],
            None, None, None, None, None,
            gen_id,
        ],
        [rid, rcid, cid, None, gen_id],
        1, 0, 1,
    ]).decode("utf-8")

    outer_payload = _json.dumps(
        [[["c8o8Fe", inner_payload, None, "generic"]]]
    ).decode("utf-8")

    params: dict = {
        "rpcids": "c8o8Fe",
        "_reqid": client._reqid,
        "rt": "c",
        "source-path": "/app",
    }
    client._reqid += 100000
    if client.build_label:
        params["bl"] = client.build_label
    if client.session_id:
        params["f.sid"] = client.session_id

    try:
        resp = await client.client.post(
            Endpoint.BATCH_EXEC,
            params=params,
            data={"at": client.access_token, "f.req": outer_payload},
            headers={
                "x-goog-ext-525001261-jspb": "[1,null,null,null,null,null,null,0,[4,4]]",
                "x-goog-ext-73010989-jspb": "[0]",
            },
            timeout=60,
        )
        if resp.status_code != 200:
            logger.warning("c8o8Fe returned status %d", resp.status_code)
            return None

        text = resp.text
        if text.startswith(")]}'"):
            text = text[4:].lstrip()
        parts, _ = parse_response_by_frame(text)
        # parse_response_by_frame returns each sublist as a separate part:
        # part = ['wrb.fr', 'c8o8Fe', '["url"]', None, None, None, 'generic']
        for part in parts:
            if isinstance(part, list) and len(part) > 2 and part[1] == "c8o8Fe":
                inner_str = part[2]
                if inner_str and isinstance(inner_str, str):
                    inner = _json.loads(inner_str)
                    if isinstance(inner, list) and inner:
                        return inner[0]
    except Exception as exc:
        logger.warning("c8o8Fe failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool(
    name="gemini_start_chat",
    annotations={
        "title": "Start Gemini Chat Session",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def gemini_start_chat(
    ctx: Context,
    model: Optional[str] = None,
) -> str:
    """Start a new multi-turn chat session with Gemini.

    The session maintains conversation history so follow-up messages
    have full context. Pass the returned session_id to gemini_chat.

    Args:
        model: Model name for this session. Defaults to gemini-3.0-flash.

    Returns:
        JSON with session_id to use in subsequent gemini_chat calls.
    """
    try:
        client = _get_client(ctx)
        chat = client.start_chat(model=model or DEFAULT_MODEL)
        session_id = str(uuid.uuid4())[:8]
        _get_sessions(ctx)[session_id] = chat
        return json.dumps({
            "session_id": session_id,
            "model": model or DEFAULT_MODEL,
            "message": f"Chat session started. Use session_id '{session_id}' in gemini_chat.",
        })
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="gemini_chat",
    annotations={
        "title": "Gemini Chat",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def gemini_chat(
    prompt: str,
    ctx: Context,
    model: Optional[str] = None,
    session_id: Optional[str] = None,
    chat_id: Optional[str] = None,
) -> str:
    """Send a prompt to Gemini. Can continue an existing Gemini chat by URL or chat_id.

    To continue an existing conversation, pass its URL
    (e.g. 'https://gemini.google.com/app/c_abc123') or raw chat ID as chat_id.
    Do NOT use gemini_analyze_url for gemini.google.com links — they are chat IDs.

    Args:
        prompt: The text prompt to send to Gemini.
        model: Model name (e.g. 'gemini-3.0-flash', 'gemini-3.0-pro',
               'gemini-3.0-flash-thinking'). Defaults to gemini-3.0-flash.
        session_id: Optional session ID from gemini_start_chat for
                    multi-turn conversation with context.
        chat_id: Optional Gemini chat ID or URL to continue an existing
                 conversation. Accepts both raw ID ('c_abc123') and full URL
                 ('https://gemini.google.com/app/c_abc123').

    Returns:
        Gemini's text response. When using flash-thinking model,
        also includes the model's reasoning process.
    """
    try:
        client = _get_client(ctx)

        if session_id:
            sessions = _get_sessions(ctx)
            chat = sessions.get(session_id)
            if not chat:
                return f"Error: Session '{session_id}' not found. Start a new one with gemini_start_chat."
            response = await chat.send_message(prompt)
        elif chat_id:
            from gemini_webapi import ChatSession
            cid = _resolve_chat_id(chat_id)
            chat = ChatSession(geminiclient=client, cid=cid, rid=None, rcid=None)
            short_id = str(uuid.uuid4())[:8]
            _get_sessions(ctx)[short_id] = chat
            response = await chat.send_message(prompt)
        else:
            response = await client.generate_content(
                prompt, model=model or DEFAULT_MODEL
            )

        text = response.text or "(empty response)"
        thoughts = response.thoughts
        suffix = ""
        if chat_id and not session_id:
            suffix = f"\n\n[session_id: {short_id}]"
        if thoughts:
            return f"**Thinking:**\n{thoughts}\n\n**Response:**\n{text}{suffix}"
        return f"{text}{suffix}"
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="gemini_generate_image",
    annotations={
        "title": "Gemini Image Generation",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def gemini_generate_image(
    prompt: str,
    ctx: Context,
    files: Optional[list[str]] = None,
    session_id: Optional[str] = None,
    chat_id: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Generate or edit images with Gemini. Can continue an existing chat by URL or chat_id.

    Without files: generates a new image from the text prompt.
    With files: edits/transforms the provided image(s) based on the prompt.
    Do NOT use gemini_analyze_url for gemini.google.com links — they are chat IDs.

    Images are saved to ~/Pictures/gemini/ and full file paths are returned.

    Args:
        prompt: Description of the image to generate, or editing instruction
                (e.g. 'change the background to blue', 'make it a cartoon').
        model: Model name. Defaults to gemini-3.0-flash-thinking
               (Nano Banana 2, supports non-square aspect ratios).
        files: Optional list of file paths to images to edit/transform.
        session_id: Optional MCP session ID from gemini_start_chat. Generates
                    images within an existing chat session (preserves context).
        chat_id: Optional Gemini chat ID or URL to continue an existing
                 conversation. Accepts both raw ID ('c_abc123') and full URL
                 ('https://gemini.google.com/app/c_abc123').

    Returns:
        JSON with generated image paths and metadata, or an error message.
    """
    global _image_mode
    try:
        client = _get_client(ctx)

        # Validate input files
        resolved_files = []
        if files:
            for f in files:
                p = Path(f).expanduser().resolve()
                if not p.exists():
                    return f"Error: File not found — {p}"
                resolved_files.append(str(p))

        # Resolve chat session
        chat = None
        if session_id:
            sessions = _get_sessions(ctx)
            chat = sessions.get(session_id)
            if not chat:
                return f"Error: Session '{session_id}' not found. Use gemini_start_chat first."
        elif chat_id:
            from gemini_webapi import ChatSession
            cid = _resolve_chat_id(chat_id)
            chat = ChatSession(geminiclient=client, cid=cid, rid=None, rcid=None)
            short_id = str(uuid.uuid4())[:8]
            _get_sessions(ctx)[short_id] = chat

        async with _image_lock:
            _image_mode = True
            try:
                kwargs = {"model": model or "gemini-3.0-flash-thinking"}
                if resolved_files:
                    kwargs["files"] = resolved_files
                gen_task = asyncio.create_task(
                    _do_generate(client, chat, prompt, **kwargs)
                )
                step = 0
                while not gen_task.done():
                    step += 1
                    await ctx.report_progress(step, total=None)
                    await ctx.info(
                        f"Image generation in progress... ({step * 15}s)"
                    )
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(gen_task), timeout=15
                        )
                    except asyncio.TimeoutError:
                        continue
                response = gen_task.result()
            finally:
                _image_mode = False

        if not response.images:
            return response.text or "No images were generated. Try rephrasing your prompt."

        # --- Try to get 2x download URLs via c8o8Fe RPC ---
        metadata = list(_last_metadata)  # snapshot before it's overwritten
        download_urls: dict[int, str] = {}
        for i, image in enumerate(response.images):
            token = _image_tokens.pop(image.url, None)
            if token and metadata:
                logger.info("Requesting 2x download URL for image %d...", i)
                dl_url = await _fetch_download_url(client, token, prompt, metadata, i)
                if dl_url:
                    download_urls[i] = dl_url
                    logger.info("Got 2x download URL for image %d", i)
        _image_tokens.clear()  # clean up any leftover tokens

        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        saved = []

        for i, image in enumerate(response.images):
            # Use 2x upscale URL from c8o8Fe if available
            has_upscale = i in download_urls
            if has_upscale:
                # Use c8o8Fe 2x URL with =s0 for full resolution (not =s2048 which downscales)
                image.url = re.sub(r"=[^/]*$", "", download_urls[i]) + "=s0"

            try:
                filepath = await image.save(
                    path=str(IMAGES_DIR),
                    filename=f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{i}.png",
                    verbose=False,
                    full_size=not has_upscale,  # c8o8Fe already has =s0; preview gets =s2048
                )
            except Exception as save_err:
                logger.warning("Image save failed for %d: %s", i, save_err)
                continue

            if not filepath:
                continue

            from PIL import Image as _PILImage
            _saved_img = _PILImage.open(filepath)
            _sw, _sh = _saved_img.size
            _saved_img.close()
            wm_scale = 2 if max(_sw, _sh) > 1500 else 1
            try:
                if _remove_watermark(filepath, scale=wm_scale):
                    logger.info("Watermark removed from %s", filepath)
            except Exception as wm_err:
                logger.warning("Watermark removal failed: %s", wm_err)
            title = getattr(image, "title", None) or f"image_{i}"
            saved.append({"title": title, "path": filepath, "dir": str(IMAGES_DIR)})

        result = {
            "text": response.text or "",
            "images_saved_to": str(IMAGES_DIR),
            "images": saved,
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="gemini_upload_file",
    annotations={
        "title": "Gemini File Upload & Analysis",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def gemini_upload_file(
    file_path: str,
    ctx: Context,
    prompt: str = "Describe this file.",
    model: Optional[str] = None,
) -> str:
    """Upload a file (image, PDF, document, video) to Gemini and ask a question about it.

    Args:
        file_path: Absolute path to the file to upload.
        prompt: Question or instruction about the file
                (e.g. 'What is shown in this image?').
        model: Model name. Defaults to gemini-3.0-flash.

    Returns:
        Gemini's text response about the uploaded file.
    """
    try:
        p = Path(file_path).expanduser().resolve()
        if not p.exists():
            return f"Error: File not found — {p}"

        client = _get_client(ctx)
        response = await client.generate_content(
            prompt, model=model or DEFAULT_MODEL, files=[str(p)]
        )
        return response.text or "(empty response)"

    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="gemini_analyze_url",
    annotations={
        "title": "Gemini URL Analysis",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def gemini_analyze_url(
    url: str,
    ctx: Context,
    prompt: str = "Summarize this content.",
    model: Optional[str] = None,
) -> str:
    """Analyze a URL — YouTube videos, webpages, articles, etc.

    NOT for gemini.google.com/app/ links — those are Gemini chat IDs;
    pass them as chat_id to gemini_chat or gemini_generate_image instead.

    Gemini can watch YouTube videos and read webpages, then answer
    questions about their content.

    Args:
        url: The URL to analyze (YouTube, article, webpage, etc.).
        prompt: Question or instruction about the content
                (e.g. 'Summarize this video', 'What are the key points?').
        model: Model name. Defaults to gemini-3.0-flash.

    Returns:
        Gemini's analysis of the URL content.
    """
    try:
        client = _get_client(ctx)
        full_prompt = f"{prompt}\n\n{url}"
        response = await client.generate_content(
            full_prompt, model=model or DEFAULT_MODEL
        )
        return response.text or "(empty response)"
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="gemini_reset",
    annotations={
        "title": "Reset Gemini Client",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def gemini_reset(ctx: Context) -> str:
    """Re-initialise the Gemini client (refresh cookies, clear state).

    Use this when you get authentication errors or want a fresh session.

    Returns:
        Confirmation message or error.
    """
    try:
        from gemini_webapi import GeminiClient

        old = _get_client(ctx)
        await old.close()

        psid, psidts = _resolve_cookies()

        new_client = GeminiClient(
            secure_1psid=psid, secure_1psidts=psidts or None
        )
        await new_client.init(timeout=600, watchdog_timeout=120, auto_close=False, auto_refresh=True)
        _patch_client(new_client)

        ctx.request_context.lifespan_context["gemini_client"] = new_client
        return "Gemini client re-initialised with fresh cookies."

    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="gemini_delete_chat",
    annotations={
        "title": "Delete Gemini Chat",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def gemini_delete_chat(
    chat_id: str,
    ctx: Context,
) -> str:
    """Delete a Gemini conversation by its chat ID.

    Args:
        chat_id: The Gemini chat ID (cid) to delete, e.g. 'c_...' from the
                 Gemini web URL or from a ChatSession.

    Returns:
        Confirmation message or error.
    """
    try:
        client = _get_client(ctx)
        await client.delete_chat(chat_id)

        # Удалить из локальных сессий, если есть
        sessions = _get_sessions(ctx)
        to_remove = [
            sid for sid, chat in sessions.items()
            if getattr(chat, "cid", None) == chat_id
        ]
        for sid in to_remove:
            del sessions[sid]

        return json.dumps({
            "deleted": chat_id,
            "removed_sessions": to_remove,
        })
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="gemini_list_sessions",
    annotations={
        "title": "List Local Gemini Sessions",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def gemini_list_sessions(ctx: Context) -> str:
    """List all chat sessions created in the current MCP server run.

    Returns:
        JSON with list of sessions: session_id, cid, model.
    """
    sessions = _get_sessions(ctx)
    result = []
    for sid, chat in sessions.items():
        result.append({
            "session_id": sid,
            "cid": getattr(chat, "cid", None),
            "model": getattr(chat, "model", None),
        })
    return json.dumps({"sessions": result, "count": len(result)})


@mcp.tool(
    name="gemini_list_chats",
    annotations={
        "title": "List Gemini Chats",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def gemini_list_chats(
    ctx: Context,
    count: int = 50,
) -> str:
    """Fetch a list of recent Gemini conversations from the API.

    Args:
        count: Maximum number of chats to return (default 50).

    Returns:
        JSON with list of chats (cid, title if available).
    """
    try:
        from gemini_webapi.constants import GRPC
        from gemini_webapi.types.grpc import RPCData
        from gemini_webapi.utils import (
            extract_json_from_response,
            get_nested_value,
        )

        client = _get_client(ctx)
        response = await client._batch_execute([
            RPCData(
                rpcid=GRPC.LIST_CHATS,
                payload=json.dumps([count, None, None]),
                identifier="list_chats",
            ),
        ])

        response_json = extract_json_from_response(response.text)

        chats = []
        for part in response_json:
            if not isinstance(part, list) or len(part) < 3:
                continue
            part_body_str = get_nested_value(part, [2])
            if not part_body_str:
                continue
            part_body = json.loads(part_body_str)
            if not isinstance(part_body, list):
                continue
            chat_items = get_nested_value(part_body, [0], [])
            if not chat_items:
                chat_items = get_nested_value(part_body, [2], [])
            if not chat_items:
                continue
            for item in chat_items:
                if not isinstance(item, list):
                    continue
                chat_info = {"cid": item[0] if len(item) > 0 else None}
                if len(item) > 1 and isinstance(item[1], str):
                    chat_info["title"] = item[1]
                elif len(item) > 2 and isinstance(item[2], str):
                    chat_info["title"] = item[2]
                chats.append(chat_info)

        return json.dumps({"chats": chats, "count": len(chats)})

    except Exception as e:
        return _handle_error(e)
