#!/usr/bin/env python3
"""
Ollama facade server.

Expose Ollama-compatible HTTP APIs locally and forward model inference requests
to an OpenAI-compatible upstream service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import ssl
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib import error, request
from urllib.parse import unquote, urlparse

DEFAULT_KEEP_ALIVE_SECONDS = 300

CAPABILITY_COMPLETION = "completion"
CAPABILITY_TOOLS = "tools"
CAPABILITY_INSERT = "insert"
CAPABILITY_VISION = "vision"
CAPABILITY_EMBEDDING = "embedding"
CAPABILITY_ORDER = [
    CAPABILITY_COMPLETION,
    CAPABILITY_TOOLS,
    CAPABILITY_INSERT,
    CAPABILITY_VISION,
    CAPABILITY_EMBEDDING,
]
ALLOWED_CAPABILITIES = set(CAPABILITY_ORDER)


class HTTPClientError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat().replace("+00:00", "Z")


def isoformat_with_tz(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def join_url(base: str, path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    base = base.rstrip("/")
    if not path:
        return base
    if path.startswith("/"):
        return f"{base}{path}"
    return f"{base}/{path}"


def normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for item in content:
            if isinstance(item, str):
                out.append(item)
                continue
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                out.append(item["text"])
                continue
            if isinstance(item.get("text"), str):
                out.append(item["text"])
        return "".join(out)
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
    return str(content)


def extract_thinking(value: dict[str, Any]) -> str:
    for key in ("thinking", "reasoning", "reasoning_content"):
        if key not in value:
            continue
        text = normalize_content(value.get(key))
        if text:
            return text
    return ""


def parse_json_if_possible(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return raw


def extract_error_message(raw: str, fallback: str) -> str:
    text = raw.strip()
    if not text:
        return fallback
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, str) and err.strip():
            return err
        if isinstance(err, dict):
            msg = err.get("message")
            if isinstance(msg, str) and msg.strip():
                return msg
        msg = payload.get("message")
        if isinstance(msg, str) and msg.strip():
            return msg
    return text


def parse_headers(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values:
        if ":" not in raw:
            raise ValueError(f"Invalid header format: {raw!r}. Expected 'Header-Name: value'.")
        name, value = raw.split(":", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            raise ValueError(f"Invalid header format: {raw!r}. Header name is empty.")
        result[name] = value
    return result


def parse_model_map(values: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Invalid --model-map: {raw!r}. Expected 'local=upstream'.")
        local, upstream = raw.split("=", 1)
        local = local.strip()
        upstream = upstream.strip()
        if not local or not upstream:
            raise ValueError(f"Invalid --model-map: {raw!r}. Expected 'local=upstream'.")
        mapping[local] = upstream
    return mapping


def parse_models(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [v.strip() for v in raw.split(",") if v.strip()]


def normalize_capabilities(raw_values: Any, *, field_name: str) -> list[str]:
    if raw_values is None:
        return []
    if isinstance(raw_values, str):
        candidates = [v.strip().lower() for v in raw_values.split(",") if v.strip()]
    elif isinstance(raw_values, list):
        candidates = []
        for item in raw_values:
            if not isinstance(item, str):
                raise ValueError(f"{field_name} must contain only strings")
            value = item.strip().lower()
            if value:
                candidates.append(value)
    else:
        raise ValueError(f"{field_name} must be a string or array")

    out: list[str] = []
    seen: set[str] = set()
    for cap in candidates:
        if cap not in ALLOWED_CAPABILITIES:
            raise ValueError(
                f"{field_name} contains unsupported capability {cap!r}; "
                f"supported: {', '.join(CAPABILITY_ORDER)}"
            )
        if cap in seen:
            continue
        seen.add(cap)
        out.append(cap)

    out.sort(key=lambda x: CAPABILITY_ORDER.index(x))
    return out


def parse_headers_from_json(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        out: dict[str, str] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("JSON headers keys must be non-empty strings")
            out[key] = str(value)
        return out
    if isinstance(raw, list):
        return parse_headers([str(v) for v in raw])
    raise ValueError("JSON headers must be an object or array")


def parse_model_map_from_json(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        out: dict[str, str] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("JSON model_map keys must be non-empty strings")
            local = key.strip()
            mapped = str(value).strip()
            if not mapped:
                raise ValueError("JSON model_map values must be non-empty")
            out[local] = mapped
        return out
    if isinstance(raw, list):
        return parse_model_map([str(v) for v in raw])
    raise ValueError("JSON model_map must be an object or array")


def parse_model_capabilities_from_json(raw: Any) -> dict[str, list[str]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("JSON model_capabilities must be an object")

    out: dict[str, list[str]] = {}
    for model_name, caps in raw.items():
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("JSON model_capabilities keys must be non-empty strings")
        local_model_name = model_name.strip()
        normalized = normalize_capabilities(caps, field_name=f"model_capabilities.{local_model_name}")
        if not normalized:
            continue
        out[local_model_name] = normalized
    return out


def parse_models_from_json(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return parse_models(raw)
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            name = str(item).strip()
            if name:
                out.append(name)
        return out
    raise ValueError("JSON models must be a string or array")


def parse_boolish(raw: Any, field: str) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    if isinstance(raw, (int, float)):
        return bool(raw)
    raise ValueError(f"Invalid boolean for {field}: {raw!r}")


def parse_keep_alive_seconds(value: Any, default_seconds: int = DEFAULT_KEEP_ALIVE_SECONDS) -> int:
    if value is None:
        return default_seconds
    if isinstance(value, bool):
        return default_seconds if value else 0
    if isinstance(value, (int, float)):
        return max(0, int(value))
    if not isinstance(value, str):
        return default_seconds

    text = value.strip().lower()
    if not text:
        return default_seconds
    if text in {"0", "0s", "0m", "0h"}:
        return 0

    # Supports values like 30s, 5m, 2h, 120, 1.5m
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(ns|us|ms|s|m|h)?", text)
    if not match:
        return default_seconds
    number = float(match.group(1))
    unit = match.group(2) or "s"

    multipliers = {
        "ns": 1e-9,
        "us": 1e-6,
        "ms": 1e-3,
        "s": 1,
        "m": 60,
        "h": 3600,
    }
    seconds = int(max(0, number * multipliers[unit]))
    return seconds


def make_digest(name: str) -> str:
    return "sha256:" + hashlib.sha256(name.encode("utf-8")).hexdigest()


def infer_family(name: str) -> str:
    if "/" in name:
        name = name.split("/", 1)[1]
    if ":" in name:
        name = name.split(":", 1)[0]
    return name or "facade"


def model_name_candidates(name: str) -> list[str]:
    raw = name.strip()
    if not raw:
        return []
    out: list[str] = []
    seen: set[str] = set()

    def add(v: str) -> None:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)

    add(raw)
    if "/" in raw:
        right = raw.split("/", 1)[1]
        add(right)
    if ":" in raw:
        left = raw.split(":", 1)[0]
        add(left)
        if "/" in left:
            add(left.split("/", 1)[1])
    if "/" in raw and ":" in raw:
        mid = raw.split("/", 1)[1]
        if ":" in mid:
            add(mid.split(":", 1)[0])
    return out


def default_details(name: str) -> dict[str, Any]:
    family = infer_family(name)
    return {
        "parent_model": "",
        "format": "openai",
        "family": family,
        "families": [family],
        "parameter_size": "unknown",
        "quantization_level": "unknown",
    }


def load_json_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            parsed = json.load(f)
    except FileNotFoundError as exc:
        raise SystemExit(f"Config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON config ({path}): {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(f"Invalid config ({path}): top-level JSON must be an object")
    return parsed


@dataclass(frozen=True)
class Config:
    listen_host: str
    listen_port: int
    ollama_version: str
    stream_default: bool
    keep_alive_default: Any

    upstream_base_url: str
    upstream_chat_path: str
    upstream_completions_path: str
    upstream_models_path: str
    upstream_embeddings_path: str
    upstream_api_key: str | None
    timeout: float

    default_model: str | None
    model_map: dict[str, str]
    static_models: list[str]
    default_capabilities: list[str]
    model_capabilities: dict[str, list[str]]
    tags_include_capabilities: bool

    extra_headers: dict[str, str]
    tls_context: ssl.SSLContext | None

    def _match_in_model_map(self, raw_name: str) -> tuple[str, str] | None:
        candidates = model_name_candidates(raw_name)
        if not candidates:
            return None

        for candidate in candidates:
            mapped = self.model_map.get(candidate)
            if mapped:
                return candidate, mapped

        lowered_candidates = {candidate.lower() for candidate in candidates}
        for local, mapped in self.model_map.items():
            for local_candidate in model_name_candidates(local):
                if local_candidate.lower() in lowered_candidates:
                    return local, mapped
        return None

    def local_to_upstream_model(self, local_name: str | None) -> str:
        if local_name:
            stripped = local_name.strip()
            matched = self._match_in_model_map(stripped)
            if matched:
                return matched[1]
            return stripped
        if self.default_model:
            return self.default_model
        return ""

    def upstream_to_local_model(self, upstream_name: str) -> str:
        if not isinstance(upstream_name, str):
            return upstream_name
        stripped = upstream_name.strip()
        if not stripped:
            return stripped

        upstream_candidates = {candidate.lower() for candidate in model_name_candidates(stripped)}
        for local, mapped in self.model_map.items():
            for mapped_candidate in model_name_candidates(mapped):
                if mapped_candidate.lower() in upstream_candidates:
                    return local
        return stripped


class ModelRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._custom_models: dict[str, dict[str, Any]] = {}
        self._deleted_models: set[str] = set()
        self._running_models: dict[str, dict[str, Any]] = {}

    def list_custom_names(self) -> list[str]:
        with self._lock:
            return list(self._custom_models.keys())

    def get_custom_model(self, name: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._custom_models.get(name)
            return deepcopy(record) if record else None

    def is_hidden(self, name: str) -> bool:
        with self._lock:
            return name in self._deleted_models

    def upsert_custom_model(self, name: str, record: dict[str, Any]) -> None:
        with self._lock:
            self._custom_models[name] = deepcopy(record)
            self._deleted_models.discard(name)

    def copy_model(self, source: str, destination: str) -> bool:
        with self._lock:
            source_record = self._custom_models.get(source)
            if source_record is None:
                source_record = {
                    "name": source,
                    "model": source,
                    "modified_at": utc_now_iso(),
                    "size": 0,
                    "digest": make_digest(source),
                    "details": default_details(source),
                    "parameters": "",
                    "template": "",
                    "license": "",
                    "capabilities": ["completion"],
                    "model_info": {},
                }
            record = deepcopy(source_record)
            record["name"] = destination
            record["model"] = destination
            record["modified_at"] = utc_now_iso()
            record["digest"] = make_digest(destination)
            self._custom_models[destination] = record
            self._deleted_models.discard(destination)
            return True

    def delete_model(self, name: str) -> None:
        with self._lock:
            self._custom_models.pop(name, None)
            self._running_models.pop(name, None)
            self._deleted_models.add(name)

    def mark_running(
        self,
        *,
        name: str,
        details: dict[str, Any],
        size: int,
        digest: str,
        context_length: int,
        keep_alive_seconds: int,
    ) -> None:
        with self._lock:
            if keep_alive_seconds <= 0:
                self._running_models.pop(name, None)
                return
            expires_at = utc_now() + timedelta(seconds=keep_alive_seconds)
            self._running_models[name] = {
                "name": name,
                "model": name,
                "size": int(size),
                "digest": digest,
                "details": deepcopy(details),
                "expires_at": isoformat_with_tz(expires_at),
                "size_vram": int(size),
                "context_length": int(context_length),
                "_expires_epoch": time.time() + keep_alive_seconds,
            }

    def list_running_models(self) -> list[dict[str, Any]]:
        now_epoch = time.time()
        with self._lock:
            expired = [name for name, item in self._running_models.items() if item.get("_expires_epoch", 0) <= now_epoch]
            for name in expired:
                self._running_models.pop(name, None)
            out: list[dict[str, Any]] = []
            for item in self._running_models.values():
                cloned = deepcopy(item)
                cloned.pop("_expires_epoch", None)
                out.append(cloned)
            return sorted(out, key=lambda x: x.get("name", ""))


class ToolCallAssembler:
    def __init__(self) -> None:
        self._by_index: dict[int, dict[str, Any]] = {}

    def ingest(self, delta_tool_calls: Any) -> None:
        if not isinstance(delta_tool_calls, list):
            return
        for i, item in enumerate(delta_tool_calls):
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            if not isinstance(idx, int):
                idx = i
            state = self._by_index.setdefault(idx, {"name": "", "description": "", "arguments": "", "arguments_obj": None})
            fn = item.get("function")
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if isinstance(name, str) and name:
                state["name"] = name
            description = fn.get("description")
            if isinstance(description, str) and description:
                state["description"] = description
            arguments = fn.get("arguments")
            if isinstance(arguments, str):
                state["arguments"] += arguments
            elif isinstance(arguments, (dict, list)):
                state["arguments_obj"] = arguments

    def to_ollama_tool_calls(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for idx in sorted(self._by_index):
            state = self._by_index[idx]
            arguments = state["arguments_obj"] if state["arguments_obj"] is not None else parse_json_if_possible(state["arguments"])
            entry: dict[str, Any] = {
                "function": {
                    "name": state["name"],
                    "arguments": arguments if arguments is not None else {},
                }
            }
            if state["description"]:
                entry["function"]["description"] = state["description"]
            out.append(entry)
        return out


class OllamaFacadeHandler(BaseHTTPRequestHandler):
    server_version = "ollama-facade/2.0"

    config: Config
    registry: ModelRegistry
    cache_lock = threading.Lock()
    upstream_models_cache: dict[str, Any] | None = None
    blob_lock = threading.Lock()
    blobs: set[str] = set()

    req_id: str

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_OPTIONS(self) -> None:
        self.req_id = uuid.uuid4().hex[:8]
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, HEAD, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_HEAD(self) -> None:
        self.req_id = uuid.uuid4().hex[:8]
        path = urlparse(self.path).path
        try:
            if path in {"/", "/api/version"}:
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Connection", "close")
                self.send_header("Content-Length", "0")
                self.end_headers()
                self.close_connection = True
                return
            if path == "/api/tags":
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Connection", "close")
                self.send_header("Content-Length", "0")
                self.end_headers()
                self.close_connection = True
                return
            blob_match = re.fullmatch(r"/api/blobs/(sha256:[a-fA-F0-9]{64})", path)
            if blob_match:
                digest = blob_match.group(1).lower()
                with self.__class__.blob_lock:
                    exists = digest in self.__class__.blobs
                self.send_response(200 if exists else 404)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Connection", "close")
                self.send_header("Content-Length", "0")
                self.end_headers()
                self.close_connection = True
                return
            raise HTTPClientError(404, f"unsupported path: {path}")
        except HTTPClientError as exc:
            self.send_error_json(exc.status_code, exc.message)
        except Exception as exc:  # pragma: no cover - safety fallback
            self.send_error_json(500, f"internal_error: {exc}")

    def do_GET(self) -> None:
        self.req_id = uuid.uuid4().hex[:8]
        path = urlparse(self.path).path
        try:
            if path == "/":
                self.send_text(200, "Ollama is running")
                return
            if path == "/api/version":
                self.send_json(200, {"version": self.config.ollama_version})
                return
            if path == "/api/tags":
                self.handle_tags()
                return
            if path == "/api/ps":
                self.handle_ps()
                return
            if path == "/v1/models":
                self.handle_v1_models()
                return
            v1_match = re.fullmatch(r"/v1/models/(.+)", path)
            if v1_match:
                self.handle_v1_model(v1_match.group(1))
                return
            raise HTTPClientError(404, f"unsupported path: {path}")
        except HTTPClientError as exc:
            self.send_error_json(exc.status_code, exc.message)
        except Exception as exc:  # pragma: no cover - safety fallback
            self.send_error_json(500, f"internal_error: {exc}")

    def do_POST(self) -> None:
        self.req_id = uuid.uuid4().hex[:8]
        path = urlparse(self.path).path
        try:
            if path == "/api/chat":
                self.handle_chat()
                return
            if path == "/api/generate":
                self.handle_generate()
                return
            if path == "/api/show":
                self.handle_show()
                return
            if path == "/api/embed":
                self.handle_embed(single=False)
                return
            if path == "/api/embeddings":
                self.handle_embed(single=True)
                return
            if path == "/api/create":
                self.handle_create()
                return
            if path == "/api/copy":
                self.handle_copy()
                return
            blob_match = re.fullmatch(r"/api/blobs/(sha256:[a-fA-F0-9]{64})", path)
            if blob_match:
                self.handle_blob_create(blob_match.group(1).lower())
                return
            if path == "/api/pull":
                self.handle_pull()
                return
            if path == "/api/push":
                self.handle_push()
                return
            if path == "/api/delete":
                self.handle_delete()
                return
            if path == "/v1/chat/completions":
                self.handle_v1_chat_completions()
                return
            if path == "/v1/completions":
                self.handle_v1_completions()
                return
            if path == "/v1/embeddings":
                self.handle_v1_embeddings()
                return
            raise HTTPClientError(404, f"unsupported path: {path}")
        except HTTPClientError as exc:
            self.send_error_json(exc.status_code, exc.message)
        except Exception as exc:  # pragma: no cover - safety fallback
            self.send_error_json(500, f"internal_error: {exc}")

    def do_DELETE(self) -> None:
        self.req_id = uuid.uuid4().hex[:8]
        path = urlparse(self.path).path
        try:
            if path == "/api/delete":
                self.handle_delete()
                return
            raise HTTPClientError(404, f"unsupported path: {path}")
        except HTTPClientError as exc:
            self.send_error_json(exc.status_code, exc.message)
        except Exception as exc:  # pragma: no cover - safety fallback
            self.send_error_json(500, f"internal_error: {exc}")

    def read_body_bytes(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def read_json_body(self, *, default_empty: bool = True) -> dict[str, Any]:
        raw = self.read_body_bytes()
        if not raw:
            if default_empty:
                return {}
            raise HTTPClientError(400, "request body is required")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPClientError(400, f"invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise HTTPClientError(400, "request body must be a JSON object")
        return payload

    def send_text(self, status: int, text: str) -> None:
        raw = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)
        self.wfile.flush()
        self.close_connection = True

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)
        self.wfile.flush()
        self.close_connection = True

    def send_ndjson_headers(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def write_ndjson_line(self, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"
        self.wfile.write(raw)
        self.wfile.flush()

    def send_error_json(self, status: int, message: str) -> None:
        self.send_json(status, {"error": message})

    def parse_iso_to_unix(self, raw: Any) -> int:
        if not isinstance(raw, str) or not raw:
            return int(time.time())
        try:
            return int(datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp())
        except ValueError:
            return int(time.time())

    def openai_model_object(self, model_name: str) -> dict[str, Any]:
        entry = self.model_entry(model_name, include_capabilities=False)
        owned_by = "library"
        if "/" in model_name:
            owned_by = model_name.split("/", 1)[0] or "library"
        return {
            "id": model_name,
            "object": "model",
            "created": self.parse_iso_to_unix(entry.get("modified_at")),
            "owned_by": owned_by,
        }

    def parse_upstream_models(self, payload: Any) -> list[str]:
        names: list[str] = []
        if isinstance(payload, dict):
            if isinstance(payload.get("data"), list):
                for item in payload["data"]:
                    if isinstance(item, dict) and isinstance(item.get("id"), str):
                        names.append(item["id"])
            elif isinstance(payload.get("models"), list):
                for item in payload["models"]:
                    if isinstance(item, dict):
                        candidate = item.get("name") or item.get("model") or item.get("id")
                        if isinstance(candidate, str):
                            names.append(candidate)
            elif isinstance(payload.get("items"), list):
                for item in payload["items"]:
                    if isinstance(item, dict):
                        candidate = item.get("name") or item.get("id")
                        if isinstance(candidate, str):
                            names.append(candidate)
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, str):
                    names.append(item)
                elif isinstance(item, dict):
                    candidate = item.get("id") or item.get("name")
                    if isinstance(candidate, str):
                        names.append(candidate)

        deduped = list(dict.fromkeys([n for n in names if n]))
        return deduped

    def upstream_request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        stream: bool = False,
    ):
        url = join_url(self.config.upstream_base_url, path)
        body = None if payload is None else json.dumps(payload).encode("utf-8")

        headers: dict[str, str] = {}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if stream:
            headers["Accept"] = "text/event-stream"
        if self.config.upstream_api_key:
            headers["Authorization"] = f"Bearer {self.config.upstream_api_key}"
        headers.update(self.config.extra_headers)

        req = request.Request(url=url, method=method, data=body, headers=headers)
        open_kwargs: dict[str, Any] = {"timeout": self.config.timeout}
        if self.config.tls_context is not None:
            open_kwargs["context"] = self.config.tls_context

        try:
            return request.urlopen(req, **open_kwargs)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            message = extract_error_message(detail, f"upstream HTTP {exc.code}")
            raise HTTPClientError(exc.code, message) from exc
        except error.URLError as exc:
            raise HTTPClientError(502, f"upstream unreachable: {exc.reason}") from exc

    def fetch_upstream_model_names(self) -> list[str]:
        now = time.time()
        with self.__class__.cache_lock:
            cached = self.__class__.upstream_models_cache
            if cached and now - cached.get("ts", 0) < 20:
                return list(cached.get("names", []))

        try:
            with self.upstream_request("GET", self.config.upstream_models_path) as resp:
                raw = resp.read().decode("utf-8")
            payload = json.loads(raw)
            upstream_names = self.parse_upstream_models(payload)
        except Exception:
            upstream_names = []

        local_names = [self.config.upstream_to_local_model(name) for name in upstream_names]
        with self.__class__.cache_lock:
            self.__class__.upstream_models_cache = {"ts": now, "names": local_names}
        return local_names

    def list_local_models(self) -> list[str]:
        names: list[str] = []
        if self.config.static_models:
            names.extend(self.config.static_models)
        else:
            names.extend(self.fetch_upstream_model_names())

        names.extend(self.config.model_map.keys())
        names.extend(self.registry.list_custom_names())

        if self.config.default_model:
            names.append(self.config.upstream_to_local_model(self.config.default_model))

        deduped: list[str] = []
        seen: set[str] = set()
        for name in names:
            if not name or name in seen or self.registry.is_hidden(name):
                continue
            deduped.append(name)
            seen.add(name)
        return deduped

    def infer_default_capabilities(self, model_name: str) -> list[str]:
        if self.config.default_capabilities:
            return list(self.config.default_capabilities)
        lowered = model_name.lower()
        if any(token in lowered for token in ("embed", "embedding", "bge", "e5", "gte")):
            return [CAPABILITY_EMBEDDING]
        return [CAPABILITY_COMPLETION]

    def resolve_model_capabilities(self, model_name: str, record: dict[str, Any] | None = None) -> list[str]:
        if record and isinstance(record.get("capabilities"), list):
            try:
                caps = normalize_capabilities(record.get("capabilities"), field_name=f"record.capabilities[{model_name}]")
                if caps:
                    return caps
            except ValueError:
                pass

        for candidate in model_name_candidates(model_name):
            if candidate in self.config.model_capabilities:
                return list(self.config.model_capabilities[candidate])

        upstream_name = self.config.local_to_upstream_model(model_name)
        for candidate in model_name_candidates(upstream_name):
            if candidate in self.config.model_capabilities:
                return list(self.config.model_capabilities[candidate])

        return self.infer_default_capabilities(model_name)

    def merge_capabilities(self, *groups: list[str]) -> list[str]:
        seen: set[str] = set()
        merged: list[str] = []
        for group in groups:
            for cap in group:
                if cap not in ALLOWED_CAPABILITIES or cap in seen:
                    continue
                seen.add(cap)
                merged.append(cap)
        merged.sort(key=lambda x: CAPABILITY_ORDER.index(x))
        return merged

    def capabilities_from_template(self, template_text: str) -> list[str]:
        if not template_text:
            return []
        lowered = template_text.lower()
        caps: list[str] = []
        if "tools" in lowered:
            caps.append(CAPABILITY_TOOLS)
        if "suffix" in lowered:
            caps.append(CAPABILITY_INSERT)
        return self.merge_capabilities(caps)

    def ensure_capabilities(self, model_name: str, required: list[str], *, action: str) -> None:
        if not required:
            return
        available = set(self.resolve_model_capabilities(model_name))
        missing = [cap for cap in required if cap not in available]
        if not missing:
            return

        if action in {"chat", "generate"} and CAPABILITY_COMPLETION in missing:
            raise HTTPClientError(400, f'"{model_name}" does not support {action}')
        if len(missing) == 1:
            raise HTTPClientError(400, f'"{model_name}" does not support {missing[0]}')
        raise HTTPClientError(400, f'"{model_name}" does not support {", ".join(missing)}')

    def model_entry(self, name: str, *, include_capabilities: bool | None = None) -> dict[str, Any]:
        record = self.registry.get_custom_model(name)
        details = deepcopy(record.get("details")) if record and isinstance(record.get("details"), dict) else default_details(name)
        modified_at = record.get("modified_at") if record else utc_now_iso()
        if not isinstance(modified_at, str):
            modified_at = utc_now_iso()
        size = record.get("size") if record else 0
        try:
            size = int(size)
        except (TypeError, ValueError):
            size = 0
        digest = record.get("digest") if record else make_digest(name)
        if not isinstance(digest, str) or not digest:
            digest = make_digest(name)

        entry = {
            "name": name,
            "model": name,
            "modified_at": modified_at,
            "size": size,
            "digest": digest,
            "details": details,
        }
        include_caps = self.config.tags_include_capabilities if include_capabilities is None else include_capabilities
        if include_caps:
            entry["capabilities"] = self.resolve_model_capabilities(name, record=record)
        return entry

    def resolve_model(self, body: dict[str, Any]) -> tuple[str, str]:
        req_model = body.get("model")
        local_model: str | None
        if isinstance(req_model, str) and req_model.strip():
            local_model = req_model.strip()
        else:
            local_model = None

        if local_model and self.registry.is_hidden(local_model):
            raise HTTPClientError(404, f"model '{local_model}' not found")

        upstream_model = self.config.local_to_upstream_model(local_model)
        if not upstream_model:
            raise HTTPClientError(400, "missing model and default_model is not configured")

        output_local = local_model or self.config.upstream_to_local_model(upstream_model)
        return output_local, upstream_model

    def usage_counts(self, payload: dict[str, Any]) -> tuple[int, int]:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return 0, 0

        prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        completion = usage.get("completion_tokens", usage.get("output_tokens", 0))
        try:
            prompt_n = int(prompt)
        except (TypeError, ValueError):
            prompt_n = 0
        try:
            completion_n = int(completion)
        except (TypeError, ValueError):
            completion_n = 0
        return max(prompt_n, 0), max(completion_n, 0)

    def usage_fields(self, *, elapsed_ns: int, prompt_tokens: int, completion_tokens: int) -> dict[str, Any]:
        return {
            "total_duration": int(max(elapsed_ns, 0)),
            "load_duration": 0,
            "prompt_eval_count": int(max(prompt_tokens, 0)),
            "prompt_eval_duration": 0,
            "eval_count": int(max(completion_tokens, 0)),
            "eval_duration": 0,
        }

    def openai_tool_calls_to_ollama(self, raw_tool_calls: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_tool_calls, list):
            return []
        out: list[dict[str, Any]] = []
        for item in raw_tool_calls:
            if not isinstance(item, dict):
                continue
            function = item.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if not isinstance(name, str):
                name = ""
            arguments = function.get("arguments")
            parsed_arguments = parse_json_if_possible(arguments)
            entry: dict[str, Any] = {
                "function": {
                    "name": name,
                    "arguments": parsed_arguments if parsed_arguments is not None else {},
                }
            }
            description = function.get("description")
            if isinstance(description, str) and description:
                entry["function"]["description"] = description
            out.append(entry)
        return out

    def apply_common_openai_fields(self, payload: dict[str, Any], body: dict[str, Any], *, stream: bool) -> None:
        passthrough = [
            "temperature",
            "top_p",
            "max_tokens",
            "stop",
            "tools",
            "tool_choice",
            "frequency_penalty",
            "presence_penalty",
            "response_format",
            "seed",
            "n",
            "logprobs",
            "top_logprobs",
        ]
        for key in passthrough:
            if key in body:
                payload[key] = body[key]

        options = body.get("options")
        if isinstance(options, dict):
            mapping = {
                "temperature": "temperature",
                "top_p": "top_p",
                "num_predict": "max_tokens",
                "stop": "stop",
                "seed": "seed",
                "logprobs": "logprobs",
                "top_logprobs": "top_logprobs",
                "presence_penalty": "presence_penalty",
                "frequency_penalty": "frequency_penalty",
            }
            for src, dst in mapping.items():
                if src in options and dst not in payload:
                    payload[dst] = options[src]

        if body.get("format") == "json" and "response_format" not in payload:
            payload["response_format"] = {"type": "json_object"}
        elif isinstance(body.get("format"), dict) and "response_format" not in payload:
            # Upstream vendors differ; preserve schema object directly.
            payload["response_format"] = body["format"]

        think = body.get("think")
        if "reasoning_effort" not in payload:
            if isinstance(think, str) and think in {"low", "medium", "high"}:
                payload["reasoning_effort"] = think
            elif think is True:
                payload["reasoning_effort"] = "medium"

        if stream:
            stream_options = payload.get("stream_options") if isinstance(payload.get("stream_options"), dict) else {}
            stream_options["include_usage"] = True
            payload["stream_options"] = stream_options

    def build_chat_payload(self, body: dict[str, Any], upstream_model: str, stream: bool) -> dict[str, Any]:
        messages = body.get("messages")
        if not isinstance(messages, list):
            raise HTTPClientError(400, "messages must be an array")

        payload: dict[str, Any] = {
            "model": upstream_model,
            "messages": messages,
            "stream": stream,
        }
        self.apply_common_openai_fields(payload, body, stream=stream)
        return payload

    def build_generate_payload(self, body: dict[str, Any], upstream_model: str, stream: bool) -> dict[str, Any]:
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            prompt = ""

        system_prompt = body.get("system")
        messages: list[dict[str, Any]] = []
        if isinstance(system_prompt, str) and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})

        images = body.get("images")
        if isinstance(images, list) and images:
            content_parts: list[dict[str, Any]] = []
            if prompt:
                content_parts.append({"type": "text", "text": prompt})
            for image_b64 in images:
                if not isinstance(image_b64, str) or not image_b64.strip():
                    continue
                # Assume base64 PNG when MIME is unknown.
                content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
            if content_parts:
                messages.append({"role": "user", "content": content_parts})
            else:
                messages.append({"role": "user", "content": prompt})
        else:
            messages.append({"role": "user", "content": prompt})

        suffix = body.get("suffix")
        if isinstance(suffix, str) and suffix.strip() and messages:
            # Chat-completions has no native suffix parameter, so preserve intent as instruction.
            if isinstance(messages[-1].get("content"), str):
                messages[-1]["content"] = (
                    f"{messages[-1]['content']}\n\nContinue the text so it naturally precedes this suffix:\n{suffix}"
                )

        payload: dict[str, Any] = {
            "model": upstream_model,
            "messages": messages,
            "stream": stream,
        }
        self.apply_common_openai_fields(payload, body, stream=stream)
        return payload

    def mark_model_running(self, model_name: str, body: dict[str, Any]) -> None:
        entry = self.model_entry(model_name)
        keep_alive_value = body.get("keep_alive", self.config.keep_alive_default)
        keep_alive_seconds = parse_keep_alive_seconds(keep_alive_value)

        context_length = 4096
        options = body.get("options")
        if isinstance(options, dict) and isinstance(options.get("num_ctx"), int):
            context_length = max(1, options["num_ctx"])

        self.registry.mark_running(
            name=model_name,
            details=entry["details"],
            size=entry["size"],
            digest=entry["digest"],
            context_length=context_length,
            keep_alive_seconds=keep_alive_seconds,
        )

    def unload_model(self, model_name: str) -> None:
        entry = self.model_entry(model_name)
        self.registry.mark_running(
            name=model_name,
            details=entry["details"],
            size=entry["size"],
            digest=entry["digest"],
            context_length=4096,
            keep_alive_seconds=0,
        )

    def request_is_explicit_unload(self, body: dict[str, Any]) -> bool:
        if "keep_alive" not in body:
            return False
        keep_alive_seconds = parse_keep_alive_seconds(body.get("keep_alive"), default_seconds=DEFAULT_KEEP_ALIVE_SECONDS)
        return keep_alive_seconds == 0

    def stream_status(self, statuses: list[str], stream: bool) -> None:
        if stream:
            self.send_ndjson_headers()
            for status in statuses:
                try:
                    self.write_ndjson_line({"status": status})
                except BrokenPipeError:
                    return
            return
        self.send_json(200, {"status": statuses[-1] if statuses else "success"})

    def relay_sse_from_upstream(self, resp, *, local_model: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        try:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace")
                stripped = line.strip()
                if stripped.startswith("data:"):
                    data = stripped[5:].strip()
                    if data and data != "[DONE]":
                        try:
                            event = json.loads(data)
                            if isinstance(event, dict) and isinstance(event.get("model"), str):
                                event["model"] = local_model
                                line = f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                        except json.JSONDecodeError:
                            pass
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()
        except BrokenPipeError:
            return

    # ---- OpenAI compatibility endpoint handlers (/v1/*) ----

    def handle_v1_models(self) -> None:
        data = [self.openai_model_object(name) for name in self.list_local_models()]
        self.send_json(200, {"object": "list", "data": data})

    def handle_v1_model(self, encoded_model_name: str) -> None:
        model_name = unquote(encoded_model_name)
        if self.registry.is_hidden(model_name):
            raise HTTPClientError(404, f"model '{model_name}' not found")
        names = set(self.list_local_models())
        if model_name not in names:
            raise HTTPClientError(404, f"model '{model_name}' not found")
        self.send_json(200, self.openai_model_object(model_name))

    def handle_v1_chat_completions(self) -> None:
        body = self.read_json_body(default_empty=False)
        if not isinstance(body.get("model"), str) or not body["model"].strip():
            raise HTTPClientError(400, "model is required")
        local_model = body["model"].strip()
        upstream_model = self.config.local_to_upstream_model(local_model)
        if not upstream_model:
            raise HTTPClientError(400, "model is required")

        payload = dict(body)
        payload["model"] = upstream_model
        stream = bool(payload.get("stream", False))

        if stream:
            with self.upstream_request("POST", self.config.upstream_chat_path, payload=payload, stream=True) as resp:
                self.relay_sse_from_upstream(resp, local_model=local_model)
            return

        with self.upstream_request("POST", self.config.upstream_chat_path, payload=payload) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPClientError(502, f"invalid upstream response: {exc}") from exc
        if isinstance(decoded, dict) and isinstance(decoded.get("model"), str):
            decoded["model"] = local_model
        self.send_json(200, decoded if isinstance(decoded, dict) else {"error": "invalid upstream response"})

    def handle_v1_completions(self) -> None:
        body = self.read_json_body(default_empty=False)
        if not isinstance(body.get("model"), str) or not body["model"].strip():
            raise HTTPClientError(400, "model is required")
        local_model = body["model"].strip()
        upstream_model = self.config.local_to_upstream_model(local_model)
        if not upstream_model:
            raise HTTPClientError(400, "model is required")

        payload = dict(body)
        payload["model"] = upstream_model
        stream = bool(payload.get("stream", False))

        if stream:
            with self.upstream_request("POST", self.config.upstream_completions_path, payload=payload, stream=True) as resp:
                self.relay_sse_from_upstream(resp, local_model=local_model)
            return

        with self.upstream_request("POST", self.config.upstream_completions_path, payload=payload) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPClientError(502, f"invalid upstream response: {exc}") from exc
        if isinstance(decoded, dict) and isinstance(decoded.get("model"), str):
            decoded["model"] = local_model
        self.send_json(200, decoded if isinstance(decoded, dict) else {"error": "invalid upstream response"})

    def handle_v1_embeddings(self) -> None:
        body = self.read_json_body(default_empty=False)
        if not isinstance(body.get("model"), str) or not body["model"].strip():
            raise HTTPClientError(400, "model is required")
        local_model = body["model"].strip()
        upstream_model = self.config.local_to_upstream_model(local_model)
        if not upstream_model:
            raise HTTPClientError(400, "model is required")

        payload = dict(body)
        payload["model"] = upstream_model

        with self.upstream_request("POST", self.config.upstream_embeddings_path, payload=payload) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPClientError(502, f"invalid upstream response: {exc}") from exc
        if isinstance(decoded, dict):
            decoded["model"] = local_model
            self.send_json(200, decoded)
            return
        self.send_json(502, {"error": "invalid upstream response"})

    def handle_blob_create(self, digest: str) -> None:
        payload = self.read_body_bytes()
        if payload:
            actual = "sha256:" + hashlib.sha256(payload).hexdigest()
            if actual != digest:
                raise HTTPClientError(400, f"digest mismatch: expected {digest}, got {actual}")
        with self.__class__.blob_lock:
            self.__class__.blobs.add(digest)

        self.send_response(201)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.send_header("Content-Length", "0")
        self.end_headers()
        self.close_connection = True

    # ---- Endpoint handlers ----

    def handle_tags(self) -> None:
        models = [self.model_entry(name, include_capabilities=True) for name in self.list_local_models()]
        models.sort(key=lambda item: item.get("modified_at", ""), reverse=True)
        self.send_json(200, {"models": models})

    def handle_ps(self) -> None:
        self.send_json(200, {"models": self.registry.list_running_models()})

    def handle_show(self) -> None:
        body = self.read_json_body(default_empty=False)
        model_name = body.get("model")
        if not isinstance(model_name, str) or not model_name.strip():
            model_name = body.get("name")
        if not isinstance(model_name, str) or not model_name.strip():
            raise HTTPClientError(400, "model is required")

        if self.registry.is_hidden(model_name):
            raise HTTPClientError(404, f"model '{model_name}' not found")

        entry = self.model_entry(model_name)
        custom = self.registry.get_custom_model(model_name) or {}

        capabilities = self.resolve_model_capabilities(model_name, record=custom if custom else None)

        template = custom.get("template") if isinstance(custom.get("template"), str) else ""
        parameters = custom.get("parameters") if isinstance(custom.get("parameters"), str) else ""
        license_text = custom.get("license") if isinstance(custom.get("license"), str) else ""
        system_text = custom.get("system") if isinstance(custom.get("system"), str) else ""
        messages = custom.get("messages") if isinstance(custom.get("messages"), list) else []
        model_info = custom.get("model_info") if isinstance(custom.get("model_info"), dict) else {}
        projector_info = custom.get("projector_info") if isinstance(custom.get("projector_info"), dict) else {}
        tensors = custom.get("tensors") if isinstance(custom.get("tensors"), list) else []

        from_model = custom.get("from") if isinstance(custom.get("from"), str) and custom.get("from").strip() else model_name
        modelfile_lines = [f"FROM {from_model}"]
        if template:
            modelfile_lines.append(f'TEMPLATE """{template}"""')
        if system_text:
            modelfile_lines.append(f'SYSTEM """{system_text}"""')
        if parameters:
            modelfile_lines.append(parameters)

        response_payload = {
            "parameters": parameters,
            "license": license_text,
            "template": template,
            "system": system_text,
            "modelfile": "\n".join(modelfile_lines),
            "capabilities": capabilities,
            "modified_at": entry["modified_at"],
            "details": entry["details"],
            "messages": messages,
            "model_info": model_info,
            "projector_info": projector_info,
            "tensors": tensors,
        }
        self.send_json(200, response_payload)

    def handle_create(self) -> None:
        body = self.read_json_body(default_empty=False)
        model_name = body.get("model")
        if not isinstance(model_name, str) or not model_name.strip():
            raise HTTPClientError(400, "model is required")

        source_model = body.get("from") if isinstance(body.get("from"), str) else model_name
        template = body.get("template") if isinstance(body.get("template"), str) else ""
        system = body.get("system") if isinstance(body.get("system"), str) else ""
        parameters = body.get("parameters")
        if isinstance(parameters, dict):
            parameter_lines = [f"{k} {v}" for k, v in parameters.items()]
            parameters_text = "\n".join(parameter_lines)
        elif isinstance(parameters, str):
            parameters_text = parameters
        else:
            parameters_text = ""

        license_field = body.get("license")
        if isinstance(license_field, list):
            license_text = "\n".join([str(v) for v in license_field])
        elif isinstance(license_field, str):
            license_text = license_field
        else:
            license_text = ""

        inherited_caps = self.resolve_model_capabilities(source_model)
        template_caps = self.capabilities_from_template(template)
        merged_caps = self.merge_capabilities(inherited_caps, template_caps)

        self.registry.upsert_custom_model(
            model_name,
            {
                "name": model_name,
                "model": model_name,
                "from": source_model,
                "modified_at": utc_now_iso(),
                "size": 0,
                "digest": make_digest(model_name),
                "details": default_details(model_name),
                "parameters": parameters_text,
                "template": template,
                "license": license_text,
                "system": system,
                "capabilities": merged_caps,
                "model_info": {},
            },
        )

        stream = body.get("stream")
        if stream is None:
            stream_enabled = True
        else:
            stream_enabled = bool(stream)

        self.stream_status(["creating model", "success"], stream_enabled)

    def handle_copy(self) -> None:
        body = self.read_json_body(default_empty=False)
        source = body.get("source")
        destination = body.get("destination")
        if not isinstance(source, str) or not source.strip():
            raise HTTPClientError(400, "source is required")
        if not isinstance(destination, str) or not destination.strip():
            raise HTTPClientError(400, "destination is required")
        source = source.strip()
        destination = destination.strip()

        source_record = self.registry.get_custom_model(source)
        if source_record is None:
            source_record = {
                "name": source,
                "model": source,
                "modified_at": utc_now_iso(),
                "size": 0,
                "digest": make_digest(source),
                "details": default_details(source),
                "parameters": "",
                "template": "",
                "license": "",
                "capabilities": self.resolve_model_capabilities(source),
                "model_info": {},
            }
        record = deepcopy(source_record)
        record["name"] = destination
        record["model"] = destination
        record["modified_at"] = utc_now_iso()
        record["digest"] = make_digest(destination)
        self.registry.upsert_custom_model(destination, record)
        self.send_json(200, {"status": "success"})

    def handle_pull(self) -> None:
        body = self.read_json_body(default_empty=False)
        model = body.get("model")
        if not isinstance(model, str) or not model.strip():
            raise HTTPClientError(400, "model is required")

        model = model.strip()
        self.registry.upsert_custom_model(
            model,
            {
                "name": model,
                "model": model,
                "modified_at": utc_now_iso(),
                "size": 0,
                "digest": make_digest(model),
                "details": default_details(model),
                "parameters": "",
                "template": "",
                "license": "",
                "capabilities": self.resolve_model_capabilities(model),
                "model_info": {},
            },
        )

        stream = body.get("stream")
        if stream is None:
            stream_enabled = True
        else:
            stream_enabled = bool(stream)

        self.stream_status(["pulling manifest", "verifying digest", "success"], stream_enabled)

    def handle_push(self) -> None:
        body = self.read_json_body(default_empty=False)
        model = body.get("model")
        if not isinstance(model, str) or not model.strip():
            raise HTTPClientError(400, "model is required")

        stream = body.get("stream")
        if stream is None:
            stream_enabled = True
        else:
            stream_enabled = bool(stream)

        self.stream_status(["pushing manifest", "success"], stream_enabled)

    def handle_delete(self) -> None:
        body = self.read_json_body(default_empty=False)
        model = body.get("model")
        if not isinstance(model, str) or not model.strip():
            raise HTTPClientError(400, "model is required")
        self.registry.delete_model(model.strip())
        self.send_json(200, {"status": "success"})

    def handle_embed(self, *, single: bool) -> None:
        body = self.read_json_body(default_empty=False)
        local_model, upstream_model = self.resolve_model(body)

        if single:
            emb_input = body.get("prompt", body.get("input", ""))
        else:
            emb_input = body.get("input", body.get("prompt", ""))

        # Ollama compatibility: empty input only loads model and returns empty vectors.
        if single:
            if not isinstance(emb_input, str) or emb_input == "":
                self.mark_model_running(local_model, body)
                self.send_json(200, {"model": local_model, "embedding": []})
                return
        else:
            is_empty = False
            if isinstance(emb_input, str):
                is_empty = emb_input == ""
            elif isinstance(emb_input, list):
                is_empty = len(emb_input) == 0
            elif emb_input is None:
                is_empty = True
            if is_empty:
                self.mark_model_running(local_model, body)
                self.send_json(200, {"model": local_model, "embeddings": []})
                return

        payload: dict[str, Any] = {
            "model": upstream_model,
            "input": emb_input,
        }
        for key in ("truncate", "dimensions"):
            if key in body:
                payload[key] = body[key]

        started_ns = time.time_ns()
        with self.upstream_request("POST", self.config.upstream_embeddings_path, payload=payload) as resp:
            raw = resp.read().decode("utf-8")
        elapsed_ns = time.time_ns() - started_ns

        try:
            upstream_payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPClientError(502, f"invalid upstream embedding response: {exc}") from exc

        data = upstream_payload.get("data")
        vectors: list[list[float]] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and isinstance(item.get("embedding"), list):
                    vectors.append(item["embedding"])

        prompt_eval_count = 0
        if isinstance(emb_input, str):
            prompt_eval_count = 1 if emb_input else 0
        elif isinstance(emb_input, list):
            prompt_eval_count = len(emb_input)

        usage = {
            "total_duration": int(max(elapsed_ns, 0)),
            "load_duration": 0,
            "prompt_eval_count": prompt_eval_count,
        }

        self.mark_model_running(local_model, body)

        if single:
            payload_single: dict[str, Any] = {
                "model": local_model,
                "embedding": vectors[0] if vectors else [],
            }
            payload_single.update(usage)
            self.send_json(200, payload_single)
            return

        payload_multi: dict[str, Any] = {
            "model": local_model,
            "embeddings": vectors,
        }
        payload_multi.update(usage)
        self.send_json(200, payload_multi)

    def handle_chat(self) -> None:
        body = self.read_json_body(default_empty=False)
        local_model, upstream_model = self.resolve_model(body)
        messages = body.get("messages")
        if not isinstance(messages, list):
            raise HTTPClientError(400, "messages must be an array")

        required_caps = [CAPABILITY_COMPLETION]
        if isinstance(body.get("tools"), list) and body.get("tools"):
            required_caps.append(CAPABILITY_TOOLS)
        self.ensure_capabilities(local_model, required_caps, action="chat")

        if len(messages) == 0 and self.request_is_explicit_unload(body):
            self.unload_model(local_model)
            self.send_json(
                200,
                {
                    "model": local_model,
                    "created_at": utc_now_iso(),
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": "unload",
                },
            )
            return

        if len(messages) == 0:
            self.mark_model_running(local_model, body)
            self.send_json(
                200,
                {
                    "model": local_model,
                    "created_at": utc_now_iso(),
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": "load",
                },
            )
            return

        stream_val = body.get("stream")
        if stream_val is None:
            stream = self.config.stream_default
        else:
            stream = bool(stream_val)

        upstream_payload = self.build_chat_payload(body, upstream_model, stream=stream)
        created_at = utc_now_iso()
        started_ns = time.time_ns()

        if not stream:
            with self.upstream_request("POST", self.config.upstream_chat_path, payload=upstream_payload) as resp:
                raw = resp.read().decode("utf-8")
            elapsed_ns = time.time_ns() - started_ns

            try:
                upstream_json = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise HTTPClientError(502, f"invalid upstream chat response: {exc}") from exc

            choices = upstream_json.get("choices")
            if not isinstance(choices, list) or not choices:
                raise HTTPClientError(502, "invalid upstream chat response: missing choices")
            choice0 = choices[0] if isinstance(choices[0], dict) else {}
            message_raw = choice0.get("message") if isinstance(choice0.get("message"), dict) else {}

            done_reason = choice0.get("finish_reason") if isinstance(choice0.get("finish_reason"), str) else "stop"
            prompt_count, eval_count = self.usage_counts(upstream_json)

            message: dict[str, Any] = {
                "role": "assistant",
                "content": normalize_content(message_raw.get("content")),
            }
            thinking = extract_thinking(message_raw)
            if thinking:
                message["thinking"] = thinking
            tool_calls = self.openai_tool_calls_to_ollama(message_raw.get("tool_calls"))
            if tool_calls:
                message["tool_calls"] = tool_calls

            response_payload = {
                "model": local_model,
                "created_at": created_at,
                "message": message,
                "done": True,
                "done_reason": done_reason,
            }
            response_payload.update(
                self.usage_fields(elapsed_ns=elapsed_ns, prompt_tokens=prompt_count, completion_tokens=eval_count)
            )
            self.send_json(200, response_payload)
            self.mark_model_running(local_model, body)
            return

        # Streamed response.
        try:
            upstream_resp = self.upstream_request("POST", self.config.upstream_chat_path, payload=upstream_payload, stream=True)
        except HTTPClientError:
            raise

        self.send_ndjson_headers()
        done_reason = "stop"
        prompt_count = 0
        eval_count = 0
        tool_calls = ToolCallAssembler()

        try:
            with upstream_resp as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if line == "[DONE]":
                        break

                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if isinstance(event.get("error"), str):
                        self.write_ndjson_line({"error": event["error"]})
                        return

                    if isinstance(event.get("usage"), dict):
                        prompt_count, eval_count = self.usage_counts(event)

                    choices = event.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue
                    choice0 = choices[0]
                    if not isinstance(choice0, dict):
                        continue

                    finish_reason = choice0.get("finish_reason")
                    if isinstance(finish_reason, str) and finish_reason:
                        done_reason = finish_reason

                    delta = choice0.get("delta")
                    if not isinstance(delta, dict):
                        continue

                    content_piece = normalize_content(delta.get("content"))
                    thinking_piece = extract_thinking(delta)
                    tool_calls.ingest(delta.get("tool_calls"))

                    if not content_piece and not thinking_piece:
                        continue

                    chunk_message: dict[str, Any] = {
                        "role": "assistant",
                        "content": content_piece,
                    }
                    if thinking_piece:
                        chunk_message["thinking"] = thinking_piece

                    chunk = {
                        "model": local_model,
                        "created_at": created_at,
                        "message": chunk_message,
                        "done": False,
                    }
                    self.write_ndjson_line(chunk)
        except BrokenPipeError:
            return
        except HTTPClientError as exc:
            try:
                self.write_ndjson_line({"error": exc.message})
            except BrokenPipeError:
                pass
            return
        except Exception as exc:
            try:
                self.write_ndjson_line({"error": f"stream_error: {exc}"})
            except BrokenPipeError:
                pass
            return

        elapsed_ns = time.time_ns() - started_ns
        final_message: dict[str, Any] = {
            "role": "assistant",
            "content": "",
        }
        final_tool_calls = tool_calls.to_ollama_tool_calls()
        if final_tool_calls:
            final_message["tool_calls"] = final_tool_calls

        final_chunk: dict[str, Any] = {
            "model": local_model,
            "created_at": created_at,
            "message": final_message,
            "done": True,
            "done_reason": done_reason,
        }
        final_chunk.update(self.usage_fields(elapsed_ns=elapsed_ns, prompt_tokens=prompt_count, completion_tokens=eval_count))

        try:
            self.write_ndjson_line(final_chunk)
        except BrokenPipeError:
            return

        self.mark_model_running(local_model, body)

    def handle_generate(self) -> None:
        body = self.read_json_body(default_empty=False)
        local_model, upstream_model = self.resolve_model(body)
        required_caps = [CAPABILITY_COMPLETION]
        suffix_val = body.get("suffix")
        if isinstance(suffix_val, str) and suffix_val.strip():
            required_caps.append(CAPABILITY_INSERT)
        self.ensure_capabilities(local_model, required_caps, action="generate")

        prompt_val = body.get("prompt")
        prompt = prompt_val if isinstance(prompt_val, str) else ""
        if prompt == "" and self.request_is_explicit_unload(body):
            self.unload_model(local_model)
            self.send_json(
                200,
                {
                    "model": local_model,
                    "created_at": utc_now_iso(),
                    "response": "",
                    "done": True,
                    "done_reason": "unload",
                },
            )
            return

        if prompt == "":
            self.mark_model_running(local_model, body)
            self.send_json(
                200,
                {
                    "model": local_model,
                    "created_at": utc_now_iso(),
                    "response": "",
                    "done": True,
                    "done_reason": "load",
                },
            )
            return

        stream_val = body.get("stream")
        if stream_val is None:
            stream = self.config.stream_default
        else:
            stream = bool(stream_val)

        upstream_payload = self.build_generate_payload(body, upstream_model, stream=stream)
        created_at = utc_now_iso()
        started_ns = time.time_ns()

        if not stream:
            with self.upstream_request("POST", self.config.upstream_chat_path, payload=upstream_payload) as resp:
                raw = resp.read().decode("utf-8")
            elapsed_ns = time.time_ns() - started_ns

            try:
                upstream_json = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise HTTPClientError(502, f"invalid upstream generate response: {exc}") from exc

            choices = upstream_json.get("choices")
            if not isinstance(choices, list) or not choices:
                raise HTTPClientError(502, "invalid upstream generate response: missing choices")
            choice0 = choices[0] if isinstance(choices[0], dict) else {}
            message_raw = choice0.get("message") if isinstance(choice0.get("message"), dict) else {}

            done_reason = choice0.get("finish_reason") if isinstance(choice0.get("finish_reason"), str) else "stop"
            prompt_count, eval_count = self.usage_counts(upstream_json)

            response_payload = {
                "model": local_model,
                "created_at": created_at,
                "response": normalize_content(message_raw.get("content")),
                "done": True,
                "done_reason": done_reason,
            }
            thinking = extract_thinking(message_raw)
            if thinking:
                response_payload["thinking"] = thinking
            response_payload.update(
                self.usage_fields(elapsed_ns=elapsed_ns, prompt_tokens=prompt_count, completion_tokens=eval_count)
            )
            self.send_json(200, response_payload)
            self.mark_model_running(local_model, body)
            return

        try:
            upstream_resp = self.upstream_request("POST", self.config.upstream_chat_path, payload=upstream_payload, stream=True)
        except HTTPClientError:
            raise

        self.send_ndjson_headers()
        done_reason = "stop"
        prompt_count = 0
        eval_count = 0

        try:
            with upstream_resp as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if line == "[DONE]":
                        break

                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if isinstance(event.get("error"), str):
                        self.write_ndjson_line({"error": event["error"]})
                        return

                    if isinstance(event.get("usage"), dict):
                        prompt_count, eval_count = self.usage_counts(event)

                    choices = event.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue
                    choice0 = choices[0]
                    if not isinstance(choice0, dict):
                        continue

                    finish_reason = choice0.get("finish_reason")
                    if isinstance(finish_reason, str) and finish_reason:
                        done_reason = finish_reason

                    delta = choice0.get("delta")
                    if not isinstance(delta, dict):
                        continue

                    content_piece = normalize_content(delta.get("content"))
                    thinking_piece = extract_thinking(delta)
                    if not content_piece and not thinking_piece:
                        continue

                    chunk: dict[str, Any] = {
                        "model": local_model,
                        "created_at": created_at,
                        "response": content_piece,
                        "done": False,
                    }
                    if thinking_piece:
                        chunk["thinking"] = thinking_piece
                    self.write_ndjson_line(chunk)
        except BrokenPipeError:
            return
        except HTTPClientError as exc:
            try:
                self.write_ndjson_line({"error": exc.message})
            except BrokenPipeError:
                pass
            return
        except Exception as exc:
            try:
                self.write_ndjson_line({"error": f"stream_error: {exc}"})
            except BrokenPipeError:
                pass
            return

        elapsed_ns = time.time_ns() - started_ns
        final_chunk: dict[str, Any] = {
            "model": local_model,
            "created_at": created_at,
            "response": "",
            "done": True,
            "done_reason": done_reason,
        }
        final_chunk.update(self.usage_fields(elapsed_ns=elapsed_ns, prompt_tokens=prompt_count, completion_tokens=eval_count))

        try:
            self.write_ndjson_line(final_chunk)
        except BrokenPipeError:
            return

        self.mark_model_running(local_model, body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local Ollama facade that forwards requests to an OpenAI-compatible upstream service."
    )
    parser.add_argument("--config", default=None, help="JSON config file path")

    parser.add_argument("--listen-host", default=None, help="Local listen host. Default: 127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=None, help="Local listen port. Default: 11434")
    parser.add_argument(
        "--ollama-version",
        default=None,
        help="Version string returned by /api/version. Default: 0.6.4",
    )

    parser.add_argument(
        "--stream-default",
        default=None,
        help="Default stream behavior when request omits stream (true/false). Default: true",
    )
    parser.add_argument(
        "--keep-alive-default",
        default=None,
        help="Default keep_alive when request omits it (e.g. 5m). Default: 5m",
    )

    parser.add_argument("--upstream-base-url", default=None, help="Upstream base URL, e.g. https://api.openai.com/v1")
    parser.add_argument("--upstream-api-key", default=None, help="Upstream API key. If omitted, no Authorization header")
    parser.add_argument("--upstream-chat-path", default=None, help="Upstream chat endpoint path")
    parser.add_argument("--upstream-completions-path", default=None, help="Upstream completions endpoint path")
    parser.add_argument("--upstream-models-path", default=None, help="Upstream models endpoint path")
    parser.add_argument("--upstream-embeddings-path", default=None, help="Upstream embeddings endpoint path")

    parser.add_argument("--default-model", default=None, help="Fallback upstream model when request body has no model")
    parser.add_argument(
        "--model-map",
        action="append",
        default=[],
        metavar="LOCAL=UPSTREAM",
        help="Map local Ollama model name to upstream model name; can be specified multiple times",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Static local model names for /api/tags, comma-separated. If omitted, query upstream /models.",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        metavar="NAME:VALUE",
        help="Extra upstream HTTP header; can be specified multiple times",
    )

    parser.add_argument("--timeout", type=float, default=None, help="Upstream request timeout in seconds")
    parser.add_argument("--insecure-skip-tls-verify", action="store_true", help="Disable upstream TLS verification")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    json_config = load_json_config(args.config)

    def pick(name: str, default: Any = None) -> Any:
        cli_value = getattr(args, name)
        if cli_value is not None:
            return cli_value
        return json_config.get(name, default)

    try:
        json_headers = parse_headers_from_json(json_config.get("headers", json_config.get("extra_headers")))
        cli_headers = parse_headers(args.header)
        extra_headers = {**json_headers, **cli_headers}

        json_model_map = parse_model_map_from_json(json_config.get("model_map"))
        cli_model_map = parse_model_map(args.model_map)
        model_map = {**json_model_map, **cli_model_map}

        default_capabilities = normalize_capabilities(
            json_config.get("default_capabilities", [CAPABILITY_COMPLETION]),
            field_name="default_capabilities",
        )
        if not default_capabilities:
            default_capabilities = [CAPABILITY_COMPLETION]
        model_capabilities = parse_model_capabilities_from_json(json_config.get("model_capabilities"))
    except ValueError as exc:
        raise SystemExit(f"Argument error: {exc}")

    try:
        if args.models is not None:
            static_models = parse_models(args.models)
        else:
            static_models = parse_models_from_json(json_config.get("models"))
    except ValueError as exc:
        raise SystemExit(f"Config error: {exc}")

    listen_host = pick("listen_host", "127.0.0.1")
    listen_port = pick("listen_port", 11434)
    ollama_version = pick("ollama_version", "0.6.4")

    stream_default_raw = pick("stream_default", True)
    keep_alive_default = pick("keep_alive_default", "5m")

    upstream_base_url = pick("upstream_base_url")
    upstream_chat_path = pick("upstream_chat_path", "/chat/completions")
    upstream_completions_path = pick("upstream_completions_path", "/completions")
    upstream_models_path = pick("upstream_models_path", "/models")
    upstream_embeddings_path = pick("upstream_embeddings_path", "/embeddings")
    upstream_api_key = pick("upstream_api_key")

    timeout = pick("timeout", 180.0)
    default_model = pick("default_model")

    if not isinstance(listen_host, str) or not listen_host.strip():
        raise SystemExit("Invalid listen_host")
    try:
        listen_port = int(listen_port)
    except (TypeError, ValueError) as exc:
        raise SystemExit("Invalid listen_port") from exc
    if listen_port <= 0 or listen_port > 65535:
        raise SystemExit("Invalid listen_port")

    if not isinstance(ollama_version, str) or not ollama_version.strip():
        raise SystemExit("Invalid ollama_version")

    try:
        stream_default = parse_boolish(stream_default_raw, "stream_default")
    except ValueError as exc:
        raise SystemExit(str(exc))

    if not isinstance(upstream_base_url, str) or not upstream_base_url.strip():
        raise SystemExit("Missing upstream_base_url (set in --config JSON or --upstream-base-url)")

    if not isinstance(upstream_chat_path, str) or not upstream_chat_path.strip():
        raise SystemExit("Invalid upstream_chat_path")
    if not isinstance(upstream_completions_path, str) or not upstream_completions_path.strip():
        raise SystemExit("Invalid upstream_completions_path")
    if not isinstance(upstream_models_path, str) or not upstream_models_path.strip():
        raise SystemExit("Invalid upstream_models_path")
    if not isinstance(upstream_embeddings_path, str) or not upstream_embeddings_path.strip():
        raise SystemExit("Invalid upstream_embeddings_path")

    try:
        timeout = float(timeout)
    except (TypeError, ValueError) as exc:
        raise SystemExit("Invalid timeout") from exc
    if timeout <= 0:
        raise SystemExit("Invalid timeout: must be > 0")

    if default_model is not None and not isinstance(default_model, str):
        default_model = str(default_model)
    if upstream_api_key is not None and not isinstance(upstream_api_key, str):
        upstream_api_key = str(upstream_api_key)

    try:
        insecure_cfg = parse_boolish(json_config.get("insecure_skip_tls_verify", False), "insecure_skip_tls_verify")
        tags_caps_cfg = parse_boolish(json_config.get("tags_include_capabilities", True), "tags_include_capabilities")
    except ValueError as exc:
        raise SystemExit(str(exc))

    insecure_skip_tls_verify = bool(args.insecure_skip_tls_verify) or insecure_cfg
    tags_include_capabilities = tags_caps_cfg

    tls_context: ssl.SSLContext | None = None
    if insecure_skip_tls_verify:
        tls_context = ssl._create_unverified_context()

    return Config(
        listen_host=listen_host,
        listen_port=listen_port,
        ollama_version=ollama_version,
        stream_default=stream_default,
        keep_alive_default=keep_alive_default,
        upstream_base_url=upstream_base_url,
        upstream_chat_path=upstream_chat_path,
        upstream_completions_path=upstream_completions_path,
        upstream_models_path=upstream_models_path,
        upstream_embeddings_path=upstream_embeddings_path,
        upstream_api_key=upstream_api_key,
        timeout=timeout,
        default_model=default_model,
        model_map=model_map,
        static_models=static_models,
        default_capabilities=default_capabilities,
        model_capabilities=model_capabilities,
        tags_include_capabilities=tags_include_capabilities,
        extra_headers=extra_headers,
        tls_context=tls_context,
    )


def main() -> int:
    args = parse_args()
    config = build_config(args)

    OllamaFacadeHandler.config = config
    OllamaFacadeHandler.registry = ModelRegistry()

    server = ThreadingHTTPServer((config.listen_host, config.listen_port), OllamaFacadeHandler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
