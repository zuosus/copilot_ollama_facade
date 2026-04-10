# ollama-facade

`ollama-facade` 是一个本地 `python3` 代理：

- 对外暴露 Ollama 兼容接口（默认 `http://127.0.0.1:11434`）
- 对内转发到 OpenAI 兼容上游（第三方）

目标场景：让 VSCode 继续按 Ollama 本地地址访问，但实际走第三方模型。

## 0.6.4 兼容能力

本项目按 `Ollama v0.6.4` 方向补齐了关键能力，包含你提到的 `capabilities`：

- `show` 响应返回 `capabilities`
- `tags` 响应可返回 `capabilities`（默认开启，可配置关闭）
- 能力校验：
  - `chat` 需要 `completion`，且带 `tools` 时还需要 `tools`
  - `generate` 需要 `completion`，带 `suffix` 时还需要 `insert`
- `chat/generate` 支持 `load/unload` 语义：
  - 空请求 + `keep_alive=0` 返回 `done_reason=unload`
  - 空请求返回 `done_reason=load`

支持的能力枚举（与 0.6.4 一致）：

- `completion`
- `tools`
- `insert`
- `vision`
- `embedding`

## 已实现端点

### Ollama API

- `HEAD /`
- `GET /`
- `HEAD /api/version`
- `GET /api/version`
- `HEAD /api/tags`
- `GET /api/tags`
- `GET /api/ps`
- `POST /api/show`
- `POST /api/chat`
- `POST /api/generate`
- `POST /api/embed`
- `POST /api/embeddings`
- `POST /api/create`
- `POST /api/copy`
- `POST /api/pull`
- `POST /api/push`
- `DELETE /api/delete`
- `POST /api/delete`（兼容调用）
- `HEAD /api/blobs/:digest`
- `POST /api/blobs/:digest`

### OpenAI 兼容入口（与 0.6.4 暴露路径一致）

- `GET /v1/models`
- `GET /v1/models/:model`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/embeddings`

## 快速开始

```bash
cd /path/to/copilot_ollama-facade
cp config.example.json config.json
# 修改 config.json
python3 ollama_facade.py --config ./config.json
```

## 配置示例

参考 [config.example.json](/path/to/copilot_ollama_facade/config.example.json)：

```json
{
  "listen_host": "127.0.0.1",
  "listen_port": 11434,
  "ollama_version": "0.6.4",
  "stream_default": true,
  "keep_alive_default": "5m",
  "upstream_base_url": "https://your-third-party.example.com/v1",
  "upstream_api_key": "YOUR_API_KEY",
  "upstream_chat_path": "/chat/completions",
  "upstream_completions_path": "/completions",
  "upstream_models_path": "/models",
  "upstream_embeddings_path": "/embeddings",
  "default_model": "thirdparty-chat-large",
  "default_capabilities": ["completion"],
  "model_map": {
    "llama3.1": "thirdparty-chat-large",
    "qwen2.5-coder": "thirdparty-coder",
    "bge-m3": "thirdparty-embedding"
  },
  "model_capabilities": {
    "llama3.1": ["completion", "tools", "insert"],
    "qwen2.5-coder": ["completion", "tools", "insert"],
    "bge-m3": ["embedding"]
  },
  "models": ["llama3.1", "qwen2.5-coder", "bge-m3"],
  "tags_include_capabilities": true,
  "headers": {},
  "timeout": 180,
  "insecure_skip_tls_verify": false
}
```

## 关键配置项

- `upstream_base_url`：上游基础地址（必填）
- `upstream_chat_path`：上游 chat 路径（默认 `/chat/completions`）
- `upstream_completions_path`：上游 completions 路径（默认 `/completions`）
- `upstream_embeddings_path`：上游 embeddings 路径（默认 `/embeddings`）
- `default_model`：请求未传 model 时的默认模型
- `model_map`：本地模型名到上游模型名映射
- `default_capabilities`：默认能力
- `model_capabilities`：按模型覆盖能力
- `tags_include_capabilities`：`/api/tags` 是否附带 `capabilities`
- `stream_default`：请求未传 `stream` 时默认是否流式
- `keep_alive_default`：请求未传 `keep_alive` 时默认值
- `ollama_version`：`/api/version` 返回值（建议 `0.6.4` 及以上）

## 基本验证

```bash
curl http://127.0.0.1:11434/api/version
curl http://127.0.0.1:11434/api/tags
curl http://127.0.0.1:11434/api/show -H 'Content-Type: application/json' -d '{"model":"llama3.1"}'
```

```bash
curl http://127.0.0.1:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama3.1","messages":[{"role":"user","content":"你好"}],"stream":false}'
```
