# MoCHi Helcyon Modal deployment

Standalone Modal deployment for the Helcyon 4o 12B Q5_K_M model used by MoCHi's Tomio Labs provider. The repository contains only the model-serving application and its deployment documentation. It does not contain model weights or credentials.

## What is implemented

- Downloads the pinned GGUF artifact into a persistent Modal Volume when the preload function is run.
- Verifies the expected file size and SHA-256 before the artifact is made available to the serving container.
- Runs the official CUDA-enabled `llama.cpp` server image with full GPU offload on an NVIDIA L4.
- Exposes the standard OpenAI-compatible `/v1/models` and `/v1/chat/completions` routes, including streaming chat completions.
- Enforces a server-side Bearer API key through a Modal Secret. The key is written to a permission-restricted file and is not placed in the process command line.

## Deployment configuration

| Setting | Value |
| --- | --- |
| Modal app | `mochi-helcyon` |
| Modal Volume | `helcyon-models` |
| Modal Secret | `helcyon-api` |
| Model source | `XeyonAI/MN-Helcyon-GPT-4o-12b-v4.5-GGUF` |
| Source revision | `42d5377229e20e9fd79ca13e49f9d778f8d1c8d9` |
| Artifact | `helcyon-gpt-4o-v4.5-Q5_K_M.gguf` |
| Artifact size | 8,727,634,272 bytes |
| Artifact SHA-256 | `510487dc7c24818066c7e009c183c6862292eb79a59db9fe7ec79969af47ecb5` |
| Runtime | `llama.cpp` CUDA server |
| GPU | NVIDIA L4 |
| Context / maximum output | 16,384 / 8,192 tokens |
| Scaling | `min_containers=0`, `max_containers=1`, 15-minute scaledown window |

The CUDA server image is pinned by digest in `modal/app.py` for reproducible deployment. Model weights remain in the Volume and are not committed to Git.

## API

After deployment, Modal prints an HTTPS endpoint. Use that value as `<modal-endpoint>`; no endpoint is hard-coded in this repository.

- `GET /health` reports llama.cpp readiness.
- `GET /v1/models` lists the served model.
- `POST /v1/chat/completions` accepts OpenAI-compatible chat completion requests.
- Set `"stream": true` to receive Server-Sent Events from llama.cpp.
- Send `Authorization: Bearer <server-managed-key>` for the authenticated model routes.

Example request shape:

```json
{
  "model": "helcyon-4o-12b-v4.5",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": true
}
```

## Request flow

```text
MoCHi client
  -> MoCHi server-side provider route
  -> authenticated Modal HTTPS endpoint
  -> llama.cpp CUDA server
  -> Helcyon GGUF in the Modal Volume
  -> OpenAI-compatible response or SSE stream
```

The API key belongs in the Modal Secret and in MoCHi's server-side provider configuration. It must never be placed in browser-visible environment variables or client JavaScript.

## Run and deploy

Install and authenticate the Modal CLI, then create the named Secret using your local secret-manager workflow. Do not commit the value or paste it into source control. The variable expected by the server is `HELCYON_API_KEY`.

```bash
python -m pip install --upgrade modal
modal setup
modal secret create helcyon-api HELCYON_API_KEY="<value supplied securely>"
modal run modal/app.py
modal deploy modal/app.py
```

`modal run` invokes the preload function and populates the persistent Volume. The deploy command publishes the authenticated server. The serving container refuses to start if the verified artifact is absent or the secret is missing.

## Performance notes

This repository does not include a reproducible Helcyon benchmark run. Cold-start latency depends on Modal scheduling, image startup, and llama.cpp model loading; warm time-to-first-token and generation throughput depend on prompt length and sampling settings. Measure those values against the deployed endpoint before publishing numbers.

## Security and release notes

- No API keys, Modal URLs, user data, or model weights are tracked.
- `.env.example` documents the required variable without containing a secret and is intentionally the only `.env*` file allowed by `.gitignore`.
- Keep the Modal endpoint behind MoCHi's server-side provider path if the model is intended for private beta access.
- Review the upstream model and runtime licenses before making a public repository or public service.
