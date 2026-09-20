# Helcyon on Modal

Standalone Modal deployment source for MoCHi's server-managed Helcyon 4o 12B
model. The deployment uses the pinned Q5_K_M GGUF, llama.cpp, one NVIDIA L4,
and an authenticated OpenAI-compatible endpoint.

- Modal app: `mochi-helcyon`
- Volume: `helcyon-models`
- Modal Secret: `helcyon-api`, containing `HELCYON_API_KEY`
- Context: 16K; maximum output: 8K
- Scaling: `min_containers=0`, `max_containers=1`, 15-minute scale-down

Model weights are downloaded and checksum-verified into the Modal Volume by the
preload step. They are not stored in Git. Credentials and MoCHi/Cloudflare
secrets are also never stored here.

```bash
modal run modal/app.py
modal deploy modal/app.py
```

Configure the resulting endpoint and the matching server-side key in MoCHi's
Cloudflare Worker. Never expose the key in browser code and do not add a
recurring health poller.
