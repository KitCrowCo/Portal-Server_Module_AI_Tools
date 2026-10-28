# AI Tools

A suite of AI-assisted modules built on Portal Server's `ai_manager` tool. Everything here talks to your own connections (Ollama, LightRAG, etc.) - no data leaves your server unless a connection you've configured points somewhere external.

## What's included

- **Athena** - managed chat. An admin defines "capabilities" (a label + connection + model + system prompt + optional knowledge base or pipeline) that end users pick from by name, never by raw model name. Built for handing to non-technical users.
- **Kimi** - knowledge base management for LightRAG. Upload/select source files, ingest them into a knowledge group, query with citations, browse the resulting entity graph. Includes batch tools (chunk/combine) for preparing large source sets without hand-editing files.
- **Tessa** - a document + pipeline workspace. Edit a document with AI assistance, build multi-step pipelines (retrieval, generation, branching, sub-pipelines), and review every AI-proposed file change through shadow-staging before it touches anything real.
- **Image** - Flux.2 Klein image generation and inpainting against a dedicated GPU node. (Requires currently unreleased modification to Flux2 interface files)

## Requirements

Each sub-tool needs at least one connection configured (AI Tools -> Settings -> Connections) before it does anything useful: i.e. an Ollama/vLLM connection for chat/generation, a LightRAG connection for knowledge features.

## License

Licensed under the [Polyform Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) - free for personal, academic, and non-commercial use. Commercial use requires registration under the Fair-Share Economic Protocol (FSEP) when available - see `FSEP.md` at the repository root.