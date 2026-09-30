# 🚀 googlemodel-samrat

**Intelligent Gemini model discovery, multi-key rotation, automatic model fallback, and LangChain integration for Python designed by Samrat Dhakal to help developers.**


\

`googlemodel-samrat` is a Python library designed  by Samrat Dhakal to make working with the Google Gemini ecosystem more resilient and convenient.

It provides utilities for:

* 🔎 Selecting available Gemini models
* 🔑 Rotating between multiple Gemini API keys
* 🔄 Automatically falling back between models
* ⏱️ Handling quota and rate-limit cooldowns
* 🦜 Integrating Gemini models with LangChain
* 📊 Inspecting rotation and failure statistics
* ⚡ Supporting streaming and asynchronous requests

The library is particularly useful for applications that need to handle API quota limits, temporary server failures, model availability changes, and multiple Gemini models without manually implementing complex fallback logic.

---

## 🆕 What's New in v0.2.0

### 🔄 Rotation Follows the Model Name

`chatmodel()` and `embeddingmodel()` now return a `RotatingModelName`.

It behaves like a normal Python `str` while also carrying the complete fallback pool.

```python
from googlemodel_samrat import ChatGoogleGenerativeAI, chatmodel

llm = ChatGoogleGenerativeAI(
    api_key="YOUR_API_KEY",
    model=chatmodel(),
)
```

The model name remains fully compatible with normal string operations:

```python
isinstance(chatmodel(), str)  # True

len(chatmodel())
chatmodel().upper()
f"{chatmodel()}"
```

The client automatically detects the rotation pool and uses it for failover.

---

### 🦜 Drop-in LangChain Shim

Existing applications using `langchain_google_genai` can migrate with a single import change:

```python
# Before
from langchain_google_genai import ChatGoogleGenerativeAI

# After
from googlemodel_samrat.langchain import ChatGoogleGenerativeAI
```

The shim also exports:

```python
GoogleGenerativeAIEmbeddings
```

No monkey-patching is performed and the original `langchain_google_genai` package is not modified.

---

### ⏱️ Google `retryDelay` Support

When Google returns a `429` response containing a structured retry delay, the library honors the provided value.

For example, if Google requests a retry after approximately 25 seconds, the corresponding key/model pair is cooled for approximately that duration instead of always waiting for the full configured cooldown.

---

### 📊 Public Failure Introspection

Inspect failed key/model combinations directly:

```python
llm.failed_pairs
```

Reset all recorded failures:

```python
llm.reset_failures()
```

The client also provides a compact debugging representation:

```python
print(llm)
```

Example:

```text
ChatGoogleGenerativeAI(keys=2, models=12, last_model='gemini-3.5-flash-lite')
```

---

### 🤫 Cleaner Terminal Output

SDK notices and warnings can be suppressed at call sites.

By default, the library keeps terminal output clean. This behavior can be disabled with:

```python
quiet_sdk_warnings=False
```

---

### 🔢 GA-First Embedding Models

`EMBEDDING_MODELS` now prioritizes:

```text
gemini-embedding-001
gemini-embedding-2
gemini-embedding-2-preview
```

This places the GA embedding model first in the rotation order.

---

## 📜 Changelog

### v0.2.0 — Current

* Added `RotatingModelName`
* `chatmodel()` and `embeddingmodel()` now carry the full rotation pool
* Added `googlemodel_samrat.langchain` drop-in shim
* Added support for Google's structured `retryDelay`
* Fixed duplicate LangSmith root traces
* Fixed mid-stream retry duplication
* Added SDK warning suppression
* Reordered embedding models to prioritize GA models
* Reordered chat models
* Added `failed_pairs`
* Added `reset_failures()`
* Added compact `__repr__`
* Improved missing API key error messages
* Added first-class `api_key=` / `model=` aliases
* Added 191 unit tests
* Added 59-check stress test
* Verified `py.typed` support

### v0.1.5

* Added per `(api_key, model)` cooldowns for `429` errors
* Added UTC midnight quota reset
* Added visible rotation logs
* Added singular `api_key=` / `model=` arguments
* Reordered chat models
* Added pair cooldown information to statistics
* Added 163 unit tests
* Added 64-check stress test

### v0.1.4

* `ChatGoogleGenerativeAI` subclasses LangChain's `BaseChatModel`
* Added LCEL compatibility
* Added `.stream()` / `.astream()`
* Added `.ainvoke()` / `.astream()`
* Added thread-safe `RateLimitManager`
* Added `GoogleGenerativeAIEmbeddings`
* Added successful model/key attribution
* Added `AllResourcesExhaustedError`
* Added daily quota "midnight sleep"
* Added complete public type hints

### v0.1.3

* Initial public model registry
* 12 model categories
* Multi-key rotation on `429`
* Model fallback on `404`
* Initial `ChatGoogleGenerativeAI` wrapper

### v0.1.2

* Expanded model registry
* Added helper model getters

### v0.1.1

* Added initial `pyproject.toml`
* Added packaging configuration

### v0.1.0

* Initial proof-of-concept release

---

# ✨ Features

## 🔑 Automatic API Key Rotation

Use multiple Gemini API keys and automatically move to another key when the current key encounters quota or rate-limit errors.

```text
API Key 1
   ↓
429 / Quota Error
   ↓
API Key 2
   ↓
429 / Quota Error
   ↓
API Key 3
   ↓
Continue Request
```

This allows applications to continue operating when an individual configured API key reaches its available quota.

---

## 🤖 Smart Model Fallback

The library maintains model lists ordered from fast models to heavier models.

When a model becomes unavailable or encounters a model-specific error, the library moves through the configured model list instead of immediately terminating the request.

Example:

```text
gemini-3.5-flash-lite
        ↓
gemini-3.8-flash
        ↓
gemini-3.1-flash-lite
        ↓
gemini-3.1-pro-preview
        ↓
        ...
        ↓
Successful Response
```

---

## 🎯 Model-Aware Rotation

`chatmodel()` and `embeddingmodel()` return a `RotatingModelName`.

```python
from googlemodel_samrat import ChatGoogleGenerativeAI, chatmodel

llm = ChatGoogleGenerativeAI(
    api_key="YOUR_API_KEY",
    model=chatmodel(),
)
```

The returned value is still a normal string:

```python
isinstance(chatmodel(), str)
# True
```

The complete fallback pool is carried by the model name and automatically detected by the client.

---

## 🔄 Intelligent Error Classification

Different error types affect different resources.

| Error                          | Resource Cooled                   |
| ------------------------------ | --------------------------------- |
| `429` / per-minute quota       | `(key, model)` pair               |
| `429` / daily quota            | Pair until next `00:00 UTC`       |
| `429` with `retryDelay`        | Pair for Google's requested delay |
| Model unavailable / deprecated | Model                             |
| `502` / `503` / timeout        | API key                           |
| Successful request             | Nothing                           |

This resource-specific cooldown system allows the same key to serve another model and the same model to be served by another key.

---

## 📢 Visible Rotation Logs

Rotation is silent by default.

Enable verbose output with:

```python
from googlemodel_samrat import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    api_key="YOUR_API_KEY",
    verbose=True,
)

llm.invoke("Hello")
```

Example:

```text
Initializing gemini-3.5-flash-lite ......
Response from model: gemini-3.5-flash-lite
```

If the first model reaches its quota:

```text
Initializing gemini-3.5-flash-lite ......
[quota] gemini-3.5-flash-lite x key ...QZYg sleeping for 60s
Initializing gemini-3.8-flash ......
Response from model: gemini-3.8-flash
```

---

# 🦜 LangChain Integration

`googlemodel-samrat` is designed to work with LangChain and LCEL.

## Basic Usage

```python
from googlemodel_samrat import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI()

response = llm.invoke(
    "What is the capital of Nepal?"
)

print(response.content)
```

---

## LCEL

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = (
    PromptTemplate.from_template("Tell me about {topic}")
    | llm
    | StrOutputParser()
)

print(
    chain.invoke({"topic": "GenAI"})
)
```

---

## Streaming

```python
for chunk in llm.stream(
    "Write a haiku about mountains."
):
    print(
        chunk.content,
        end="",
        flush=True,
    )
```

---

## Async

```python
import asyncio

async def main():
    response = await llm.ainvoke("Say hi.")
    print(response.content)

asyncio.run(main())
```

---

# 📦 Installation

Install the published package from PyPI:

```bash
pip install googlemodel-samrat
```

> **Note:** The PyPI distribution name is `googlemodel-samrat`, while the Python import namespace is `googlemodel_samrat`.

### Development Dependencies

```bash
pip install "googlemodel-samrat[dev]"
```

### Upgrade

```bash
pip install -U googlemodel-samrat
```

---

# 🧩 Model Categories

`googlemodel-samrat` organizes Gemini-related models into 12 categories.

|  # | Category         | Helper               | Constant              | Purpose                            |
| -: | ---------------- | -------------------- | --------------------- | ---------------------------------- |
|  1 | 💬 Chat          | `chatmodel()`        | `CHAT_MODELS`         | Conversational and multimodal LLMs |
|  2 | 📝 Text          | —                    | `TEXT_MODELS`         | Text-focused and legacy models     |
|  3 | 🎙️ Audio        | `audiomodel()`       | `AUDIO_MODELS`        | Speech, transcription, and audio   |
|  4 | 🖼️ Image        | `imagemodel()`       | `IMAGE_MODELS`        | Image generation and visual models |
|  5 | 🎬 Video         | `videomodel()`       | `VIDEO_MODELS`        | Video generation and processing    |
|  6 | 🔢 Embedding     | `embeddingmodel()`   | `EMBEDDING_MODELS`    | Vector embeddings                  |
|  7 | 🎵 Music         | `musicmodel()`       | `MUSIC_MODELS`        | Music generation                   |
|  8 | 🤖 Robotics      | `roboticsmodel()`    | `ROBOTICS_MODELS`     | Robotics and embodied reasoning    |
|  9 | 🖥️ Computer Use | `computerusemodel()` | `COMPUTER_USE_MODELS` | UI and computer interaction        |
| 10 | 🔬 Research      | `researchmodel()`    | `RESEARCH_MODELS`     | Research and long-form analysis    |
| 11 | 🧠 Agent         | `agentmodel()`       | `AGENT_MODELS`        | Agents and autonomous workflows    |
| 12 | 🦙 Gemma         | `gemmamodel()`       | `GEMMA_MODELS`        | Open-weight Gemma models           |

---

# 📋 Model Registry

The registry lists models in rotation priority order.

## 1. 💬 Chat Models

**Constant:** `CHAT_MODELS`

```text
gemini-3.5-flash-lite
gemini-3.8-flash
gemini-3.1-flash-lite
gemini-3.1-pro-preview
gemini-2.5-flash-lite
gemini-3-flash-preview
gemini-2.5-flash
gemini-3.7-flash
gemini-3.5-flash
gemini-3.6-flash
gemini-2.5-pro
gemini-1.5-flash-latest
```

Get the default chat model:

```python
from googlemodel_samrat import chatmodel

print(chatmodel())
```

---

## 2. 📝 Text Models

**Constant:** `TEXT_MODELS`

Text-centric and legacy text endpoints.

`TEXT_MODELS` aliases the chat model list.

---

## 3. 🎙️ Audio Models

**Constant:** `AUDIO_MODELS`

```text
gemini-3.8-live
gemini-3.8-live-extended-thinking
gemini-3.5-live-translate-preview
gemini-3.1-flash-live-preview
gemini-3.1-flash-tts-preview
gemini-3.5-transcribe
gemini-3.5-transcribe-live
gemini-2.5-flash-native-audio-preview-12-2025
gemini-2.5-flash-preview-tts
gemini-2.5-pro-preview-tts
```

---

## 4. 🖼️ Image Models

**Constant:** `IMAGE_MODELS`

```text
gemini-3.1-flash-image
gemini-3.1-flash-lite-image
gemini-3-pro-image
gemini-2.5-flash-image
```

---

## 5. 🎬 Video Models

**Constant:** `VIDEO_MODELS`

```text
veo-3.1-generate-preview
veo-3.1-fast-generate-preview
veo-3.1-lite-generate-preview
gemini-omni-1.1-flash
```

---

## 6. 🔢 Embedding Models

**Constant:** `EMBEDDING_MODELS`

The registry prioritizes the GA embedding model:

```text
gemini-embedding-001
gemini-embedding-2
gemini-embedding-2-preview
```

Get the default embedding model:

```python
from googlemodel_samrat import embeddingmodel

print(embeddingmodel())
```

---

## 7. 🎵 Music Models

**Constant:** `MUSIC_MODELS`

```text
lyria-3.5
lyria-3-clip-preview
lyria-3-pro-preview
lyria-realtime-exp
```

---

## 8. 🤖 Robotics Models

**Constant:** `ROBOTICS_MODELS`

```text
gemini-robotics-er-2-preview
gemini-robotics-er-2-streaming-preview
```

---

## 9. 🖥️ Computer Use Models

**Constant:** `COMPUTER_USE_MODELS`

```text
gemini-2.5-computer-use-preview-10-2025
```

---

## 10. 🔬 Research Models

**Constant:** `RESEARCH_MODELS`

```text
deep-research-max-preview-04-2026
deep-research-preview-04-2026
```

---

## 11. 🧠 Agent Models

**Constant:** `AGENT_MODELS`

```text
antigravity-preview-09-2026
```

---

## 12. 🦙 Gemma Models

**Constant:** `GEMMA_MODELS`

```text
gemma-4-31b-it
gemma-4-26b-a4b-it
```

---

# 🚀 Quick Start

## Get the Latest Model From Every Category

```python
from googlemodel_samrat import (
    chatmodel,
    audiomodel,
    imagemodel,
    videomodel,
    embeddingmodel,
    musicmodel,
    roboticsmodel,
    computerusemodel,
    researchmodel,
    agentmodel,
    gemmamodel,
)

print(f"Chat:          {chatmodel()}")
print(f"Audio:         {audiomodel()}")
print(f"Image:         {imagemodel()}")
print(f"Video:         {videomodel()}")
print(f"Embedding:     {embeddingmodel()}")
print(f"Music:         {musicmodel()}")
print(f"Robotics:      {roboticsmodel()}")
print(f"Computer Use:  {computerusemodel()}")
print(f"Research:      {researchmodel()}")
print(f"Agent:         {agentmodel()}")
print(f"Gemma:         {gemmamodel()}")
```

---

# 🧠 Using Chat Models

Load your API key from `.env`:

```python
import os

from dotenv import load_dotenv
from googlemodel_samrat import (
    ChatGoogleGenerativeAI,
    chatmodel,
)

load_dotenv()

llm = ChatGoogleGenerativeAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    model=chatmodel(),
)

response = llm.invoke(
    "What is the capital of Nepal?"
)

print(response.content)
```

Because `chatmodel()` carries the complete fallback pool, the client can automatically rotate between configured models.

---

# 🔢 Using Embeddings

```python
import os

from dotenv import load_dotenv
from googlemodel_samrat import (
    GoogleGenerativeAIEmbeddings,
    embeddingmodel,
)

load_dotenv()

embedding_model = GoogleGenerativeAIEmbeddings(
    api_key=os.getenv("GEMINI_API_KEY"),
    model=embeddingmodel(),
)

embedding = embedding_model.embed_query(
    "My name is Samrat Dhakal."
)

print(embedding[:5])
```

---

# 🔐 Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_API_KEY2=your_backup_api_key_here
```

The second key is optional.

Load environment variables:

```python
from dotenv import load_dotenv

load_dotenv()
```

---

# ⚠️ Security

**Never commit API keys to GitHub or publish them inside source code.**

Add `.env` to `.gitignore`:

```gitignore
.env
```

If an API key is accidentally exposed:

1. Revoke the exposed key immediately.
2. Generate a replacement key.
3. Update your environment variables.

Never include credentials in:

* Source code
* README files
* Git commits
* Public repositories
* Issue reports
* Screenshots

---

# 🔄 Multi-Key Rotation

Multiple API keys can be supplied directly:

```python
from googlemodel_samrat import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    api_keys=[
        "YOUR_PRIMARY_API_KEY",
        "YOUR_BACKUP_API_KEY",
    ],
    temperature=0.8,
    max_output_tokens=500,
)

response = llm.invoke(
    "Write a creative science-fiction story opening."
)

print(response.content)
```

When a key reaches its quota, the library rotates to the next healthy `(key, model)` combination.

---

# 💬 Multi-Turn Conversations

The client supports LangChain message history:

```python
from googlemodel_samrat import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
)

llm = ChatGoogleGenerativeAI(
    api_key="YOUR_API_KEY"
)

conversation_history = [
    HumanMessage(
        content="Hi, I'm learning Python."
    ),
    AIMessage(
        content="That's awesome! How can I help you with Python today?"
    ),
    HumanMessage(
        content="Can you write a quick Hello World program?"
    ),
]

response = llm.invoke(
    conversation_history
)

print(response.content)
```

---

# 📊 Rotation & Failover Statistics

Inspect the current rotation state:

```python
from googlemodel_samrat import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    api_key="YOUR_API_KEY"
)

llm.invoke("Test query")

stats = llm.get_rotation_stats()

print(stats)
```

Example:

```python
{
    "total_keys": 2,
    "total_models": 12,
    "active_keys": 2,
    "active_models": 11,
    "cooling_keys": 0,
    "cooling_models": 1,
    "cooling_pairs": 1,
    "available_combinations": 22,
    "current_key_index": 1,
    "current_model_index": 5,
    "key_rotation_enabled": True,
    "model_rotation_enabled": True,
}
```

---

## Successful Model & Key Attribution

After a request:

```python
print(llm.last_successful_model)
print(llm.last_successful_key_index)
```

Example:

```text
gemini-3.5-flash-lite
0
```

---

## Inspect Failed Pairs

```python
print(llm.failed_pairs)
```

Example:

```python
[
    ("key1", "gemini-3.8-flash"),
]
```

Reset all recorded failures:

```python
llm.reset_failures()
```

Inspect the client:

```python
print(llm)
```

Example:

```text
ChatGoogleGenerativeAI(
    keys=2,
    models=12,
    last_model='gemini-3.5-flash-lite'
)
```

---

# 🏗️ How the Rotation System Works

The core idea is to treat API keys and models as a Cartesian pool of request combinations.

For example:

```text
Key 1 × Model 1
Key 1 × Model 2
Key 1 × Model 3
       ↓
Key 2 × Model 1
Key 2 × Model 2
Key 2 × Model 3
       ↓
Key 3 × Model 1
Key 3 × Model 2
Key 3 × Model 3
```

When a request fails:

```text
429 / quota
    ↓
(key, model) pair cools
```

```text
404 / model unavailable
    ↓
model cools for every key
```

```text
5xx / timeout
    ↓
key cools for every model
```

This means:

* A single key with multiple models can still rotate.
* Multiple keys with one model can still rotate.
* Multiple keys × multiple models can rotate across the entire pool.

---

# 🛠️ Intended Use Cases

`googlemodel-samrat` can be useful for:

* 🤖 AI chatbots
* 💬 Conversational applications
* 📚 RAG applications
* 🔎 Semantic search systems
* 🧠 AI agents
* 🖥️ Computer-use experiments
* 🎙️ Voice applications
* 🖼️ Image-generation workflows
* 🎬 Video-generation workflows
* 🧪 AI experimentation
* 🎓 Academic and student projects
* 🏗️ Prototypes requiring model fallback
* 🔄 Applications using multiple Gemini API keys

---

# ⚠️ Important Notes

## API Quotas

API key rotation does **not** remove Google's API quotas or usage policies.

It provides application-level handling for multiple configured API keys.

---

## Model Availability

Google may introduce, rename, replace, deprecate, or remove models.

The model lists included in this package should therefore be treated as a snapshot/configuration rather than a permanent guarantee that every listed endpoint will remain available.

---

## Preview Models

Models containing `-preview` may change or become unavailable as their lifecycle progresses.

---

## API Compatibility

Not every model supports every Gemini API capability.

A model appearing in a category should not automatically be assumed to support every LangChain operation.

---

## Client State

Rotation state is maintained per client instance and in memory.

This includes:

* Cooldowns
* Daily sleeps
* Failed pairs

Create the client once and reuse it across requests:

```python
llm = ChatGoogleGenerativeAI(...)
```

Avoid creating a new client inside every request handler.

State is not shared across processes. Multiple workers maintain their own cooldown tracking.

---

# 📋 Requirements

Typical dependencies include:

```text
langchain-google-genai>=4.0.0
python-dotenv>=1.0.0
google-api-core>=2.15.0
langchain-core>=0.2.0
pydantic>=2.0
```

Upgrade the package:

```bash
pip install -U googlemodel-samrat
```

---

# 🧪 Development

Clone the repository:

```bash
git clone https://github.com/samrat-dhakal-11/googlemodel-samrat.git
cd googlemodel-samrat
```

Create a virtual environment:

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest tests/ -v
```

Run verification:

```bash
python3 scripts/verify.py
```

Run the stress test:

```bash
python3 scripts/stress_test.py
```

Expected results from the current project configuration:

```text
pytest tests/ -q
→ 191 passed (offline)

python3 scripts/verify.py
→ all PASS / 0 FAIL (live optional)

python3 scripts/stress_test.py
→ 59 PASS (offline, ~0.5 s)
```

---

# 📦 Publishing

Clean previous build artifacts:

```bash
rm -rf dist/ build/ *.egg-info
```

Build the package:

```bash
python -m build
```

Validate the package:

```bash
twine check dist/*
```

Upload to PyPI:

```bash
twine upload dist/*
```

The build produces:

```text
dist/
├── googlemodel_samrat-<version>.tar.gz
└── googlemodel_samrat-<version>-py3-none-any.whl
```

> **Security:** Never place PyPI API tokens directly inside shell history, README files, source code, or public repositories. Use Twine's interactive authentication mechanism or another secure credential method.

---

# 🗺️ Roadmap

## Completed

* [x] Async API support
* [x] Streaming support
* [x] Expanded test coverage
* [x] Per-pair cooldowns
* [x] UTC midnight reset
* [x] Rotation follows the model name
* [x] `RotatingModelName`
* [x] Drop-in LangChain shim
* [x] Google `retryDelay` support
* [x] Public failure introspection
* [x] `failed_pairs`
* [x] `reset_failures()`

## Planned

* [ ] Automatic model-list synchronization
* [ ] Automatic Gemini API model discovery
* [ ] Persistent key health tracking
* [ ] Redis/file-backed health tracking
* [ ] Configurable retry policies
* [ ] Better telemetry and diagnostics
* [ ] Model capability detection
* [ ] Automatic deprecated-model removal
* [ ] `.env` configuration support
* [ ] CLI utilities
* [ ] Documentation website

---

# 🤝 Contributing

Create a feature branch:

```bash
git checkout -b feature/my-feature
```

Make your changes and add tests.

Run the test suite:

```bash
pytest tests/ -q
```

Commit your changes:

```bash
git commit -m "feat: my feature"
```

Push your branch:

```bash
git push origin feature/my-feature
```

---

## 🐛 Reporting Issues

When reporting an issue, include:

* Python version
* `googlemodel-samrat` version
* Operating system
* Model being used
* Relevant error message
* Minimal reproducible example

**Never include API keys, tokens, passwords, or other credentials in an issue report.**

---

# 📄 License

This project is licensed under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

---

# 👨‍💻 Author

**Samrat Dhakal**

Python • Generative AI • Gemini • LangChain • RAG

---

# ⭐ Support the Project

If you find `googlemodel-samrat` useful:

* ⭐ Star the repository
* 🐛 Report bugs
* 💡 Suggest improvements
* 🤝 Contribute improvements
* 📦 Share the package with other developers

---

# ⚡ Quick Reference

## Install

```bash
pip install googlemodel-samrat
```

## Get the Latest Chat Model

```python
from googlemodel_samrat import chatmodel

print(chatmodel())
```

## Get the Latest Embedding Model

```python
from googlemodel_samrat import embeddingmodel

print(embeddingmodel())
```

## Use with LangChain

```python
from googlemodel_samrat import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    api_key="YOUR_API_KEY"
)

response = llm.invoke(
    "Hello, Gemini!"
)

print(response.content)
```

## Migrate Existing `langchain_google_genai` Code

```python
from googlemodel_samrat.langchain import ChatGoogleGenerativeAI
```

## LCEL Chain

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = (
    PromptTemplate.from_template("Tell me about {topic}")
    | llm
    | StrOutputParser()
)

print(
    chain.invoke({"topic": "RAG"})
)
```

---

<div align="center">

### 🚀 googlemodel-samrat

**Making Gemini model selection, API-key rotation, and fallback simpler for Python developers.**

Made with ❤️ by **Samrat Dhakal**

</div>
