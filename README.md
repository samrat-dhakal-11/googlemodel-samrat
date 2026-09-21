# 🚀 googlemodel-samrat

> **Intelligent Gemini model discovery, multi-key rotation, automatic model fallback, and LangChain integration for Python.**

`googlemodel-samrat` is a Python library designed to make working with the **Google Gemini ecosystem** more resilient and convenient.

It provides utilities for selecting the latest available Gemini models, rotating between multiple API keys, falling back between models when errors occur, and integrating Gemini models into **LangChain-based applications**.

The library is particularly useful for applications that need to handle API quota limits, temporary server failures, model availability changes, and multiple Gemini models without manually implementing complex fallback logic.

---

## ✨ Features

### 🔑 Automatic API Key Rotation

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

This helps applications remain operational when an individual API key reaches its available quota.

---

### 🤖 Smart Model Fallback

The library maintains model lists ordered from **latest to oldest**.

When a model becomes unavailable or encounters a model-specific error, the system can move through the configured model list instead of immediately terminating the request.

```text
Latest Model
     ↓
Model Error
     ↓
Next Model
     ↓
Model Error
     ↓
Older Stable Model
     ↓
Successful Response
```

---

### 🔄 Intelligent Error Classification

The library is designed to distinguish between different types of failures.

| Error Type                     | Typical Response        |
| ------------------------------ | ----------------------- |
| `429` / Quota                  | Rotate API key          |
| Model unavailable / deprecated | Rotate model            |
| `502` / `503`                  | Rotate key and/or model |
| Successful request             | Continue normally       |

This provides a more resilient request strategy than relying on a single API key and model.

---

### 🦜 LangChain Integration

Designed to work with the LangChain Gemini ecosystem and provide a convenient interface for conversational applications.

Example:

```python
from googlemodel_samrat import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI()

response = llm.invoke("What is the capital of Nepal?")

print(response.content)
```

---

# 📦 Installation

Install the published package from PyPI:

```bash
pip install googlemodel-samrat
```

> **Note:** `googlemodel-samrat` is the PyPI distribution name. The Python import namespace is `googlemodel_samrat`.

After installation:

```python
import googlemodel_samrat
```

---

# 🧩 Model Categories

`googlemodel-samrat` organizes Gemini-related models into **12 categories**.

Each category provides a constant containing its model list and, where applicable, a helper function for retrieving the highest-priority model.

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

## 1. 💬 Chat Models

**Constant:** `CHAT_MODELS`

Primary conversational and multimodal models.

```text
gemini-3.8-flash
gemini-3.5-flash
gemini-3.1-pro-preview
gemini-3-flash-preview
gemini-2.5-pro
gemini-2.5-flash
gemini-2.5-flash-lite
```

Get the highest-priority model:

```python
from googlemodel_samrat import chatmodel

model = chatmodel()

print(model)
```

---

## 2. 📝 Text Models

**Constant:** `TEXT_MODELS`

Text-centric and legacy text endpoints.

```text
gemini-1.5-pro
gemini-1.5-flash
gemini-pro
```

---

## 3. 🎙️ Audio Models

**Constant:** `AUDIO_MODELS`

Models intended for real-time audio, transcription, and speech generation.

```text
gemini-3.8-live
gemini-3.8-live-extended-thinking
gemini-3.5-transcribe
gemini-3.1-flash-live-preview
gemini-3.1-flash-tts-preview
gemini-2.5-flash-native-audio-preview-12-2025
gemini-2.5-flash-preview-tts
gemini-2.5-pro-preview-tts
```

Get the highest-priority audio model:

```python
from googlemodel_samrat import audiomodel

print(audiomodel())
```

---

## 4. 🖼️ Image Models

**Constant:** `IMAGE_MODELS`

Visual generation models.

```text
gemini-3.1-flash-image
gemini-3.1-flash-lite-image
gemini-3-pro-image
```

Get the highest-priority image model:

```python
from googlemodel_samrat import imagemodel

print(imagemodel())
```

---

## 5. 🎬 Video Models

**Constant:** `VIDEO_MODELS`

Video generation and processing models.

```text
veo-3.1-generate-preview
veo-3.1-lite-generate-preview
```

Get the highest-priority video model:

```python
from googlemodel_samrat import videomodel

print(videomodel())
```

---

## 6. 🔢 Embedding Models

**Constant:** `EMBEDDING_MODELS`

Embedding models for semantic search, RAG systems, vector databases, and similarity applications.

```text
gemini-embedding-2-preview
gemini-embedding-001
```

Get the highest-priority embedding model:

```python
from googlemodel_samrat import embeddingmodel

print(embeddingmodel())
```

---

## 7. 🎵 Music Models

**Constant:** `MUSIC_MODELS`

Specialized music-generation models.

```text
music-fx-001
lyria-preview
```

Get the highest-priority music model:

```python
from googlemodel_samrat import musicmodel

print(musicmodel())
```

---

## 8. 🤖 Robotics Models

**Constant:** `ROBOTICS_MODELS`

Models designed for robotics and embodied reasoning applications.

```text
gemini-robotics-er-2-preview
gemini-robotics-er-1.6-preview
```

Get the highest-priority robotics model:

```python
from googlemodel_samrat import roboticsmodel

print(roboticsmodel())
```

---

## 9. 🖥️ Computer Use Models

**Constant:** `COMPUTER_USE_MODELS`

Models intended for UI navigation and computer interaction.

```text
gemini-computer-use-preview
gemini-desktop-agent-001
```

Get the highest-priority computer-use model:

```python
from googlemodel_samrat import computerusemodel

print(computerusemodel())
```

---

## 10. 🔬 Research Models

**Constant:** `RESEARCH_MODELS`

Models intended for deep analysis and research workflows.

```text
gemini-3.1-pro-preview
gemini-deep-research-1.0
```

Get the highest-priority research model:

```python
from googlemodel_samrat import researchmodel

print(researchmodel())
```

---

## 11. 🧠 Agent Models

**Constant:** `AGENT_MODELS`

Models intended for multi-step workflows and agent-based applications.

```text
gemini-3.8-flash
gemini-agent-engine-001
```

Get the highest-priority agent model:

```python
from googlemodel_samrat import agentmodel

print(agentmodel())
```

---

## 12. 🦙 Gemma Models

**Constant:** `GEMMA_MODELS`

Open-weight Gemma models for local deployment and customized applications.

```text
gemma-4
gemma-3-27b
gemma-3-9b
gemma-2-2b
```

Get the highest-priority Gemma model:

```python
from googlemodel_samrat import gemmamodel

print(gemmamodel())
```

---

# 🚀 Quick Start

## Get the Latest Model From Every Category

You can import the model getters and dynamically select the highest-priority model for each modality.

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

The model getter can be used directly with `ChatGoogleGenerativeAI`.

```python
import os

from dotenv import load_dotenv
from googlemodel_samrat import chatmodel
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model=chatmodel(),
    api_key=api_key,
)

response = llm.invoke(
    "What is the capital of Nepal?"
)

print(response.content)
```

---

# 🔢 Using Embeddings

The same approach can be used for Gemini embeddings.

```python
import os

from dotenv import load_dotenv
from googlemodel_samrat import embeddingmodel
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

embedding_model = GoogleGenerativeAIEmbeddings(
    model=embeddingmodel(),
    api_key=api_key,
)

embedding = embedding_model.embed_query(
    "My name is Samrat Dhakal."
)

print(embedding[:5])
```

---

# 🔐 Environment Variables

For applications that use a single API key, store your key in an environment variable rather than hard-coding it.

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

Then load it:

```python
from dotenv import load_dotenv

load_dotenv()
```

### ⚠️ Security

**Never commit API keys to GitHub or publish them inside source code.**

Add `.env` to `.gitignore`:

```gitignore
.env
```

If an API key is accidentally exposed, revoke it and create a replacement key.

---

# 🔄 Multi-Key Rotation

Applications that use multiple Gemini API keys can configure a key pool.

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

The library can rotate between the configured keys when supported failures occur.

> **Security:** Never publish real API keys in README files, GitHub repositories, screenshots, or package source code.

---

# 💬 Multi-Turn Conversations

Because the package is designed around the LangChain ecosystem, it can be used with LangChain message objects.

```python
from googlemodel_samrat import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatGoogleGenerativeAI(
    api_keys=[
        "YOUR_API_KEY_1",
        "YOUR_API_KEY_2",
    ]
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

response = llm.generate_messages(
    conversation_history
)

print(response)
```

---

# 📊 Rotation & Failover Statistics

The rotation system can expose statistics about the current key/model state.

```python
from googlemodel_samrat import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    api_keys=[
        "YOUR_API_KEY_1",
        "YOUR_API_KEY_2",
    ]
)

llm.invoke("Test query")

stats = llm.get_rotation_stats()

print(stats)
```

Example structure:

```python
{
    "total_keys": 2,
    "total_models": 15,
    "failed_keys": 0,
    "failed_models": 0,
    "available_combinations": 30,
    "current_key_index": 0,
    "current_model_index": 0,
}
```

---

# 🏗️ How the Rotation System Works

The core idea is to treat API keys and models as a pool of available request combinations.

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

When a request fails because of a supported quota, model, or server issue, the library can move to another available combination.

This allows applications to continue operating without manually implementing every fallback path.

---

# 🧱 Architecture

```text
                    ┌──────────────────────┐
                    │   Application/User   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ googlemodel-samrat   │
                    └──────────┬───────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
          ┌──────────┐  ┌────────────┐  ┌─────────────┐
          │ API Keys │  │ Model Pool │  │ Error Logic │
          └────┬─────┘  └─────┬──────┘  └──────┬──────┘
               │              │                │
               └──────────────┼────────────────┘
                              ▼
                    ┌──────────────────┐
                    │ Gemini / LangChain│
                    └──────────────────┘
```

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

### API Quotas

API key rotation does **not** remove Google's API quotas or usage policies. It only provides application-level handling for multiple configured keys.

### Model Availability

Google may introduce, rename, replace, deprecate, or remove models.

The model lists included in this package should therefore be treated as a snapshot/configuration rather than a guarantee that every listed endpoint will remain available indefinitely.

### Preview Models

Models containing identifiers such as:

```text
-preview
```

may change or become unavailable as their lifecycle progresses.

### API Compatibility

Not every model supports every Gemini API capability. A model listed in a category should not automatically be assumed to support every LangChain operation.

---

# 📋 Requirements

The package is designed to work with the Google Gemini and LangChain ecosystem.

Typical dependencies include:

```text
langchain-google-genai
google-genai
google-api-core
langchain-core
python-dotenv
```

Install or update the relevant dependencies with:

```bash
pip install -U googlemodel-samrat
```

---

# 🧪 Development

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd googlemodel-samrat
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install the project:

```bash
pip install -e .
```

---

# 📦 Publishing

Build the package:

```bash
python -m build
```

This produces:

```text
dist/
├── googlemodel_samrat-<version>.tar.gz
└── googlemodel_samrat-<version>-py3-none-any.whl
```

Upload to PyPI using your preferred publishing workflow.

> **Never place PyPI API tokens directly inside shell history, README files, source code, or public repositories.**

---

# 🗺️ Roadmap

Potential future improvements include:

* [ ] Automatic model-list synchronization
* [ ] Automatic Gemini API model discovery
* [ ] Persistent key health tracking
* [ ] Configurable retry policies
* [ ] Async API support
* [ ] Streaming support
* [ ] Better telemetry and diagnostics
* [ ] Model capability detection
* [ ] Automatic deprecated-model removal
* [ ] Configuration through `.env`
* [ ] CLI utilities
* [ ] Expanded test coverage
* [ ] Documentation website

---

# 🤝 Contributing

Contributions, issues, and suggestions are welcome.

A typical contribution workflow:

```bash
git checkout -b feature/my-feature
```

Make your changes, test them, and submit a pull request.

When reporting an issue, include:

* Python version
* Package version
* Operating system
* Model being used
* Relevant error message
* Minimal reproducible example

**Never include API keys or other credentials in an issue report.**

---

# 📄 License

Add your project's license here.

Example:

```text
MIT License
```

If your project uses a different license, replace the above with the appropriate license information.

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

## 🚀 Quick Reference

### Install

```bash
pip install googlemodel-samrat
```

### Get the latest chat model

```python
from googlemodel_samrat import chatmodel

print(chatmodel())
```

### Get the latest embedding model

```python
from googlemodel_samrat import embeddingmodel

print(embeddingmodel())
```

### Use with LangChain

```python
from googlemodel_samrat import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI()

response = llm.invoke("Hello, Gemini!")

print(response.content)
```

---

> **googlemodel-samrat — making Gemini model selection and fallback simpler for Python developers.**
