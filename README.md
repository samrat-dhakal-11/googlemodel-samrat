# 🚀 googlemodel-samrat

> **Intelligent Gemini model discovery, multi-key rotation, automatic model fallback, and LangChain integration for Python.**

`googlemodel-samrat` is a Python library designed to make working with the **Google Gemini ecosystem** more resilient and convenient.

It provides utilities for selecting the latest available Gemini models, rotating between multiple API keys, falling back between models when errors occur, and integrating Gemini models into **LangChain-based applications**.

The library is particularly useful for applications that need to handle API quota limits, temporary server failures, model availability changes, and multiple Gemini models without manually implementing complex fallback logic.

---

## 🆕 What's new in v0.1.5

- **Per (key, model) pair cooldowns** — a 429 on one pair no longer blocks the same key on other models, nor the same model on other keys. Rotation now works in *both* directions, even with a single key.
- **UTC midnight reset** — daily-quota 429s now sleep the pair until the next **00:00 UTC**, matching Google's actual reset window more closely for users worldwide.
- **Visible rotation logs** — `Initializing <model> ......` and `[quota] <model> x key ...XXXX sleeping for Ns` are printed on stdout, so users see exactly when rotation happens.
- **Singular `api_key=` / `model=`** — sugar for `api_keys=[...]` / `models=[...]`. Both forms work on `ChatGoogleGenerativeAI` and `GoogleGenerativeAIEmbeddings`.
- **`chatmodel()` now returns `gemini-3.5-flash-lite`** — fast, high-quota, low latency. Heavier models are fallbacks.
- **Alternating fast/heavy model priority** — the registry orders models as *fast → heavy → fast → heavy* so the first answer is snappy and the first fallback is still capable.
- **163 unit tests + 64-check stress test** — `pytest tests/ -q` runs in ~1 second, `python3 scripts/stress_test.py` runs 64 checks in ~0.5 s (offline).
- **Timezone-aware `datetime`** — no deprecation warnings on Python 3.12+.
- **`shortest_wait()` and `stats`** now account for pair cooldowns.

---

## 📜 Changelog

### v0.1.5 — *current*
- Per `(api_key, model)` pair cooldowns for 429s
- `_next_midnight_ts()` returns next 00:00 UTC (timezone-aware)
- Visible rotation logs on stdout (`Initializing ...`, `[quota] ...`, `Response from model: ...`)
- `api_key=` / `model=` singular sugar for both chat and embeddings
- `CHAT_MODELS` reordered: alternating fast/heavy, `gemini-3.5-flash-lite` first
- `shortest_wait()` and `stats` include pair cooldowns
- 163 unit tests; `scripts/stress_test.py` (64 checks)
- Type-safe: `py.typed`, fully annotated public API

### v0.1.4
- `ChatGoogleGenerativeAI` subclasses LangChain's `BaseChatModel` — **LCEL native** (`prompt | llm | parser`)
- `.stream()` / `.astream()` with mid-stream fallback
- `.ainvoke()` / `.astream()` async support
- Thread-safe `RateLimitManager` with timestamped per-resource cooldowns
- New `GoogleGenerativeAIEmbeddings` with rotation
- Attribution: `llm.last_successful_model`, `llm.last_successful_key_index`
- Friendly `AllResourcesExhaustedError` with troubleshooting tips
- "Midnight sleep" for daily-quota 429s
- `py.typed` + full type hints
- 36 unit tests + `scripts/verify.py`

### v0.1.3
- Initial public release of the model registry (12 categories, ~45 models)
- Multi-key rotation on 429
- Model fallback on 404
- Basic `ChatGoogleGenerativeAI` wrapper

### v0.1.2
- Registry expansion
- Helper getters (`chatmodel()`, `embeddingmodel()`, etc.)

### v0.1.1
- Initial `pyproject.toml` and packaging

### v0.1.0
- Proof-of-concept release

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
````


This helps applications remain operational when an individual API key reaches its available quota.

---

### 🤖 Smart Model Fallback

The library maintains model lists ordered from **fastest to heaviest** (alternating).

When a model becomes unavailable or encounters a model-specific error, the system moves through the configured model list instead of immediately terminating the request.


```
gemini-3.5-flash-lite     (fast, first choice)
    ↓
gemini-3.8-flash          (heavy, next)
    ↓
gemini-3.1-flash-lite     (fast)
    ↓
gemini-3.1-pro-preview    (heavy)
    ↓
...
    ↓
Successful Response
```


---

### 🔄 Intelligent Error Classification

The library distinguishes between different failure types and cools the appropriate resource.

| Error TypeWhat cools down      |                                |
| ------------------------------ | ------------------------------ |
| `429` / per-minute quota       | The `(key, model)` pair        |
| `429` / *daily* quota          | The pair, until next 00:00 UTC |
| Model unavailable / deprecated | The model only                 |
| `502` / `503` / timeout        | The key only                   |
| Successful request             | Nothing — normal flow          |

Cooling a **pair** means: the same key can immediately serve a *different* model, and the same model can immediately be served by a *different* key. This is what makes single-key multi-model rotation work.

---

### 📢 Visible Rotation Logs

By default, rotation is silent. With `verbose=True`, every fallback step is printed to stdout:


```
from googlemodel_samrat import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(api_key="...", verbose=True)
llm.invoke("Hello")
```


Output:


```
Initializing gemini-3.5-flash-lite ......
Response from model: gemini-3.5-flash-lite
```


If the first model is quota-capped:


```
Initializing gemini-3.5-flash-lite ......
[quota] gemini-3.5-flash-lite x key ...QZYg sleeping for 60s
Initializing gemini-3.8-flash ......
Response from model: gemini-3.8-flash
```


---

### 🦜 LangChain Integration

Designed to work with the LangChain Gemini ecosystem and provide a convenient interface for conversational applications.


```
from googlemodel_samrat import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI()

response = llm.invoke("What is the capital of Nepal?")

print(response.content)
```


Fully LCEL-compatible:


```
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = PromptTemplate.from_template("Tell me about {topic}") | llm | StrOutputParser()
print(chain.invoke({"topic": "GenAI"}))
```


---

# 📦 Installation

Install the published package from PyPI:


```
pip install googlemodel-samrat
```


> **Note:** `googlemodel-samrat` is the PyPI distribution name. The Python import namespace is `googlemodel_samrat`.

Optional dev dependencies:


```
pip install "googlemodel-samrat[dev]"
```


After installation:


```
import googlemodel_samrat
```


---

# 🧩 Model Categories

`googlemodel-samrat` organizes Gemini-related models into **12 categories**.

Each category provides a constant containing its model list and, where applicable, a helper function for retrieving the highest-priority model.

| #CategoryHelperConstantPurpose |                  |                      |                       |                                    |
| ------------------------------ | ---------------- | -------------------- | --------------------- | ---------------------------------- |
| 1                              | 💬 Chat          | `chatmodel()`        | `CHAT_MODELS`         | Conversational and multimodal LLMs |
| 2                              | 📝 Text          | —                    | `TEXT_MODELS`         | Text-focused and legacy models     |
| 3                              | 🎙️ Audio        | `audiomodel()`       | `AUDIO_MODELS`        | Speech, transcription, and audio   |
| 4                              | 🖼️ Image        | `imagemodel()`       | `IMAGE_MODELS`        | Image generation and visual models |
| 5                              | 🎬 Video         | `videomodel()`       | `VIDEO_MODELS`        | Video generation and processing    |
| 6                              | 🔢 Embedding     | `embeddingmodel()`   | `EMBEDDING_MODELS`    | Vector embeddings                  |
| 7                              | 🎵 Music         | `musicmodel()`       | `MUSIC_MODELS`        | Music generation                   |
| 8                              | 🤖 Robotics      | `roboticsmodel()`    | `ROBOTICS_MODELS`     | Robotics and embodied reasoning    |
| 9                              | 🖥️ Computer Use | `computerusemodel()` | `COMPUTER_USE_MODELS` | UI and computer interaction        |
| 10                             | 🔬 Research      | `researchmodel()`    | `RESEARCH_MODELS`     | Research and long-form analysis    |
| 11                             | 🧠 Agent         | `agentmodel()`       | `AGENT_MODELS`        | Agents and autonomous workflows    |
| 12                             | 🦙 Gemma         | `gemmamodel()`       | `GEMMA_MODELS`        | Open-weight Gemma models           |

---

# 📋 Model Registry

The registry lists are ordered by **rotation priority**, alternating fast and heavy models so that the first request is fast and the first fallback is still capable.

## 1. 💬 Chat Models

**Constant:** `CHAT_MODELS`


```
gemini-3.5-flash-lite       (fast, high RPD)
gemini-3.8-flash            (heavy)
gemini-3.1-flash-lite       (fast)
gemini-3.1-pro-preview      (heavy)
gemini-2.5-flash-lite       (fast)
gemini-3-flash-preview      (heavy)
gemini-2.5-flash            (fast)
gemini-3.7-flash            (heavy)
gemini-3.5-flash            (fast)
gemini-3.6-flash            (heavy)
gemini-2.5-pro              (heavy)
gemini-1.5-flash-latest     (last resort)
```


Get the highest-priority model:


```
from googlemodel_samrat import chatmodel
print(chatmodel())     # gemini-3.5-flash-lite
```


---

## 2. 📝 Text Models

**Constant:** `TEXT_MODELS` (aliases `CHAT_MODELS`)

Text-centric and legacy text endpoints.

---

## 3. 🎙️ Audio Models

**Constant:** `AUDIO_MODELS`


```
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



```
from googlemodel_samrat import audiomodel
print(audiomodel())
```


---

## 4. 🖼️ Image Models

**Constant:** `IMAGE_MODELS`


```
gemini-3.1-flash-image
gemini-3.1-flash-lite-image
gemini-3-pro-image
gemini-2.5-flash-image
```



```
from googlemodel_samrat import imagemodel
print(imagemodel())
```


---

## 5. 🎬 Video Models

**Constant:** `VIDEO_MODELS`


```
veo-3.1-generate-preview
veo-3.1-fast-generate-preview
veo-3.1-lite-generate-preview
gemini-omni-1.1-flash
```



```
from googlemodel_samrat import videomodel
print(videomodel())
```


---

## 6. 🔢 Embedding Models

**Constant:** `EMBEDDING_MODELS`


```
gemini-embedding-2-preview
gemini-embedding-2
gemini-embedding-001
```



```
from googlemodel_samrat import embeddingmodel
print(embeddingmodel())
```


---

## 7. 🎵 Music Models

**Constant:** `MUSIC_MODELS`


```
lyria-3.5
lyria-3-clip-preview
lyria-3-pro-preview
lyria-realtime-exp
```



```
from googlemodel_samrat import musicmodel
print(musicmodel())
```


---

## 8. 🤖 Robotics Models

**Constant:** `ROBOTICS_MODELS`


```
gemini-robotics-er-2-preview
gemini-robotics-er-2-streaming-preview
```



```
from googlemodel_samrat import roboticsmodel
print(roboticsmodel())
```


---

## 9. 🖥️ Computer Use Models

**Constant:** `COMPUTER_USE_MODELS`


```
gemini-2.5-computer-use-preview-10-2025
```



```
from googlemodel_samrat import computerusemodel
print(computerusemodel())
```


---

## 10. 🔬 Research Models

**Constant:** `RESEARCH_MODELS`


```
deep-research-max-preview-04-2026
deep-research-preview-04-2026
```



```
from googlemodel_samrat import researchmodel
print(researchmodel())
```


---

## 11. 🧠 Agent Models

**Constant:** `AGENT_MODELS`


```
antigravity-preview-09-2026
```



```
from googlemodel_samrat import agentmodel
print(agentmodel())
```


---

## 12. 🦙 Gemma Models

**Constant:** `GEMMA_MODELS`


```
gemma-4-31b-it
gemma-4-26b-a4b-it
```



```
from googlemodel_samrat import gemmamodel
print(gemmamodel())
```


---

# 🚀 Quick Start

## Get the Latest Model From Every Category


```
from googlemodel_samrat import (
    chatmodel, audiomodel, imagemodel, videomodel,
    embeddingmodel, musicmodel, roboticsmodel,
    computerusemodel, researchmodel, agentmodel, gemmamodel,
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


```
import os
from dotenv import load_dotenv
from googlemodel_samrat import ChatGoogleGenerativeAI, chatmodel

load_dotenv()

llm = ChatGoogleGenerativeAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    model=chatmodel(),
)

response = llm.invoke("What is the capital of Nepal?")
print(response.content)
```


With streaming:


```
for chunk in llm.stream("Write a haiku about mountains."):
    print(chunk.content, end="", flush=True)
```


With async:


```
import asyncio

async def main():
    msg = await llm.ainvoke("Say hi.")
    print(msg.content)

asyncio.run(main())
```


---

# 🔢 Using Embeddings


```
import os
from dotenv import load_dotenv
from googlemodel_samrat import GoogleGenerativeAIEmbeddings, embeddingmodel

load_dotenv()

embedding_model = GoogleGenerativeAIEmbeddings(
    api_key=os.getenv("GEMINI_API_KEY"),
    model=embeddingmodel(),
)

embedding = embedding_model.embed_query("My name is Samrat Dhakal.")
print(embedding[:5])
```


---

# 🔐 Environment Variables

Create a `.env` file:


```
GEMINI_API_KEY=your_api_key_here
GEMINI_API_KEY2=your_backup_api_key_here   # optional
```


Load it:


```
from dotenv import load_dotenv
load_dotenv()
```


### ⚠️ Security

**Never commit API keys to GitHub or publish them inside source code.**

Add `.env` to `.gitignore`:


```
.env
```


If a key is accidentally exposed, revoke it immediately and generate a replacement.

---

# 🔄 Multi-Key Rotation


```
from googlemodel_samrat import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    api_keys=[
        "YOUR_PRIMARY_API_KEY",
        "YOUR_BACKUP_API_KEY",
    ],
    temperature=0.8,
    max_output_tokens=500,
)

response = llm.invoke("Write a creative science-fiction story opening.")
print(response.content)
```


When a key hits quota, the library rotates to the next healthy `(key, model)` pair.

---

# 💬 Multi-Turn Conversations


```
from googlemodel_samrat import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatGoogleGenerativeAI(api_key="YOUR_API_KEY")

conversation_history = [
    HumanMessage(content="Hi, I'm learning Python."),
    AIMessage(content="That's awesome! How can I help you with Python today?"),
    HumanMessage(content="Can you write a quick Hello World program?"),
]

response = llm.invoke(conversation_history)
print(response.content)
```


---

# 📊 Rotation & Failover Statistics


```
from googlemodel_samrat import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(api_key="YOUR_API_KEY")
llm.invoke("Test query")

stats = llm.get_rotation_stats()
print(stats)
```


Example structure:


```
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


Attribution after every call:


```
print(llm.last_successful_model)         # e.g. "gemini-3.5-flash-lite"
print(llm.last_successful_key_index)     # e.g. 0
```


---

# 🏗️ How the Rotation System Works

The core idea is to treat API keys and models as a Cartesian pool of request combinations.


```
Key 1 × Model 1    ← cools together on 429 (per-pair)
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

- **429 (quota)** → the pair `(key, model)` cools. Every other combination stays available.
- **404 (model gone)** → the model cools for every key.
- **5xx / timeout** → the key cools for every model.

This means: a single key with N models still rotates; M keys with a single model still rotates; M keys × N models rotates in all directions.

---

# 🧱 Architecture


```
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

- 🤖 AI chatbots
- 💬 Conversational applications
- 📚 RAG applications
- 🔎 Semantic search systems
- 🧠 AI agents
- 🖥️ Computer-use experiments
- 🎙️ Voice applications
- 🖼️ Image-generation workflows
- 🎬 Video-generation workflows
- 🧪 AI experimentation
- 🎓 Academic and student projects
- 🏗️ Prototypes requiring model fallback
- 🔄 Applications using multiple Gemini API keys

---

# ⚠️ Important Notes

### API Quotas

API key rotation does **not** remove Google's API quotas or usage policies. It only provides application-level handling for multiple configured keys.

### Model Availability

Google may introduce, rename, replace, deprecate, or remove models.

The model lists in this package should be treated as a **snapshot/configuration**, not a guarantee that every listed endpoint will remain available indefinitely.

### Preview Models

Models containing `-preview` may change or become unavailable as their lifecycle progresses.

### API Compatibility

Not every model supports every Gemini API capability. A model listed in a category should not automatically be assumed to support every LangChain operation.

### Client State

Rotation state (cooldowns, daily sleeps, failed pairs) is **per client instance** and **in-memory**.

- Create the client **once** and reuse it across requests — do not construct `ChatGoogleGenerativeAI(...)` inside a request handler.
- State is **not** shared across processes. Multiple workers each keep their own cooldown tracking.

---

# 📋 Requirements

Typical dependencies:


```
langchain-google-genai>=4.0.0
python-dotenv>=1.0.0
google-api-core>=2.15.0
langchain-core>=0.2.0
pydantic>=2.0
```


Upgrade:


```
pip install -U googlemodel-samrat
```


---

# 🧪 Development


```
git clone https://github.com/samrat-dhakal-11/googlemodel-samrat.git
cd googlemodel-samrat

python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows

pip install -e ".[dev]"
pytest tests/ -v
python3 scripts/verify.py
python3 scripts/stress_test.py
```


Expected:

- `pytest tests/ -q` → **163 passed** (offline)
- `python3 scripts/verify.py` → **16 PASS / 0 FAIL** (live optional)
- `python3 scripts/stress_test.py` → **64 PASS** (offline, \~0.5 s)

---

# 📦 Publishing


```
rm -rf dist/ build/ *.egg-info
python -m build
twine check dist/*
twine upload dist/*
```


Produces:


```
dist/
├── googlemodel_samrat-<version>.tar.gz
└── googlemodel_samrat-<version>-py3-none-any.whl
```


> **Never place PyPI API tokens directly inside shell history, README files, source code, or public repositories.** Paste tokens only at `twine`'s interactive prompt.

---

# 🗺️ Roadmap

- ☑  

  Async API support
- ☑  

  Streaming support
- ☑  

  Expanded test coverage
- ☑  

  Per-pair cooldowns
- ☑  

  UTC midnight reset
- □  

  Automatic model-list synchronization
- □  

  Automatic Gemini API model discovery
- □  

  Persistent key health tracking (Redis / file)
- □  

  Configurable retry policies
- □  

  Better telemetry and diagnostics
- □  

  Model capability detection
- □  

  Automatic deprecated-model removal
- □  

  Configuration through `.env`
- □  

  CLI utilities
- □  

  Documentation website

---

# 🤝 Contributing


```
git checkout -b feature/my-feature
# ... make changes, add tests ...
pytest tests/ -q
git commit -m "feat: my feature"
git push origin feature/my-feature
```


When reporting an issue, include:

- Python version
- Package version
- Operating system
- Model being used
- Relevant error message
- Minimal reproducible example

**Never include API keys or other credentials in an issue report.**

---

# 📄 License

MIT License. See [LICENSE](https://license/) for details.

---

# 👨‍💻 Author

**Samrat Dhakal**

Python • Generative AI • Gemini • LangChain • RAG

---

# ⭐ Support the Project

If you find `googlemodel-samrat` useful:

- ⭐ Star the repository
- 🐛 Report bugs
- 💡 Suggest improvements
- 🤝 Contribute improvements
- 📦 Share the package with other developers

---

## 🚀 Quick Reference

### Install


```
pip install googlemodel-samrat
```


### Get the latest chat model


```
from googlemodel_samrat import chatmodel
print(chatmodel())     # gemini-3.5-flash-lite
```


### Get the latest embedding model


```
from googlemodel_samrat import embeddingmodel
print(embeddingmodel())     # gemini-embedding-2-preview
```


### Use with LangChain


```
from googlemodel_samrat import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(api_key="...")
response = llm.invoke("Hello, Gemini!")
print(response.content)
```


### LCEL chain


```
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = PromptTemplate.from_template("Tell me about {topic}") | llm | StrOutputParser()
print(chain.invoke({"topic": "RAG"}))
```


---

> **googlemodel-samrat — making Gemini model selection and fallback simpler for Python developers.**
