# 🗺️ googlemodel-samrat Roadmap

The roadmap for **`googlemodel-samrat`** is organized around a simple semantic versioning mental model.

## 🧠 Versioning Philosophy

We keep version numbers meaningful:

| Version | Meaning            | Typical Scope                                   | Release Cadence |
| ------- | ------------------ | ----------------------------------------------- | --------------- |
| `0.1.x` | 🐛 Bugfix          | One-off fixes, no new public API                | Days            |
| `0.x.0` | ✨ Feature          | New constructor parameters, methods, or modules | Weeks           |
| `x.0.0` | 💥 Breaking Change | Removed APIs or changed defaults                | Rarely          |

> **Principle:** Bugfix releases should remain low-risk, feature releases should be intentional, and breaking changes should be rare and clearly communicated.

---

# 🚀 v0.2.0 — SHIPPED

**v0.2.0 is released and includes the complete `0.1.6` bugfix batch.**

This release combines the original bugfix scope with the flagship `0.2.0` features.

---

## 🐛 Bugfixes

The following items were originally scoped for `0.1.6` and are now included in `v0.2.0`.

### Logging & Warnings

* [x] Silence Google SDK AFC logs across the full `google_genai.*` logger tree
* [x] Silence fixed-sampling `UserWarning` at call sites
* [x] `quiet_sdk_warnings()` wraps both construction and every inner call

### Model Registry

* [x] Reorder `EMBEDDING_MODELS`

  * `gemini-embedding-001` first
* [x] Reorder `CHAT_MODELS`

  * `gemini-3.5-flash-lite`
  * `gemini-3.1-flash-lite`
  * Flow models

### Retry & Rotation

* [x] Honor Google's `retryDelay` hint from `429` responses
* [x] Apply `min(hint + 1s, cooldown_seconds)`
* [x] Fix mid-stream retry duplication
* [x] Use first-chunk-only rotation to prevent already-yielded output from being duplicated

### LangSmith

* [x] Fix duplicate LangSmith root traces
* [x] Drive the inner client through protected `_generate()` / `_stream()`
* [x] Ensure a single root trace per invocation
* [x] Preserve correct token accounting

### Constructor Handling

* [x] Fix singular `api_key=` / `model=` values leaking into inner client kwargs

### Packaging & CI

* [x] Verify `py.typed`
* [x] Add `verify.py` package check
* [x] Add wheel verification in CI

### Public Diagnostics

* [x] Add public `failed_pairs` property to both clients
* [x] Add public `reset_failures()` proxy to both clients
* [x] Improve client `__repr__`
* [x] Improve `"no API key found"` `ConfigurationError` message

---

# ✨ v0.2.0 Flagship Features

The primary features introduced by `v0.2.0` are:

### 🎯 `RotatingModelName`

Rotation now follows the model name itself.

`chatmodel()` and `embeddingmodel()` return a model name that behaves like a normal `str` while carrying the complete rotation pool.

This allows rotation-aware clients to automatically discover and use the fallback pool.

```python
from googlemodel_samrat import (
    ChatGoogleGenerativeAI,
    chatmodel,
)

llm = ChatGoogleGenerativeAI(
    api_key="YOUR_API_KEY",
    model=chatmodel(),
)
```

---

### 🦜 LangChain Import Shim

Added:

```python
googlemodel_samrat.langchain
```

This enables a one-line migration from `langchain_google_genai`:

```python
# Before
from langchain_google_genai import ChatGoogleGenerativeAI

# After
from googlemodel_samrat.langchain import ChatGoogleGenerativeAI
```

---

### 🚫 No Automatic Monkey-Patching

`googlemodel-samrat` does **not** monkey-patch third-party libraries by default.

This behavior is explicitly covered by:

```text
test_no_monkey_patch_by_default
```

---

# 🔮 v0.2.x — Next Train

The next `0.2.x` releases focus on production capabilities, observability, shared state, and developer tooling.

---

## 🗄️ Shared Cooldown State

Add optional shared cooldown state across processes and workers.

### Planned API

```python
shared_state=...
```

### Planned Architecture

```text
StateBackend
├── In-Memory
├── File
└── Redis
```

* [ ] Redis-backed shared cooldown state
* [ ] File-backed shared cooldown state
* [ ] `StateBackend` protocol
* [ ] Opt-in `shared_state=` configuration

---

## 🌍 Timezone-Aware Daily Reset

Improve daily quota reset handling.

* [ ] Add `daily_reset_tz`
* [ ] Support Python `ZoneInfo`
* [ ] Example:

  ```python
  ZoneInfo("America/Los_Angeles")
  ```
* [ ] Add configurable UTC reset hour
* [ ] Add `GMS_DAILY_RESET_HOUR_UTC` environment variable override

---

## 📡 Rotation Events & Metrics

Add structured rotation events for production observability.

### Planned API

```python
RotationEvent
```

and:

```python
on_rotate=...
```

Use cases include:

* Monitoring
* Metrics
* Logging
* Alerting
* Production dashboards

---

## 📊 Structured Statistics

Replace the current dictionary-based statistics implementation with a frozen dataclass while maintaining backward compatibility.

### Planned API

```python
@dataclass(frozen=True)
class RotationStats:
    ...
```

Maintain compatibility through:

```python
stats.to_dict()
```

---

## 📦 Batch Operations

Add batch operations while preserving rotation behavior.

Planned APIs:

```python
.batch()
.abatch()
```

* [ ] `.batch()`
* [ ] `.abatch()`
* [ ] Preserve model rotation
* [ ] Preserve key rotation
* [ ] Preserve cooldown behavior

---

## 🔊 Environment-Based Verbose Mode

Add an environment variable override:

```env
GMS_VERBOSE=1
```

This will allow verbose rotation logging without changing application code.

---

## 📚 Documentation Website

Create a dedicated documentation site using:

```text
MkDocs
    ↓
GitHub Pages
```

Planned documentation:

* Installation
* Quickstart
* API reference
* Rotation architecture
* Error handling
* LangChain integration
* Examples
* Configuration
* Troubleshooting

---

## 🖥️ CLI Utility

Introduce a command-line interface:

```bash
gms list-models
gms verify-keys
gms chat
```

### Planned Commands

| Command           | Purpose                    |
| ----------------- | -------------------------- |
| `gms list-models` | Display registered models  |
| `gms verify-keys` | Verify configured API keys |
| `gms chat`        | Interactive Gemini chat    |

---

## 🧠 Model Capability Detection

Add capability-aware model selection.

Instead of simply rotating through a model pool, the client could pre-filter models based on the requested operation.

Conceptually:

```text
Requested Operation
        ↓
Capability Detection
        ↓
Compatible Models
        ↓
Rotation Pool
```

* [ ] Capability metadata
* [ ] Operation-aware filtering
* [ ] Pre-filter rotation pools
* [ ] Unsupported-operation avoidance

---

# 📋 Backlog

These items are planned but are not currently assigned to a specific release.

---

## 🔄 Automatic Model Registry Refresh

Automatically refresh the model registry using Google's `ListModels` endpoint.

* [ ] Fetch available models
* [ ] Detect newly available models
* [ ] Detect removed models
* [ ] Update registry safely
* [ ] Preserve manual overrides

---

## 📈 Benchmarking

Add:

```text
scripts/bench.py
```

The benchmark should report:

* `p50`
* `p95`
* Requests per day (`RPD`)
* Rotation overhead
* Model response characteristics

Example output:

```text
Model                    p50       p95       RPD
---------------------------------------------------
gemini-3.5-flash-lite    ...       ...       ...
gemini-3.1-flash-lite    ...       ...       ...
...
```

---

## 🧪 Property-Based Testing

Introduce Hypothesis-based tests for the rotation manager.

Potential test areas:

* Key rotation
* Model rotation
* Pair cooldowns
* Resource exhaustion
* Cooldown expiration
* Concurrent access
* Reset behavior

---

## 💾 Persistent Telemetry

Optional local history:

```text
~/.googlemodel_samrat/history.jsonl
```

Potential information:

* Rotation events
* Failed models
* Failed keys
* Successful models
* Cooldown events
* Request metadata

This feature should remain optional and privacy-conscious.

---

## 🧹 Automatic Deprecated-Model Cleanup

Perform a deprecation cleanup pass during releases.

Potential behavior:

```text
Model Registry
      ↓
Check availability
      ↓
404 / deprecated
      ↓
Remove from active registry
```

---

## 🧪 Example Gallery

Create a dedicated:

```text
examples/
```

directory containing practical examples:

* [ ] Quickstart
* [ ] RAG
* [ ] Streamlit
* [ ] FastAPI
* [ ] Batch processing
* [ ] Multi-key rotation
* [ ] LangChain
* [ ] Async usage
* [ ] Streaming

---

## 🔌 Optional Explicit Monkey-Patching

Potential future API:

```python
install_monkey_patch()
```

Rules:

* [ ] Explicit opt-in only
* [ ] Never enabled automatically
* [ ] Clearly documented
* [ ] Easy to uninstall/reverse
* [ ] Covered by tests

---

# 🧪 CI & Quality

The project already has several quality gates in CI.

## Current CI Guarantees

* [x] Coverage gate ≥ **90%**
* [x] `py.typed` present in the built wheel
* [x] `verify.py` executed on every push
* [x] `stress_test.py` executed on every push

---

# 🏁 Release Philosophy

`googlemodel-samrat` follows a deliberately conservative release strategy.

```text
                    googlemodel-samrat
                           │
            ┌──────────────┼──────────────┐
            │              │              │
          Bugfix         Feature        Breaking
            │              │              │
          0.1.x           0.x.0           x.0.0
            │              │              │
          Days            Weeks          Rarely
```

### 🐛 Bugfix Releases

Use `0.1.x` for:

* Internal fixes
* Regression fixes
* Reliability improvements
* No new public API

Target: **ship within days**.

### ✨ Feature Releases

Use `0.x.0` for:

* New constructor parameters
* New methods
* New modules
* New developer-facing functionality

Target: **ship within weeks**.

### 💥 Breaking Releases

Use `x.0.0` for:

* Removing public methods
* Changing established defaults
* Breaking API behavior
* Major architectural changes

Target: **ship rarely and deliberately**.

---

# 📌 Status Legend

| Symbol | Meaning                      |
| ------ | ---------------------------- |
| `[x]`  | Completed                    |
| `[ ]`  | Planned                      |
| 🚀     | Released                     |
| 🔮     | Planned for upcoming release |
| 📋     | Backlog                      |

---

# 🎯 Current Focus

The current development direction is:

```text
v0.2.0
   │
   ├── Rotation-aware model names
   ├── LangChain shim
   ├── Reliability fixes
   ├── Better diagnostics
   └── Cleaner SDK behavior
        │
        ▼
v0.2.x
   │
   ├── Shared state
   ├── Timezone-aware resets
   ├── Rotation events
   ├── Structured statistics
   ├── Batch APIs
   ├── CLI
   ├── Documentation site
   └── Capability detection
        │
        ▼
Backlog
   │
   ├── Automatic model discovery
   ├── Benchmarks
   ├── Property-based testing
   ├── Persistent telemetry
   ├── Example gallery
   └── Optional explicit monkey-patching
```

---

<div align="center">

### 🚀 googlemodel-samrat

**Reliable Gemini model selection, rotation, and fallback for Python.**

Built incrementally. Tested continuously. Designed for resilience.

</div>
