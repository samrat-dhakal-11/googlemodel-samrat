# Changelog

All notable changes to **googlemodel-samrat** are documented here.

---

## [0.2.0] — 2026-09-30

### Fixed

#### Duplicate LangSmith Root Traces

Fixed duplicate LangSmith root traces where a single `llm.invoke()` could previously produce two top-level runs, resulting in double-counted tokens, costs, and latency.

The wrapper now drives the inner client through its protected:

* `_generate`
* `_stream`
* `_agenerate`
* `_astream`

This makes our `BaseChatModel` run the single top-level trace while `run_manager` preserves token accounting on that trace.

> **LangGraph:** Calls made inside LangGraph already nest under the graph's parent trace. This fix also makes standalone calls clean.

---

#### Mid-Stream Retry Duplicated Output

Fixed an issue where a mid-stream failure could cause already-yielded output to be generated again after rotation.

Rotation is now **first-chunk-only**:

* Before the first chunk is delivered → rotation/retry is allowed.
* After output has been delivered → errors propagate normally.
* Already-yielded content is never silently regenerated.

---

#### AFC Log Spam

Silenced the complete `google_genai.*` logger tree.

This removes noisy SDK messages such as:

```text
Direct use of automatic function calling ... is not recommended
```

---

#### Fixed-Sampling `UserWarning`

Warnings related to fixed sampling are now suppressed at the **actual SDK call site**.

`quiet_sdk_warnings()` wraps:

* Client construction
* Every inner SDK call

This ensures suppression survives:

* `warnings.resetwarnings()`
* pytest warning-filter management

To disable suppression and expose all warnings:

```python
suppress_warnings=False
```

---

#### Singular `api_key=` / `model=` Arguments

Fixed an issue where the documented singular arguments:

```python
api_key=
model=
```

were silently lost inside `model_extra` and could leak into the inner client keyword arguments.

Both are now handled as real aliases.

---

#### Google `retryDelay` Ignored for 429 Cooldowns

Fixed cooldown handling for Google API `429` responses.

The effective cooldown now honors Google's `retryDelay` hint:

```text
effective_cooldown = min(retryDelay + 1s, cooldown_seconds)
```

This behavior was confirmed through live testing. For example, when Google returned a `25s` retry hint, the client previously cooled the pair for `60s`; it now honors the shorter server-provided delay.

---

### Changed

#### Embedding Model Priority

Reordered `EMBEDDING_MODELS` so the GA model is attempted first:

```text
gemini-embedding-001
```

This follows live testing that encountered back-to-back `429` responses when preview models were placed first.

---

#### Chat Model Priority

Updated `CHAT_MODELS` ordering:

```text
gemini-3.5-flash-lite
→ gemini-3.1-flash-lite
→ alternating fast/heavy flow
```

---

### Added

#### `RotatingModelName`

Added `RotatingModelName`, a `str` subclass that carries its failover pool.

`chatmodel()` and `embeddingmodel()` can now return a rotating model name:

```python
model = chatmodel()
```

Pass it directly to our clients:

```python
llm = ChatGoogleGenerativeAI(model=chatmodel())
```

The complete failover pool can then rotate automatically.

It remains fully backward compatible:

```python
isinstance(model, str)  # True
```

---

#### `googlemodel_samrat.langchain`

Added a one-line LangChain migration shim:

```python
from googlemodel_samrat.langchain import ChatGoogleGenerativeAI
```

This prevents the common situation where users import `ChatGoogleGenerativeAI` expecting automatic rotation but accidentally receive the upstream implementation without our rotation behavior.

**No monkey-patching is used.**

This behavior is explicitly asserted by tests.

---

#### Public Failure Introspection

Added:

```python
llm.failed_pairs
```

to both clients.

This provides public access to the currently failed model/key pairs.

---

#### Public Failure Reset

Added:

```python
llm.reset_failures()
```

to both clients.

This provides a public way to reset recorded failures.

---

#### Improved `__repr__`

Added improved `__repr__` implementations to both the chat and embedding clients for clearer debugging and introspection.

---

#### `py.typed` Verification

Added `py.typed` verification through:

```text
scripts/verify.py
```

and a corresponding CI wheel check to ensure the marker is included in built distributions.

---

#### Timezone-Aware Daily Reset

Added `daily_reset_hour_utc` and `daily_reset_tz` to both clients.

The daily quota reset no longer has to fire at UTC midnight:

```python
llm = ChatGoogleGenerativeAI(
    api_keys=[...],
    daily_reset_hour_utc=9,
    daily_reset_tz="Asia/Kathmandu",
)
```

Defaults are unchanged (`0`, `"UTC"`), so existing behavior is fully preserved.

---

#### `on_rotate` Rotation Callback

Added an `on_rotate` callback to both clients and `RateLimitManager`:

```python
def on_rotate(failed_key, failed_model, error):
    print(f"rotating away from {failed_model} / {failed_key}: {error}")

llm = ChatGoogleGenerativeAI(api_keys=[...], on_rotate=on_rotate)
```

The callback fires in all four execution paths (sync, sync streaming, async, async streaming) whenever the client rotates away from a failed `(key, model)` pair — but never on the final attempt, since there is nothing left to rotate to.

Exceptions raised inside the callback are swallowed and logged, so a broken monitoring hook can never kill a request.

---

#### `GMS_VERBOSE` Environment Flag

`verbose` is now `Optional[bool]` and defaults to `None`:

* `verbose=None` (default) → read the `GMS_VERBOSE` environment variable (`1`, `true`, `yes`, `on` — case-insensitive).
* `verbose=True` / `verbose=False` → always wins over the environment.

As part of this change, the `Initializing <model> ......` progress print is now shown only when verbose output is enabled. Set `verbose=True` or `GMS_VERBOSE=1` to restore the old always-on behavior.

---

### Tests

Added and expanded test coverage for:

* `tests/test_model_helper_rotation.py`
* Google `retryDelay` behavior
* Warning suppression
* LangSmith single-trace behavior
* Mid-stream retry behavior
* Singular `api_key=` / `model=` aliases
* `failed_pairs`
* `reset_failures()`
* `__repr__`
* `RotatingModelName`
* LangChain import shim
* No-monkey-patching guarantee
* `tests/test_timezone_config.py`
* `tests/test_verbose_on_rotate.py`

---

# Final Release Summary

|  # | Item                                                    | File(s)                                   | Status |
| -: | ------------------------------------------------------- | ----------------------------------------- | :----: |
|  1 | AFC log silenced — full `google_genai.*` tree           | `core.py`                                 |    ✅   |
|  2 | Fixed-sampling warning silenced at call sites           | `core.py`, `chat.py`, `embeddings.py`     |    ✅   |
|  3 | `EMBEDDING_MODELS` GA-first reorder                     | `registry.py`                             |    ✅   |
| 3b | `CHAT_MODELS` 3.5-lite → 3.1-lite → flow                | `registry.py`                             |    ✅   |
|  4 | Google `retryDelay` hint honored                        | `core.py` + 4 tests                       |    ✅   |
|  5 | Duplicate LangSmith root traces fixed                   | `chat.py` + 2 tests                       |    ✅   |
|  6 | Mid-stream retry duplication fixed                      | `core.py` mixin + 1 test                  |    ✅   |
|  7 | Singular `api_key=` / `model=` leak fixed               | `chat.py` + tests                         |    ✅   |
|  8 | `py.typed` verification                                 | `verify.py`, CI                           |    ✅   |
|  9 | Improved `__repr__` on both clients                     | `chat.py`, `embeddings.py` + test         |    ✅   |
| 10 | Public `failed_pairs`                                   | Mixin + test                              |    ✅   |
| 11 | Public `reset_failures()`                               | Mixin + test                              |    ✅   |
|  ⭐ | **`RotatingModelName` — flagship feature**              | `registry.py`, `chat.py`, `embeddings.py` |    ✅   |
|  ⭐ | **`googlemodel_samrat.langchain` shim**                 | New module + 4 tests                      |    ✅   |
|  ⭐ | **No monkey-patching guarantee**                        | Test assertion                            |    ✅   |
| 📋 | `ROADMAP.md` + `CHANGELOG.md`                           | Repository root                           |    ✅   |
| 🤖 | CI — 90% coverage gate, wheel `py.typed`, advisory mypy | `.github/workflows/ci.yml`                |    ✅   |
| ⭐ | **Timezone-aware daily reset** | `core.py`, `chat.py`, `embeddings.py` | ✅ |
| ⭐ | **`on_rotate` rotation callback** | `core.py`, `chat.py`, `embeddings.py` | ✅ |
| ⭐ | **`GMS_VERBOSE` env flag** | `chat.py`, `embeddings.py` | ✅ |

---

## Release Highlights

Version **0.2.0** focuses on making model/key rotation more reliable, observable, and LangChain-friendly.

The release introduces three major capabilities:

1. **`RotatingModelName`** for portable failover pools.
2. **A dedicated LangChain import shim** with explicit, non-invasive integration.
3. **More reliable retry and rotation behavior**, including Google's `retryDelay`, first-chunk-only streaming rotation, and clean LangSmith tracing.
4. **Timezone-aware resets with observability hooks** — configure the daily
   quota reset per timezone, observe rotation via `on_rotate`, and control
   output with `GMS_VERBOSE`.

It also strengthens the package with public failure introspection, reset controls, improved diagnostics, warning management, model-priority updates, and CI verification.

---

**Release:** `0.2.0`
**Date:** `2026-09-30`
