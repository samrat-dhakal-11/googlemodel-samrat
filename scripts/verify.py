#!/usr/bin/env python3
"""
End-to-end smoke test for googlemodel-samrat.

Usage:
    python3 scripts/verify.py

Exit codes:
    0  -> all executed checks passed (or safely skipped)
    1  -> at least one check failed

Live API calls only run when GEMINI_API_KEY is present. They are
kept minimal (1 invoke + 1 embedding) because the Gemini free tier
allows only a handful of requests per minute.
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings('ignore', message=r'.*fixed sampling defaults.*')
warnings.filterwarnings('ignore', message=r'.*sampling parameter.*will be ignored.*')
warnings.filterwarnings('ignore', category=UserWarning, module=r'langchain_google_genai.*')
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from googlemodel_samrat import (
    AllResourcesExhaustedError,
    ChatGoogleGenerativeAI,
    ConfigurationError,
    GoogleGenerativeAIEmbeddings,
    chatmodel,
    embeddingmodel,
    get_model_count,
    model_exists,
)

load_dotenv()

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
_counts = {PASS: 0, FAIL: 0, SKIP: 0}


def section(title: str) -> None:
    print(f"\n{'=' * 64}\n  {title}\n{'=' * 64}")


def check(name: str, ok: Optional[bool], detail: str = "") -> None:
    tag = SKIP if ok is None else (PASS if ok else FAIL)
    _counts[tag] += 1
    marker = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}[tag]
    line = f"{marker} {name}"
    if detail:
        line += f"  ->  {detail}"
    print(line)


def extract_text(msg: Any) -> str:
    """
    Get plain text from an AIMessage across LangChain versions.

    LangChain 1.x / langchain-google-genai 4.x may return `content`
    as either a plain string OR a list of content blocks like:
        [{"type": "text", "text": "...", ...}]
    """
    # Prefer the built-in .text property (langchain-core 0.3+)
    if hasattr(msg, "text"):
        try:
            t = msg.text
            if isinstance(t, str):
                return t
        except Exception:
            pass

    c = getattr(msg, "content", msg)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for block in c:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(c)


# ─────────────────────────────────────────────────────────────────────
# Test 1 — Registry
# ─────────────────────────────────────────────────────────────────────
section("Test 1: Model Registry")

cm = chatmodel()
check("chatmodel() returns a non-empty str", isinstance(cm, str) and bool(cm), cm)

em = embeddingmodel()
check("embeddingmodel() returns a non-empty str", isinstance(em, str) and bool(em), em)

count = get_model_count()
check("get_model_count() > 0", count > 0, f"{count} models registered")

check("model_exists(chatmodel()) is True", model_exists(cm))


# ─────────────────────────────────────────────────────────────────────
# Test 2 — Single key: no key rotation
# ─────────────────────────────────────────────────────────────────────
section("Test 2: Single-key mode (no key rotation)")

single = ChatGoogleGenerativeAI(
    api_keys=["dummy-single-key"],
    models=["gemini-3.8-flash", "gemini-2.5-flash"],
)
stats = single.get_rotation_stats()

check("total_keys == 1", stats["total_keys"] == 1, str(stats["total_keys"]))
check("key_rotation_enabled is False", stats["key_rotation_enabled"] is False)
check(
    "model_rotation_enabled is True",
    stats["model_rotation_enabled"] is True,
    "model fallback still active",
)
check(
    "available_combinations == 2",
    stats["available_combinations"] == 2,
    str(stats["available_combinations"]),
)


# ─────────────────────────────────────────────────────────────────────
# Test 3 — Multi-key: key rotation enabled
# ─────────────────────────────────────────────────────────────────────
section("Test 3: Multi-key mode (key rotation enabled)")

multi = ChatGoogleGenerativeAI(
    api_keys=["dummy-key-a", "dummy-key-b", "dummy-key-c"],
    models=["gemini-3.8-flash", "gemini-2.5-flash"],
)
stats = multi.get_rotation_stats()

check("total_keys == 3", stats["total_keys"] == 3, str(stats["total_keys"]))
check("key_rotation_enabled is True", stats["key_rotation_enabled"] is True)
check(
    "available_combinations == 6",
    stats["available_combinations"] == 6,
    str(stats["available_combinations"]),
)


# ─────────────────────────────────────────────────────────────────────
# Test 4 — Friendly configuration errors
# ─────────────────────────────────────────────────────────────────────
section("Test 4: Friendly configuration errors")

# Temporarily clear env vars so the fallback path doesn't kick in
_saved = {k: os.environ.pop(k, None) for k in ("GEMINI_API_KEY", "GEMINI_API_KEY2")}
try:
    ChatGoogleGenerativeAI(api_keys=[], models=["gemini-3.8-flash"])
    check("empty api_keys raises ConfigurationError", False, "no error raised")
except ConfigurationError:
    check("empty api_keys raises ConfigurationError", True)
finally:
    for k, v in _saved.items():
        if v is not None:
            os.environ[k] = v


# ─────────────────────────────────────────────────────────────────────
# Test 5 — Live API calls (minimal to avoid free-tier exhaustion)
# ─────────────────────────────────────────────────────────────────────
section("Test 5: Live invoke + attribution + embedding")

keys = [k for k in (os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_API_KEY2")) if k]

if not keys:
    reason = "no GEMINI_API_KEY in environment"
    for name in (
        "invoke() returns text content",
        "last_successful_model populated",
        "last_successful_key_index populated",
        "embed_query() returns a list of floats",
    ):
        check(f"{name} (skipped: {reason})", None)
else:
    print(f"  Using {len(keys)} API key(s) from environment.")
    print("  (Free tier allows only a few requests/min — 1 invoke + 1 embed only.)\n")

    llm = ChatGoogleGenerativeAI(
        api_keys=keys,
        max_output_tokens=256,   # room for reasoning + output
        suppress_warnings=True,
        verbose=True,
    )

    invoke_succeeded = False
    try:
        msg = llm.invoke("Reply with exactly: ok")
        text = extract_text(msg).strip()
        ok = len(text) > 0
        check("invoke() returns text content", ok, repr(text[:40]))
        invoke_succeeded = ok
    except AllResourcesExhaustedError as e:
        check("invoke() succeeded", False, str(e).splitlines()[0])
    except Exception as e:
        check("invoke() succeeded", False, f"{type(e).__name__}: {str(e)[:60]}")

    if invoke_succeeded:
        check(
            "last_successful_model populated",
            bool(llm.last_successful_model),
            llm.last_successful_model or "",
        )
        check(
            "last_successful_key_index populated",
            llm.last_successful_key_index is not None,
            str(llm.last_successful_key_index),
        )
    else:
        check("last_successful_model populated", None, "invoke failed")
        check("last_successful_key_index populated", None, "invoke failed")

    # Embedding — separate client, separate model, separate key pool
    try:
        emb = GoogleGenerativeAIEmbeddings(api_keys=keys)
        vec = emb.embed_query("hello world")
        check(
            "embed_query() returns a list of floats",
            isinstance(vec, list) and len(vec) > 0 and isinstance(vec[0], float),
            f"dim={len(vec)}",
        )
    except AllResourcesExhaustedError:
        check("embed_query() succeeded", None, "rate-limited (free tier)")
    except Exception as e:
        check("embed_query() succeeded", False, f"{type(e).__name__}: {str(e)[:60]}")


# ─────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────
section("Summary")
print(
    f"  PASS: {_counts[PASS]}    "
    f"FAIL: {_counts[FAIL]}    "
    f"SKIP: {_counts[SKIP]}"
)
if _counts[FAIL] == 0:
    print("\n  All checks passed. googlemodel-samrat is ready.")
else:
    print("\n  Some checks failed. See [FAIL] lines above.")
print()

sys.exit(1 if _counts[FAIL] else 0)
