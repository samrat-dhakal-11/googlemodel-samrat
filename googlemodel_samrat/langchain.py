"""
Drop-in import shim: rotation-enforcing replacements for
langchain_google_genai classes.

THE PROBLEM THIS SOLVES
------------------------
`chatmodel()` / `embeddingmodel()` only return a model NAME. Rotation
is enforced by OUR wrapper classes — stock LangChain clients bypass
the rotation engine entirely, so this real-world pattern 429s with no
fallback::

    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from googlemodel_samrat import embeddingmodel

    emb = GoogleGenerativeAIEmbeddings(model=embeddingmodel(), ...)
    emb.embed_documents(texts)   # <- no rotation, 429 propagates

THE FIX — a one-line import change
----------------------------------
    # before (no rotation)
    from langchain_google_genai import ChatGoogleGenerativeAI

    # after (rotation enforced)
    from googlemodel_samrat.langchain import ChatGoogleGenerativeAI

Everything else in your code stays identical.

WHY NOT MONKEY-PATCH?
---------------------
We deliberately do NOT patch `langchain_google_genai` on import.
Silent global patching breaks import-order independence, misroutes
bug reports to LangChain, and depends on internal class names.
If it is ever wanted, it will ship as an explicit opt-in
`googlemodel_samrat.install_monkey_patch()` — never automatic.
(`tests/test_model_helper_rotation.py::test_no_monkey_patch_by_default`
asserts this stays true.)
"""
from .chat import ChatGoogleGenerativeAI
from .embeddings import GoogleGenerativeAIEmbeddings
from .registry import chatmodel, embeddingmodel

__all__ = [
    "ChatGoogleGenerativeAI",
    "GoogleGenerativeAIEmbeddings",
    "chatmodel",
    "embeddingmodel",
]