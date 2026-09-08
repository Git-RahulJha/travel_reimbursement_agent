# -*- coding: utf-8 -*-
"""Travel Reimbursement Approval Agent — reference implementation.

The design in one line: a language model reads documents and writes sentences;
deterministic code owns every rupee.
"""

import os

from .contracts import Claim, Decision, Outcome
from .knowledge import PolicyKnowledgeBase, build_embeddings
from .llm import ModelBackend, OfflineBackend, OpenAIBackend, build_backend
from .repository import ClaimRepository
from .workflow import ReimbursementAgent

__version__ = "1.0.0"


def load_env(path: str = None) -> bool:
    """Read OPENAI_API_KEY and friends from a local .env, if one exists.

    Values already in the environment win, so an exported key is never
    overwritten. The file is read, never written and never logged; keep it out
    of version control.
    """
    path = path or os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), ".env")
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    return True


load_env()


__all__ = ["Claim", "Decision", "Outcome", "PolicyKnowledgeBase",
           "build_embeddings", "ModelBackend", "OfflineBackend", "OpenAIBackend",
           "build_backend", "ClaimRepository", "ReimbursementAgent", "load_env",
           "__version__"]
