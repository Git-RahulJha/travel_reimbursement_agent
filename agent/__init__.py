# -*- coding: utf-8 -*-
"""Travel Reimbursement Approval Agent — reference implementation."""

from dotenv import load_dotenv

from .contracts import Claim, Decision, Outcome
from .knowledge import PolicyKnowledgeBase, build_embeddings
from .llm import ModelBackend, OfflineBackend, OpenAIBackend, build_backend
from .repository import ClaimRepository
from .workflow import ReimbursementAgent

__version__ = "1.0.0"

load_dotenv()


__all__ = ["Claim", "Decision", "Outcome", "PolicyKnowledgeBase",
           "build_embeddings", "ModelBackend", "OfflineBackend", "OpenAIBackend",
           "build_backend", "ClaimRepository", "ReimbursementAgent", 
           "__version__"]
