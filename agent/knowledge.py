# -*- coding: utf-8 -*-
"""Policy knowledge base and RAG retrieval.

The corpus is the policy PDF in ./policies. It is extracted, split into one
chunk per clause, embedded with OpenAI and indexed in FAISS. Retrieval is
filtered on category and on the policy version in force for the travel dates,
so an obsolete clause and a current one can never reach one decision.

    PDF  ->  PdfClauseLoader  ->  clause chunks + metadata
         ->  OpenAIEmbeddings ->  FAISS index
         ->  similarity search, filtered  ->  RetrievedClause

Embeddings default to OpenAI (text-embedding-3-small). If no API key is
present, a deterministic TF-IDF embedding is used instead so the demo still
runs offline; the vector-store code is identical either way, because both are
LangChain Embeddings implementations.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .contracts import PolicyRule, RetrievedClause

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CORPUS = os.path.normpath(
    os.path.join(HERE, "..", "policies", "TRV-POL_v1.0.pdf"))

EMBEDDING_MODEL = "text-embedding-3-small"
MAX_CHUNK_CHARS = 1200          # clauses are smaller than this; the splitter is
CHUNK_OVERLAP = 120             # insurance for a future longer clause

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "under", "must", "not", "any",
    "are", "was", "were", "may", "who", "which", "from", "into", "such", "than",
    "then", "there", "their", "have", "has", "had", "been", "being", "its",
    "per", "all", "one", "two", "where", "what", "when", "each", "every",
    "policy", "clause", "reimbursed", "reimbursement", "reimbursable",
}


# ------------------------------------------------------------------ loading

class PdfClauseLoader:
    """Extracts the policy PDF and yields one LangChain Document per clause.

    A clause heading reads '<ID> — <title>'; the line directly beneath it is
    the metadata line 'category: ... | type: ... | effective: ...'. Both survive
    PDF text extraction, so the PDF alone is a sufficient knowledge base.
    """

    HEAD = re.compile(r"^([A-Z]{3}-\d{3})\s*[—-]\s*(.+?)\s*$")
    META = re.compile(r"^category:\s*.+\|\s*type:\s*", re.I)
    SECTION = re.compile(r"^\d+\.\s+[A-Z]")
    FOOTER = re.compile(r"^TRV-POL\s+v[\d.]+\s*[·|]")

    def __init__(self, path: str = DEFAULT_CORPUS):
        self.path = path
        if not os.path.exists(path):
            raise FileNotFoundError("policy corpus not found: %s" % path)

    def _lines(self) -> List[str]:
        import pymupdf
        with pymupdf.open(self.path) as pdf:
            raw = "\n".join(page.get_text() for page in pdf)
        out = []
        for line in raw.split("\n"):
            line = line.replace(" ", " ").strip()
            if not line or self.FOOTER.match(line):
                continue
            out.append(line)
        return out

    def load(self) -> List[Document]:
        docs: List[Document] = []
        meta: Optional[Dict[str, str]] = None
        body: List[str] = []

        def close():
            if meta is not None:
                text = re.sub(r"\s+", " ", " ".join(body)).strip()
                if text:
                    docs.append(Document(page_content=text, metadata=dict(meta)))

        for line in self._lines():
            head = self.HEAD.match(line)
            if head:
                close()
                meta = {
                    "clause_id": head.group(1),
                    "title": head.group(2).strip(),
                    "category": "ALL",
                    "clause_type": "context",
                    "effective_from": "",
                    "document": "TRV-POL",
                    "version": "v1.0",
                    "source": os.path.basename(self.path),
                }
                body = []
                continue

            if meta is None:
                continue                       # front matter, before clause 1

            if self.META.match(line):
                for part in line.split("|"):
                    if ":" not in part:
                        continue
                    key, value = (s.strip() for s in part.split(":", 1))
                    key = key.lower()
                    if key == "effective":
                        meta["effective_from"] = value
                    elif key == "type":
                        meta["clause_type"] = value
                    elif key == "category":
                        meta["category"] = value
                continue

            if self.SECTION.match(line):
                continue                       # section heading, not clause text

            body.append(line)

        close()
        if not docs:
            raise RuntimeError("no clauses parsed from %s" % self.path)
        return docs


# --------------------------------------------------------------- embeddings

class TfidfEmbeddings(Embeddings):
    """Offline fallback used only when no OpenAI key is available.

    Real vector similarity over a real vocabulary, but lexical rather than
    semantic. It is a LangChain Embeddings implementation, so nothing else in
    the pipeline changes when it is swapped for the hosted model.
    """

    name = "tfidf-offline"

    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {}
        self.idf: List[float] = []

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return [w for w in re.findall(r"[a-z]{3,}", text.lower())
                if w not in STOPWORDS]

    def _vector(self, text: str) -> List[float]:
        counts = Counter(t for t in self._tokens(text) if t in self.vocab)
        vec = [0.0] * len(self.vocab)
        for term, n in counts.items():
            i = self.vocab[term]
            vec[i] = (1.0 + math.log(n)) * self.idf[i]
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        docs = [self._tokens(t) for t in texts]
        vocabulary = sorted({term for d in docs for term in d})
        self.vocab = {term: i for i, term in enumerate(vocabulary)}
        n = len(docs)
        seen = Counter(term for d in docs for term in set(d))
        self.idf = [math.log((1 + n) / (1 + seen[t])) + 1.0 for t in vocabulary]
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        if not self.vocab:
            raise RuntimeError("index not built - embed_documents must run first")
        return self._vector(text)


def build_embeddings(prefer_hosted: bool = True) -> Embeddings:
    """OpenAI embeddings when a key is present, TF-IDF otherwise."""
    if prefer_hosted and os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return TfidfEmbeddings()


# ---------------------------------------------------------------- knowledge

class PolicyKnowledgeBase:
    """The policy corpus: loaded, chunked, embedded and queryable."""

    def __init__(self, corpus_path: str = DEFAULT_CORPUS,
                 embeddings: Optional[Embeddings] = None,
                 prefer_hosted: bool = True):
        self.corpus_path = corpus_path
        self.clauses = PdfClauseLoader(corpus_path).load()
        self.documents = self._chunk(self.clauses)
        self.embeddings = embeddings or build_embeddings(prefer_hosted)
        self.store = FAISS.from_documents(self.documents, self.embeddings)
        self.by_id = {d.metadata["clause_id"]: d for d in self.clauses}

    @staticmethod
    def _chunk(clauses: List[Document]) -> List[Document]:
        """One chunk per clause, splitting only a clause too long to embed well.

        Every sub-chunk keeps the full clause metadata, so a hit on any part of
        a clause still cites the clause.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_CHUNK_CHARS, chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", ". ", " ", ""])
        chunks: List[Document] = []
        for clause in clauses:
            parts = splitter.split_documents([clause])
            for i, part in enumerate(parts):
                part.metadata = dict(clause.metadata)
                part.metadata["part"] = i
                part.metadata["parts"] = len(parts)
                chunks.append(part)
        return chunks

    # -- introspection ----------------------------------------------------

    @property
    def clause_count(self) -> int:
        return len(self.clauses)

    @property
    def chunk_count(self) -> int:
        return len(self.documents)

    @property
    def embedding_name(self) -> str:
        """OpenAIEmbeddings is a pydantic v1 model and will not take an
        attribute, so read the model name it already carries."""
        embeddings = self.embeddings
        if hasattr(embeddings, "name"):
            return embeddings.name
        model = getattr(embeddings, "model", None)
        return "openai:" + model if model else type(embeddings).__name__

    @property
    def priced_clauses(self) -> List[str]:
        return [d.metadata["clause_id"] for d in self.clauses
                if d.metadata["clause_type"] == "priced"]

    def clause(self, clause_id: str) -> RetrievedClause:
        return self._to_clause(self.by_id[clause_id], 1.0)

    def text(self, clause_id: str) -> str:
        return self.by_id[clause_id].page_content

    # -- retrieval --------------------------------------------------------

    @staticmethod
    def _to_clause(doc: Document, score: float) -> RetrievedClause:
        m = doc.metadata
        return RetrievedClause(
            clause_id=m["clause_id"], title=m["title"], category=m["category"],
            clause_type=m["clause_type"], effective_from=m["effective_from"],
            document=m["document"], version=m["version"],
            score=round(max(0.0, min(1.0, score)), 3))

    def search(self, query: str, k: int = 5, category: Optional[str] = None,
               as_of: Optional[str] = None) -> List[RetrievedClause]:
        """Semantic search over the clauses that could govern this line."""
        def keep(meta: dict) -> bool:
            if category and meta.get("category") not in (category, "ALL"):
                return False
            if as_of and meta.get("effective_from", "") > as_of:
                return False               # not in force on the travel dates
            return True

        hits = self.store.similarity_search_with_score(query, k=k, filter=keep)
        best: Dict[str, RetrievedClause] = {}
        for doc, distance in hits:
            # FAISS IndexFlatL2 returns the SQUARED L2 distance, and the
            # vectors are unit length, so cosine = 1 - d_squared / 2.
            clause = self._to_clause(doc, 1.0 - distance / 2.0)
            keep_it = best.get(clause.clause_id)
            if keep_it is None or clause.score > keep_it.score:
                best[clause.clause_id] = clause
        return sorted(best.values(), key=lambda c: c.score, reverse=True)

    def governing_clause(self, query: str, category: str,
                         as_of: Optional[str] = None) -> Optional[RetrievedClause]:
        """The clause that actually prices this category, if one does.

        A topically relevant clause is not authority to pay. Only a priced one
        is - anything else is evidence that a person is needed.
        """
        for hit in self.search(query, k=6, category=category, as_of=as_of):
            if hit.is_authority_to_pay:
                return hit
        return None


# ------------------------------------------------------------------- rules

class PolicyRuleResolver:
    """Turns a retrieved clause into the rule the engine will apply.

    This is where RAG stops being decorative. The limits are not constants in
    the code: they are read out of the clause the retriever returned, so editing
    the policy PDF changes what the system pays, with no code release.

    Every extracted figure is **verified against the clause text** before it may
    price anything. If the number the model reports does not appear in the
    clause it claims to have read it from, the rule is not verified, and an
    unverified rule prices nothing - the line is held for a human.
    """

    def __init__(self, knowledge: "PolicyKnowledgeBase", backend):
        self.knowledge = knowledge
        self.backend = backend
        self._cache: Dict[str, PolicyRule] = {}

    @staticmethod
    def _appears_in(amount: Decimal, text: str) -> bool:
        """Is this figure written in the clause, in any normal formatting?"""
        whole = int(amount)
        candidates = {str(whole), "{:,}".format(whole)}
        if amount != whole:
            candidates.add(str(amount))
        return any(c in text for c in candidates)

    def by_clause(self, clause_id: str, category: str = "ALL") -> PolicyRule:
        """Read - and verify - the rule stated by one clause. Cached per clause."""
        if clause_id in self._cache:
            return self._cache[clause_id]

        text = self.knowledge.text(clause_id)
        rule = self.backend.extract_rule(clause_id, category, text)

        rule.verified = bool(
            rule.limit_amount is not None
            and self._appears_in(rule.limit_amount, text)
            and (not rule.source_phrase or rule.source_phrase[:12].lower() in text.lower())
        )
        rule.tiers_verified = self._tiers_appear_in(rule.tiers, text)
        self._cache[clause_id] = rule
        return rule

    def _tiers_appear_in(self, tiers: List[str], text: str) -> bool:
        """Every ceiling and every approver title must be written in the clause.

        The approval ladder decides who may authorise a payment, so it gets the
        same verbatim check the limits get. An unverified ladder is discarded
        rather than trusted, and the matrix service then escalates everything to
        the highest tier - the safe direction to fail in.
        """
        if not tiers:
            return False
        lowered = text.lower()
        for entry in tiers:
            ceiling, _, approver = entry.partition(":")
            if ceiling != "*":
                try:
                    amount = Decimal(ceiling)
                except Exception:                      # noqa: BLE001
                    return False
                if not self._appears_in(amount, text):
                    return False
            title = approver.replace("_", " ").strip().lower()
            if not title or title not in lowered:
                return False
        return True

    def for_line(self, category: str, query: str,
                 as_of: Optional[str] = None) -> Tuple[Optional[RetrievedClause],
                                                       Optional[PolicyRule]]:
        """Retrieve the governing clause for a claim line, and read its rule.

        Returns (clause, rule). The clause may be present with no rule - that is
        the INC-001 case: topically relevant, prices nothing, cannot pay.
        """
        clause = self.knowledge.governing_clause(query, category, as_of)
        if clause is None:
            nearest = self.knowledge.search(query, k=1, category=category, as_of=as_of)
            return (nearest[0] if nearest else None), None
        return clause, self.by_clause(clause.clause_id, category)

    # -- claim-level rules, also read from the document ---------------------

    def receipt_threshold(self) -> Optional[Decimal]:
        rule = self.by_clause("RCP-001")
        return rule.limit_amount if rule.verified else None

    def agent_delegation(self) -> Optional[Decimal]:
        rule = self.by_clause("AGT-001")
        return rule.limit_amount if rule.verified else None

    def approval_tiers(self) -> List[Tuple[Optional[Decimal], str]]:
        """The APR-001 ladder, as (ceiling, approver); ceiling None means open."""
        rule = self.by_clause("APR-001")
        if not rule.tiers_verified:
            return []                  # discarded, not trusted - the caller escalates
        ladder = []
        for entry in rule.tiers:
            ceiling, _, approver = entry.partition(":")
            ladder.append((None if ceiling == "*" else Decimal(ceiling), approver))
        return ladder

    def summary(self) -> List[PolicyRule]:
        """Every rule resolved so far - what the run actually read from the PDF."""
        return [self._cache[k] for k in sorted(self._cache)]
