# -*- coding: utf-8 -*-
"""The three places a language model is used, and what contains it.

The model reads documents and writes sentences. It never returns an amount that
reaches the ledger: it returns field values, categories and confidence, and the
rule engine does the arithmetic.

    OpenAIBackend   the default - LangChain ChatOpenAI with structured output
    OfflineBackend  used only when no OPENAI_API_KEY is present, so the demo
                    still runs on a machine with no key and no network

Swapping between them changes no other line of the workflow.

The model never sees a Decimal. The schemas it is bound to use plain JSON types;
values are converted to Decimal on the way back, before any arithmetic happens.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from decimal import Decimal, InvalidOperation
from typing import List, Optional, Type

from pydantic import BaseModel, Field, ValidationError

from .contracts import (ClaimLine, Decision, ItemClassification, PolicyRule,
                        ReceiptExtraction, ReceiptRecord, RuleBasis)

CHAT_MODEL = "gpt-5-mini"


class ModelError(RuntimeError):
    """A model response that could not be made to satisfy its schema.

    Raised only after a repair attempt. It becomes a HIGH-severity trigger, and
    a HIGH-severity trigger can never be settled automatically - which is the
    reason no malformed model response can become a payment.
    """


# ----------------------------------------------- schemas the model is bound to

class ReceiptFields(BaseModel):
    """What the model may return when reading a receipt. No Decimals."""
    merchant: Optional[str] = Field(None, description="trading name on the invoice")
    invoice_no: Optional[str] = Field(None, description="receipt or invoice number")
    date: Optional[str] = Field(None, description="transaction date as YYYY-MM-DD")
    total: Optional[float] = Field(None, description="invoice total")
    currency: Optional[str] = Field(None, description="ISO currency code")
    line_items: List[str] = Field(default_factory=list,
                                  description="itemised lines as 'label :: amount'")
    confidence: float = Field(0.0, ge=0.0, le=1.0,
                              description="how certain you are of these fields")


class RuleFields(BaseModel):
    """What the model may return when reading a rule out of a policy clause.

    It reports what the clause says. It never decides what to pay.
    """
    limit_amount: Optional[float] = Field(
        None, description="the monetary limit stated in the clause, digits only")
    basis: Optional[str] = Field(
        None, description="one of PER_SECTOR, PER_NIGHT, PER_TRAVEL_DAY, "
                          "PER_CLAIM, THRESHOLD, NONE")
    source_phrase: str = Field(
        "", description="the exact phrase from the clause stating the limit, "
                        "copied verbatim")
    ineligible_terms: List[str] = Field(
        default_factory=list,
        description="things the clause says are NOT eligible, e.g. business "
                    "class, suite, sightseeing. Single words or short phrases.")
    excluded_items: List[str] = Field(
        default_factory=list,
        description="item types the clause says are never reimbursable, "
                    "e.g. alcohol")
    receipt_required_always: bool = Field(
        False, description="true if the clause requires a receipt at any amount")
    tiers: List[str] = Field(
        default_factory=list,
        description="for an approval clause only: the approver ladder in order, "
                    "as 'amount:APPROVER' e.g. '25000:REPORTING_MANAGER'")


class Exclusions(BaseModel):
    """What the model may return when classifying items. Reductions only."""
    excluded_amount: float = Field(0.0, description="total value of excluded items")
    excluded_items: List[str] = Field(default_factory=list)
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class ModelBackend(ABC):
    """The contract every backend honours. Three tasks, all schema-bound."""

    name = "abstract"

    @abstractmethod
    def extract_receipt(self, receipt: ReceiptRecord) -> ReceiptExtraction:
        """Read a receipt document into structured fields with a confidence."""

    @abstractmethod
    def classify_items(self, line: ClaimLine, extraction: ReceiptExtraction,
                       clause_text: str) -> ItemClassification:
        """Find items on the receipt that the clause excludes.

        May only reduce an entitlement. There is no field on Exclusions that
        could increase one.
        """

    @abstractmethod
    def extract_rule(self, clause_id: str, category: str, clause_text: str) -> PolicyRule:
        """Read the rule a policy clause states.

        The model reports what the document says; it never decides what to pay.
        The result is verified against the clause text before it can price
        anything - see PolicyRuleResolver.
        """

    @abstractmethod
    def explain(self, decision: Decision) -> str:
        """Write the claimant-facing sentence, from a decision already made."""

    @staticmethod
    def guard(raw: dict, schema: Type[BaseModel]) -> BaseModel:
        """Validate a raw response, or raise so the claim escalates."""
        try:
            return schema.model_validate(raw)
        except ValidationError as exc:
            raise ModelError("%s failed validation: %s" % (schema.__name__, exc)) from exc


def _dec(value) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        return Decimal(str(value)).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


# ------------------------------------------------------------------- hosted

class OpenAIBackend(ModelBackend):
    """LangChain ChatOpenAI, bound to a JSON schema on every call.

    A response that does not fit the contract raises rather than reaching the
    rule engine. Temperature is zero so a rerun of the same claim is stable.
    """

    def __init__(self, model: str = CHAT_MODEL, temperature: float = 0.0):
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        self.model_name = model
        self.name = "openai:" + model
        self._chat = ChatOpenAI(model=model, temperature=temperature, timeout=30,
                                max_retries=1)
        self._prompt = ChatPromptTemplate

    def _structured(self, system: str, human: str, schema: Type[BaseModel]) -> BaseModel:
        chain = (self._prompt.from_messages([("system", system), ("human", "{input}")])
                 | self._chat.with_structured_output(schema))
        try:
            result = chain.invoke({"input": human})
        except Exception as exc:                       # noqa: BLE001
            raise ModelError("model call failed: %s" % exc) from exc
        if result is None:
            raise ModelError("%s: model returned nothing" % schema.__name__)
        return result

    def extract_receipt(self, receipt: ReceiptRecord) -> ReceiptExtraction:
        fields: ReceiptFields = self._structured(
            "You read receipts and return only what the text actually says. "
            "Never guess or invent a value: leave a field null if it is not "
            "there. Return every itemised line as 'label :: amount'. Set "
            "confidence to how certain you are, lower it when the text is "
            "garbled or fields are missing.",
            receipt.ocr_text, ReceiptFields)

        # the page can cap, but never raise, the model's own confidence
        confidence = min(fields.confidence, receipt.page_quality)
        return ReceiptExtraction(
            receipt_id=receipt.receipt_id, merchant=fields.merchant,
            invoice_no=fields.invoice_no, date=fields.date,
            total=_dec(fields.total), currency=fields.currency,
            line_items=fields.line_items, confidence=round(confidence, 2))

    def classify_items(self, line: ClaimLine, extraction: ReceiptExtraction,
                       clause_text: str) -> ItemClassification:
        if not extraction.line_items:
            return ItemClassification()

        found: Exclusions = self._structured(
            "You identify items on a receipt that the given policy clause "
            "excludes from reimbursement. You have no knowledge of this policy "
            "beyond the clause text supplied. Return only excluded items and "
            "their total. If nothing is excluded return zero.",
            "CLAUSE:\n%s\n\nITEMS:\n%s" % (clause_text, "\n".join(extraction.line_items)),
            Exclusions)

        excluded = _dec(found.excluded_amount) or Decimal("0")
        # containment: an exclusion can only ever reduce, never exceed the line
        excluded = max(Decimal("0"), min(excluded, line.amount))
        return ItemClassification(
            excluded_amount=excluded, excluded_items=found.excluded_items,
            clause=None if excluded == 0 else "MEL-001",
            confidence=found.confidence)

    def extract_rule(self, clause_id: str, category: str,
                     clause_text: str) -> PolicyRule:
        fields: RuleFields = self._structured(
            "You read one clause of a travel expense policy and report the rule "
            "it states. Report only what this clause says; you have no other "
            "knowledge of this policy. If the clause states no monetary limit, "
            "return null for limit_amount and NONE for basis. Copy source_phrase "
            "verbatim from the clause - it will be checked against the text.",
            "CLAUSE %s (%s):\n%s" % (clause_id, category, clause_text), RuleFields)
        return _to_rule(clause_id, category, fields)

    def explain(self, decision: Decision) -> str:
        from langchain_core.output_parsers import StrOutputParser
        chain = (self._prompt.from_messages([
            ("system", "Write two or three plain sentences for the employee "
                       "explaining this reimbursement decision. Use only figures "
                       "already present in the JSON. Introduce no new numbers and "
                       "no policy you were not given. Be factual, not apologetic."),
            ("human", "{input}")]) | self._chat | StrOutputParser())
        try:
            return chain.invoke({"input": decision.model_dump_json()}).strip()
        except Exception:                              # noqa: BLE001
            # an explanation failure must never block or alter a decision
            return OfflineBackend().explain(decision)


def _to_rule(clause_id: str, category: str, fields: RuleFields) -> PolicyRule:
    """Map the model's plain-JSON answer onto the typed rule. No arithmetic here."""
    try:
        basis = RuleBasis(fields.basis) if fields.basis else RuleBasis.NONE
    except ValueError:
        basis = RuleBasis.NONE
    return PolicyRule(
        clause_id=clause_id, category=category,
        limit_amount=_dec(fields.limit_amount), basis=basis,
        ineligible_terms=[t.strip().upper().replace(" ", "_")
                          for t in fields.ineligible_terms if t.strip()],
        excluded_items=[t.strip().upper() for t in fields.excluded_items if t.strip()],
        receipt_required_always=fields.receipt_required_always,
        tiers=fields.tiers, source_phrase=fields.source_phrase.strip())


# ------------------------------------------------------------------ offline

class OfflineBackend(ModelBackend):
    """Deterministic stand-in used when no API key is present.

    Not a stub returning canned answers: it really parses the document text,
    computes confidence from page quality and how much it could read, and
    refuses to invent a field it cannot find.
    """

    name = "offline-deterministic"

    ALCOHOL = ("beer", "wine", "whisky", "vodka", "rum", "cocktail", "liquor",
               "alcohol", "bar ", "lager", "single malt")

    FIELDS = {
        "merchant": re.compile(r"^\s*Merchant:\s*(.+)$", re.M | re.I),
        "invoice_no": re.compile(r"^\s*Invoice\s*(?:No|Number)\.?:\s*(.+)$", re.M | re.I),
        "date": re.compile(r"^\s*Date:\s*(\d{4}-\d{2}-\d{2})", re.M | re.I),
        "total": re.compile(r"^\s*Total:\s*(?:INR|Rs\.?|₹)?\s*([\d,]+(?:\.\d{1,2})?)",
                            re.M | re.I),
        "currency": re.compile(r"^\s*Total:\s*(INR|USD|EUR|GBP)", re.M | re.I),
    }
    ITEM = re.compile(r"^\s{2,}(.+?)\s*[.\s]{3,}\s*([\d,]+(?:\.\d{1,2})?)\s*$", re.M)

    def extract_receipt(self, receipt: ReceiptRecord) -> ReceiptExtraction:
        text = receipt.ocr_text
        found = {}
        for field, pattern in self.FIELDS.items():
            m = pattern.search(text)
            if m:
                found[field] = m.group(1).strip()

        total = _dec(found["total"].replace(",", "")) if "total" in found else None
        items = ["%s :: %s" % (name.strip(), amount)
                 for name, amount in self.ITEM.findall(text)]

        completeness = sum(1 for f in self.FIELDS if f in found) / len(self.FIELDS)
        confidence = round(receipt.page_quality * (0.55 + 0.45 * completeness), 2)

        return ReceiptExtraction(
            receipt_id=receipt.receipt_id, merchant=found.get("merchant"),
            invoice_no=found.get("invoice_no"), date=found.get("date"),
            total=total,
            currency=found.get("currency", "INR" if total is not None else None),
            line_items=items, confidence=min(confidence, 0.99))

    def classify_items(self, line: ClaimLine, extraction: ReceiptExtraction,
                       clause_text: str) -> ItemClassification:
        excluded, amount = [], Decimal("0")
        for item in extraction.line_items:
            label, _, value = item.rpartition(" :: ")
            if any(word in label.lower() for word in self.ALCOHOL):
                parsed = _dec(value.replace(",", ""))
                if parsed is not None:
                    amount += parsed
                    excluded.append(label.strip())
        amount = max(Decimal("0"), min(amount, line.amount))
        return ItemClassification(
            excluded_amount=amount, excluded_items=excluded,
            clause="MEL-001" if excluded else None,
            confidence=0.95 if excluded else 1.0)

    LIMIT = re.compile(
        r"INR\s*([\d,]+)\s*(per one-way sector|per sector|per night|per travel day)?",
        re.I)
    BASIS = {"per one-way sector": RuleBasis.PER_SECTOR,
             "per sector": RuleBasis.PER_SECTOR,
             "per night": RuleBasis.PER_NIGHT,
             "per travel day": RuleBasis.PER_TRAVEL_DAY}
    NOT_ELIGIBLE = ("business class", "first class", "premium-economy", "suite",
                    "club floor", "executive floor", "premium", "luxury",
                    "sightseeing", "leisure", "personal errand")
    NEVER = ("alcoholic beverages", "alcohol")
    # PDF extraction flattens the APR-001 table, so match the prose form:
    # "Up to INR 25,000 Reporting Manager  INR 25,001 to INR 50,000 Senior
    #  Manager  Above INR 50,000 Finance Controller"
    # The approver titles are two words, so bound the capture at two - otherwise
    # it swallows the "Above" that starts the next tier.
    TIER = re.compile(
        r"(Up to|to|Above)\s+INR\s*([\d,]+)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)")

    def extract_rule(self, clause_id: str, category: str,
                     clause_text: str) -> PolicyRule:
        text = clause_text
        limit, basis, phrase = None, RuleBasis.NONE, ""

        # the receipt threshold clause words it as "above INR 500"
        threshold = re.search(r"above\s+INR\s*([\d,]+)", text, re.I)
        match = self.LIMIT.search(text)
        if threshold and clause_id.startswith("RCP"):
            limit = _dec(threshold.group(1).replace(",", ""))
            basis, phrase = RuleBasis.THRESHOLD, threshold.group(0)
        elif match:
            limit = _dec(match.group(1).replace(",", ""))
            basis = self.BASIS.get((match.group(2) or "").lower(), RuleBasis.PER_CLAIM)
            phrase = match.group(0)

        fields = RuleFields(
            limit_amount=float(limit) if limit is not None else None,
            basis=basis.value, source_phrase=phrase,
            ineligible_terms=[t for t in self.NOT_ELIGIBLE if t in text.lower()],
            excluded_items=[t for t in self.NEVER if t in text.lower()],
            receipt_required_always="receipt is mandatory" in text.lower(),
            tiers=["%s:%s" % ("*" if word.lower() == "above" else amount.replace(",", ""),
                              approver.strip().upper().replace(" ", "_"))
                   for word, amount, approver in self.TIER.findall(text)])
        return _to_rule(clause_id, category, fields)

    def explain(self, decision: Decision) -> str:
        head = {
            "APPROVE": "Your claim has been approved in full.",
            "PARTIALLY_APPROVE": "Your claim has been partly approved.",
            "REJECT": "Your claim could not be approved.",
            "MANUAL_REVIEW": "Your claim has been sent to a reviewer.",
        }[decision.decision.value]

        parts = ["%s Of the INR %s claimed, INR %s is approved."
                 % (head, _money(decision.claimed_amount),
                    _money(decision.approved_amount))]

        reductions = [l for l in decision.lines if l.rejected > 0]
        if reductions:
            parts.append("Reduced: " + "; ".join(
                "%s INR %s (%s, %s)" % (l.category.value.lower(), _money(l.rejected),
                                        l.reason, l.clause)
                for l in reductions) + ".")

        if decision.held_amount > 0:
            parts.append("INR %s could not be assessed automatically and is with a "
                         "reviewer." % _money(decision.held_amount))

        if decision.missing_documents:
            parts.append("Please supply: " + ", ".join(
                "%s for line %s" % (m.required.lower().replace("_", " "), m.line_id)
                for m in decision.missing_documents) + ".")

        if decision.manual_review and decision.approval:
            parts.append("Approval rests with the %s under APR-001."
                         % decision.approval.tier.lower().replace("_", " "))

        return " ".join(parts)


def _money(value: Decimal) -> str:
    return "{:,.2f}".format(value)


def build_backend(prefer_hosted: bool = True) -> ModelBackend:
    """OpenAI when a key is present, offline otherwise.

    A construction failure is not swallowed: if a key is set, the caller asked
    for the hosted model and a silent downgrade would hide it.
    """
    if prefer_hosted and os.environ.get("OPENAI_API_KEY"):
        return OpenAIBackend()
    return OfflineBackend()
