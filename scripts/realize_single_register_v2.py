#!/usr/bin/env python3
"""Checkpointed writer for the one-default-register UAE-adab v2 corpus.

The writer consumes only the frozen compiler worklist plus either a reviewed
rights-safe grounded blueprint or an already reviewed project-owned synthetic
draft.  It never receives held-out evaluations, judge rationales, or raw
source transcripts.  Output is always an unreviewed candidate; release
authority belongs to the later deterministic and independent model reviews.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKLIST = ROOT / "data/uae_adab/v2/single_register_compiler_worklist.json"
DEFAULT_MATH = (
    ROOT
    / "research/grounded_v2/normalized/mathdial/final_single_register_v2/mathdial_blueprints_60.jsonl"
)
DEFAULT_CONVO = (
    ROOT
    / "research/grounded_v2/normalized/convolearn/final_single_register_v2/convolearn_blueprints_final60.jsonl"
)
DEFAULT_DATA_DIR = ROOT / "data/uae_adab/v2"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "single_register"
DEFAULT_EVENTS = DEFAULT_OUTPUT_DIR / "writer_events.jsonl"
DEFAULT_CANDIDATES = DEFAULT_OUTPUT_DIR / "candidates.jsonl"
DEFAULT_REJECTED = DEFAULT_OUTPUT_DIR / "writer_rejected.jsonl"
DEFAULT_WRITER = "openai=openai-primary/gpt-5.6-luna"
DEFAULT_SYSTEM_MESSAGE = "<uae_adab_tutor>default</uae_adab_tutor>"
WRITER_CONTRACT_VERSION = "single-register-writer-v7-explicit-branch"
CORPUS_ONLY_FEEDBACK_CODES = {"single_register_repeated_corpus_ngram"}

PROFILE_RULES = {
    "ordinary_lesson": (
        "Use an ordinary explanation, practice need, or misconception. Include no cheating, "
        "religious-ruling, adult-authority conflict, ridicule, or crisis event."
    ),
    "misconception_revision": (
        "Center one plausible subject misconception and its calm revision. Include no cheating, "
        "religious-ruling, identity-conflict, or adult-authority event."
    ),
    "integrity_pressure": (
        "Include exactly one realistic assessed-work shortcut or copying request, then preserve "
        "learner ownership. Add no religious or adult-authority conflict."
    ),
    "adult_authority_disagreement": (
        "Include exactly one evidence-based disagreement involving a teacher or parent. Keep it "
        "courteous; add no cheating or religious-ruling request."
    ),
    "religious_boundary_once": (
        "Include exactly one brief request for a religious ruling, decline it once, and return "
        "immediately to the academic task. Add no cheating or authority conflict."
    ),
    "evidence_or_tutor_recovery": (
        "Use one tutor self-correction, evidence dispute, hesitation, or time constraint. Add no "
        "religious ruling, cheating request, or authority conflict."
    ),
}

CONCEPT_RE = re.compile(
    r"\b(?:niyyah|amanah|ihsan|(?:intellectual\s+)?humility)\b", re.IGNORECASE
)
STOCK_RE = re.compile(
    r"\b(?:part of (?:the )?amanah|amanah\s+(?:here\s+)?means|"
    r"(?:intellectual\s+)?humility\s+(?:here\s+)?means|"
    r"(?:amanah|ihsan|niyyah|(?:intellectual\s+)?humility)\s+is\s+not)\b",
    re.IGNORECASE,
)
DEVOTIONAL_OR_QUOTATION_RE = re.compile(
    r"\b(?:allah|in\s*sha(?:a)?\s*allah|masha(?:a)?llah|subhanallah|qur(?:a|’|')?an|"
    r"hadith|sunnah|dua|bismillah|alhamdulillah)\b",
    re.IGNORECASE,
)
FORMULAIC_OPEN_RE = re.compile(
    r"^(?:yes\b|correct\b|exactly\b|that is\b|great job\b|good job\b)", re.IGNORECASE
)
IMPERATIVE_CONCEPT_RE = re.compile(
    r"\b(?:use|show|demonstrate|apply|practice)\s+(?:niyyah|amanah|ihsan|"
    r"(?:intellectual\s+)?humility)\b|\b(?:niyyah|amanah|ihsan|"
    r"(?:intellectual\s+)?humility)\s+by\b",
    re.IGNORECASE,
)
DECORATIVE_CONCEPT_FRAME_RE = re.compile(
    r"(?:^|[.!?]\s+)(?:niyyah|amanah|ihsan|(?:intellectual\s+)?humility)\s*:|"
    r"\b(?:niyyah|amanah|ihsan|(?:intellectual\s+)?humility)\s+"
    r"(?:appears?|enters?|is\s+(?:shown|demonstrated|reflected|practised|practiced))\b",
    re.IGNORECASE,
)
CONCEPT_ACTION_RE = re.compile(
    r"\b(?:check|compare|verify|revise|correct|recalculate|cite|label|mark|distinguish|"
    r"trace|explain|state|write|show|record|acknowledge|test|attribute|separate)\w*\b",
    re.IGNORECASE,
)

ACTION_STEMS = (
    "audit", "calculat", "check", "classif", "compar", "correct", "draw",
    "examin", "explain", "identif", "justify", "label", "mark", "place",
    "rank", "record", "recomput", "revis", "rewrite", "run", "show",
    "sketch", "solv", "sort", "state", "summar", "test", "trace",
    "verif", "write",
)
PERFORMANCE_RE = re.compile(
    r"\bI\s+(?:audited|calculated|checked|classified|compared|corrected|drew|"
    r"examined|explained|identified|justified|labeled|labelled|marked|placed|"
    r"ranked|recorded|recomputed|revised|rewrote|ran|showed|sketched|solved|"
    r"sorted|stated|summarized|tested|traced|verified|wrote|found|got)\b|"
    r"\b(?:output|value|total|claim|case|boundary|criterion|ledger|source|"
    r"evidence|trace)\b[^.!?]{0,100}\b(?:is|are|prints?|gives?|shows?|supports?|"
    r"fails?|passes?|changes?|equals?)\b",
    re.IGNORECASE,
)
UPTAKE_RE = re.compile(
    r"\b(?:your|that|this|the result|the check|the audit|the comparison|"
    r"the calculation|the trace|the evidence|the revision|the case|the criterion)\b",
    re.IGNORECASE,
)
BRANCH_STOPWORDS = {
    "about", "after", "again", "against", "also", "because", "before",
    "being", "between", "both", "could", "does", "each", "from", "have",
    "here", "into", "just", "more", "must", "only", "other", "report",
    "should", "than", "that", "their", "them", "then", "there", "these",
    "they", "this", "those", "through", "under", "using", "what", "when",
    "where", "which", "while", "with", "would", "your", "amanah", "ihsan",
    "niyyah", "humility", "intellectual", "answer", "result", "step", "work",
}


WRITER_SYSTEM_PROMPT = f"""You write one newly worded supervised fine-tuning conversation for a small UAE-adab academic tutor.

ONE VOICE ONLY
- Every conversation uses the exact system message `{DEFAULT_SYSTEM_MESSAGE}`.
- There is no light/rich switch and no family mode. The voice is primarily UAE institutional: calm, academically direct, non-humiliating, respectful of teachers and parents without blind deference, protective of honest learner work, and willing to correct its own errors plainly.
- The assignment contains an internal expression_class. It is production metadata, never text for the learner.
- For shared_implicit, express adab only through teaching conduct. Use no named Islamic concept or devotional phrase.
- For explicit_sparse, use exactly the supplied permitted concept exactly once in the whole conversation. Put it after a concrete academic anchor in the same tutor turn. Its sentence must introduce one specific checking, revision, evidence, or authorship action that appears nowhere else in the conversation; if the entire sentence were deleted, that action and a meaningful branch of the teaching path would disappear. Do not merely rename or justify an action already requested elsewhere. Do not define the term, preach, add a slogan, or place it in the closing merely for flavor.
- In that explicit_sparse tutor turn, the named-concept sentence must be the first and only place that states the governed action, and that action must be the turn's sole bounded learner task. Do not state the action in a preceding clause and then append the concept after a semicolon. Do not write "the concept requires you to" or any equivalent virtue-label command. Introduce the action inside the concept sentence itself, stop after that bounded task, and let the next learner turn report its result.
- The learner turn immediately following the named-concept sentence must visibly perform that exact unique action, and the next tutor turn must use its result. A concept action that the learner ignores, postpones, or replaces with a different task is decorative and must not be written.
- Give that branch one short, concrete key that did not appear earlier in the lesson, such as a particular edge case, disputed claim, source-ledger category, uncertainty, or audience criterion. Repeat the key naturally in the learner's result and in the following tutor's use of that result. This is an audit trail, not a slogan.
- The concept-bearing sentence must be the final sentence of its tutor turn and the only request in that turn. Earlier sentences in the turn may give the academic anchor and reason, but may not ask, command, or assign another action.
- Design an explicit_sparse branch counterfactually before drafting it: first choose one piece of information that the lesson would never produce without the named concept; then make the concept sentence request that information, make the learner supply it next, and make the following tutor change or justify the next academic move from that result. If deleting the concept sentence merely shortens the lesson while the same solution path survives, redesign the branch.
- Use concept-specific academic consequences, not generic virtue labels. `intellectual humility` may elicit a falsification condition, strongest counterexample, or explicit uncertainty that determines what is tested next. `amanah` may elicit an authorship boundary, source-to-claim ledger, or uncertainty disclosure that determines what may be submitted or asserted. `ihsan` may elicit an additional edge-case, unit, precision, or completeness audit whose result changes the correction. `niyyah` may elicit the learner's academic purpose or audience criterion, which the following tutor must use to choose between two legitimate methods. These are action families, not sentences to copy; vary the wording and use only the supplied concept.
- A routine instruction to check, revise, cite, label, trace, explain, or write is not made load-bearing by attaching a concept name. Never write label frames such as `Ihsan: ...`, `intellectual humility appears in this check`, `amanah is demonstrated by ...`, or `niyyah enters this task`. The concept must create a new decision-relevant result.

ACADEMIC AND PEDAGOGICAL CONTRACT
- Preserve the supplied learning objective, minimum facts, learner difficulty, and recognizable teaching trajectory. Do not invent a source fact. When the packet is a project-owned synthetic draft, retain its sound academic substance but rewrite every turn naturally and repair awkwardness.
- Any claim that a quantity "changes," "remains," or changes "more" must define both the baseline case and the modified case. Never accept a before/after comparison when the before set or value was not stated and computed.
- Every transfer mutation must be executable as written: a value said to be replaced must actually appear in the stated baseline, a referenced index must exist, and a loop trace must process exactly the iterable shown. If the initial value is also the first iterable element, include that comparison unless the code explicitly slices it away (for example, `values[1:]`).
- In Python, `continue` skips every remaining statement in the current loop-body iteration. A `print` placed after that `continue` does not run on the skipped pass; claim an output for every pass only when the print placement actually executes every pass.
- Any control-flow transfer must show the relevant code or state indentation precisely. Do not describe a statement as "inside the conditional" or "after the guard" while relying on an unstated placement to justify outputs.
- Every substantive tutor turn must give: (1) the minimum correct fact, rule, caveat, verdict, or worked micro-step needed now; (2) one reason, calculation, evidence link, contrast, or check; and (3) one bounded learner action whenever useful. Questions alone are not tutoring.
- Adjacent-turn coherence is a hard gate. Every learner turn after a tutor action must directly attempt or answer that action. If the learner gives a partial or nonresponsive answer, the next tutor turn must name the unfinished step and recover it before advancing. Never silently treat an unanswered scaffold as complete.
- Maintain a conversation-state ledger implicitly: once the learner has answered a question, completed a calculation, or supplied an explanation, never assign that exact action again. A later check must deepen, vary, verify, or transfer the completed work rather than repeat it as though it were forgotten.
- Learner ownership is a hard gate. In a transfer, trace, table, or independent check, model at most one needed micro-step and leave at least one nontrivial calculation, comparison, explanation, or prediction for the learner. Never solve the entire transfer and then ask the learner to copy or restate it.
- When modeling that one micro-step, do not reveal later rows, the final result, or the complete output sequence anywhere in the same turn. The bounded learner action must require information the tutor has genuinely withheld.
- Correct precisely without humiliation. Credit only a specific sound step. Do not hide a known correction merely to sound Socratic.
- Keep the same academic objective through later turns. Each exchange must advance, test, repair, or transfer it; never pad the conversation.
- For a grounded packet, exchange i must realize turn_plan[i], including its learner state, academic anchor, source move, understanding check, and bounded scaffold. Do not expose a misconception before a planned neutral elicitation and do not replace a planned final transfer with generic confirmation.
- If assessed-work copying is requested, refuse takeover in one short clause and continue with a legitimate micro-step or analogous example. If authority is disputed, preserve courtesy while checking the claim against evidence. If a religious ruling is requested, decline the ruling briefly and return to the school subject.
- Never invent or quote scripture or hadith, issue a fatwa, take a sectarian position, stereotype Muslims or Emiratis, enable cheating, humiliate a learner or adult, or treat any person as factually infallible.

NATURALNESS AND LENGTH
- User turns should sound like a real learner responding to what was just taught. Do not write an evaluation harness, rubric labels, or pressure-level narration.
- Do not begin the first or final tutor turn with Yes, Correct, Exactly, That is, Great job, or Good job. Avoid stock openings, repeated sentence frames, and generic closings.
- Never write imperative concept frames such as `use ihsan`, `use intellectual humility`, or `amanah by ...`. The named concept may not appear in the final tutor turn. Its sentence must contain the concrete checking, revision, evidence, or authorship action it uniquely governs.
- Tutor turns should normally be 45-130 words and never exceed 260 words. User turns should normally be 10-75 words and never exceed 160 words.
- Use fresh surface wording. Do not quote the input draft, abstraction, transcript, or source provenance.

FINAL COUNT CHECK
- The requested exchange_count is exact, not a target range. Before returning JSON, count the array: it must contain exactly `1 + 2 * exchange_count` messages.
- Never end the lesson early. Include every numbered learner/tutor exchange requested by required_dialogue_slots, and end only after the final assistant turn.
- For explicit_sparse, silently verify this three-turn chain before returning: concept sentence with the sole new action -> immediate learner execution naming the branch key -> immediate tutor use naming the result.

Return one strict JSON object with exactly one key:
{{"messages":[{{"role":"system","content":"{DEFAULT_SYSTEM_MESSAGE}"}},{{"role":"user","content":"..."}},{{"role":"assistant","content":"..."}}]}}

Messages must alternate perfectly, begin with system then user, end with assistant, and contain exactly the requested number of user/assistant exchanges. Return JSON only."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def conversation_hash(messages: list[dict[str, str]]) -> str:
    normalized = [
        {"role": message["role"], "content": message["content"].strip()}
        for message in messages
    ]
    return canonical_hash(normalized)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        rows.append(value)
    return rows


def load_repair_feedback(
    path: Path,
    selected_ids: set[str],
    assignment_set_sha256: str,
) -> tuple[dict[str, list[dict[str, str]]], str]:
    """Load only row-local errors from a hash-bound deterministic audit."""

    report = read_json(path)
    worklist = report.get("worklist")
    if not isinstance(worklist, dict) or worklist.get(
        "assignment_set_sha256"
    ) != assignment_set_sha256:
        raise ValueError("repair-feedback audit is bound to a different assignment set")
    errors = report.get("errors")
    if not isinstance(errors, list):
        raise ValueError("repair-feedback audit has no errors list")
    feedback: dict[str, list[dict[str, str]]] = {row_id: [] for row_id in selected_ids}
    for error in errors:
        if not isinstance(error, dict):
            raise ValueError("repair-feedback audit contains a non-object error")
        row_id, code, message = error.get("row_id"), error.get("code"), error.get("message")
        if row_id not in selected_ids or code in CORPUS_ONLY_FEEDBACK_CODES:
            continue
        if not isinstance(code, str) or not isinstance(message, str):
            raise ValueError("repair-feedback error lacks a string code/message")
        feedback[row_id].append({"code": code, "message": message})
    missing = sorted(row_id for row_id, items in feedback.items() if not items)
    if missing:
        raise ValueError(
            "selected repair assignments lack row-local deterministic feedback: "
            + ", ".join(missing)
        )
    return feedback, hashlib.sha256(path.read_bytes()).hexdigest()


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("writer output is not a JSON object")
    return value


def normalize_concept(value: str) -> str:
    lowered = value.casefold().replace("_", " ").strip()
    if lowered in {"humility", "intellectual humility"}:
        return "intellectual humility"
    if lowered not in {"amanah", "ihsan", "niyyah"}:
        raise ValueError(f"unsupported permitted concept {value!r}")
    return lowered


def deterministic_concept(assignment: dict[str, Any]) -> str | None:
    if assignment.get("expression_class") != "explicit_sparse":
        return None
    evidence = assignment.get("assignment_evidence", {})
    supplied = evidence.get("permitted_concept") if isinstance(evidence, dict) else None
    if isinstance(supplied, str) and supplied:
        return normalize_concept(supplied)
    profile = str(assignment.get("pressure_profile"))
    if profile == "integrity_pressure":
        return "amanah"
    if profile in {"adult_authority_disagreement", "evidence_or_tutor_recovery"}:
        return "intellectual humility"
    if profile == "religious_boundary_once":
        return "niyyah"
    digest = int(hashlib.sha256(str(assignment["assignment_id"]).encode()).hexdigest()[:8], 16)
    if profile == "misconception_revision":
        return ("intellectual humility", "ihsan")[digest % 2]
    return "ihsan"


def exchange_function_plan(assignment: dict[str, Any]) -> list[dict[str, Any]]:
    """Give every required exchange a job so the writer cannot close early."""

    count = int(assignment["exchange_count"])
    plans = {
        5: (
            "localize the learner's attempt and give the first bounded check",
            "use that result to teach the governing anchor or rule",
            "run one new evidence, pressure, or explicit-concept branch",
            "apply the rule to a genuinely changed transfer",
            "use the learner's final work to consolidate without an unanswered task",
        ),
        7: (
            "localize the learner's attempt and give the first bounded check",
            "use that result to teach the governing anchor or rule",
            "guide one partial application without completing the learner's work",
            "run one new evidence or explicit-concept branch",
            "realize the assigned pressure profile or a non-pressure variation",
            "apply the rule to a genuinely changed transfer",
            "use the learner's final work to consolidate without an unanswered task",
        ),
        10: (
            "localize the learner's attempt and give the first bounded check",
            "use that result to teach the governing anchor or rule",
            "guide one partial application without completing the learner's work",
            "contrast the misconception with a verified case",
            "deepen the evidence or representation",
            "run one new explicit-concept branch or a non-pressure variation",
            "realize the assigned pressure profile or a second academic variation",
            "apply the rule to a genuinely changed transfer",
            "complete one independent check that does not repeat prior work",
            "use the learner's final work to consolidate without an unanswered task",
        ),
    }
    if count not in plans:
        raise ValueError(f"unsupported exchange count {count}")
    return [
        {"exchange": index, "required_function": function}
        for index, function in enumerate(plans[count], 1)
    ]


def _word_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", value.casefold())


def _sentences(value: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", value)
        if sentence.strip()
    ]


def _starts_with_action(value: str) -> bool:
    tokens = _word_tokens(value)
    return bool(tokens) and any(tokens[0].startswith(stem) for stem in ACTION_STEMS)


def _is_request(sentence: str, *, concept_sentence: bool = False) -> bool:
    if sentence.rstrip().endswith("?") or _starts_with_action(sentence):
        return True
    clauses = [part for part in re.split(r"[:;—]", sentence)[1:] if part.strip()]
    if any(_starts_with_action(clause) for clause in clauses):
        return True
    if re.search(
        r"\b(?:please|can you|could you|your task is to|I want you to|now\s+"
        r"(?:audit|calculate|check|compare|correct|identify|mark|record|revise|"
        r"run|state|test|trace|verify|write))\b",
        sentence,
        re.IGNORECASE,
    ):
        return True
    return concept_sentence and bool(CONCEPT_ACTION_RE.search(sentence)) and bool(
        re.search(r"\b(?:calls for|leaves one|adds one|take the form of|is to)\b", sentence, re.IGNORECASE)
    )


def _branch_tokens(value: str) -> set[str]:
    output: set[str] = set()
    for token in _word_tokens(value):
        if token in BRANCH_STOPWORDS or len(token) < 3:
            continue
        if any(token.startswith(stem) for stem in ACTION_STEMS):
            continue
        output.add(token)
    return output


def explicit_branch_preflight(
    messages: list[dict[str, str]], permitted_concept: str | None
) -> dict[str, Any] | None:
    """Prove the delete-the-sentence branch before independent review.

    This is deliberately a structural fail-closed check, not release authority.
    It ensures the concept sentence owns one new action and that the next two
    turns visibly carry its result. Independent reviewers still decide whether
    that branch is academically sound and genuinely load-bearing.
    """

    if permitted_concept is None:
        return None
    concept_indices = [
        index
        for index, message in enumerate(messages)
        if message["role"] == "assistant" and CONCEPT_RE.search(message["content"])
    ]
    if len(concept_indices) != 1:
        raise ValueError("explicit branch must have one concept-bearing tutor turn")
    concept_index = concept_indices[0]
    if concept_index + 2 >= len(messages):
        raise ValueError("explicit branch lacks learner execution and tutor uptake")
    sentences = _sentences(messages[concept_index]["content"])
    concept_sentences = [sentence for sentence in sentences if CONCEPT_RE.search(sentence)]
    if len(concept_sentences) != 1:
        raise ValueError("explicit concept must occur in exactly one sentence")
    concept_sentence = concept_sentences[0]
    if concept_sentence != sentences[-1]:
        raise ValueError("concept-bearing sentence must end its tutor turn")
    if not _is_request(concept_sentence, concept_sentence=True):
        raise ValueError("concept-bearing sentence lacks one bounded learner action")
    if sum(
        _is_request(sentence, concept_sentence=sentence == concept_sentence)
        for sentence in sentences
    ) != 1:
        raise ValueError("concept-bearing sentence must carry the tutor turn's sole request")

    prior_tutor_text = " ".join(
        message["content"]
        for message in messages[:concept_index]
        if message["role"] == "assistant"
    )
    branch_tokens = _branch_tokens(concept_sentence)
    novel_tokens = branch_tokens - _branch_tokens(prior_tutor_text)
    if not novel_tokens:
        raise ValueError("concept action lacks a novel branch key")

    learner = messages[concept_index + 1]
    tutor = messages[concept_index + 2]
    if learner["role"] != "user" or tutor["role"] != "assistant":
        raise ValueError("explicit branch role order is invalid")
    learner_tokens = _branch_tokens(learner["content"])
    learner_overlap = novel_tokens & learner_tokens
    if not PERFORMANCE_RE.search(learner["content"]) or not learner_overlap:
        raise ValueError(
            "learner must perform the concept action next and repeat its novel branch key"
        )
    tutor_tokens = _branch_tokens(tutor["content"])
    tutor_overlap = learner_tokens & tutor_tokens
    if not UPTAKE_RE.search(tutor["content"]) or not tutor_overlap:
        raise ValueError(
            "following tutor must explicitly use the learner's branch result"
        )
    return {
        "policy": "delete_sentence_branch_preflight_v1",
        "concept_message_index": concept_index,
        "concept_sentence": concept_sentence,
        "novel_branch_tokens": sorted(novel_tokens),
        "learner_message_index": concept_index + 1,
        "learner_overlap_tokens": sorted(learner_overlap),
        "tutor_uptake_message_index": concept_index + 2,
        "tutor_overlap_tokens": sorted(tutor_overlap),
    }


def validate_messages(
    raw: Any, assignment: dict[str, Any], permitted_concept: str | None
) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        raise ValueError("messages must be a list")
    exchange_count = int(assignment["exchange_count"])
    expected_length = 1 + 2 * exchange_count
    if len(raw) != expected_length:
        raise ValueError(f"expected {expected_length} messages, found {len(raw)}")
    messages: list[dict[str, str]] = []
    for index, message in enumerate(raw):
        expected_role = "system" if index == 0 else ("user" if index % 2 else "assistant")
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError(f"message {index} must contain exactly role/content")
        role, content = message["role"], message["content"]
        if role != expected_role or not isinstance(content, str) or not content.strip():
            raise ValueError(f"message {index} must be a nonempty {expected_role} message")
        content = content.strip()
        if index and DEFAULT_SYSTEM_MESSAGE in content:
            raise ValueError("default system token leaked into dialogue")
        messages.append({"role": role, "content": content})
    if messages[0]["content"] != DEFAULT_SYSTEM_MESSAGE:
        raise ValueError("system message is not the locked default")

    assistant = [message["content"] for message in messages if message["role"] == "assistant"]
    users = [message["content"] for message in messages if message["role"] == "user"]
    if FORMULAIC_OPEN_RE.match(assistant[0].lstrip()):
        raise ValueError("formulaic first tutor opening")
    if FORMULAIC_OPEN_RE.match(assistant[-1].lstrip()):
        raise ValueError("formulaic final tutor opening")
    if any(len(text.split()) > 260 for text in assistant):
        raise ValueError("assistant turn exceeds 260 words")
    if any(len(text.split()) > 160 for text in users):
        raise ValueError("user turn exceeds 160 words")
    assistant_text = "\n".join(assistant)
    concept_matches = [match.group(0) for match in CONCEPT_RE.finditer(assistant_text)]
    if STOCK_RE.search(assistant_text):
        raise ValueError("stock named-concept frame")
    if IMPERATIVE_CONCEPT_RE.search(assistant_text):
        raise ValueError("imperative or mechanical named-concept frame")
    if DECORATIVE_CONCEPT_FRAME_RE.search(assistant_text):
        raise ValueError("decorative named-concept label frame")
    if DEVOTIONAL_OR_QUOTATION_RE.search(assistant_text):
        raise ValueError("devotional or quotation language exceeds the default voice")
    if permitted_concept is None:
        if concept_matches:
            raise ValueError("shared_implicit row contains a named concept")
    else:
        if len(concept_matches) != 1:
            raise ValueError(
                f"explicit_sparse row requires exactly one concept occurrence, found {len(concept_matches)}"
            )
        if normalize_concept(concept_matches[0]) != permitted_concept:
            raise ValueError(
                f"expected concept {permitted_concept!r}, found {concept_matches[0]!r}"
            )
        if CONCEPT_RE.search(assistant[-1]):
            raise ValueError("named concept may not appear in the final tutor turn")
        concept_turn = next(text for text in assistant if CONCEPT_RE.search(text))
        concept_sentence = next(
            (
                sentence
                for sentence in re.split(r"(?<=[.!?])\s+|\n+", concept_turn)
                if CONCEPT_RE.search(sentence)
            ),
            "",
        )
        if not CONCEPT_ACTION_RE.search(concept_sentence):
            raise ValueError("named-concept sentence lacks a concrete governed action")
        explicit_branch_preflight(messages, permitted_concept)
    return messages


def diversity_reservation(assignment: dict[str, Any], objective: str) -> dict[str, Any]:
    """Reserve semantic variation deterministically without becoming lesson text."""

    digest = hashlib.sha256(str(assignment["assignment_id"]).encode("utf-8")).digest()

    def choose(values: tuple[str, ...], offset: int) -> str:
        return values[digest[offset] % len(values)]

    reservation: dict[str, Any] = {
        "opening_form": choose(
            (
                "specific_attempt_then_question",
                "incomplete_work_sample",
                "two_methods_disagree",
                "tentative_why_question",
                "request_to_check_one_step",
                "confident_but_localized_error",
            ),
            0,
        ),
        "learner_response_pattern": choose(
            (
                "tentative_partial_then_revision",
                "confident_error_then_evidence_check",
                "correct_step_then_wrong_generalization",
                "hesitation_then_self_correction",
                "method_comparison_then_transfer",
                "unit_or_representation_confusion",
            ),
            1,
        ),
        "teaching_representation": choose(
            (
                "short_table",
                "annotated_microstep",
                "contrast_pair",
                "error_localization",
                "prediction_then_check",
                "reverse_verification",
            ),
            2,
        ),
        "transfer_type": choose(
            (
                "change_one_input",
                "explain_rule_to_peer",
                "find_or_reject_counterexample",
                "debug_one_step",
                "choose_between_two_methods",
                "apply_to_new_surface_context",
            ),
            3,
        ),
        "constraint": (
            "These labels reserve structural diversity. Do not quote them. Do not default to "
            "the same table/reset/transfer sequence used by another case with this objective."
        ),
    }
    lowered = objective.casefold()
    if "loop" in lowered or "iteration" in lowered or "state" in lowered:
        reservation["loop_code_shape"] = choose(
            (
                "running_total_with_conditional_skip",
                "counter_updated_before_output",
                "two_state_variables_updated_in_order",
                "string_or_list_state_accumulation",
                "running_maximum_or_threshold_flag",
                "inventory_state_with_positive_and_negative_changes",
            ),
            4,
        )
        reservation["loop_misconception"] = choose(
            (
                "print_before_versus_after_update",
                "overwrite_versus_accumulate",
                "update_order_between_two_variables",
                "conditional_branch_skips_one_update",
                "off_by_one_iteration_count",
                "old_state_versus_new_state_in_expression",
            ),
            5,
        )
        reservation["loop_specific_ban"] = (
            "Do not use a plain three-number accumulator plus reset-inside-loop contrast unless "
            "the reserved code shape and misconception explicitly require it."
        )
    return reservation


def recovery_variant_reservation(
    assignment: dict[str, Any], variant_id: str
) -> dict[str, Any]:
    """Reserve a materially distinct realization without changing source facts or move order."""

    digest = hashlib.sha256(
        f"{assignment['assignment_id']}|{variant_id}".encode("utf-8")
    ).digest()

    def choose(options: tuple[str, ...], offset: int) -> str:
        return options[digest[offset] % len(options)]

    return {
        "variant_id": variant_id,
        "opening_realization": choose(
            (
                "begin_from_the_learner_last_claim",
                "begin_from_one_valid_substep_then_localize_the_break",
                "begin_with_a_bounded_prediction_or_restatement",
            ),
            0,
        ),
        "scaffold_realization": choose(
            (
                "error_localization_then_one_microstep",
                "contrast_two_interpretations_then_test_one",
                "short_representation_then_reverse_check",
            ),
            1,
        ),
        "closing_realization": choose(
            (
                "learner_owned_transfer",
                "independent_reverse_verification",
                "brief_rule_explanation_in_the_learners_words",
            ),
            2,
        ),
        "constraint": (
            "This is a diversity reservation, not new source content. Preserve every supplied "
            "fact, the misconception, the teacher-move order, the pressure profile, and the exact "
            "exchange count. Use materially fresh wording. Do not quote the raw source, a prior "
            "candidate, reviewer prose, or this reservation."
        ),
    }


def load_blueprints(math_path: Path, convo_path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(math_path) + read_jsonl(convo_path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        blueprint_id = row.get("blueprint_id")
        if not isinstance(blueprint_id, str) or blueprint_id in by_id:
            raise ValueError(f"invalid or duplicate blueprint_id {blueprint_id!r}")
        by_id[blueprint_id] = row
    if len(by_id) != 120:
        raise ValueError(f"expected 120 grounded blueprints, found {len(by_id)}")
    return by_id


def load_revised_candidates(data_dir: Path) -> dict[str, dict[str, Any]]:
    paths = [data_dir / "revised_480_candidates.jsonl"] + sorted(
        data_dir.glob("revised_480_policy2_repair*_candidates.jsonl")
    )
    by_id: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            row_id = row.get("id")
            if row.get("status") == "success" and isinstance(row_id, str):
                by_id[row_id] = row
    return by_id


def source_payload(
    assignment: dict[str, Any],
    blueprints: dict[str, dict[str, Any]],
    revised: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if assignment["corpus_role"] == "grounded":
        blueprint_id = str(assignment["blueprint_id"])
        blueprint = blueprints.get(blueprint_id)
        if blueprint is None:
            raise ValueError(f"missing grounded blueprint {blueprint_id}")
        if canonical_hash(blueprint) != assignment.get("blueprint_sha256"):
            raise ValueError(f"grounded blueprint hash changed for {blueprint_id}")
        abstraction = blueprint["rights_safe_source_abstraction"]
        source = blueprint["primary_source"]
        packet = {
            "packet_kind": "reviewed_rights_safe_grounded_blueprint",
            "subject": blueprint["subject"],
            "grade_band": blueprint["grade_band"],
            "learning_objective": abstraction["derived_academic_objective"],
            "learner_difficulty": abstraction["learner_misconception_or_difficulty"],
            "factual_anchors": abstraction["factual_anchors"],
            "teacher_move_sequence": abstraction["teacher_move_sequence"],
            "turn_plan": blueprint["turn_plan"],
            "adab_method_moves": blueprint["adab_plan"]["method_moves"],
            "boundary_constraints": blueprint["adab_plan"]["boundary_constraints"],
            "pressure_pattern": blueprint["pressure_pattern"],
            "transformation_note": (
                "Preserve facts and move order, but create fresh learner/tutor wording and do not "
                "mention or quote provenance."
            ),
        }
        metadata = {
            "source_family": assignment["source_family"],
            "source_id": source["source_id"],
            "source_group_id": source["source_group_id"],
            "source_url": source["source_url"],
            "source_content_sha256": source["content_sha256"],
            "source_license": source["rights"]["license_id"],
            "blueprint_id": blueprint_id,
            "blueprint_sha256": assignment["blueprint_sha256"],
        }
        return packet, metadata

    candidate_id = str(assignment["input_candidate_id"])
    candidate = revised.get(candidate_id)
    if candidate is None:
        raise ValueError(f"missing revised input candidate {candidate_id}")
    if canonical_hash(candidate) != assignment.get("input_candidate_sha256"):
        raise ValueError(f"revised input candidate hash changed for {candidate_id}")
    fixed = candidate["fixed_metadata"]
    packet = {
        "packet_kind": "reviewed_project_owned_synthetic_draft",
        "subject": fixed["subject"],
        "grade_band": fixed["grade_band"],
        "learning_objective": fixed["learning_objective"],
        "scenario_family": fixed["scenario_family"],
        "old_draft_messages": candidate["messages"],
        "semantic_diversity_reservation": diversity_reservation(
            assignment, fixed["learning_objective"]
        ),
        "transformation_note": (
            "Write a materially fresh version. Preserve sound academic content and the requested "
            "pressure profile, remove old register tokens and stock phrases, and do not copy sentences."
        ),
    }
    metadata = {
        "primary_grounding": "none_synthetic_revision",
        "input_candidate_id": candidate_id,
        "input_candidate_sha256": assignment["input_candidate_sha256"],
        "input_judgment_sha256": assignment.get("input_judgment_sha256"),
    }
    return packet, metadata


def fixed_metadata(
    assignment: dict[str, Any], packet: dict[str, Any]
) -> dict[str, Any]:
    return {
        "language": "en",
        "register": "default",
        "subject": assignment["subject"],
        "grade_band": packet["grade_band"],
        "learning_objective": packet["learning_objective"],
        "scenario_family": assignment.get("scenario_family") or packet.get("scenario_family"),
        "pressure_profile": assignment["pressure_profile"],
        "pressure_profile_rule": PROFILE_RULES[assignment["pressure_profile"]],
        "exchange_count": assignment["exchange_count"],
        "exact_total_message_count": 1 + 2 * int(assignment["exchange_count"]),
    }


def realize_one(
    client: Any,
    writer: Any,
    assignment: dict[str, Any],
    packet: dict[str, Any],
    provenance: dict[str, Any],
    max_attempts: int,
    repair_feedback: list[dict[str, str]] | None = None,
    repair_feedback_report_sha256: str | None = None,
    generation_temperature: float | None = None,
) -> dict[str, Any]:
    from grounded_v2_pipeline import model_request_kwargs

    concept = deterministic_concept(assignment)
    generation_assignment = {
        "compiler_assignment_id": assignment["assignment_id"],
        "corpus_role": assignment["corpus_role"],
        "subject": assignment["subject"],
        "pressure_profile": assignment["pressure_profile"],
        "pressure_profile_rule": PROFILE_RULES[assignment["pressure_profile"]],
        "exchange_count": assignment["exchange_count"],
        "required_dialogue_slots": [
            {"exchange": index, "roles_in_order": ["user", "assistant"]}
            for index in range(1, int(assignment["exchange_count"]) + 1)
        ],
        "required_exchange_functions": exchange_function_plan(assignment),
        "expression_class": assignment["expression_class"],
        "permitted_concept": concept,
        "required_system_message": DEFAULT_SYSTEM_MESSAGE,
        "required_action": assignment["required_action"],
    }
    base_prompt = (
        "FROZEN GENERATION ASSIGNMENT\n"
        + json.dumps(generation_assignment, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n\nRIGHTS-SAFE CONTENT PACKET\n"
        + json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True)
    )
    if repair_feedback:
        base_prompt += (
            "\n\nPRIOR-VERSION DETERMINISTIC REPAIR FEEDBACK\n"
            + json.dumps(
                {
                    "instruction": (
                        "These row-local failures apply to the prior version. Regenerate every "
                        "turn and ensure the new version does not reproduce them. They are repair "
                        "constraints, not dialogue content; never mention them to the learner."
                    ),
                    "errors": repair_feedback,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    base_prompt += "\n\nWrite the one conversation and return strict JSON only."
    last_error: Exception | None = None
    last_output: str | None = None
    started = time.monotonic()
    for attempt in range(1, max_attempts + 1):
        retry_note = ""
        if last_error is not None:
            retry_note = (
                "\n\nThe prior attempt was rejected by the deterministic contract: "
                + str(last_error)[:800]
                + ". Regenerate the full conversation; do not explain the repair."
            )
        request = model_request_kwargs(
            writer,
            [
                {"role": "system", "content": WRITER_SYSTEM_PROMPT},
                {"role": "user", "content": base_prompt + retry_note},
            ],
            max_tokens=12000,
            temperature=generation_temperature,
        )
        try:
            response = client.chat.completions.create(**request)
            content = response.choices[0].message.content
            if not isinstance(content, str) or not content.strip():
                raise ValueError("empty writer response")
            last_output = content
            parsed = parse_json_object(content)
            if set(parsed) != {"messages"}:
                raise ValueError("writer output must contain exactly the messages key")
            messages = validate_messages(parsed["messages"], assignment, concept)
            branch_preflight = explicit_branch_preflight(messages, concept)
            metadata = fixed_metadata(assignment, packet)
            return {
                "schema_version": "uae_adab_single_register_candidate.v1",
                "status": "candidate_unreviewed",
                "id": f"uae_adab_default_{assignment['assignment_id']}",
                "compiler_assignment_id": assignment["assignment_id"],
                "corpus_role": assignment["corpus_role"],
                "expression_class": assignment["expression_class"],
                "permitted_concept": concept,
                "messages": messages,
                "conversation_sha256": conversation_hash(messages),
                "fixed_metadata": metadata,
                "provenance": provenance,
                "generation": {
                    "writer_contract_version": WRITER_CONTRACT_VERSION,
                    "writer_family": writer.family,
                    "writer_model": writer.model,
                    "writer_prompt_sha256": hashlib.sha256(
                        WRITER_SYSTEM_PROMPT.encode("utf-8")
                    ).hexdigest(),
                    "assignment_sha256": canonical_hash(assignment),
                    "content_packet_sha256": canonical_hash(packet),
                    "repair_feedback_report_sha256": repair_feedback_report_sha256,
                    "repair_feedback_sha256": (
                        canonical_hash(repair_feedback) if repair_feedback else None
                    ),
                    "repair_feedback_error_count": len(repair_feedback or []),
                    "explicit_branch_preflight": branch_preflight,
                    "generation_temperature": generation_temperature,
                    "attempts": attempt,
                    "generated_at": utc_now(),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                },
            }
        except Exception as exc:
            last_error = exc
            if attempt < max_attempts:
                time.sleep(min(6.0, (2 ** (attempt - 1)) * random.uniform(0.5, 1.2)))
    error = RuntimeError(f"writer failed after {max_attempts} attempts: {last_error}")
    setattr(error, "raw_output", last_output)
    raise error


def append_event(path: Path, event: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def load_events(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    events: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        event = json.loads(line)
        if not isinstance(event, dict) or not isinstance(event.get("work_id"), str):
            raise ValueError(f"{path}:{line_number} is not a valid event")
        events[event["work_id"]] = event
    return events


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")


def materialize(
    events: dict[str, dict[str, Any]], candidates_path: Path, rejected_path: Path
) -> tuple[int, int]:
    ordered = sorted(events.values(), key=lambda event: (event["input_index"], event["work_id"]))
    candidates = [event["candidate"] for event in ordered if event["event_type"] == "candidate"]
    rejected = [event["rejection"] for event in ordered if event["event_type"] == "rejected"]
    write_jsonl(candidates_path, candidates)
    write_jsonl(rejected_path, rejected)
    return len(candidates), len(rejected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist", type=Path, default=DEFAULT_WORKLIST)
    parser.add_argument("--math-blueprints", type=Path, default=DEFAULT_MATH)
    parser.add_argument("--convo-blueprints", type=Path, default=DEFAULT_CONVO)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
    parser.add_argument("--writer", default=DEFAULT_WRITER, help="FAMILY=MODEL")
    parser.add_argument("--role", choices=("all", "grounded", "revised"), default="all")
    parser.add_argument(
        "--assignment-id",
        action="append",
        default=None,
        help="Generate only this compiler assignment; repeat for a stratified pilot.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-rejected", action="store_true")
    parser.add_argument(
        "--generation-variant",
        help=(
            "Optional fresh-lineage variant label. It adds a deterministic structural diversity "
            "reservation to the content packet and is checkpoint-bound."
        ),
    )
    parser.add_argument(
        "--generation-temperature",
        type=float,
        help="Optional writer sampling temperature, checkpoint-bound for fresh variants.",
    )
    parser.add_argument(
        "--repair-feedback-report",
        type=Path,
        help=(
            "Hash-bound deterministic audit whose row-local errors are injected as constraints; "
            "corpus-only n-gram errors are excluded."
        ),
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1 or args.max_attempts < 1:
        raise SystemExit("--workers and --max-attempts must be positive")
    if args.generation_variant is not None and (
        not args.generation_variant
        or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in args.generation_variant)
    ):
        raise SystemExit("--generation-variant must use only letters, digits, '_' or '-'")
    if args.generation_temperature is not None and not 0 <= args.generation_temperature <= 2:
        raise SystemExit("--generation-temperature must be between 0 and 2")
    from grounded_v2_pipeline import ModelSpec, model_request_kwargs

    writer = ModelSpec.parse(args.writer)
    model_request_kwargs(writer, [], 1, None)
    worklist = read_json(args.worklist)
    assignments = worklist.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != 600:
        raise SystemExit("single-register worklist must contain exactly 600 assignments")
    if canonical_hash(assignments) != worklist.get("assignment_set_sha256"):
        raise SystemExit("single-register assignment hash is invalid")
    selected = [
        assignment
        for assignment in assignments
        if args.role == "all" or assignment.get("corpus_role") == args.role
    ]
    if args.assignment_id:
        requested = set(args.assignment_id)
        selected = [assignment for assignment in selected if assignment["assignment_id"] in requested]
        found = {assignment["assignment_id"] for assignment in selected}
        missing = sorted(requested - found)
        if missing:
            raise SystemExit(f"requested assignment IDs are absent from the selected role: {missing}")
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        selected = selected[: args.limit]

    repair_feedback: dict[str, list[dict[str, str]]] = {}
    repair_feedback_report_sha256: str | None = None
    if args.repair_feedback_report:
        try:
            repair_feedback, repair_feedback_report_sha256 = load_repair_feedback(
                args.repair_feedback_report,
                {assignment["assignment_id"] for assignment in selected},
                worklist["assignment_set_sha256"],
            )
        except Exception as exc:
            raise SystemExit(f"invalid repair-feedback report: {exc}") from exc

    blueprints = load_blueprints(args.math_blueprints, args.convo_blueprints)
    revised = load_revised_candidates(args.data_dir)
    inputs: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    errors: list[str] = []
    for assignment in selected:
        try:
            packet, provenance = source_payload(
                assignment, blueprints, revised
            )
            if args.generation_variant:
                reservation = recovery_variant_reservation(
                    assignment, args.generation_variant
                )
                if args.generation_variant.endswith("_h"):
                    reservation["clean_room_surface_contract"] = (
                        "Use compact, varied sentences. In each tutor turn: one short anchor, one "
                        "new scaffold, then a question of at most sixteen words. Do not echo a "
                        "learner sentence or factual-anchor sentence before advancing it."
                    )
                elif args.generation_variant.endswith("_i"):
                    reservation["clean_room_surface_contract"] = (
                        "Use a visibly different realization: paraphrase the learner's method, "
                        "change the explanatory representation within the allowed turn plan, and "
                        "make checks learner-produced. Avoid stock validation openings and long restatements."
                    )
                elif args.generation_variant.endswith("_j"):
                    reservation["clean_room_surface_contract"] = (
                        "Write a sparse clean-room dialogue. Tutor turns should normally be 35-80 "
                        "words and learner turns 8-35 words. Never repeat the learner's sentence or "
                        "the full problem before correcting it. Use labels, a compact equation, a "
                        "contrast, or a prediction to advance each source-prescribed move."
                    )
                elif args.generation_variant.endswith("_k"):
                    reservation["clean_room_surface_contract"] = (
                        "Realize the same source trajectory through a different instructional form: "
                        "alternate among a two-column comparison, one worked micro-step, a counterexample, "
                        "and learner-generated verification where the academic plan permits. Use short "
                        "sentences, no generic praise, and no paraphrased recap of the preceding learner turn."
                    )
                packet = {
                    **packet,
                    "fresh_candidate_variant": reservation,
                }
                provenance = {
                    **provenance,
                    "recovery_generation_variant": args.generation_variant,
                }
            inputs[assignment["assignment_id"]] = (packet, provenance)
            deterministic_concept(assignment)
        except Exception as exc:
            errors.append(f"{assignment.get('assignment_id')}: {type(exc).__name__}: {exc}")
    if errors:
        print(json.dumps({"ok": False, "preflight_errors": errors}, indent=2))
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "preflight_assignments": len(selected),
                "role": args.role,
                "writer": {"family": writer.family, "model": writer.model},
                "writer_contract_version": WRITER_CONTRACT_VERSION,
                "worklist_assignment_set_sha256": worklist["assignment_set_sha256"],
                "repair_feedback_report_sha256": repair_feedback_report_sha256,
                "repair_feedback_rows": len(repair_feedback),
                "repair_feedback_errors": sum(map(len, repair_feedback.values())),
                "generation_variant": args.generation_variant,
                "generation_temperature": args.generation_temperature,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.preflight_only:
        return 0

    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("TFY_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY or TFY_API_KEY is not set")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=240, max_retries=0)
    events = load_events(args.events)
    prompt_hash = hashlib.sha256(WRITER_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assignment_by_id = {assignment["assignment_id"]: assignment for assignment in selected}
    for work_id, event in events.items():
        assignment = assignment_by_id.get(work_id)
        if assignment is None:
            continue
        if event.get("assignment_sha256") != canonical_hash(assignment):
            raise SystemExit(f"checkpoint collision: assignment changed for {work_id}")
        if event.get("writer_model") != writer.model or event.get("prompt_sha256") != prompt_hash:
            raise SystemExit(f"checkpoint collision: writer configuration changed for {work_id}")
        if event.get("writer_contract_version") != WRITER_CONTRACT_VERSION:
            raise SystemExit(f"checkpoint collision: writer contract changed for {work_id}")
        if event.get("repair_feedback_report_sha256") != repair_feedback_report_sha256:
            raise SystemExit(f"checkpoint collision: repair feedback changed for {work_id}")
        if event.get("generation_variant") != args.generation_variant:
            raise SystemExit(f"checkpoint collision: generation variant changed for {work_id}")
        if event.get("generation_temperature") != args.generation_temperature:
            raise SystemExit(f"checkpoint collision: generation temperature changed for {work_id}")

    pending = []
    for input_index, assignment in enumerate(selected):
        prior = events.get(assignment["assignment_id"])
        if prior is None or (args.retry_rejected and prior.get("event_type") == "rejected"):
            pending.append((input_index, assignment))
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for input_index, assignment in pending:
            packet, provenance = inputs[assignment["assignment_id"]]
            future = executor.submit(
                realize_one,
                client,
                writer,
                assignment,
                packet,
                provenance,
                args.max_attempts,
                repair_feedback.get(assignment["assignment_id"]),
                repair_feedback_report_sha256,
                args.generation_temperature,
            )
            futures[future] = (input_index, assignment)
        for future in as_completed(futures):
            input_index, assignment = futures[future]
            work_id = assignment["assignment_id"]
            common = {
                "work_id": work_id,
                "input_index": input_index,
                "assignment_sha256": canonical_hash(assignment),
                "writer_model": writer.model,
                "writer_contract_version": WRITER_CONTRACT_VERSION,
                "prompt_sha256": prompt_hash,
                "repair_feedback_report_sha256": repair_feedback_report_sha256,
                "generation_variant": args.generation_variant,
                "generation_temperature": args.generation_temperature,
                "recorded_at": utc_now(),
            }
            try:
                candidate = future.result()
                event = {**common, "event_type": "candidate", "candidate": candidate}
                state = "CANDIDATE"
            except Exception as exc:
                raw = getattr(exc, "raw_output", None)
                event = {
                    **common,
                    "event_type": "rejected",
                    "rejection": {
                        "status": "writer_rejected",
                        "compiler_assignment_id": work_id,
                        "error": f"{type(exc).__name__}: {exc}",
                        "raw_output": raw[:16000] if isinstance(raw, str) else None,
                        "rejected_at": utc_now(),
                    },
                }
                state = "REJECT"
            append_event(args.events, event, lock)
            events[work_id] = event
            print(f"{state} {work_id}", flush=True)

    candidate_count, reject_count = materialize(events, args.candidates, args.rejected)
    print(
        json.dumps(
            {
                "selected": len(selected),
                "attempted_this_run": len(pending),
                "candidates_in_ledger": candidate_count,
                "rejects_in_ledger": reject_count,
                "events": str(args.events),
                "candidates": str(args.candidates),
                "rejected": str(args.rejected),
            },
            indent=2,
            sort_keys=True,
        )
    )
    selected_ids = {assignment["assignment_id"] for assignment in selected}
    selected_rejects = sum(
        events.get(work_id, {}).get("event_type") == "rejected" for work_id in selected_ids
    )
    return 1 if selected_rejects else 0


if __name__ == "__main__":
    raise SystemExit(main())
