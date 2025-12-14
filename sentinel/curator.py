import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from camel.agents import ChatAgent
from camel.messages import BaseMessage


CURATOR_SYSTEM_PROMPT = r"""
You are "Curator", a strict memory filter for a personal assistant.

Your job:
Given a user's message, extract ONLY the parts that are worth remembering long-term.
Return:
1) clean_text: a short rewrite that contains only memory-worthy information.
2) candidates: a structured list of candidate facts (triples) grounded ONLY in the user's text.
3) notes: brief notes about what you removed and why.

ABSOLUTE RULES:
- Do NOT invent facts.
- Do NOT add common knowledge not stated by the user.
- Do NOT infer geography, taxonomy, institutions, or relationships that are not explicitly stated.
- If unsure, omit it.
- Output JSON only, no markdown.

WHAT IS MEMORY-WORTHY:
A) Stable personal profile facts:
- name, age, origin, hometown, nationality (if explicitly stated)
- current living location (if explicitly stated)
- education/work status (degree type, job role) if explicitly stated
- preferences and persistent likes/dislikes ("I love X", "I prefer Y")

B) Plans, commitments, and events:
- meetings, appointments, deadlines, reminders, scheduled times
- tasks the user intends to do
- IMPORTANT: keep the purpose/topic of events/tasks even if phrased casually,
  e.g. "it'll be about X", "it's about X", "to discuss X", "so we can X".
  This is NOT filler; it is the event/task's meaning.

C) Relationships between explicit entities that the user states directly:
- "X is my hometown"
- "I live in Y"
- "I study in Z"
- "My research area is A and B"

WHAT TO REMOVE (NOT MEMORY-WORTHY):
- greetings and smalltalk directed at the assistant ("how are you", "what's up")
- rhetorical questions
- the user's opinions/questions about external things unless they express a stable preference
  Example: "Is Melbourne worth living?" -> remove (question)
  Example: "I love living in Melbourne" -> keep (preference)
- assistant-directed requests that are not durable facts ("can you answer this")

RELATION LABELS:
Prefer these labels (use UPPER_SNAKE_CASE):
- NAME, AGE
- FROM, HOMETOWN, NATIONALITY
- LIVES_IN
- STUDIES_IN, WORKS_AS
- DEGREE, PROGRAM, RESEARCH_AREA
- HAS_EVENT, HAS_TASK
- EVENT_TIME, TASK_TIME
- EVENT_TOPIC, TASK_TOPIC
- REMINDER_REQUESTED

IMPORTANT IDENTITY RULE:
Use subj as "USER" for the user. Do NOT use assistant as a subject.
Use obj as short natural strings (not normalized).

OUTPUT SCHEMA (strict JSON object):
{
  "clean_text": "string",
  "candidates": [
    {"subj":"USER","rel":"REL","obj":"OBJ","confidence":0.0-1.0}
  ],
  "notes": ["string", ...]
}

EXAMPLES:

Example 1 (smalltalk removed)
User: "Hello, how are you doing today?"
Output:
{
  "clean_text": "",
  "candidates": [],
  "notes": ["Removed smalltalk: greeting and questions directed at the assistant."]
}

Example 2 (profile + location + degree)
User: "My name is Paul. I'm 24. I'm from Vietnam. I'm living in Melbourne, Australia doing my PhD in AI and multi-agent systems."
Output:
{
  "clean_text": "User name is Paul. User is 24 years old. User is from Vietnam. User lives in Melbourne, Australia. User is doing a PhD in AI and multi-agent systems.",
  "candidates": [
    {"subj":"USER","rel":"NAME","obj":"Paul","confidence":0.95},
    {"subj":"USER","rel":"AGE","obj":"24","confidence":0.90},
    {"subj":"USER","rel":"FROM","obj":"Vietnam","confidence":0.85},
    {"subj":"USER","rel":"LIVES_IN","obj":"Melbourne, Australia","confidence":0.90},
    {"subj":"USER","rel":"DEGREE","obj":"PhD","confidence":0.85},
    {"subj":"USER","rel":"RESEARCH_AREA","obj":"AI","confidence":0.80},
    {"subj":"USER","rel":"RESEARCH_AREA","obj":"multi-agent systems","confidence":0.80}
  ],
  "notes": ["Removed questions/opinions not stated as stable preferences."]
}

Example 3 (event with topic kept)
User: "I have a meeting today around 4PM, can you set a reminder for me then? it'll be about discussing what to cook for tonight."
Output:
{
  "clean_text": "User has a meeting today at 4PM and wants a reminder. The meeting is about discussing what to cook for tonight.",
  "candidates": [
    {"subj":"USER","rel":"HAS_EVENT","obj":"meeting","confidence":0.85},
    {"subj":"USER","rel":"EVENT_TIME","obj":"today 4PM","confidence":0.90},
    {"subj":"USER","rel":"REMINDER_REQUESTED","obj":"true","confidence":0.90},
    {"subj":"USER","rel":"EVENT_TOPIC","obj":"discussing what to cook for tonight","confidence":0.85}
  ],
  "notes": ["Removed filler smalltalk; kept event topic because it defines the event."]
}

TASK:
Read the user's message and produce JSON in the exact schema.
"""


@dataclass
class CuratorResult:
    clean_text: str
    candidates: List[Dict[str, Any]]
    notes: List[str]


def _extract_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        return t
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return m.group(0).strip() if m else None


def _safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    js = _extract_json_object(text)
    if not js:
        return None
    try:
        obj = json.loads(js)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def make_curator_agent(model) -> ChatAgent:
    system_msg = BaseMessage.make_assistant_message(
        role_name="Curator",
        content=CURATOR_SYSTEM_PROMPT,
    )
    return ChatAgent(system_message=system_msg, model=model)


def run_curator(curator_agent: ChatAgent, user_text: str) -> CuratorResult:
    prompt = (
        "User message:\n"
        f"{user_text}\n\n"
        "Return JSON only following the schema."
    )
    resp = curator_agent.step(prompt)
    raw = resp.msg.content if resp and resp.msg else ""

    obj = _safe_json_parse(raw)
    if not obj:
        # Hard fallback: return empty rather than hallucinate
        return CuratorResult(
            clean_text="",
            candidates=[],
            notes=["Curator output was not valid JSON; stored nothing."],
        )

    clean_text = str(obj.get("clean_text", "") or "").strip()
    candidates = obj.get("candidates", []) or []
    notes = obj.get("notes", []) or []

    if not isinstance(candidates, list):
        candidates = []
    if not isinstance(notes, list):
        notes = []

    # Normalize candidate dicts
    norm_cands: List[Dict[str, Any]] = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        subj = str(c.get("subj", "USER")).strip() or "USER"
        rel = str(c.get("rel", "FACT")).strip()
        objv = str(c.get("obj", "")).strip()
        if not rel or not objv:
            continue
        try:
            conf = float(c.get("confidence", 0.8))
        except Exception:
            conf = 0.8
        conf = max(0.0, min(conf, 1.0))
        norm_cands.append({"subj": subj, "rel": rel, "obj": objv, "confidence": conf})

    norm_notes = [str(n) for n in notes if str(n).strip()]

    return CuratorResult(clean_text=clean_text, candidates=norm_cands, notes=norm_notes)