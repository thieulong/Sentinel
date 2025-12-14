import json
import re
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from camel.agents import ChatAgent
from camel.messages import BaseMessage

from .extract import Triplet, norm, normalize_relation


ENRICHER_SYSTEM_PROMPT = r"""
You are "Enricher", a graph architect for a personal knowledge graph.

Input:
- A "clean_text" summary (already filtered by Curator).
- A list of Curator "candidates" in the form:
  {"subj":"USER","rel":"RELATION","obj":"OBJECT","confidence":0..1}

Goal:
Transform flat candidates into a better-structured graph by:
1) Reifying composite facts into entity-centric structure.
   Example: instead of USER -> MEETING_TIME -> today 4pm,
   create an EVENT node and attach time to that EVENT node.

2) Splitting combined strings into smaller nodes when explicitly present.
   Example: "Melbourne, Australia" may be split into "Melbourne" and "Australia" and add:
     Melbourne -[LOCATED_IN]-> Australia
   This is allowed because the country is explicitly present in the same text.

3) Creating topical branches.
   Example: "PhD in AI and multi-agent systems" should become:
     USER -[HAS_PROGRAM]-> PhD
     PhD -[HAS_FIELD]-> AI
     PhD -[HAS_FIELD]-> multi-agent systems
   Only do this when the text explicitly mentions these components.

CRITICAL RULES:
- Do NOT add new facts that are not supported by the inputs.
- Do NOT use web search.
- Common-sense is allowed only for restructuring, not for inventing new entities.
- If something is ambiguous, keep the flat version rather than guessing.
- Output JSON only, no markdown.

OUTPUT SCHEMA (strict JSON object):
{
  "entities": [
    {
      "key": "string",               // stable key you create, like "event_1", "program_1"
      "label": "string",             // human label like "meeting", "PhD"
      "type": "EVENT|TASK|PROGRAM|PLACE|CONCEPT|OTHER",
      "time_text": "string|null",    // optional, like "today 4PM"
      "details": {"k":"v"}           // optional dictionary
    }
  ],
  "relations": [
    {
      "subj": "string",
      "rel": "string",
      "obj": "string",
      "confidence": 0.0-1.0,
      "derived": true
    }
  ],
  "notes": ["string", ...]
}

IDENTITY:
- "USER" refers to the human user (not the assistant).

RELATION VOCABULARY (prefer these):
- HAS_EVENT, HAS_TASK, HAS_PROGRAM
- AT_TIME, ABOUT, LOCATED_IN
- HAS_FIELD, HAS_TOPIC
- NAME, AGE, FROM, HOMETOWN, LIVES_IN (keep if already correct)

EXAMPLES:

Example A (event structuring)
clean_text:
"User has a meeting today at 4PM and wants a reminder about cooking."
candidates:
[
 {"subj":"USER","rel":"HAS_MEETING","obj":"meeting","confidence":0.85},
 {"subj":"USER","rel":"MEETING_TIME","obj":"today 4PM","confidence":0.90},
 {"subj":"USER","rel":"MEETING_TOPIC","obj":"discuss what to cook for tonight","confidence":0.80}
]
Good output:
{
 "entities":[{"key":"event_1","label":"meeting","type":"EVENT","time_text":"today 4PM","details":{"topic":"discuss what to cook for tonight"}}],
 "relations":[
   {"subj":"USER","rel":"HAS_EVENT","obj":"event_1","confidence":0.90,"derived":true},
   {"subj":"event_1","rel":"AT_TIME","obj":"today 4PM","confidence":0.90,"derived":true},
   {"subj":"event_1","rel":"ABOUT","obj":"discuss what to cook for tonight","confidence":0.80,"derived":true}
 ],
 "notes":["Reified meeting into an EVENT node; attached time/topic to the event."]
}

Example B (place split)
clean_text:
"User lives in Melbourne, Australia."
candidates:
[
 {"subj":"USER","rel":"LIVES_IN","obj":"Melbourne, Australia","confidence":0.90}
]
Good output:
{
 "entities":[
   {"key":"place_1","label":"Melbourne","type":"PLACE","time_text":null,"details":{}},
   {"key":"place_2","label":"Australia","type":"PLACE","time_text":null,"details":{}}
 ],
 "relations":[
   {"subj":"USER","rel":"LIVES_IN","obj":"place_1","confidence":0.90,"derived":true},
   {"subj":"place_1","rel":"LOCATED_IN","obj":"place_2","confidence":0.90,"derived":true}
 ],
 "notes":["Split explicit 'Melbourne, Australia' into two places and added containment."]
}

Example C (program fields)
clean_text:
"User is doing a PhD in AI and multi-agent systems."
candidates:
[
 {"subj":"USER","rel":"STUDIES_AT","obj":"PhD in AI and multi-agent systems","confidence":0.90}
]
Good output:
{
 "entities":[
   {"key":"program_1","label":"PhD","type":"PROGRAM","time_text":null,"details":{}},
   {"key":"concept_1","label":"AI","type":"CONCEPT","time_text":null,"details":{}},
   {"key":"concept_2","label":"multi-agent systems","type":"CONCEPT","time_text":null,"details":{}}
 ],
 "relations":[
   {"subj":"USER","rel":"HAS_PROGRAM","obj":"program_1","confidence":0.90,"derived":true},
   {"subj":"program_1","rel":"HAS_FIELD","obj":"concept_1","confidence":0.85,"derived":true},
   {"subj":"program_1","rel":"HAS_FIELD","obj":"concept_2","confidence":0.85,"derived":true}
 ],
 "notes":["Extracted PROGRAM and fields from explicit text."]
}

TASK:
Given clean_text and candidates, output JSON with a well-structured graph.
If you cannot improve structure, return empty entities and just rewrite relations that mirror candidates.
"""


@dataclass(frozen=True)
class EnricherResult:
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
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


def make_enricher_agent(model) -> ChatAgent:
    system_msg = BaseMessage.make_assistant_message(
        role_name="Enricher",
        content=ENRICHER_SYSTEM_PROMPT,
    )
    return ChatAgent(system_message=system_msg, model=model)


def run_enricher(enricher_agent: ChatAgent, clean_text: str, candidates: List[Dict[str, Any]]) -> EnricherResult:
    payload = {
        "clean_text": clean_text,
        "candidates": candidates or [],
    }

    prompt = (
        "Input JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n\n"
        "Output JSON only in the required schema."
    )

    resp = enricher_agent.step(prompt)
    raw = resp.msg.content if resp and resp.msg else ""
    obj = _safe_json_parse(raw)

    if not obj:
        return EnricherResult(entities=[], relations=[], notes=["Enricher output was not valid JSON."])

    entities = obj.get("entities", [])
    relations = obj.get("relations", [])
    notes = obj.get("notes", [])

    if not isinstance(entities, list):
        entities = []
    if not isinstance(relations, list):
        relations = []
    if not isinstance(notes, list):
        notes = []

    # Normalize relations to a safe shape
    norm_relations: List[Dict[str, Any]] = []
    for r in relations:
        if not isinstance(r, dict):
            continue
        subj = str(r.get("subj", "")).strip()
        rel = str(r.get("rel", "")).strip()
        objv = str(r.get("obj", "")).strip()
        if not subj or not rel or not objv:
            continue
        try:
            conf = float(r.get("confidence", 0.8))
        except Exception:
            conf = 0.8
        conf = max(0.0, min(conf, 1.0))
        derived = bool(r.get("derived", True))
        norm_relations.append({"subj": subj, "rel": rel, "obj": objv, "confidence": conf, "derived": derived})

    norm_notes = [str(n) for n in notes if str(n).strip()]

    return EnricherResult(entities=entities, relations=norm_relations, notes=norm_notes)


def _stable_entity_id(entity_type: str, label: str) -> str:
    """
    Create a stable-ish id for an entity based on type+label.
    Used to map Enricher keys to Neo4j-safe ids.
    """
    base = f"{entity_type}:{label}".strip().lower()
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:10]
    return norm(f"{entity_type.lower()}_{label}_{h}")


def build_triplets_from_enricher(
    enricher: EnricherResult,
    user_canonical_id: str,
) -> List[Triplet]:
    """
    Convert Enricher entities/relations into Triplet objects.

    We map:
    - subject/object "USER" -> user_canonical_id
    - entity keys like "event_1" -> stable node id derived from entity type+label
    """
    key_to_node: Dict[str, str] = {}

    # Map entity keys to stable ids
    for e in enricher.entities:
        if not isinstance(e, dict):
            continue
        key = str(e.get("key", "")).strip()
        label = str(e.get("label", "")).strip()
        etype = str(e.get("type", "OTHER")).strip().upper()
        if not key or not label:
            continue
        key_to_node[key] = _stable_entity_id(etype, label)

    out: List[Triplet] = []

    def map_node(x: str) -> str:
        x = x.strip()
        if x.upper() == "USER":
            return norm(user_canonical_id)
        if x in key_to_node:
            return key_to_node[x]
        return norm(x)

    for r in enricher.relations:
        subj = map_node(str(r.get("subj", "")))
        rel = normalize_relation(str(r.get("rel", "RELATED_TO")))
        objv = map_node(str(r.get("obj", "")))

        if not subj or not objv:
            continue
        if subj == objv:
            continue

        out.append(Triplet(subj=subj, rel=rel, obj=objv))

    # Deduplicate preserve order
    seen = set()
    deduped = []
    for t in out:
        if t in seen:
            continue
        seen.add(t)
        deduped.append(t)
    return deduped