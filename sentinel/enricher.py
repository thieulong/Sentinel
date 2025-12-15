import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from camel.agents import ChatAgent
from camel.messages import BaseMessage

from .extract import Triplet, norm, normalize_relation


ENRICHER_SYSTEM_PROMPT = r"""
You are Enricher, a knowledge-graph structuring agent.

INPUT:
- clean_text: a cleaned factual summary
- candidates: list of extracted factual relations

GOAL:
Restructure facts into better graph form WITHOUT inventing new facts.

STRICT RULES:
- DO NOT invent information.
- DO NOT use web search.
- DO NOT guess missing links.
- ONLY reorganize facts explicitly present.
- If unsure, keep the structure flat.

IMPORTANT RULES:
- USER always refers to the same real person.
- DO NOT create abstract placeholder entities.
- For NAME and AGE, keep literal values (e.g., "Paul", "24").
- If object is "City, Country", you MAY also add City LOCATED_IN Country.
- DO NOT merge unrelated concepts.

OUTPUT:
Return JSON ONLY in this schema:

{
  "relations": [
    {
      "subj": "USER|string",
      "rel": "string",
      "obj": "string",
      "confidence": 0.0-1.0,
      "derived": true
    }
  ],
  "notes": ["string"]
}

Preferred relations:
NAME, AGE, FROM, HOMETOWN, LIVES_IN, STUDIES_AT, WORKS_AS,
DEGREE, PROGRAM,
RESEARCH_AREA, HAS_FIELD, HAS_BACKGROUND, LOCATED_IN
"""


@dataclass(frozen=True)
class EnricherResult:
    relations: List[Dict[str, Any]]
    notes: List[str]


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def make_enricher_agent(model) -> ChatAgent:
    system_msg = BaseMessage.make_assistant_message(
        role_name="Enricher",
        content=ENRICHER_SYSTEM_PROMPT,
    )
    return ChatAgent(system_message=system_msg, model=model)


def run_enricher(
    agent: ChatAgent,
    clean_text: str,
    candidates: List[Dict[str, Any]],
) -> EnricherResult:
    payload = {
        "clean_text": clean_text,
        "candidates": candidates or [],
    }

    prompt = (
        "Input:\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n\n"
        "Return JSON in the required schema only."
    )

    resp = agent.step(prompt)
    raw = resp.msg.content if resp and resp.msg else ""
    data = _extract_json(raw)

    if not data:
        return EnricherResult(relations=[], notes=["Invalid JSON from Enricher."])

    relations = data.get("relations", []) or []
    notes = data.get("notes", []) or []

    clean_relations: List[Dict[str, Any]] = []
    for r in relations:
        if not isinstance(r, dict):
            continue
        subj = str(r.get("subj", "")).strip()
        rel = str(r.get("rel", "")).strip()
        obj = str(r.get("obj", "")).strip()
        if not subj or not rel or not obj:
            continue
        try:
            conf = float(r.get("confidence", 0.8))
        except Exception:
            conf = 0.8
        clean_relations.append(
            {
                "subj": subj,
                "rel": rel,
                "obj": obj,
                "confidence": max(0.0, min(conf, 1.0)),
                "derived": bool(r.get("derived", True)),
            }
        )

    return EnricherResult(relations=clean_relations, notes=[str(n) for n in notes])


# ---------------------------
# Triplet building fixes
# ---------------------------

_PLACE_RELS = {"LIVES_IN", "FROM", "HOMETOWN", "LOCATED_IN"}
_PROGRAM_HINT_RELS = {"DEGREE", "PROGRAM"}  # from Curator/Enricher
_FIELD_RELS = {"RESEARCH_AREA", "HAS_FIELD"}  # we normalize under program when available


def _split_city_country(text: str) -> Optional[Tuple[str, str]]:
    # Conservative: split only once at the first comma
    if not text or "," not in text:
        return None
    left, right = [p.strip() for p in text.split(",", 1)]
    if not left or not right:
        return None
    return left, right


def _split_field_list(text: str) -> List[str]:
    """
    Split a field list like:
      "LLMs, multi-agent systems, and knowledge graph"
    into ["LLMs", "multi-agent systems", "knowledge graph"].
    Conservative: only splits on commas and ' and '.
    """
    t = (text or "").strip()
    if not t:
        return []
    t = t.replace(" & ", " and ")
    t = t.replace(", and ", ", ")
    parts = []
    for chunk in t.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        # Further split "A and B" if present
        if " and " in chunk:
            subparts = [p.strip() for p in chunk.split(" and ") if p.strip()]
            parts.extend(subparts)
        else:
            parts.append(chunk)
    # Dedup preserve order
    seen = set()
    out = []
    for p in parts:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def build_triplets_from_enricher(
    enricher: EnricherResult,
    user_canonical_id: str,
) -> List[Triplet]:
    """
    Convert Enricher relations into Triplets with:
    - Place normalization: USER->City plus City LOCATED_IN Country (no City__Country nodes)
    - Program normalization: USER->HAS_PROGRAM->Program and Program->HAS_FIELD->Fields
    - Prevent accidental comma-splitting for non-place relations
    """
    user_node = norm(user_canonical_id)

    # 1) Read relations and collect program + fields first
    degree_program: Optional[str] = None
    collected_fields: List[str] = []

    # We also keep “raw triplets” temporarily for relations we will store as-is
    raw_pairs: List[Tuple[str, str, str]] = []

    for r in enricher.relations:
        subj_raw = str(r.get("subj", "USER")).strip()
        rel_raw = str(r.get("rel", "")).strip()
        obj_raw = str(r.get("obj", "")).strip()

        if not rel_raw or not obj_raw:
            continue

        subj = user_node if subj_raw.upper() == "USER" else norm(subj_raw)
        rel = normalize_relation(rel_raw)

        # Program detection
        if rel in _PROGRAM_HINT_RELS and subj == user_node:
            # e.g. DEGREE -> "PhD"
            degree_program = obj_raw.strip()
            continue

        # Also detect packed "PhD in X, Y, Z"
        if rel in _FIELD_RELS and subj == user_node:
            low = obj_raw.lower()
            if low.startswith("phd in "):
                degree_program = "PhD"
                fields_part = obj_raw[7:].strip()
                collected_fields.extend(_split_field_list(fields_part))
                continue

        # Collect research areas as fields (we may attach under program later)
        if rel in _FIELD_RELS and subj == user_node:
            collected_fields.append(obj_raw)
            continue

        # Everything else store later
        raw_pairs.append((subj, rel, obj_raw))

    # 2) Build final triplets
    out: List[Triplet] = []

    # 2a) Store place relations with splitting
    for subj, rel, obj_raw in raw_pairs:
        if rel in _PLACE_RELS:
            split = _split_city_country(obj_raw)
            if split:
                city, country = split
                city_id = norm(city)
                country_id = norm(country)

                # USER -> City (not City__Country)
                out.append(Triplet(subj=subj, rel=rel, obj=city_id))

                # City -> Country
                out.append(Triplet(subj=city_id, rel="LOCATED_IN", obj=country_id))
                continue

        # default: store as-is (no splitting!)
        out.append(Triplet(subj=subj, rel=rel, obj=norm(obj_raw)))

    # 2b) Program + fields normalization
    if degree_program:
        program_node = norm(degree_program)
        out.append(Triplet(subj=user_node, rel="HAS_PROGRAM", obj=program_node))

        # Attach fields under program
        for f in _split_field_list(", ".join(collected_fields)) if len(collected_fields) > 1 else collected_fields:
            f = (f or "").strip()
            if not f:
                continue
            out.append(Triplet(subj=program_node, rel="HAS_FIELD", obj=norm(f)))
    else:
        # No program found, keep fields directly under user (still useful)
        for f in collected_fields:
            f = (f or "").strip()
            if not f:
                continue
            out.append(Triplet(subj=user_node, rel="RESEARCH_AREA", obj=norm(f)))

    # 3) Deduplicate preserve order and avoid trivial NAME self-loop
    seen = set()
    deduped: List[Triplet] = []
    for t in out:
        if t.rel == "NAME" and t.subj == t.obj:
            continue
        if t in seen:
            continue
        seen.add(t)
        deduped.append(t)

    return deduped