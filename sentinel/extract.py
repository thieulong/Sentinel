from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
import re

from camel.agents import KnowledgeGraphAgent
from camel.loaders import UnstructuredIO
from camel.storages import Neo4jGraph

from .utils import norm

@dataclass(frozen=True)
class Triplet:
    subj: str
    rel: str
    obj: str


def make_timestamp() -> str:
    from datetime import datetime
    return datetime.now().astimezone().isoformat(timespec="seconds")


def extract_relationships(
    kg_agent: KnowledgeGraphAgent,
    uio: UnstructuredIO,
    text: str,
    element_id: str,
):
    element = uio.create_element_from_text(text=text, element_id=element_id)
    graph_element = kg_agent.run(element, parse_graph_elements=True)
    return graph_element.relationships


def normalize_relation(rel_type: str) -> str:
    """Canonicalize relation type into Neo4j-friendly UPPER_SNAKE_CASE."""
    r = (rel_type or "").strip()
    if not r:
        return "RELATED_TO"

    # Turn camelCase / PascalCase into snake_case first: StudiesIn -> Studies_In
    r = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", r)

    # Now normalize to uppercase snake-like format
    r = "".join(ch if ch.isalnum() else "_" for ch in r).upper()
    while "__" in r:
        r = r.replace("__", "_")

    r = r.strip("_")
    return r or "RELATED_TO"


def is_garbage_node(x: str) -> bool:
    if not x:
        return True
    bad = {
        "OBJECT", "THING", "SOMETHING", "SOMEONE", "UNKNOWN", "NONE", "NULL",
        "NODE", "ITEM",
    }
    return x.strip().upper() in bad


_USER_REF_PATTERNS = [
    r"^i$",
    r"^me$",
    r"^my$",
    r"^myself$",
    r"^user$",
    r"^the\s*user$",
    r"^speaker$",
    r"^the\s*speaker$",
]


def _normalize_surface(text: str) -> str:
    """Normalize raw ids/surface forms for matching (not for storage)."""
    t = (text or "").strip().lower()
    # Treat underscores and punctuation like spaces for matching.
    t = re.sub(r"[^a-z0-9]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_user_reference(raw_subj_id: str) -> bool:
    """Return True if the extracted subject likely refers to the human user."""
    s = _normalize_surface(raw_subj_id)
    if not s:
        return True
    for pat in _USER_REF_PATTERNS:
        if re.match(pat, s):
            return True
    return False


def to_triplets(
    relationships,
    user_canonical_id: str,
) -> List[Triplet]:
    out: List[Triplet] = []

    for rel in relationships:
        s_raw = (rel.subj.id or "user").strip()
        o_raw = (rel.obj.id or "object").strip()

        s_id = norm(s_raw)
        o_id = norm(o_raw)

        # Canonicalize subject if the extractor output refers to the human user
        if is_user_reference(s_raw):
            s_id = user_canonical_id

        rel_type = normalize_relation(rel.type)

        # Drop garbage objects
        if is_garbage_node(o_id):
            continue

        # Drop self-loops that are almost always noise
        if s_id == o_id:
            continue

        out.append(Triplet(subj=s_id, rel=rel_type, obj=o_id))

    # Deduplicate within a turn (preserve order)
    seen = set()
    deduped: List[Triplet] = []
    for t in out:
        if t in seen:
            continue
        seen.add(t)
        deduped.append(t)
    return deduped


def store_triplets(
    neo: Neo4jGraph,
    triplets: Iterable[Triplet],
    timestamp: str,
) -> List[Tuple[str, str, str]]:
    stored = []
    for t in triplets:
        neo.add_triplet(subj=t.subj, obj=t.obj, rel=t.rel, timestamp=timestamp)
        stored.append((t.subj, t.rel, t.obj))
    return stored