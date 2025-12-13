import time
from typing import List, Tuple, Optional

from camel.agents import KnowledgeGraphAgent
from camel.loaders import UnstructuredIO
from camel.storages import Neo4jGraph

from .utils import norm

def extract_relationships(kg_agent: KnowledgeGraphAgent, uio: UnstructuredIO, text: str, element_id: str):
    element = uio.create_element_from_text(text=text, element_id=element_id)
    graph_element = kg_agent.run(element, parse_graph_elements=True)
    return graph_element.relationships


def store_relationships(
    neo: Neo4jGraph,
    relationships,
    timestamp: str,
    user_canonical_id: str,
) -> List[Tuple[str, str, str]]:

    stored = []
    for rel in relationships:
        s_id_raw = rel.subj.id or "user"
        o_id_raw = rel.obj.id or "object"

        s_id = norm(s_id_raw)
        o_id = norm(o_id_raw)
        rel_type = rel.type

        subj_lower = s_id.lower()
        if subj_lower in ("speaker", "user", "i", "me", "myself", "my"):
            s_id = user_canonical_id

        neo.add_triplet(subj=s_id, obj=o_id, rel=rel_type, timestamp=timestamp)
        stored.append((s_id, rel_type, o_id))

    return stored


def make_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())