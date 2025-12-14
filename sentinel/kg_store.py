from typing import Dict, List

from camel.storages import Neo4jGraph

from .config import load_neo4j_config


def connect_neo4j(clear: bool = False) -> Neo4jGraph:
    cfg = load_neo4j_config()

    uri = cfg["NEO4J_URI"]
    user = cfg["NEO4J_USERNAME"]
    pwd = cfg["NEO4J_PASSWORD"]
    database = cfg.get("NEO4J_DATABASE", "neo4j")

    print(f"[INFO] Connecting to Neo4j URI: {uri}, DB: {database}, user: {user}")

    neo = Neo4jGraph(
        url=uri,
        username=user,
        password=pwd,
        database=database,
    )

    if clear:
        print("[KG] Erasing entire knowledge graph...")
        neo.query("MATCH (n) DETACH DELETE n")
        print("[KG] Knowledge graph erased.")

    return neo


def get_all_triplets(neo: Neo4jGraph) -> List[Dict]:
    return neo.get_triplet()


def show_recent_triplets(neo: Neo4jGraph, limit: int = 10) -> None:
    triplets = get_all_triplets(neo)
    if not triplets:
        print("[KG] No memories stored yet.")
        return

    triplets_sorted = sorted(
        triplets,
        key=lambda t: t.get("timestamp", ""),
        reverse=True,
    )

    print(f"[KG] Showing {min(limit, len(triplets_sorted))} most recent memories:")
    for i, t in enumerate(triplets_sorted[:limit], start=1):
        print(
            f"  {i}. [{t.get('timestamp', 'no-time')}] "
            f"{t.get('subj')} -[{t.get('rel')}]-> {t.get('obj')}"
        )


def remove_knowledge(neo: Neo4jGraph, pattern: str) -> None:
    pattern_l = pattern.lower()
    triplets = get_all_triplets(neo)
    to_delete = []

    for t in triplets:
        s = t.get("subj", "")
        r = t.get("rel", "")
        o = t.get("obj", "")
        ts = t.get("timestamp", "")

        if (pattern_l in s.lower()) or (pattern_l in r.lower()) or (pattern_l in o.lower()):
            to_delete.append((s, r, o, ts))

    if not to_delete:
        print(f"[KG] No memories matched pattern: {pattern!r}")
        return

    print(f"[KG] Removing {len(to_delete)} memories matching {pattern!r}:")
    for s, r, o, ts in to_delete:
        print(f"  [{ts}] {s} -[{r}]-> {o}")
        cypher = f"""
        MATCH (a {{id: $subj}})-[rel:`{r}` {{timestamp: $ts}}]->(b {{id: $obj}})
        DELETE rel
        """
        neo.query(cypher, {"subj": s, "obj": o, "ts": ts})


def triplet_exists(neo: Neo4jGraph, subj: str, rel: str, obj: str) -> bool:
    """
    Check whether (subj)-[rel]->(obj) exists, without triggering Neo4j warnings
    for unseen relationship types.
    """
    cypher = """
    MATCH (a {id: $subj})-[r]->(b {id: $obj})
    WHERE type(r) = $rel
    RETURN count(r) AS c
    """
    rows = neo.query(cypher, {"subj": subj, "obj": obj, "rel": rel}) or []
    try:
        return (rows[0].get("c", 0) if rows else 0) > 0
    except Exception:
        return False