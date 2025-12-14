from typing import Dict, List, Optional, Set

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.storages import Neo4jGraph

SINGLE_VALUED_RELATIONS: Set[str] = {
    "LIVES_IN",
    "WORKS_AT",
    "STUDIES_AT",
    "HAS_PHONE_NUMBER",
    "HAS_EMAIL",
    "HAS_BIRTHDATE",
    "CURRENT_ROLE",
    "CURRENT_JOB",
}


def is_single_valued(rel_type: str) -> bool:
    return (rel_type or "").upper() in SINGLE_VALUED_RELATIONS


def detect_conflicts(neo: Neo4jGraph, subj: str, rel_type: str, obj: str) -> List[Dict]:
    if not is_single_valued(rel_type):
        return []

    triplets = neo.get_triplet()
    conflicts = []

    rel_norm = (rel_type or "").lower()

    for t in triplets:
        t_subj = t.get("subj", "")
        t_rel = (t.get("rel", "") or "").lower()
        t_obj = t.get("obj", "")

        if t_subj == subj and t_rel == rel_norm and t_obj != obj:
            conflicts.append(t)

    return conflicts


def xai_explain_conflict(
    base_model,
    conflicts: List[Dict],
    new_subj: str,
    new_rel: str,
    new_obj: str,
    new_ts: str,
) -> str:
    conflict_lines = []
    for t in conflicts:
        ts = t.get("timestamp", "no-time")
        s = t.get("subj")
        r = t.get("rel")
        o = t.get("obj")
        conflict_lines.append(f"[{ts}] {s} -[{r}]-> {o}")
    conflict_text = "\n".join(conflict_lines)

    system_msg = BaseMessage.make_assistant_message(
        role_name="ConflictExplainer",
        content=(
            "You are an explainable AI module. The system has detected a conflict "
            "between previously stored facts and a new fact about the same subject and relation.\n"
            "You must:\n"
            "1) Briefly explain the conflict in natural language.\n"
            "2) Offer the user 3 clear options, exactly labeled 'A', 'B', and 'C', such as:\n"
            "   - A: The old fact is outdated; the new fact is now correct.\n"
            "   - B: Both facts are true (for example, different time periods or multiple roles).\n"
            "   - C: The new fact is incorrect; keep the old one.\n"
            "Keep your answer short and clear, and end by asking the user to reply with A, B, or C."
        ),
    )

    agent = ChatAgent(system_message=system_msg, model=base_model)

    prompt = (
        "Previously stored facts:\n"
        f"{conflict_text}\n\n"
        "New fact:\n"
        f"[{new_ts}] {new_subj} -[{new_rel}]-> {new_obj}\n\n"
        "Generate the explanation and the options as instructed."
    )

    resp = agent.step(prompt)
    return resp.msg.content


def interpret_conflict_choice(user_text: str) -> Optional[str]:
    t = user_text.strip().lower()

    if t in ("a", "option a") or "outdated" in t or "new fact is correct" in t:
        return "A"
    if t in ("b", "option b") or "both" in t or "both true" in t:
        return "B"
    if t in ("c", "option c") or "keep the old" in t or "wrong" in t or "incorrect" in t:
        return "C"

    if t.startswith("a "):
        return "A"
    if t.startswith("b "):
        return "B"
    if t.startswith("c "):
        return "C"

    return None


def apply_conflict_resolution(neo: Neo4jGraph, choice: str, conflict: Dict) -> None:
    subj = conflict["subj"]
    rel = conflict["rel"]
    new_obj = conflict["new_obj"]
    new_ts = conflict["new_ts"]
    old_list = conflict["old"]

    if choice == "A":
        print("Applying resolution A: mark old facts as past, new fact as current.")
        for t in old_list:
            o = t.get("obj")
            ts = t.get("timestamp")
            cypher = f"""
            MATCH (a {{id: $subj}})-[r:`{rel}` {{timestamp: $ts}}]->(b {{id: $obj}})
            SET r.status = 'past'
            """
            neo.query(cypher, {"subj": subj, "obj": o, "ts": ts})

        cypher_new = f"""
        MATCH (a {{id: $subj}})-[r:`{rel}` {{timestamp: $ts}}]->(b {{id: $obj}})
        SET r.status = 'current'
        """
        neo.query(cypher_new, {"subj": subj, "obj": new_obj, "ts": new_ts})
        print("Conflict resolved: new fact current; old facts past.")
        return

    if choice == "B":
        print("Applying resolution B: keep both as current.")
        for t in old_list:
            o = t.get("obj")
            ts = t.get("timestamp")
            cypher = f"""
            MATCH (a {{id: $subj}})-[r:`{rel}` {{timestamp: $ts}}]->(b {{id: $obj}})
            SET r.status = 'current'
            """
            neo.query(cypher, {"subj": subj, "obj": o, "ts": ts})

        cypher_new = f"""
        MATCH (a {{id: $subj}})-[r:`{rel}` {{timestamp: $ts}}]->(b {{id: $obj}})
        SET r.status = 'current'
        """
        neo.query(cypher_new, {"subj": subj, "obj": new_obj, "ts": new_ts})
        print("Conflict resolved: all facts kept current.")
        return

    if choice == "C":
        print("Applying resolution C: delete the new fact.")
        cypher_del = f"""
        MATCH (a {{id: $subj}})-[r:`{rel}` {{timestamp: $ts}}]->(b {{id: $obj}})
        DELETE r
        """
        neo.query(cypher_del, {"subj": subj, "obj": new_obj, "ts": new_ts})
        print("Conflict resolved: new fact removed.")
        return