from typing import Optional

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.storages import Neo4jGraph

from .kg_store import get_all_triplets
from .utils import detect_time_window, parse_iso_ts


def run_kg_qa(
    neo: Neo4jGraph,
    base_model,
    question: str,
    max_records: int = 50,
) -> str:
    triplets = get_all_triplets(neo)
    if not triplets:
        return "I do not have any stored memories yet."

    window = detect_time_window(question)
    if window is not None:
        start_w, end_w = window
        filtered = []
        for t in triplets:
            ts = t.get("timestamp")
            dt = parse_iso_ts(ts) if ts else None
            if dt is not None and start_w <= dt <= end_w:
                filtered.append(t)
        if filtered:
            triplets = filtered

    triplets_sorted = sorted(triplets, key=lambda t: t.get("timestamp", ""))

    if len(triplets_sorted) > max_records:
        triplets_sorted = triplets_sorted[-max_records:]

    memory_lines = []
    for t in triplets_sorted:
        ts = t.get("timestamp", "no-time")
        s = t.get("subj")
        r = t.get("rel")
        o = t.get("obj")
        memory_lines.append(f"[{ts}] {s} -[{r}]-> {o}")

    memory_block = "\n".join(memory_lines)

    system_msg = BaseMessage.make_assistant_message(
        role_name="KGMemoryAnswerAgent",
        content=(
            "You are an assistant that answers questions using ONLY the provided "
            "memory log, which is a list of time-stamped facts in the form:\n"
            "[timestamp] subject -[relation]-> object\n\n"
            "Very important identity rule:\n"
            "- Any subject named 'user', 'I', 'you', 'speaker', or similar "
            "refers to the SAME real-world person: the human user.\n"
            "- That person is NOT you. You are a separate AI assistant.\n\n"
            "When you answer, describe what the user told you in the second person, "
            "for example: 'You told me that you like cooking Vietnamese dishes', "
            "not 'I cook Vietnamese dishes'.\n\n"
            "You must not invent facts that are not logically supported by this log. "
            "If the answer cannot be determined, say you are not sure."
        ),
    )

    kg_qa_agent = ChatAgent(system_message=system_msg, model=base_model)

    prompt = (
        "Here is the memory log:\n"
        "------------------------\n"
        f"{memory_block}\n"
        "------------------------\n\n"
        f"Question: {question}\n"
        "Answer based ONLY on the memory log above."
    )

    resp = kg_qa_agent.step(prompt)
    return resp.msg.content