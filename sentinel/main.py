from camel.agents import ChatAgent, KnowledgeGraphAgent
from camel.loaders import UnstructuredIO
from camel.messages import BaseMessage

from .commands import handle_kg_command
from .config import load_user_canonical_id
from .conflicts import (
    detect_conflicts,
    xai_explain_conflict,
    interpret_conflict_choice,
    apply_conflict_resolution,
)
from .extract import extract_relationships, store_relationships, make_timestamp
from .kg_store import connect_neo4j
from .llm import create_local_model


def main():
    neo = connect_neo4j(clear=False)

    user_canonical_id = load_user_canonical_id()
    print(f"[INFO] Canonical user id: {user_canonical_id}")

    base_model = create_local_model()
    kg_agent = KnowledgeGraphAgent(model=base_model)
    uio = UnstructuredIO()

    system_msg = BaseMessage.make_assistant_message(
        role_name="PersonalKGAIAgent",
        content=(
            "You are an AI assistant that is getting to know the user over time.\n"
            "You chat naturally, ask follow-up questions, and remember what the user "
            "tells you, but the actual storage of memory is handled by an external "
            "knowledge graph system. Focus on having a natural, helpful conversation.\n"
            "Do not mention Neo4j or knowledge graphs unless the user explicitly asks."
        ),
    )

    chat_agent = ChatAgent(system_message=system_msg, model=base_model)

    print("\n[!] PERSONAL KG CHAT [!]")
    print("Type your messages and press Enter.")
    print("Commands:")
    print('  - "/kg clean"           → wipe stored KG (MATCH (n) DETACH DELETE n)')
    print('  - "/kg show recent"     → show recent KG memories (triplets)')
    print('  - "/kg remove <pattern>"→ remove memories whose subj/rel/obj matches <pattern>')
    print('  - "/kg <question>"      → ask a memory question answered from KG only')
    print('  - "exit" / "quit"       → end the session.\n')

    turn_index = 1
    pending_conflict = None

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Exiting chat.")
            break

        if not user_input:
            continue

        lower = user_input.lower()

        # If pending conflict, interpret user's choice
        if pending_conflict is not None and not lower.startswith("/kg"):
            choice = interpret_conflict_choice(user_input)
            if choice is not None:
                apply_conflict_resolution(neo, choice, pending_conflict)
                pending_conflict = None
                print("Assistant (KG): Thank you, I have updated my memory based on your decision.\n")
                continue

        # /kg commands
        if lower.startswith("/kg"):
            handled, neo = handle_kg_command(neo, base_model, user_input)
            if handled:
                continue

        # exit
        if lower in ("exit", "quit", "/exit", "/quit"):
            print("[INFO] Goodbye.")
            break

        # Normal chat
        resp = chat_agent.step(user_input)
        assistant_msg = resp.msg.content
        print(f"\nAssistant: {assistant_msg}\n")

        combined = f"User: {user_input}"
        element_id = f"turn-{turn_index}"

        relationships = extract_relationships(
            kg_agent=kg_agent,
            uio=uio,
            text=combined,
            element_id=element_id,
        )

        timestamp = make_timestamp()

        print("[KG] Extracted relationships for this turn:")
        if not relationships:
            print("  (none)")
        else:
            for rel in relationships:
                s_id_raw = rel.subj.id or "user"
                o_id_raw = rel.obj.id or "object"

                # NOTE: We normalize here only for printing/conflict detection.
                from .utils import norm
                s_id = norm(s_id_raw)
                o_id = norm(o_id_raw)
                rel_type = rel.type

                if s_id.lower() in ("speaker", "user", "i", "me", "myself", "my"):
                    s_id = user_canonical_id

                print(f"  {s_id} -[{rel_type} @ {timestamp}]-> {o_id}")

                conflicts = detect_conflicts(neo, s_id, rel_type, o_id)

                # write new fact
                neo.add_triplet(subj=s_id, obj=o_id, rel=rel_type, timestamp=timestamp)

                if conflicts:
                    explanation = xai_explain_conflict(
                        base_model,
                        conflicts,
                        new_subj=s_id,
                        new_rel=rel_type,
                        new_obj=o_id,
                        new_ts=timestamp,
                    )
                    print("Assistant (memory/XAI):")
                    print(explanation)
                    print()

                    pending_conflict = {
                        "subj": s_id,
                        "rel": rel_type,
                        "new_obj": o_id,
                        "new_ts": timestamp,
                        "old": conflicts,
                    }

        print()
        turn_index += 1

    print("\n[!] SESSION ENDED [!]")
    print("Your knowledge graph remains in Neo4j until you erase it.")


if __name__ == "__main__":
    main()