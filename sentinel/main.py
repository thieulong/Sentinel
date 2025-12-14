from camel.agents import ChatAgent
from camel.messages import BaseMessage

from .commands import handle_kg_command
from .config import load_user_canonical_id
from .conflicts import (
    detect_conflicts,
    xai_explain_conflict,
    interpret_conflict_choice,
    apply_conflict_resolution,
)
from .curator import make_curator_agent, run_curator
from .enricher import make_enricher_agent, run_enricher, build_triplets_from_enricher
from .extract import make_timestamp
from .kg_store import connect_neo4j, triplet_exists
from .llm import create_chat_model, create_curator_model, create_enricher_model


def _pretty_print_curator(curator_result):
    clean = curator_result.clean_text.strip()
    candidates = curator_result.candidates or []
    notes = curator_result.notes or []

    print("[CURATOR] clean_text:", clean if clean else "(empty)")

    if candidates:
        print("[CURATOR] candidates:")
        for c in candidates:
            subj = c.get("subj", "USER")
            rel = c.get("rel", "FACT")
            obj = c.get("obj", "")
            conf = c.get("confidence", None)
            if conf is None:
                print(f"  - {subj}  {rel}  {obj}")
            else:
                try:
                    print(f"  - {subj}  {rel}  {obj}   (conf={float(conf):.2f})")
                except Exception:
                    print(f"  - {subj}  {rel}  {obj}   (conf={conf})")
    else:
        print("[CURATOR] candidates: (none)")

    if notes:
        print("[CURATOR] notes:", "; ".join(notes))
    else:
        print("[CURATOR] notes: (none)")
    print()


def _pretty_print_enricher(enricher_result):
    ents = enricher_result.entities or []
    rels = enricher_result.relations or []
    notes = enricher_result.notes or []

    if ents:
        print("[ENRICHER] entities:")
        for e in ents:
            key = e.get("key", "")
            label = e.get("label", "")
            etype = e.get("type", "")
            time_text = e.get("time_text", None)
            if time_text:
                print(f"  - {key}  ({etype})  {label}   time={time_text}")
            else:
                print(f"  - {key}  ({etype})  {label}")
    else:
        print("[ENRICHER] entities: (none)")

    if rels:
        print("[ENRICHER] relations:")
        for r in rels:
            subj = r.get("subj", "")
            rel = r.get("rel", "")
            obj = r.get("obj", "")
            conf = r.get("confidence", None)
            if conf is None:
                print(f"  - {subj}  {rel}  {obj}")
            else:
                try:
                    print(f"  - {subj}  {rel}  {obj}   (conf={float(conf):.2f})")
                except Exception:
                    print(f"  - {subj}  {rel}  {obj}   (conf={conf})")
    else:
        print("[ENRICHER] relations: (none)")

    if notes:
        print("[ENRICHER] notes:", "; ".join(notes))
    else:
        print("[ENRICHER] notes: (none)")
    print()


def main():
    neo = connect_neo4j(clear=False)

    user_canonical_id = load_user_canonical_id()
    print(f"[INFO] Canonical user id: {user_canonical_id}")

    chat_model = create_chat_model()
    curator_model = create_curator_model()
    enricher_model = create_enricher_model()

    curator_agent = make_curator_agent(curator_model)
    enricher_agent = make_enricher_agent(enricher_model)

    system_msg = BaseMessage.make_assistant_message(
        role_name="PersonalKGAIAgent",
        content=(
            "You are an AI assistant that is getting to know the user over time.\n"
            "You chat naturally, ask follow-up questions, and remember what the user "
            "tells you, but memory storage is handled externally.\n"
            "Focus on a natural, helpful conversation.\n"
            "Do not mention Neo4j or knowledge graphs unless the user explicitly asks."
        ),
    )
    chat_agent = ChatAgent(system_message=system_msg, model=chat_model)

    print("\n[!] PERSONAL KG CHAT [!]")
    print("Commands:")
    print('  - "/kg clean"              -> wipe stored KG')
    print('  - "/kg show recent"        -> show recent KG memories')
    print('  - "/kg remove <pattern>"   -> remove memories')
    print('  - "/kg <question>"         -> ask a memory question')
    print('  - "/curator on|off"        -> toggle Curator output (default: ON)')
    print('  - "/enricher on|off"       -> toggle Enricher output (default: ON)')
    print('  - "exit" / "quit"          -> end the session.\n')

    pending_conflict = None
    show_curator = True
    show_enricher = True

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Exiting chat.")
            break

        if not user_input:
            continue

        lower = user_input.lower()

        if lower in ("/curator on", "curator on"):
            show_curator = True
            print("[INFO] Curator output: ON\n")
            continue
        if lower in ("/curator off", "curator off"):
            show_curator = False
            print("[INFO] Curator output: OFF\n")
            continue

        if lower in ("/enricher on", "enricher on"):
            show_enricher = True
            print("[INFO] Enricher output: ON\n")
            continue
        if lower in ("/enricher off", "enricher off"):
            show_enricher = False
            print("[INFO] Enricher output: OFF\n")
            continue

        if pending_conflict is not None and not lower.startswith("/kg"):
            choice = interpret_conflict_choice(user_input)
            if choice is not None:
                apply_conflict_resolution(neo, choice, pending_conflict)
                pending_conflict = None
                print("Assistant (KG): Memory updated based on your decision.\n")
                continue

        if lower.startswith("/kg"):
            handled, neo = handle_kg_command(neo, chat_model, user_input)
            if handled:
                continue

        if lower in ("exit", "quit", "/exit", "/quit"):
            print("[INFO] Goodbye.")
            break

        resp = chat_agent.step(user_input)
        assistant_msg = resp.msg.content
        print(f"\nAssistant: {assistant_msg}\n")

        # 1) Curator
        curator_result = run_curator(curator_agent, user_input)
        if show_curator:
            _pretty_print_curator(curator_result)

        # If nothing to remember, stop
        if not curator_result.candidates:
            continue

        # 2) Enricher (graph architect)
        enricher_result = run_enricher(
            enricher_agent,
            clean_text=curator_result.clean_text,
            candidates=curator_result.candidates,
        )
        if show_enricher:
            _pretty_print_enricher(enricher_result)

        # 3) Convert enriched relations to Triplets and store
        timestamp = make_timestamp()
        triplets = build_triplets_from_enricher(enricher_result, user_canonical_id=user_canonical_id)

        # Fallback: if Enricher produced nothing, store Curator candidates flat (rare, but safe)
        if not triplets:
            # Minimal fallback: store only the original Curator relations
            for c in curator_result.candidates:
                subj = user_canonical_id if str(c.get("subj", "USER")).upper() == "USER" else str(c.get("subj", "")).strip()
                rel = str(c.get("rel", "FACT")).strip()
                obj = str(c.get("obj", "")).strip()
                if not subj or not obj:
                    continue
                from .extract import Triplet, norm, normalize_relation
                t = Triplet(subj=norm(subj), rel=normalize_relation(rel), obj=norm(obj))
                triplets.append(t)

        for t in triplets:
            if triplet_exists(neo, t.subj, t.rel, t.obj):
                continue

            print(f"[KG] STORE [{timestamp}] {t.subj} -[{t.rel}]-> {t.obj}")

            conflicts = detect_conflicts(neo, t.subj, t.rel, t.obj)
            neo.add_triplet(subj=t.subj, obj=t.obj, rel=t.rel, timestamp=timestamp)

            if conflicts:
                explanation = xai_explain_conflict(
                    chat_model,
                    conflicts,
                    new_subj=t.subj,
                    new_rel=t.rel,
                    new_obj=t.obj,
                    new_ts=timestamp,
                )
                print("Assistant (memory/XAI):")
                print(explanation)
                print()

                pending_conflict = {
                    "subj": t.subj,
                    "rel": t.rel,
                    "new_obj": t.obj,
                    "new_ts": timestamp,
                    "old": conflicts,
                }

    print("\n[!] SESSION ENDED [!]")
    print("Your knowledge graph remains in Neo4j until you erase it.")


if __name__ == "__main__":
    main()