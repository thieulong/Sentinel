from camel.storages import Neo4jGraph

from .kg_store import connect_neo4j, show_recent_triplets, remove_knowledge
from .kg_qa import run_kg_qa

def handle_kg_command(neo: Neo4jGraph, base_model, user_input: str):

    sub = user_input[3:].strip()  
    if not sub:
        print('Assistant (KG): Please provide a subcommand or question after "/kg".')
        return True, neo

    sub_lower = sub.lower()

    if sub_lower == "clean":
        neo = connect_neo4j(clear=True)
        print("Assistant (KG): I have erased everything in the knowledge graph.")
        return True, neo

    if sub_lower.startswith("show"):
        show_recent_triplets(neo, limit=10)
        return True, neo

    if sub_lower.startswith("remove"):
        parts = sub.split(maxsplit=1)
        if len(parts) == 1:
            print('Assistant (KG): Please provide a pattern to remove, e.g. "/kg remove War Thunder".')
        else:
            remove_knowledge(neo, parts[1].strip())
        return True, neo

    question = sub.strip()
    answer = run_kg_qa(neo, base_model, question)
    print(f"Assistant (KG):\n{answer}\n")
    return True, neo