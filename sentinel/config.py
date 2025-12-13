import json
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
NEO4J_CONFIG_PATH = ROOT / "neo4j_config.json"
USER_PROFILE_PATH = ROOT / "user_profile.json"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_user_canonical_id(default: str = "User") -> str:
    try:
        data = load_json(USER_PROFILE_PATH)
        name = data.get("canonical_id")
        if name:
            from .utils import norm
            return norm(str(name))
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return default


def load_neo4j_config(path: Path = NEO4J_CONFIG_PATH) -> Dict[str, Any]:
    print(f"[DEBUG] Loading Neo4j config from: {path}")
    return load_json(path)