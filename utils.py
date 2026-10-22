# utils.py – shared helpers for Neo4j connection and config
"""
Utility functions used across the KG loader scripts.
Provides:
* `get_neo4j_config(args)` – builds a dict from CLI args / env vars.
* `connect_neo4j(config)` – returns a Neo4j driver (or a mock object when the driver
  library is not installed).
The implementation mirrors the logic that existed in each loader but without
terminating the process on import.
"""
import os
import sys
from typing import Dict, Any

try:
    from neo4j import GraphDatabase, Driver
except ImportError:  # pragma: no cover – optional dependency
    GraphDatabase = None  # type: ignore
    Driver = None  # type: ignore


def get_neo4j_config(args: Any = None, uri: str = None, username: str = None, password: str = None, database: str = None) -> Dict[str, str]:
    """Return a configuration dict for Neo4j.
    Values are taken from explicit arguments / args namespace, then environment variables,
    finally defaults.
    """
    if args is not None and not isinstance(args, str):
        uri = getattr(args, "uri", None) or uri
        username = getattr(args, "username", None) or username
        password = getattr(args, "password", None) or password
        database = getattr(args, "database", None) or database
    elif isinstance(args, str):
        uri = args

    return {
        "uri": uri or os.getenv("NEO4J_URI") or "bolt://localhost:7687",
        "username": username or os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER") or "neo4j",
        "password": password or os.getenv("NEO4J_PASSWORD") or "password",
        "database": database or os.getenv("NEO4J_DATABASE") or "neo4j",
    }


def connect_neo4j(config: Dict[str, str]):
    """Create a Neo4j driver if the library is available.
    Returns a driver object or ``None`` when the driver cannot be instantiated.
    """
    if GraphDatabase is None:
        print("[WARN] neo4j driver not installed – running in mock mode.")
        return None
    try:
        driver: Driver = GraphDatabase.driver(config["uri"], auth=(config["username"], config["password"]))
        return driver
    except Exception as e:  # pragma: no cover – runtime connection issue
        print(f"[ERROR] Failed to connect to Neo4j: {e}")
        return None
