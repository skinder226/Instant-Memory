import re
from concurrent.futures import ThreadPoolExecutor

from bson.objectid import ObjectId
from langchain_core.messages import SystemMessage, HumanMessage
from rich import print as rprint
from rich.markup import escape

from src.memori.llms import RetrievalRouter, query_converter
from src.memori.prompts import RetrivalRouter_plan, query_converter_prompt
from src.memori.schema import RetrievalPlan, ResolvedAnchor
from src.memori.embeddings import embeddings
from src.memori.vector_str import index
from src.memori.Neo4j_connect import Neo4jGraph
from src.memori.mongodb_connection import db

DEBUG = True      # set to False to silence debug output
PARALLEL = False  # set to True to run several queries in parallel

# NOTE: all logging below uses Python's built-in print(), NOT rich's print.
# Rich treats text like "[debug]" or "[red]" inside strings as markup and can
# swallow or alter the output, which hides exactly the lines you need to see.
print("=== retrieval.py v3 loaded ===")

# Relationship names are never inserted into the Cypher text; they are passed
# as a query parameter. So a regex check is enough for safety and we do NOT
# depend on an ALLOWED_RELATIONSHIPS list (if that variable was a string or
# had different names, every relationship got silently rejected).
REL_NAME_RE = re.compile(r"^[a-z_][a-z0-9_]*$")

VALID_MEMORY_TYPES = {
    "profile", "fact", "preference", "semantic", "episodic",
    "procedural", "goal", "event", "task", "project", "temporal",
}

SOCIAL = [
    "knows",
    "has_friend",
    "works_with",
    "married_to",
    "partner_of",
    "sibling_of",
    "parent_of",
]

MAX_HOPS_CAP = 3


import time
from openai import APIStatusError


def invoke_with_retry(runnable, messages, retries=3):
    for attempt in range(retries):
        try:
            return runnable.invoke(messages)

        except APIStatusError as e:
            if e.status_code != 503 or attempt == retries - 1:
                raise

            wait = 2 ** attempt
            print(
                f"LLM temporarily unavailable (503). "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)

def debug(*args):
    if DEBUG:
        print("DEBUG:", *args)


# ---------------------------------------------------------------------------
# Retrieval plan
# ---------------------------------------------------------------------------
def get_retrieval_plan(user_query: str, retries: int = 3):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return RetrievalRouter.invoke([
                SystemMessage(content=RetrivalRouter_plan),
                HumanMessage(content=user_query),
            ])
        except Exception as e:  # parser errors, network errors, API errors
            last_error = e
            rprint(f"[red]Error getting retrieval plan: {escape(str(e))}[/red]")
            print(f"Retrying {attempt}/{retries}...")
    raise last_error


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def expand_relationships(relationship) -> list[str]:
    """Lowercase relationship names. 'knows' expands to all social
    relationships. Malformed names are dropped."""
    if not relationship:
        return []

    rel = str(relationship).strip().lower()
    expanded = SOCIAL if rel == "knows" else [rel]

    return [r for r in expanded if REL_NAME_RE.match(r)]


def get_mongo_doc_from_memory_id(memory_id, memory_type, user_id):
    if not memory_id or not ObjectId.is_valid(str(memory_id)):
        return None
    if memory_type not in VALID_MEMORY_TYPES:
        return None
    return db[memory_type].find_one({
        "_id": ObjectId(str(memory_id)),
        "user_id": user_id,
    })


def user_has_memories(user_id: str) -> bool:
    """True if the user has at least one memory document in ANY collection.
    Stops at the first match, so it is cheap. Create an index on user_id in
    each collection to keep it fast: db[type].create_index("user_id")"""

    if db["outbox"].find_one({"user_id": user_id}, {"_id": 1}):
        debug(f"User {user_id!r} has memories.")
        return True
    debug(f"User {user_id!r} has NO memories in MongoDB.")
    return False


def _normalize_types(node_type) -> list[str]:
    if not node_type:
        return []
    if isinstance(node_type, str):
        return [node_type]
    return list(node_type)


# ---------------------------------------------------------------------------
# Pinecone
# ---------------------------------------------------------------------------
def Retrieval_for_Pinecone(
    plan: RetrievalPlan,
    user_id: str,
    top_k: int | None = None,
) -> list[dict]:
    filters = {"user_id": user_id}

    # Only filter is_current when the router explicitly provides it.
    if plan.is_current is not None:
        filters["is_current"] = plan.is_current

    if plan.memory_type:
        filters["memory_type"] = plan.memory_type

    embedding = embeddings.embed_query(plan.query)

    results = index.query(
        vector=embedding,
        top_k=top_k or plan.top_k,
        namespace=user_id,
        filter=filters,
        include_metadata=True,
    )

    candidates = []
    for match in results.matches:
        memory_id = match.metadata.get("memory_id")
        if not memory_id:
            continue
        candidates.append({
            "memory_id": memory_id,
            "score": match.score,
            "metadata": match.metadata,
        })

    return candidates


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------
def Anchor_from_Pinecone(
    plan: RetrievalPlan,
    user_id: str,
    top_k: int = 5,
) -> list[ResolvedAnchor]:
    """Find anchor nodes from the entities inside the memories Pinecone
    returns. Neo4j is queried ONCE for all unique (name, type) pairs."""
    results = Retrieval_for_Pinecone(plan, user_id, top_k=top_k)
    if not results:
        return []

    anchor_type = plan.anchor_entity.entity_type if plan.anchor_entity else None
    anchor_type_lc = anchor_type.lower() if anchor_type else None

    pairs = {}
    for result in results[:top_k]:
        memory_type = result["metadata"].get("memory_type")
        doc = get_mongo_doc_from_memory_id(result["memory_id"], memory_type, user_id)
        if not doc:
            continue

        for entity in doc.get("entities", []):
            for side in ("source", "target"):
                node = entity.get(side)
                if not node:
                    continue

                node_name = node.get("name")
                if not node_name:
                    continue

                node_types_lc = [t.lower() for t in _normalize_types(node.get("type"))]

                if anchor_type_lc and anchor_type_lc not in node_types_lc:
                    continue

                pairs[node_name.lower()] = {
                    "name": node_name,
                    "types": [anchor_type_lc] if anchor_type_lc else node_types_lc,
                }

    if not pairs:
        return []

    query = """
    UNWIND $pairs AS p
    MATCH (n {user_id: $user_id})
    WHERE toLower(n.name) = toLower(p.name)
      AND (
          size(p.types) = 0
          OR ANY(label IN labels(n) WHERE toLower(label) IN p.types)
      )
    RETURN n, labels(n) AS node_labels
    """

    neo4j_result = Neo4jGraph.query(
        query,
        params={"user_id": user_id, "pairs": list(pairs.values())},
    )

    unique = {}
    for row in neo4j_result or []:
        neo4j_node = row["n"]
        labels = row.get("node_labels") or []

        entity_type = anchor_type or (labels[0] if labels else None)

        anchor = ResolvedAnchor(
            name=neo4j_node["name"],
            entity_type=entity_type,
            user_id=user_id,
            source="pinecone_mongo_discovery",
        )
        unique[(anchor.name.lower(), (anchor.entity_type or "").lower())] = anchor

    return list(unique.values())[:top_k]


def Anchor_Resolver(
    user_id: str,
    plan: RetrievalPlan,
) -> list[ResolvedAnchor]:
    resolved = {}

    exact_query = """
    MATCH (n {user_id: $user_id})
    WHERE toLower(n.name) = toLower($name)
      AND (
          $entity_type IS NULL
          OR ANY(label IN labels(n) WHERE toLower(label) = toLower($entity_type))
      )
    RETURN n
    LIMIT 1
    """

    user_query = """
    MATCH (n {user_id: $user_id})
    WHERE (
        $entity_type IS NULL
        OR ANY(label IN labels(n) WHERE toLower(label) = toLower($entity_type))
    )
    RETURN n
    LIMIT 1
    """

    for entity in plan.entities:
        if entity.name == "__current_user__":
            # Resolve the real User node; its name is not necessarily user_id.
            result = Neo4jGraph.query(
                user_query,
                params={"user_id": user_id, "entity_type": entity.entity_type},
            )
            source = "user_context"
        else:
            result = Neo4jGraph.query(
                exact_query,
                params={
                    "user_id": user_id,
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                },
            )
            source = "neo4j_exact"

        if not result:
            debug(
                f"Anchor NOT found: name={entity.name!r} "
                f"type={entity.entity_type!r} user_id={user_id!r}"
            )
            continue

        node = result[0]["n"]
        anchor = ResolvedAnchor(
            name=node["name"],
            entity_type=entity.entity_type,
            user_id=user_id,
            source=source,
        )
        resolved[(anchor.name.lower(), (anchor.entity_type or "").lower())] = anchor

    return list(resolved.values())


# ---------------------------------------------------------------------------
# Graph traversal
# ---------------------------------------------------------------------------
def _build_hops(plan: RetrievalPlan) -> list[dict]:
    """Use plan.path when given. Otherwise fall back to max_hops generic
    hops (any relationship, any direction); at least 1 for graph queries."""
    if plan.path:
        return [
            {
                "relationship": getattr(hop, "relationship", None),
                "direction": getattr(hop, "direction", None),
            }
            for hop in plan.path
        ]

    n = min(max(plan.max_hops or 1, 1), MAX_HOPS_CAP)
    return [{"relationship": None, "direction": "both"}] * n


def _build_pattern(direction: str | None) -> str:
    """Relationship TYPES are not put in the pattern. They are filtered in the
    WHERE clause (case-insensitive, parameterised)."""
    if direction == "outgoing":
        return "-[r]->"
    if direction == "incoming":
        return "<-[r]-"
    return "-[r]-"


def Graph_Traversal(
    plan: RetrievalPlan,
    resolved_anchors: list[ResolvedAnchor],
    user_id: str,
) -> list[dict]:
    # None means "both current and historical", same as Pinecone.
    is_current = plan.is_current

    hops = _build_hops(plan)
    all_results = []
    seen_rows = set()

    for anchor in resolved_anchors:
        current_nodes = [{
            "name": anchor.name,
            "entity_type": anchor.entity_type,
        }]
        visited = {anchor.name.lower()}

        for hop in hops:
            if hop["relationship"]:
                rels_lc = expand_relationships(hop["relationship"])
                if not rels_lc:
                    debug(f"Invalid relationship {hop['relationship']!r}, stopping.")
                    break
            else:
                rels_lc = []  # generic hop: any relationship

            pattern = _build_pattern(hop["direction"])

            query = f"""
            MATCH (start {{user_id: $user_id}})
            WHERE toLower(start.name) = toLower($start_name)
              AND (
                  $start_entity_type IS NULL
                  OR ANY(label IN labels(start) WHERE toLower(label) = toLower($start_entity_type))
              )

            MATCH (start){pattern}(target)
            WHERE coalesce(r.user_id, $user_id) = $user_id
              AND (size($rels) = 0 OR toLower(type(r)) IN $rels)
              AND ($is_current IS NULL OR coalesce(r.is_current, true) = $is_current)

            RETURN
                start.name AS starting_node,
                labels(start) AS starting_labels,
                type(r) AS relationship_type,
                r.memory_id AS memory_id,
                r.memory_type AS memory_type,
                target.name AS target_node,
                labels(target) AS target_labels
            """

            next_nodes = []

            for current_node in current_nodes:
                result = Neo4jGraph.query(
                    query,
                    params={
                        "user_id": user_id,
                        "start_name": current_node["name"],
                        "start_entity_type": current_node["entity_type"],
                        "rels": rels_lc,
                        "is_current": is_current,
                    },
                )

                debug(
                    f"Hop {hop} from {current_node['name']!r}: "
                    f"{len(result or [])} rows"
                )

                for row in result or []:
                    row_key = (
                        row["starting_node"],
                        row["relationship_type"],
                        row["target_node"],
                        row["memory_id"],
                    )
                    if row_key in seen_rows:
                        continue
                    seen_rows.add(row_key)

                    all_results.append({
                        "starting_node": row["starting_node"],
                        "relationship_type": row["relationship_type"],
                        "memory_id": row["memory_id"],
                        "target_node": row["target_node"],
                        "target_labels": row["target_labels"],
                        "memory_type": row["memory_type"],
                    })

                    target_key = (row["target_node"] or "").lower()
                    if target_key and target_key not in visited:
                        visited.add(target_key)
                        next_nodes.append({
                            "name": row["target_node"],
                            # Match by name only on later hops; labels[0] is
                            # not reliable when nodes have several labels.
                            "entity_type": None,
                        })

            current_nodes = next_nodes
            if not current_nodes:
                break

    return all_results


# ---------------------------------------------------------------------------
# Retrieval strategies
# ---------------------------------------------------------------------------
def _get_anchors(plan: RetrievalPlan, user_id: str) -> list[ResolvedAnchor]:
    if plan.entities:
        anchors = Anchor_Resolver(user_id, plan)
    else:
        anchors = Anchor_from_Pinecone(plan, user_id, 5)
    debug("Anchors:", anchors)
    return anchors


def hybrid_retrieval(plan: RetrievalPlan, user_id: str):
    pinecone_results = Retrieval_for_Pinecone(plan, user_id)
    anchors = _get_anchors(plan, user_id)
    graph_results = Graph_Traversal(plan, anchors, user_id)
    return {"pinecone": pinecone_results, "graph": graph_results}


def semantic_retrieval(plan: RetrievalPlan, user_id: str):
    return {
        "pinecone": Retrieval_for_Pinecone(plan, user_id),
        "graph": [],
    }


def graph_retrieval(plan: RetrievalPlan, user_id: str):
    anchors = _get_anchors(plan, user_id)
    graph_results = Graph_Traversal(plan, anchors, user_id)
    return {"pinecone": [], "graph": graph_results}


def retrieve(plan: RetrievalPlan, user_id: str):
    """Returns (results, effective_query_type).

    If a graph query finds nothing, falls back to semantic search so the
    user still gets an answer."""
    if plan.query_type == "semantic":
        return semantic_retrieval(plan, user_id), "semantic"

    if plan.query_type == "graph":
        results = graph_retrieval(plan, user_id)
        if not results["graph"]:
            debug("Graph returned nothing, falling back to semantic.")
            return semantic_retrieval(plan, user_id), "semantic"
        return results, "graph"

    if plan.query_type == "hybrid":
        return hybrid_retrieval(plan, user_id), "hybrid"

    raise ValueError(f"Unknown query type: {plan.query_type}")


# ---------------------------------------------------------------------------
# Results -> Mongo documents
# ---------------------------------------------------------------------------
def convert_results_to_mongo_docs(results: dict, query_type: str, user_id: str):
    refs = []

    if query_type in ("semantic", "hybrid"):
        for r in results.get("pinecone", []):
            refs.append((r["memory_id"], r["metadata"].get("memory_type")))

    if query_type in ("graph", "hybrid"):
        for r in results.get("graph", []):
            refs.append((r.get("memory_id"), r.get("memory_type")))

    final_docs = []
    seen = set()

    for memory_id, memory_type in refs:
        if not memory_id or not memory_type:
            debug(f"Skipping result without memory_id/memory_type: {memory_id!r}, {memory_type!r}")
            continue
        key = (str(memory_id), memory_type)
        if key in seen:
            continue
        seen.add(key)

        doc = get_mongo_doc_from_memory_id(memory_id, memory_type, user_id)
        if doc:  # never append None
            final_docs.append(doc)
        else:
            debug(f"Mongo doc not found: {memory_type}/{memory_id}")

    return final_docs


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def _process_single_query(q_text: str, user_id: str) -> list[dict]:
    try:
        retrieval_plan = get_retrieval_plan(q_text)
        print(f"Retrieval Plan: {retrieval_plan}")

        results, effective_type = retrieve(retrieval_plan, user_id)
        print(f"Results ({effective_type}): {results}\n")

        return convert_results_to_mongo_docs(results, effective_type, user_id)
    except Exception as e:
        # One failing query should not kill the others.
        print(f"Query failed ({q_text!r}): {type(e).__name__}: {e}")
        return []


def retrieve_memory(user_query: str, user_id: str):
    # Nothing stored yet (e.g. a brand-new user): skip every LLM call,
    # Pinecone query and Neo4j query.
    if not user_has_memories(user_id):
        print("No memories stored for this user, skipping retrieval.")
        return []

    converter_response = invoke_with_retry(
    query_converter,
    [
        SystemMessage(content=query_converter_prompt),
        HumanMessage(content=user_query),
    ],
)
    print(f"Query Converter Output: {converter_response}")

    query_texts = [q.query for q in converter_response.queries if q.query]
    if not query_texts:
        return []

    if PARALLEL and len(query_texts) > 1:
        with ThreadPoolExecutor(max_workers=min(len(query_texts), 4)) as pool:
            per_query_docs = list(
                pool.map(lambda t: _process_single_query(t, user_id), query_texts)
            )
    else:
        per_query_docs = [_process_single_query(t, user_id) for t in query_texts]

    # Dedupe across queries by Mongo _id.
    seen_ids = set()
    unique_docs = []
    for docs in per_query_docs:
        for doc in docs:
            doc_id = str(doc["_id"])
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            unique_docs.append(doc)

    print(f"Final Results: {len(unique_docs)} unique documents retrieved.\n")
    return unique_docs