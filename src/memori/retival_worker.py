import logging
import re
from concurrent.futures import ThreadPoolExecutor

from bson.objectid import ObjectId
from langchain_core.messages import SystemMessage, HumanMessage

from .llms import RetrievalRouter, query_converter
from .prompts import RetrivalRouter_plan, query_converter_prompt
from .schema import RetrievalPlan, ResolvedAnchor
from .embeddings import embeddings
from .vector_str import index
from .Neo4j_connect import Neo4jGraph
from .neo4j_learn import ensure_neo4j_indexes
from .worker import invoke_with_retry
from .mongodb_connection import db, ensure_indexes

logger = logging.getLogger(__name__)
DEBUG = False
PARALLEL = True
MAX_GRAPH_ROWS_PER_HOP = 500


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

MAX_HOPS_CAP = 6



def debug(*args):
    if DEBUG:
        logger.debug("%s", " ".join(str(arg) for arg in args))


# ---------------------------------------------------------------------------
# Retrieval plan
# ---------------------------------------------------------------------------
def get_retrieval_plan(user_query: str):
    """Ask the router once; invoke_with_retry already handles transient 503s.

    A second retry loop here multiplied LLM calls (up to 9 attempts per query).
    """
    return invoke_with_retry(RetrievalRouter, [
        SystemMessage(content=RetrivalRouter_plan),
        HumanMessage(content=user_query),
    ])

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
    memory = db["memories"].find_one({
        "_id": ObjectId(str(memory_id)),
        "user_id": user_id,
    })
    return memory if memory and memory.get("type") == memory_type else None


def user_has_memories(user_id: str) -> bool:
    """True if the user has a memory document or an outbox record."""
    ensure_indexes(memory_types=[], include_outbox="basic")
    if db["memories"].find_one({"user_id": user_id}, {"_id": 1}) or db["outbox"].find_one({"user_id": user_id}, {"_id": 1}):
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
    candidates: list[dict] | None = None,
) -> list[ResolvedAnchor]:
    """Find graph anchors from existing Pinecone candidates when available.

    Passing candidates avoids repeating the same embedding and Pinecone query.
    """
    results = candidates if candidates is not None else Retrieval_for_Pinecone(
        plan, user_id, top_k=top_k
    )
    if not results:
        return []

    anchor_type = plan.anchor_entity.entity_type if plan.anchor_entity else None
    anchor_type_lc = anchor_type.lower() if anchor_type else None

    # Fetch candidate memories in batches instead of one MongoDB round trip
    # per Pinecone match.
    ids_by_type = {}
    for result in results[:top_k]:
        memory_type = result["metadata"].get("memory_type")
        memory_id = result.get("memory_id")
        if memory_type not in VALID_MEMORY_TYPES or not ObjectId.is_valid(str(memory_id or "")):
            continue
        ids_by_type.setdefault(memory_type, set()).add(ObjectId(str(memory_id)))

    all_ids = set().union(*ids_by_type.values()) if ids_by_type else set()
    docs_by_key = {}
    if all_ids:
        for doc in db["memories"].find(
            {"_id": {"$in": list(all_ids)}, "user_id": user_id},
            {"entities": 1, "type": 1},
        ):
            docs_by_key[(doc.get("type"), str(doc["_id"]))] = doc

    pairs = {}
    for result in results[:top_k]:
        memory_type = result["metadata"].get("memory_type")
        doc = docs_by_key.get((memory_type, str(result.get("memory_id"))))
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

                node_types = _normalize_types(node.get("type"))
                node_types_lc = [t.lower() for t in node_types]

                if anchor_type_lc and anchor_type_lc not in node_types_lc:
                    continue

                pairs[node_name.lower()] = {
                    "name": node_name,
                    "types": [anchor_type] if anchor_type else node_types,
                }

    if not pairs:
        return []

    label_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    pairs_by_label = {}
    for pair in pairs.values():
        types = pair.get("types") or []
        label = types[0] if types and label_re.fullmatch(types[0]) else None
        pairs_by_label.setdefault(label, []).append(pair)

    neo4j_result = []
    for label, label_pairs in pairs_by_label.items():
        if label:
            ensure_neo4j_indexes([label])
            query = f"""
            UNWIND $pairs AS p
            MATCH (n:{label} {{user_id: $user_id}})
            WHERE n.name = p.name
            RETURN n, labels(n) AS node_labels
            LIMIT $row_limit
            """
        else:
            query = """
            UNWIND $pairs AS p
            MATCH (n {user_id: $user_id})
            WHERE toLower(n.name) = toLower(p.name)
            RETURN n, labels(n) AS node_labels
            LIMIT $row_limit
            """
        neo4j_result.extend(Neo4jGraph.query(
            query,
            params={
                "user_id": user_id,
                "pairs": label_pairs,
                "row_limit": top_k,
            },
        ) or [])

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
    label_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    for entity in plan.entities:
        entity_type = entity.entity_type
        label = entity_type if entity_type and label_re.fullmatch(entity_type) else None
        if label:
            ensure_neo4j_indexes([label])
            if entity.name == "__current_user__":
                query = f"""
                MATCH (n:{label} {{user_id: $user_id}})
                RETURN n
                LIMIT 1
                """
                params = {"user_id": user_id}
                source = "user_context"
                result = Neo4jGraph.query(query, params=params)
            else:
                query = f"""
                MATCH (n:{label} {{user_id: $user_id}})
                WHERE n.name = $name
                RETURN n
                LIMIT 1
                """
                params = {"user_id": user_id, "name": entity.name}
                source = "neo4j_exact"
                result = Neo4jGraph.query(query, params=params)
                if not result:
                    # Preserve case-insensitive matching as a fallback; the
                    # normal exact path can use the (user_id, name) index.
                    query = f"""
                    MATCH (n:{label} {{user_id: $user_id}})
                    WHERE toLower(n.name) = toLower($name)
                    RETURN n
                    LIMIT 1
                    """
            result = result if entity.name == "__current_user__" else (
                result if result else Neo4jGraph.query(query, params=params)
            )
        else:
            # Unknown entity types cannot safely be interpolated as labels.
            if entity.name == "__current_user__":
                query = """
                MATCH (n {user_id: $user_id})
                WHERE $entity_type IS NULL
                   OR ANY(label IN labels(n) WHERE toLower(label) = toLower($entity_type))
                RETURN n LIMIT 1
                """
                params = {"user_id": user_id, "entity_type": entity_type}
                source = "user_context"
            else:
                query = """
                MATCH (n {user_id: $user_id})
                WHERE toLower(n.name) = toLower($name)
                  AND ($entity_type IS NULL OR
                       ANY(label IN labels(n) WHERE toLower(label) = toLower($entity_type)))
                RETURN n LIMIT 1
                """
                params = {
                    "user_id": user_id, "name": entity.name,
                    "entity_type": entity_type,
                }
                source = "neo4j_exact"
            result = Neo4jGraph.query(query, params=params)

        if not result:
            debug(
                f"Anchor NOT found: name={entity.name!r} "
                f"type={entity_type!r} user_id={user_id!r}"
            )
            continue

        node = result[0]["n"]
        anchor = ResolvedAnchor(
            name=node["name"],
            entity_type=entity_type,
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
    """Batch each hop by start-node label and cap returned graph rows."""
    is_current = plan.is_current
    hops = _build_hops(plan)
    all_results = []
    seen_rows = set()
    visited_by_root = {}
    current_nodes = []
    label_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    for root_id, anchor in enumerate(resolved_anchors):
        visited_by_root[root_id] = {anchor.name.lower()}
        current_nodes.append({
            "root_id": root_id,
            "name": anchor.name,
            "entity_type": anchor.entity_type,
        })

    for hop in hops:
        if not current_nodes:
            break
        if hop["relationship"]:
            rels_lc = expand_relationships(hop["relationship"])
            if not rels_lc:
                debug("Invalid relationship " + repr(hop["relationship"]) + ", stopping.")
                break
        else:
            rels_lc = []

        pattern = _build_pattern(hop["direction"])
        grouped_starts = {}
        for node in current_nodes:
            label = node.get("entity_type")
            if not label or not label_re.fullmatch(label):
                label = None
            grouped_starts.setdefault(label, []).append(node)

        rows = []
        for label, starts in grouped_starts.items():
            remaining = MAX_GRAPH_ROWS_PER_HOP - len(rows)
            if remaining <= 0:
                break
            if label:
                ensure_neo4j_indexes([label])
                match_start = "MATCH (start:" + label + " {user_id: $user_id})"
                name_filter = "start.name = s.name"
            else:
                match_start = "MATCH (start {user_id: $user_id})"
                name_filter = "toLower(start.name) = toLower(s.name)"

            query = f"""
            UNWIND $starts AS s
            {match_start}
            WHERE {name_filter}
            MATCH (start){pattern}(target)
            WHERE coalesce(r.user_id, $user_id) = $user_id
              AND (size($rels) = 0 OR toLower(type(r)) IN $rels)
              AND ($is_current IS NULL OR coalesce(r.is_current, true) = $is_current)
              AND target.user_id = $user_id
            UNWIND coalesce(r.memory_ids, [r.memory_id]) AS memory_id
            RETURN
                s.root_id AS root_id,
                start.name AS starting_node,
                labels(start) AS starting_labels,
                type(r) AS relationship_type,
                memory_id AS memory_id,
                r.memory_type AS memory_type,
                target.name AS target_node,
                labels(target) AS target_labels
            LIMIT $row_limit
            """
            rows.extend(Neo4jGraph.query(
                query,
                params={
                    "user_id": user_id,
                    "starts": starts,
                    "rels": rels_lc,
                    "is_current": is_current,
                    "row_limit": remaining,
                },
            ) or [])

        next_nodes = []
        for row in rows:
            root_id = row["root_id"]
            row_key = (
                row.get("starting_node"),
                row.get("relationship_type"),
                row.get("target_node"),
                row.get("memory_id"),
            )
            if row_key not in seen_rows:
                seen_rows.add(row_key)
                all_results.append({
                    "starting_node": row.get("starting_node"),
                    "relationship_type": row.get("relationship_type"),
                    "memory_id": row.get("memory_id"),
                    "target_node": row.get("target_node"),
                    "target_labels": row.get("target_labels"),
                    "memory_type": row.get("memory_type"),
                })

            target_name = row.get("target_node")
            target_key = (target_name or "").lower()
            if target_key and target_key not in visited_by_root.setdefault(root_id, set()):
                visited_by_root[root_id].add(target_key)
                target_labels = row.get("target_labels") or []
                target_type = target_labels[0] if target_labels else None
                next_nodes.append({
                    "root_id": root_id,
                    "name": target_name,
                    "entity_type": target_type,
                })

        if len(rows) >= MAX_GRAPH_ROWS_PER_HOP:
            logger.warning(
                "Graph traversal reached the %d-row hop limit for user_id=%s",
                MAX_GRAPH_ROWS_PER_HOP, user_id,
            )
        current_nodes = next_nodes

    return all_results

# ---------------------------------------------------------------------------
# Retrieval strategies
# ---------------------------------------------------------------------------
def _get_anchors(
    plan: RetrievalPlan,
    user_id: str,
    pinecone_candidates: list[dict] | None = None,
) -> list[ResolvedAnchor]:
    if plan.entities:
        anchors = Anchor_Resolver(user_id, plan)
    else:
        anchors = Anchor_from_Pinecone(
            plan, user_id, 5, candidates=pinecone_candidates
        )
    debug("Anchors:", anchors)
    return anchors


def hybrid_retrieval(plan: RetrievalPlan, user_id: str):
    if plan.entities:
        # Explicit graph anchors mean Pinecone and graph retrieval are
        # independent, so overlap their network latency.
        with ThreadPoolExecutor(max_workers=2) as pool:
            pinecone_future = pool.submit(Retrieval_for_Pinecone, plan, user_id)
            graph_future = pool.submit(_retrieve_graph_results, plan, user_id)
            pinecone_results = pinecone_future.result()
            graph_results = graph_future.result()
    else:
        # Without explicit anchors, reuse the Pinecone candidates to discover
        # graph anchors rather than embedding and querying Pinecone twice.
        pinecone_results = Retrieval_for_Pinecone(plan, user_id)
        anchors = _get_anchors(plan, user_id, pinecone_results)
        graph_results = Graph_Traversal(plan, anchors, user_id)

    return {"pinecone": pinecone_results, "graph": graph_results}


def _retrieve_graph_results(
    plan: RetrievalPlan,
    user_id: str,
    pinecone_candidates: list[dict] | None = None,
) -> list[dict]:
    anchors = _get_anchors(plan, user_id, pinecone_candidates)
    return Graph_Traversal(plan, anchors, user_id)


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
        # Reuse Pinecone candidates if needed to discover graph anchors; do not
        # embed and query Pinecone a second time on graph fallback.
        if not plan.entities:
            candidates = Retrieval_for_Pinecone(plan, user_id)
            anchors = _get_anchors(plan, user_id, candidates)
            graph_results = Graph_Traversal(plan, anchors, user_id)
            results = {"pinecone": candidates, "graph": graph_results}
        else:
            results = graph_retrieval(plan, user_id)
        if not results["graph"]:
            debug("Graph returned nothing, falling back to semantic.")
            return {"pinecone": results.get("pinecone", []), "graph": []}, "semantic"
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

    # ObjectIds are preserved during migration, so fetch all memory types
    # in one query against the unified source-of-truth collection.
    unique_refs = []
    seen = set()
    all_ids = set()
    for memory_id, memory_type in refs:
        if not memory_id or memory_type not in VALID_MEMORY_TYPES:
            continue
        key = (str(memory_id), memory_type)
        if key in seen or not ObjectId.is_valid(str(memory_id)):
            continue
        seen.add(key)
        unique_refs.append(key)
        all_ids.add(ObjectId(str(memory_id)))

    docs_by_key = {}
    if all_ids:
        for doc in db["memories"].find(
            {"_id": {"$in": list(all_ids)}, "user_id": user_id}
        ):
            docs_by_key[(str(doc["_id"]), doc.get("type"))] = doc

    final_docs = []
    for memory_id, memory_type in unique_refs:
        doc = docs_by_key.get((memory_id, memory_type))
        if doc:
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
        logger.debug("Retrieval plan for %r: %s", q_text, retrieval_plan)

        results, effective_type = retrieve(retrieval_plan, user_id)
        logger.debug("Retrieval results (%s) for %r: %s", effective_type, q_text, results)

        return convert_results_to_mongo_docs(results, effective_type, user_id)
    except Exception as e:
        # One failing query should not kill the others.
        logger.exception("Query failed (%r)", q_text)
        return []


def retrieve_memory(user_query: str, user_id: str, converter_response=None):
    # Nothing stored yet (e.g. a brand-new user): skip every LLM call,
    # Pinecone query and Neo4j query.
    if not user_has_memories(user_id):
        logger.debug("No memories stored for user_id=%s; skipping retrieval", user_id)
        return []

    if converter_response is None:
        converter_response = invoke_with_retry(
            query_converter,
            [
                SystemMessage(content=query_converter_prompt),
                HumanMessage(content=user_query),
            ],
        )
    logger.debug("Query converter output: %s", converter_response)

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
            unique_docs.append(doc["content"] if "content" in doc else doc)

    logger.debug("Final retrieval result count: %d unique documents", len(unique_docs))
    return unique_docs