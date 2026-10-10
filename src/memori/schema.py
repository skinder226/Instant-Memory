from typing import Annotated, Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator,model_validator


# ============================================================
# MESSAGE ROLE
# ============================================================

MessageRole = Literal[
    "user",
    "assistant",
    "tool",
    "system",
    "other",
]


MemoryTypes = Literal[
    "profile",
    "fact",
    "preference",
    "semantic",
    "episodic",
    "procedural",
    "goal",
    "event",
    "task",
    "project",
    "temporal",
]

ALLOWED_RELATIONSHIPS = Literal[
    "has_name", "has_age", "has_email", "has_phone", "has_role", "has_skill", "speaks", "interested_in",
    "lives_in", "born_in", "located_in", "visited",
    "works_at", "studies_at", "graduated_from", "studies", "teaches", "founded", "member_of", "leads", "manages", "reports_to", "mentors",
    "works_on", "built_at", "built_for", "part_of", "depends_on", "deployed_on", "integrates_with", "based_on", "replaces",
    "uses", "prefers", "wants_to_learn", "learns", "likes", "dislikes", "subscribed_to",
    "works_with", "knows", "married_to", "sibling_of", "parent_of", "partner_of",
    "has_goal", "has_task", "attended", "plans_to","has_friend","has_pet",
    "owns", "created",
]
# ============================================================
# GRAPH ENTITY / EDGE
# ============================================================

class RelationshipProperties(BaseModel):
    context_reference: Optional[str] = Field(
        default=None,
        description=(
            "Reference to the source context this relationship was derived "
            "from (e.g. a conversation ID, message ID, or snippet of the "
            "original text). Useful for tracing back why this relationship "
            "was created or for resolving ambiguity later."
        )
    )

    # AI can still add any other free-form scalar keys, like before
    model_config = ConfigDict(extra="allow")


class Node(BaseModel):
    """
    Represents the source or target node/entity of a graph relationship.
    """

    name: str = Field(
        ...,
        description=(
            "The name of the node/entity. "
            "Example: User, Microsoft, New York, etc."
        )
    )

    type: List[str] = Field(
        ...,
        description=(
            "The type or label of the node/entity. "
            "Use a short label such as Person, User, Company, Location, "
            "Product, Event, Name, etc."
        )
    )


class Entity(BaseModel):
    """
    Represents a graph relationship.

    The relationship can contain additional contextual properties.

    Example:
        User --WORKS_AT {role: "CEO"}--> Microsoft

    source = {name: "User", type: ["Person", "User"]}
    relationship = "works_at"
    target = {name: "Microsoft", type: ["Company"]}
    properties = {
        "role": "CEO"
    }
    """
    source: Node = Field(
        ...,
        description=(
            "The source node/entity of the relationship. "
            "Example: User"
        )
    )

    relationship: ALLOWED_RELATIONSHIPS = Field(
        ...,
        description=(
            "The relationship/edge connecting source and target. "
            "Use a short relationship name such as has_name, works_at, "
            "uses, owns, lives_in, studies_at, etc."
        )
    )

    target: Node = Field(
        ...,
        description=(
            "The target node/entity of the relationship. "
            "Example: Microsoft"
        )
    )

    


# ============================================================
# TEMPORAL INFORMATION
# ============================================================

class TemporalInfo(BaseModel):
    """
    Information about when a memory is valid.
    """

    valid_from: Optional[str] = Field(
        default=None,
        description="When the memory became valid, if known"
    )

    valid_until: Optional[str] = Field(
        default=None,
        description="When the memory stopped being valid, if known"
    )

    is_current: bool = Field(
        default=True,
        description="Whether this memory is currently valid"
    )


# ============================================================
# MEMORY ITEM
# ============================================================

# ============================================================
# SUPPORTED PLACEHOLDER NAMES
# ============================================================
#
# Placeholders are now OPEN-ENDED — the LLM can invent as many distinct
# placeholder names as it needs (user_company, user_location,
# user_school, user_pet_name, user_favorite_language, ... — no fixed
# count or fixed list). To keep this safe and machine-parseable instead
# of a free-for-all, every placeholder name MUST still follow a strict
# naming convention, enforced here at the schema level:
#
#   - lowercase snake_case
#   - MUST start with "user_"
#   - only letters, digits, underscores after that
#
# This means "user_company", "user_favorite_ide", and
# "user_childhood_city" are all valid, but "Company", "user company",
# "USER_COMPANY", or "the_company" are rejected outright by Pydantic
# before they ever reach your storage layer.

PlaceholderName = Annotated[
    str,
    StringConstraints(pattern=r"^user_[a-z][a-z0-9_]*$"),
]


# ============================================================
# MEMORY ITEM
# ============================================================

class Placeholder(BaseModel):
    name: PlaceholderName = Field(
        ...,
        description=(
            "The name of the placeholder, snake_case, always starting "
            "with 'user_' (e.g. 'user_company', 'user_location', "
            "'user_favorite_language'). Any unresolved personal attribute "
            "can become a placeholder this way — there is no fixed list "
            "or maximum count."
        )
    )


class MemoryItem(BaseModel):
    type: MemoryTypes = Field(
        ...,
        description="The category classification of the memory"
    )

    content: str = Field(
        ...,
        description="The natural language representation of the memory"
    )


    id: Optional[str] = Field(
        default=None,
        description=(
            "Unique identifier for the memory item. If you are editing an "
            "existing memory, reuse the SAME id so it updates in place "
            "instead of creating a new memory."
        )
    )


    placeholders: List[Placeholder] = Field(
        default_factory=list,
        description=(
            "A list of placeholder values that can be used to fill in "
            "missing information in the memory. Make sure this is the "
            "same name as the name used in the content/entities of the "
            "memory (e.g. {'name': 'user_company'})."
        )
    )

    entities: List[Entity] = Field(
        default_factory=list,
        description="Structured representation of the memory as a graph of entities and relationships"
    )

    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Confidence that the extracted memory is supported by the latest user message."
    )

    importance: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="How important this memory is for future personalization and retrieval."
    )

    @field_validator("entities", mode="before")
    @classmethod
    def validate_entities(cls, value):
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError("entities must be a list")
        cleaned = []
        for entity in value:
            if entity in (None, ""):
                continue
            if isinstance(entity, (Entity, dict)):
                cleaned.append(entity)
            else:
                raise TypeError("each entity must be an object")
        return cleaned

    source_role: MessageRole = Field(
        ...,
        description="The role of the message from which this memory was extracted"
    )

    temporal: Optional[TemporalInfo] = Field(
        default=None,
        description="Temporal information when the memory changes over time"
    )


# ============================================================
# LIGHTWEIGHT PLACEHOLDER-ONLY UPDATE
# ============================================================

class FilledPlaceholder(BaseModel):
    """
    A single resolved placeholder: which one, and its real value.
    """

    name: PlaceholderName = Field(
        ...,
        description=(
            "The placeholder being resolved, snake_case starting with "
            "'user_' (e.g. 'user_company', 'user_location', "
            "'user_favorite_language'). Any placeholder name is allowed "
            "as long as it matches an existing memory's placeholder."
        )
    )

    value: str = Field(
        ...,
        description="The now-known real value for this placeholder, e.g. 'Nexora'."
    )


class PlaceholderUpdate(BaseModel):
    """
    Used ONLY when a placeholder is being resolved and NOTHING else about
    any memory changes — content, entities, and placeholders on the
    affected memories stay exactly as they were.

    Placeholder names are open-ended (see PlaceholderName) — there is no
    fixed list and no maximum count. A single response can contain as
    many PlaceholderUpdate entries as there are distinct placeholder
    names resolved in that message.

    Example (four different placeholders resolved in one response):
        [
            {"filled_placeholders": {"name": "user_company", "value": "Nexora"}},
            {"filled_placeholders": {"name": "user_location", "value": "Lahore"}},
            {"filled_placeholders": {"name": "user_school", "value": "LUMS"}},
            {"filled_placeholders": {"name": "user_favorite_language", "value": "Rust"}}
        ]
    """

    filled_placeholders: FilledPlaceholder = Field(
        ...,
        description=(
            "The resolved placeholder as {'name': ..., 'value': ...}. "
            "Applied to every currently-unresolved memory sharing this "
            "placeholder name for the user — content/entities/placeholders "
            "on those memories are left untouched."
        )
    )


# ============================================================
# FINAL EXTRACTION RESULT
# ============================================================

class MemorySchema(BaseModel):
    memories: List[MemoryItem] = Field(
        default_factory=list,
        description=(
            "New or updated FULL memory items — i.e. anything where the "
            "content, entities, type, or other core fields are being "
            "created or changed. Do NOT put pure placeholder resolutions "
            "here — use placeholder_updates for those instead."
        )
    )

    placeholder_updates: List[PlaceholderUpdate] = Field(
        default_factory=list,
        description=(
            "Lightweight placeholder-only resolutions, used when the ONLY "
            "thing changing is that a placeholder (e.g. 'user_company') "
            "now has a real value, and nothing else about any memory "
            "(content/entities/type) changes. Each entry is just the "
            "resolved {name, value} pair — it has no memory id, so it is "
            "applied to every currently-unresolved memory sharing that "
            "placeholder for this user. Do not also return a full "
            "MemoryItem in `memories` for a memory that only needed its "
            "placeholder resolved."
        )
    )




class MemoryDecision(BaseModel):
    needs_retrieval: bool = Field(
        ...,
        description=(
            "True if the user is asking something (or making a request) "
            "whose answer depends on stored personal memories."
        ),
    )
    needs_saving: bool = Field(
        ...,
        description=(
            "True if the message states NEW durable personal information "
            "that should be saved to memory."
        ),
    )
    saving_needs_context: bool = Field(
        ...,
        description=(
            "Only meaningful when needs_saving is true. True if existing "
            "memories are needed to save this message correctly (detect "
            "duplicates, updates, known people/projects, or unresolved "
            "placeholders). Must be false when needs_saving is false."
        ),
    )
 
    @model_validator(mode="after")
    def _context_only_when_saving(self):
        if not self.needs_saving:
            self.saving_needs_context = False
        return self
 





class ContextDecision(BaseModel):
    """Whether the latest message needs previous conversation context."""

    needs_context: bool = Field(...)


class MemoryRetrieval(BaseModel):

    query: str = Field(
        ...,
        description=(
            "The specific information to search for within this memory type. "
            "Rewrite the relevant part of the user's query into a focused "
            "retrieval query."
        )
    )


class RetivalTypes(BaseModel):
    queries: List[MemoryRetrieval] = Field(
        ...,
        description=(
            "A list of independent queries, each representing a distinct memory retrieval requirement. "
            "Create one entry for each distinct memory requirement in the "
            "user's query."
        )
    )




class GraphEntity(BaseModel):
    name: str
    entity_type: str | None = None


class GraphHop(BaseModel):
    relationship: ALLOWED_RELATIONSHIPS
    direction: Literal["outgoing", "incoming", "both"]


class RetrievalPlan(BaseModel):
    query_type: Literal[
        "semantic",
        "graph",
        "hybrid",
    ]

    query: str


    memory_type: MemoryTypes = Field(
        ...,
        description="The type of memory to retrieve."
    )
    # Entities explicitly identifiable from the user's question.
    # These are NOT necessarily the final Neo4j starting nodes.
    entities: list[GraphEntity] = Field(default_factory=list)

    # Relationship path required for graph retrieval.
    path: list[GraphHop] = Field(default_factory=list)

    # What kind of node the answer is looking for.
    target_entity: GraphEntity | None = None

    anchor_entity: GraphEntity | None = None

    max_hops: int = Field(default=0, ge=0, le=6)

    # Whether retrieval should prefer currently valid memories.
    is_current: bool | None = Field(
        default=True,
        description="True for current-only, false for historical-only, null for current and historical."
    )

    # Number of semantic candidates to retrieve from Pinecone.
    top_k: int = Field(default=20, ge=1, le=50)



class ResolvedAnchor(BaseModel):
    name: str
    entity_type: str | None = None
    user_id: str
    source: str

# ============================================================
# Self-test — run this file directly to see both cases in action.
# ============================================================
if __name__ == "__main__":
    # Case 1: a brand-new / fully-updated memory -> goes in `memories`
    full_memory_case = {
        "memories": [
            {
                "id": None,
                "type": "profile",
                "content": "The user's name is Skinder.",
                "placeholders": [],
                "source_role": "user",
                "entities": [
                    {
                        "source": {"name": "user", "type": ["Person", "User"]},
                        "relationship": "has_name",
                        "target": {"name": "Skinder", "type": ["Name"]},
                        "properties": {},
                    }
                ],
                "temporal": {
                    "valid_from": "2026-09-04T12:25:56.586088+00:00",
                    "valid_until": None,
                    "is_current": True,
                },
            }
        ],
        "placeholder_updates": [],
    }

    # Case 2: ONLY resolving a placeholder (no memory id — applied to
    # every currently-unresolved 'user_company' memory for the user)
    # -> goes in `placeholder_updates`, memories stays empty
    placeholder_only_case = {
        "memories": [],
        "placeholder_updates": [
            {
                "filled_placeholders": {"name": "user_company", "value": "Nexora"},
            }
        ],
    }

    print("=== Full memory case ===")
    print(MemorySchema.model_validate(full_memory_case).model_dump_json(indent=2))

    print("\n=== Placeholder-only case ===")
    print(MemorySchema.model_validate(placeholder_only_case).model_dump_json(indent=2))

    # Case 3: a single message resolves FOUR different placeholders at
    # once — proving there's no fixed cap on how many are supported
    four_placeholder_case = {
        "memories": [],
        "placeholder_updates": [
            {"filled_placeholders": {"name": "user_company", "value": "Nexora"}},
            {"filled_placeholders": {"name": "user_location", "value": "Lahore"}},
            {"filled_placeholders": {"name": "user_school", "value": "LUMS"}},
            {"filled_placeholders": {"name": "user_favorite_language", "value": "Rust"}},
        ],
    }
    print("\n=== Four-placeholder case (open-ended) ===")
    print(MemorySchema.model_validate(four_placeholder_case).model_dump_json(indent=2))

    # Case 4: an invalid placeholder name is still rejected — naming
    # convention is enforced even though the set is open-ended
    print("\n=== Invalid placeholder name (should raise) ===")
    try:
        MemorySchema.model_validate({
            "memories": [],
            "placeholder_updates": [
                {"filled_placeholders": {"name": "Company", "value": "Nexora"}}
            ],
        })
        print("ERROR: this should have failed validation but didn't!")
    except Exception as e:
        print("Correctly rejected:", str(e).splitlines()[0])