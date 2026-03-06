"""Query and intent data models."""

from pydantic import BaseModel, Field


class UserQuery(BaseModel):
    """Incoming user query."""

    text: str = Field(..., description="User's natural language question")
    conversation_id: str | None = Field(None, description="Conversation ID for multi-turn context")


class ProcessedQuery(BaseModel):
    """Query after preprocessing."""

    original_text: str
    intent: str = Field(description="Classified intent: 'regulation_lookup' | 'explanation' | 'comparison' | 'general'")
    entities: list[str] = Field(default_factory=list, description="Extracted entities (regulation numbers, topics)")
    keywords: list[str] = Field(default_factory=list, description="Key terms for retrieval")
