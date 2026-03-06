"""Response and confidence data models."""

from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    """Citation linking response to a specific SEBI document section."""

    document_name: str
    section: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class ConfidenceScore(BaseModel):
    """Confidence assessment of a response."""

    overall: float = Field(ge=0.0, le=1.0, description="Overall confidence score")
    retrieval_quality: float = Field(ge=0.0, le=1.0, description="Quality of retrieved context")
    answer_groundedness: float = Field(ge=0.0, le=1.0, description="How grounded the answer is in sources")
    coverage: float = Field(ge=0.0, le=1.0, description="How completely the query is addressed")


class FailureFlag(BaseModel):
    """A detected failure mode flag."""

    mode: str = Field(description="Failure mode identifier")
    description: str
    severity: str = Field(description="'critical' | 'warning' | 'info'")
    mitigation_applied: str


class FinSenseResponse(BaseModel):
    """Complete response from FinSense AI."""

    answer: str = Field(description="User-friendly response text")
    confidence: ConfidenceScore
    citations: list[SourceCitation] = Field(default_factory=list)
    failure_flags: list[FailureFlag] = Field(default_factory=list)
    disclaimer: str | None = Field(None, description="Disclaimer if confidence is low")
    conversation_id: str
