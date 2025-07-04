"""
Core data models and types for the ReasonIt LLM reasoning architecture.

This module defines the fundamental data structures used throughout the system,
including reasoning requests, results, memory entries, and tool interactions.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, validator


class ReasoningStrategy(str, Enum):
    """Available reasoning strategies for the LLM system."""

    CHAIN_OF_THOUGHT = "cot"
    TREE_OF_THOUGHTS = "tot"
    MONTE_CARLO_TREE_SEARCH = "mcts"
    SELF_ASK = "self_ask"
    REFLEXION = "reflexion"
    ADAPTIVE = "adaptive"  # Let the controller choose


class ContextVariant(str, Enum):
    """Context variation types for prompt engineering."""

    MINIFIED = "minified"      # Core information only
    STANDARD = "standard"      # Original prompt
    ENRICHED = "enriched"      # Enhanced with examples and context
    SYMBOLIC = "symbolic"      # Abstract/mathematical representation
    EXEMPLAR = "exemplar"      # Rich with examples and patterns


class ToolType(str, Enum):
    """Available tool types for the reasoning system."""

    PYTHON_EXECUTOR = "python_executor"
    SEARCH = "search"
    CALCULATOR = "calculator"
    KNOWLEDGE_BASE = "knowledge_base"
    VERIFIER = "verifier"
    CODE_GENERATOR = "code_generator"


class OutcomeType(str, Enum):
    """Possible outcomes for reasoning attempts."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    ERROR = "error"


class ReasoningRequest(BaseModel):
    """Request for the reasoning system to process."""

    query: str = Field(..., description="The question or task to reason about")
    strategy: ReasoningStrategy | None = Field(
        default=None,
        description="Specific reasoning strategy to use, or None for adaptive selection"
    )
    context_variant: ContextVariant = Field(
        default=ContextVariant.STANDARD,
        description="Context variation to use for prompt engineering"
    )
    max_cost: float | None = Field(
        default=None,
        description="Maximum cost in dollars for this request"
    )
    max_time: int | None = Field(
        default=None,
        description="Maximum time in seconds for this request"
    )
    use_tools: bool = Field(
        default=True,
        description="Whether to allow tool usage during reasoning"
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence required before accepting result"
    )
    enable_reflection: bool = Field(
        default=True,
        description="Whether to enable reflexion/learning from this request"
    )
    session_id: str | None = Field(
        default_factory=lambda: str(uuid4()),
        description="Session identifier for tracking conversations"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the reasoning request"
    )

    @validator('max_cost')
    def validate_max_cost(cls, v):
        if v is not None and v <= 0:
            raise ValueError("max_cost must be positive")
        return v

    @validator('max_time')
    def validate_max_time(cls, v):
        if v is not None and v <= 0:
            raise ValueError("max_time must be positive")
        return v


class ToolResult(BaseModel):
    """Result from a tool execution."""

    tool_name: str = Field(..., description="Name of the tool that was executed")
    tool_type: ToolType = Field(..., description="Type of the tool")
    input_data: dict[str, Any] = Field(..., description="Input parameters to the tool")
    output_data: Any = Field(..., description="Output from the tool execution")
    success: bool = Field(..., description="Whether the tool execution was successful")
    error_message: str | None = Field(
        default=None,
        description="Error message if execution failed"
    )
    execution_time: float = Field(
        ge=0.0,
        description="Time taken for tool execution in seconds"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the tool was executed"
    )
    cost: float = Field(
        default=0.0,
        ge=0.0,
        description="Cost associated with this tool usage"
    )


class ReasoningStep(BaseModel):
    """Individual step in a reasoning trace."""

    step_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this reasoning step"
    )
    step_number: int = Field(
        ge=0,
        description="Sequential number of this step in the reasoning process"
    )
    strategy: ReasoningStrategy = Field(..., description="Strategy used for this step")
    content: str = Field(..., description="The reasoning content/thought for this step")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for this step"
    )
    cost: float = Field(
        ge=0.0,
        description="Cost incurred for this step"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this step was executed"
    )
    tools_used: list[ToolResult] = Field(
        default_factory=list,
        description="Tools that were used in this step"
    )
    intermediate_result: str | None = Field(
        default=None,
        description="Intermediate result or conclusion for this step"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for this step"
    )


class ReasoningResult(BaseModel):
    """Complete result from a reasoning request."""

    request: ReasoningRequest = Field(..., description="Original request that generated this result")
    final_answer: str = Field(..., description="Final answer or conclusion")
    reasoning_trace: list[ReasoningStep] = Field(
        default_factory=list,
        description="Complete trace of reasoning steps taken"
    )
    total_cost: float = Field(
        ge=0.0,
        description="Total cost for this reasoning session"
    )
    total_time: float = Field(
        ge=0.0,
        description="Total time taken in seconds"
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence in the final answer"
    )
    strategies_used: list[ReasoningStrategy] = Field(
        default_factory=list,
        description="All strategies that were used during reasoning"
    )
    outcome: OutcomeType = Field(
        default=OutcomeType.SUCCESS,
        description="Overall outcome of the reasoning process"
    )
    reflection: str | None = Field(
        default=None,
        description="Reflection on the reasoning process for learning"
    )
    lessons_learned: list[str] = Field(
        default_factory=list,
        description="Key lessons extracted from this reasoning session"
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if reasoning failed"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this result was generated"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the reasoning process"
    )


class MemoryEntry(BaseModel):
    """Entry in the episodic memory system for learning."""

    entry_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this memory entry"
    )
    query: str = Field(..., description="Original query that led to this memory")
    query_embedding: list[float] | None = Field(
        default=None,
        description="Vector embedding of the query for similarity search"
    )
    strategy: ReasoningStrategy = Field(
        ...,
        description="Strategy that was used for this reasoning attempt"
    )
    outcome: OutcomeType = Field(
        ...,
        description="Outcome of the reasoning attempt"
    )
    confidence_achieved: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score that was achieved"
    )
    cost_incurred: float = Field(
        ge=0.0,
        description="Cost that was incurred for this attempt"
    )
    time_taken: float = Field(
        ge=0.0,
        description="Time taken for this reasoning attempt"
    )
    reflection: str = Field(
        ...,
        description="Reflection on what happened and why"
    )
    lessons: list[str] = Field(
        default_factory=list,
        description="Specific lessons learned from this experience"
    )
    error_patterns: list[str] = Field(
        default_factory=list,
        description="Error patterns identified in this attempt"
    )
    success_patterns: list[str] = Field(
        default_factory=list,
        description="Success patterns identified in this attempt"
    )
    context_used: ContextVariant = Field(
        ...,
        description="Context variant that was used"
    )
    tools_used: list[str] = Field(
        default_factory=list,
        description="Names of tools that were used"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this memory entry was created"
    )
    relevance_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Relevance score for memory consolidation"
    )
    access_count: int = Field(
        default=0,
        ge=0,
        description="Number of times this memory has been accessed"
    )
    last_accessed: datetime | None = Field(
        default=None,
        description="When this memory was last accessed"
    )


class ThoughtNode(BaseModel):
    """Node in a Tree of Thoughts reasoning structure."""

    node_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this thought node"
    )
    content: str = Field(..., description="The thought content")
    parent_id: str | None = Field(
        default=None,
        description="ID of the parent node"
    )
    children_ids: list[str] = Field(
        default_factory=list,
        description="IDs of child nodes"
    )
    depth: int = Field(
        ge=0,
        description="Depth of this node in the tree"
    )
    value_score: float = Field(
        default=0.0,
        description="Value/quality score for this thought"
    )
    visits: int = Field(
        default=0,
        ge=0,
        description="Number of times this node has been visited (for MCTS)"
    )
    is_terminal: bool = Field(
        default=False,
        description="Whether this is a terminal/final thought"
    )
    is_solution: bool = Field(
        default=False,
        description="Whether this thought represents a valid solution"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for this thought node"
    )


class UsageMetrics(BaseModel):
    """Metrics for tracking API usage and costs."""

    session_id: str = Field(..., description="Session identifier")
    model_name: str = Field(..., description="Name of the model used")
    input_tokens: int = Field(ge=0, description="Number of input tokens")
    output_tokens: int = Field(ge=0, description="Number of output tokens")
    total_tokens: int = Field(ge=0, description="Total tokens used")
    cost: float = Field(ge=0.0, description="Cost for this usage")
    timestamp: datetime = Field(default_factory=datetime.now)
    tool_calls: int = Field(default=0, ge=0, description="Number of tool calls made")
    reasoning_strategy: ReasoningStrategy | None = Field(
        default=None,
        description="Strategy used for this usage"
    )

    @validator('total_tokens', always=True)
    def validate_total_tokens(cls, v, values):
        input_tokens = values.get('input_tokens', 0)
        output_tokens = values.get('output_tokens', 0)
        expected_total = input_tokens + output_tokens
        if v != expected_total:
            return expected_total
        return v


class SystemConfiguration(BaseModel):
    """System-wide configuration settings."""

    primary_model: str = Field(default="gpt-4o-mini")
    fallback_model: str = Field(default="gpt-4")
    coach_model: str = Field(default="gpt-4")
    max_daily_cost: float = Field(default=10.0, ge=0.0)
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    enable_tools: bool = Field(default=True)
    enable_memory: bool = Field(default=True)
    enable_reflection: bool = Field(default=True)
    max_reasoning_depth: int = Field(default=10, ge=1)
    memory_max_entries: int = Field(default=10000, ge=1)
    log_level: str = Field(default="INFO")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
