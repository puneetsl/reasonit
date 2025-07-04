"""
Reflexion memory system for learning from reasoning experiences.

This module implements episodic memory, error analysis, and pattern recognition
to help reasoning agents learn and improve from past experiences.
"""

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict, Counter
import hashlib
import uuid

from models import ReasoningResult, ReasoningRequest, ReasoningStrategy, OutcomeType

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory entries."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL_SUCCESS = "partial_success"
    ERROR = "error"
    PATTERN = "pattern"
    INSIGHT = "insight"


class ErrorCategory(Enum):
    """Categories of reasoning errors."""
    LOGICAL_ERROR = "logical_error"
    FACTUAL_ERROR = "factual_error"
    MATHEMATICAL_ERROR = "mathematical_error"
    PROCEDURAL_ERROR = "procedural_error"
    CONFIDENCE_ERROR = "confidence_error"
    STRATEGY_ERROR = "strategy_error"
    TOOL_ERROR = "tool_error"
    CONTEXT_ERROR = "context_error"


@dataclass
class ErrorPattern:
    """Represents an identified error pattern."""
    
    pattern_id: str = ""
    category: ErrorCategory = ErrorCategory.LOGICAL_ERROR
    description: str = ""
    frequency: int = 0
    
    # Pattern characteristics
    triggers: List[str] = field(default_factory=list)  # What tends to cause this error
    indicators: List[str] = field(default_factory=list)  # How to recognize this error
    solutions: List[str] = field(default_factory=list)  # How to fix/avoid this error
    
    # Statistics
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    contexts: List[str] = field(default_factory=list)  # Where this pattern appears
    
    def __post_init__(self):
        if not self.pattern_id:
            self.pattern_id = f"pattern_{id(self)}"


@dataclass
class MemoryEntry:
    """Represents a single memory entry in the Reflexion system."""
    
    # Entry identification
    entry_id: str = ""
    memory_type: MemoryType = MemoryType.SUCCESS
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Original request and result
    request: Optional[ReasoningRequest] = None
    result: Optional[ReasoningResult] = None
    
    # Analysis and insights
    summary: str = ""
    key_insights: List[str] = field(default_factory=list)
    error_analysis: Optional[str] = None
    error_category: Optional[ErrorCategory] = None
    
    # Pattern information
    identified_patterns: List[str] = field(default_factory=list)  # Pattern IDs
    contributes_to_patterns: List[str] = field(default_factory=list)
    
    # Context and metadata
    strategy_used: Optional[ReasoningStrategy] = None
    confidence_achieved: float = 0.0
    cost_incurred: float = 0.0
    time_taken: float = 0.0
    context_used: str = ""
    
    # Learning value
    learning_weight: float = 1.0  # How much to weight this experience
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.entry_id:
            self.entry_id = f"memory_{uuid.uuid4().hex[:8]}_{int(self.timestamp.timestamp())}"


@dataclass
class ReflexionInsight:
    """Represents a learned insight from memory analysis."""
    
    insight_id: str = ""
    insight_type: str = "general"  # general, strategy-specific, domain-specific
    description: str = ""
    confidence: float = 0.0
    
    # Supporting evidence
    supporting_entries: List[str] = field(default_factory=list)  # Memory entry IDs
    evidence_strength: float = 0.0
    
    # Applicability
    applicable_strategies: List[ReasoningStrategy] = field(default_factory=list)
    applicable_domains: List[str] = field(default_factory=list)
    triggers: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_validated: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    success_rate: float = 0.0
    
    def __post_init__(self):
        if not self.insight_id:
            self.insight_id = f"insight_{id(self)}"


class ReflexionMemorySystem:
    """
    Reflexion memory system for learning from reasoning experiences.
    
    This system provides:
    1. Episodic memory storage and retrieval
    2. Error pattern analysis and recognition
    3. Insight generation and validation
    4. Performance trend analysis
    5. Strategy effectiveness tracking
    """
    
    def __init__(
        self,
        memory_db_path: str = "reflexion_memory.db",
        max_memory_entries: int = 10000,
        retention_days: int = 365,
        auto_cleanup: bool = True
    ):
        self.memory_db_path = Path(memory_db_path)
        self.max_memory_entries = max_memory_entries
        self.retention_days = retention_days
        self.auto_cleanup = auto_cleanup
        
        # In-memory caches for performance
        self.recent_entries: List[MemoryEntry] = []
        self.error_patterns: Dict[str, ErrorPattern] = {}
        self.insights: Dict[str, ReflexionInsight] = {}
        
        # Analysis counters
        self.strategy_performance: Dict[ReasoningStrategy, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.error_frequency: Dict[ErrorCategory, int] = defaultdict(int)
        
        # Initialize database
        self._init_database()
        self._load_cache()
    
    def _init_database(self) -> None:
        """Initialize the SQLite database for persistent storage."""
        
        with sqlite3.connect(self.memory_db_path) as conn:
            cursor = conn.cursor()
            
            # Memory entries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_entries (
                    entry_id TEXT PRIMARY KEY,
                    memory_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    request_data TEXT,
                    result_data TEXT,
                    summary TEXT,
                    key_insights TEXT,
                    error_analysis TEXT,
                    error_category TEXT,
                    strategy_used TEXT,
                    confidence_achieved REAL,
                    cost_incurred REAL,
                    time_taken REAL,
                    context_used TEXT,
                    learning_weight REAL,
                    tags TEXT,
                    identified_patterns TEXT
                )
            """)
            
            # Error patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS error_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    description TEXT,
                    frequency INTEGER,
                    triggers TEXT,
                    indicators TEXT,
                    solutions TEXT,
                    first_seen TEXT,
                    last_seen TEXT,
                    contexts TEXT
                )
            """)
            
            # Insights table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS insights (
                    insight_id TEXT PRIMARY KEY,
                    insight_type TEXT,
                    description TEXT,
                    confidence REAL,
                    supporting_entries TEXT,
                    evidence_strength REAL,
                    applicable_strategies TEXT,
                    applicable_domains TEXT,
                    triggers TEXT,
                    created_at TEXT,
                    last_validated TEXT,
                    usage_count INTEGER,
                    success_rate REAL
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON memory_entries(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_entries(memory_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_strategy ON memory_entries(strategy_used)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pattern_category ON error_patterns(category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_insight_type ON insights(insight_type)")
            
            conn.commit()
    
    def _load_cache(self) -> None:
        """Load recent data into memory caches for performance."""
        
        # Load recent memory entries
        recent_cutoff = datetime.now() - timedelta(days=7)
        self.recent_entries = self.retrieve_memories(
            since=recent_cutoff,
            limit=100
        )
        
        # Load error patterns
        self.error_patterns = self._load_error_patterns()
        
        # Load insights
        self.insights = self._load_insights()
        
        # Update performance statistics
        self._update_performance_stats()
    
    async def store_memory(
        self,
        request: ReasoningRequest,
        result: ReasoningResult,
        error_analysis: Optional[str] = None
    ) -> MemoryEntry:
        """Store a reasoning experience in memory."""
        
        # Determine memory type based on outcome
        memory_type = self._determine_memory_type(result)
        
        # Analyze the result for insights
        summary = await self._generate_summary(request, result)
        key_insights = await self._extract_key_insights(request, result)
        
        # Identify error category if applicable
        error_category = None
        if result.outcome in [OutcomeType.ERROR, OutcomeType.FAILURE]:
            error_category = await self._categorize_error(request, result, error_analysis)
        
        # Create memory entry
        entry = MemoryEntry(
            memory_type=memory_type,
            request=request,
            result=result,
            summary=summary,
            key_insights=key_insights,
            error_analysis=error_analysis,
            error_category=error_category,
            strategy_used=result.strategies_used[0] if result.strategies_used else None,
            confidence_achieved=result.confidence_score,
            cost_incurred=result.total_cost,
            time_taken=result.total_time,
            context_used=getattr(request, 'context_variant', '').value if hasattr(getattr(request, 'context_variant', ''), 'value') else '',
            learning_weight=self._calculate_learning_weight(result),
            tags=self._generate_tags(request, result)
        )
        
        # Store in database
        await self._persist_memory_entry(entry)
        
        # Add to cache
        self.recent_entries.append(entry)
        if len(self.recent_entries) > 100:
            self.recent_entries.pop(0)
        
        # Update performance stats with new entry
        self._update_performance_stats()
        
        # Analyze for patterns and insights
        await self._analyze_for_patterns(entry)
        await self._update_insights(entry)
        
        # Cleanup if needed
        if self.auto_cleanup:
            await self._cleanup_old_entries()
        
        logger.info(f"Stored memory entry: {entry.entry_id} ({memory_type.value})")
        return entry
    
    def _determine_memory_type(self, result: ReasoningResult) -> MemoryType:
        """Determine the type of memory based on the result."""
        
        if result.outcome == OutcomeType.SUCCESS:
            if result.confidence_score >= 0.8:
                return MemoryType.SUCCESS
            else:
                return MemoryType.PARTIAL_SUCCESS
        elif result.outcome == OutcomeType.ERROR:
            return MemoryType.ERROR
        elif result.outcome == OutcomeType.FAILURE:
            return MemoryType.FAILURE
        elif result.outcome == OutcomeType.PARTIAL:
            return MemoryType.PARTIAL_SUCCESS
        else:
            return MemoryType.PARTIAL_SUCCESS
    
    async def _generate_summary(
        self,
        request: ReasoningRequest,
        result: ReasoningResult
    ) -> str:
        """Generate a concise summary of the reasoning experience."""
        
        strategy = result.strategies_used[0].value if result.strategies_used else "unknown"
        outcome = result.outcome.value
        confidence = result.confidence_score
        
        summary_parts = [
            f"Used {strategy} strategy to solve: {request.query[:100]}...",
            f"Outcome: {outcome} (confidence: {confidence:.2f})",
        ]
        
        if result.total_cost > 0:
            summary_parts.append(f"Cost: ${result.total_cost:.4f}")
        
        if result.error_message:
            summary_parts.append(f"Error: {result.error_message[:100]}...")
        
        return " | ".join(summary_parts)
    
    async def _extract_key_insights(
        self,
        request: ReasoningRequest,
        result: ReasoningResult
    ) -> List[str]:
        """Extract key insights from the reasoning experience."""
        
        insights = []
        
        # Strategy effectiveness insight
        if result.confidence_score >= 0.8:
            strategy = result.strategies_used[0].value if result.strategies_used else "unknown"
            insights.append(f"{strategy} strategy worked well for this type of problem")
        
        # Cost efficiency insight
        if result.total_cost < 0.01 and result.confidence_score >= 0.7:
            insights.append("Achieved good results with low cost")
        
        # Tool usage insight
        if any("tools_used" in step.metadata for step in result.reasoning_trace if step.metadata):
            insights.append("Tool usage contributed to successful reasoning")
        
        # Reflection insight
        if result.reflection and "high confidence" in result.reflection.lower():
            insights.append("Self-reflection indicated high confidence in solution")
        
        # Error pattern insight
        if result.outcome in [OutcomeType.ERROR, OutcomeType.FAILURE]:
            insights.append("Need to investigate failure patterns for this problem type")
        
        return insights[:5]  # Limit to top 5 insights
    
    async def _categorize_error(
        self,
        request: ReasoningRequest,
        result: ReasoningResult,
        error_analysis: Optional[str] = None
    ) -> Optional[ErrorCategory]:
        """Categorize the type of error that occurred."""
        
        if not result.error_message and result.outcome != OutcomeType.FAILURE:
            return None
        
        error_text = (result.error_message or "").lower()
        error_analysis_text = (error_analysis or "").lower()
        
        # Combine both error sources for categorization
        combined_error_text = f"{error_text} {error_analysis_text}".strip()
        
        # Check for specific error patterns
        if any(keyword in combined_error_text for keyword in ["syntax", "parse", "format"]):
            return ErrorCategory.PROCEDURAL_ERROR
        
        if any(keyword in combined_error_text for keyword in ["tool", "execute", "api"]):
            return ErrorCategory.TOOL_ERROR
        
        if any(keyword in combined_error_text for keyword in ["math", "calculation", "arithmetic"]):
            return ErrorCategory.MATHEMATICAL_ERROR
        
        if any(keyword in combined_error_text for keyword in ["fact", "information", "knowledge"]):
            return ErrorCategory.FACTUAL_ERROR
        
        if any(keyword in combined_error_text for keyword in ["logic", "reasoning", "contradiction"]):
            return ErrorCategory.LOGICAL_ERROR
        
        if any(keyword in combined_error_text for keyword in ["confidence", "threshold", "uncertain"]):
            return ErrorCategory.CONFIDENCE_ERROR
        
        if any(keyword in combined_error_text for keyword in ["strategy", "approach", "method"]):
            return ErrorCategory.STRATEGY_ERROR
        
        if any(keyword in combined_error_text for keyword in ["context", "prompt", "understanding"]):
            return ErrorCategory.CONTEXT_ERROR
        
        # Default categorization based on outcome
        if result.outcome == OutcomeType.FAILURE:
            return ErrorCategory.STRATEGY_ERROR
        else:
            return ErrorCategory.PROCEDURAL_ERROR
    
    def _calculate_learning_weight(self, result: ReasoningResult) -> float:
        """Calculate how much weight to give this experience for learning."""
        
        base_weight = 1.0
        
        # Failures and errors are more valuable for learning
        if result.outcome in [OutcomeType.ERROR, OutcomeType.FAILURE]:
            base_weight *= 1.5
        
        # High confidence successes are also valuable
        elif result.outcome == OutcomeType.SUCCESS and result.confidence_score >= 0.9:
            base_weight *= 1.2
        
        # Unique or novel experiences are more valuable
        if result.total_cost > 0.05:  # Expensive = likely complex/novel
            base_weight *= 1.1
        
        # Recent experiences are more relevant
        # (this would be adjusted based on time since occurrence)
        
        return min(base_weight, 2.0)  # Cap at 2x normal weight
    
    def _generate_tags(
        self,
        request: ReasoningRequest,
        result: ReasoningResult
    ) -> List[str]:
        """Generate tags for categorizing and searching memories."""
        
        tags = []
        
        # Strategy tags
        if result.strategies_used:
            tags.extend([s.value for s in result.strategies_used])
        
        # Outcome tags
        tags.append(result.outcome.value)
        
        # Confidence tags
        if result.confidence_score >= 0.9:
            tags.append("high_confidence")
        elif result.confidence_score <= 0.5:
            tags.append("low_confidence")
        
        # Cost tags
        if result.total_cost <= 0.01:
            tags.append("low_cost")
        elif result.total_cost >= 0.10:
            tags.append("high_cost")
        
        # Problem type tags (basic heuristics)
        query_lower = request.query.lower()
        if any(keyword in query_lower for keyword in ["calculate", "math", "number"]):
            tags.append("mathematical")
        
        if any(keyword in query_lower for keyword in ["logic", "if", "then", "all", "some"]):
            tags.append("logical")
        
        if any(keyword in query_lower for keyword in ["fact", "when", "where", "who"]):
            tags.append("factual")
        
        if any(keyword in query_lower for keyword in ["plan", "strategy", "approach"]):
            tags.append("planning")
        
        return list(set(tags))  # Remove duplicates
    
    async def _persist_memory_entry(self, entry: MemoryEntry) -> None:
        """Persist memory entry to database."""
        
        with sqlite3.connect(self.memory_db_path) as conn:
            cursor = conn.cursor()
            
            # Serialize complex objects
            request_data = entry.request.model_dump_json() if entry.request else None
            result_data = entry.result.model_dump_json() if entry.result else None
            
            cursor.execute("""
                INSERT OR REPLACE INTO memory_entries (
                    entry_id, memory_type, timestamp, request_data, result_data,
                    summary, key_insights, error_analysis, error_category,
                    strategy_used, confidence_achieved, cost_incurred, time_taken,
                    context_used, learning_weight, tags, identified_patterns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.entry_id,
                entry.memory_type.value,
                entry.timestamp.isoformat(),
                request_data,
                result_data,
                entry.summary,
                json.dumps(entry.key_insights),
                entry.error_analysis,
                entry.error_category.value if entry.error_category else None,
                entry.strategy_used.value if entry.strategy_used else None,
                entry.confidence_achieved,
                entry.cost_incurred,
                entry.time_taken,
                entry.context_used,
                entry.learning_weight,
                json.dumps(entry.tags),
                json.dumps(entry.identified_patterns)
            ))
            
            conn.commit()
    
    async def _analyze_for_patterns(self, entry: MemoryEntry) -> None:
        """Analyze a memory entry for error patterns."""
        
        if entry.memory_type not in [MemoryType.ERROR, MemoryType.FAILURE]:
            return
        
        # Look for existing patterns this entry might match
        for pattern_id, pattern in self.error_patterns.items():
            if self._entry_matches_pattern(entry, pattern):
                pattern.frequency += 1
                pattern.last_seen = entry.timestamp
                entry.identified_patterns.append(pattern_id)
                
                # Update pattern contexts
                if entry.context_used and entry.context_used not in pattern.contexts:
                    pattern.contexts.append(entry.context_used)
        
        # Try to identify new patterns if this doesn't match existing ones
        if not entry.identified_patterns:
            new_pattern = await self._try_create_new_pattern(entry)
            if new_pattern:
                self.error_patterns[new_pattern.pattern_id] = new_pattern
                entry.identified_patterns.append(new_pattern.pattern_id)
                await self._persist_error_pattern(new_pattern)
    
    def _entry_matches_pattern(self, entry: MemoryEntry, pattern: ErrorPattern) -> bool:
        """Check if a memory entry matches an existing error pattern."""
        
        # Must be same error category
        if entry.error_category != pattern.category:
            return False
        
        # Check for trigger keywords in the request
        if pattern.triggers and entry.request:
            query_text = entry.request.query.lower()
            if not any(trigger.lower() in query_text for trigger in pattern.triggers):
                return False
        
        # Check for indicator keywords in error/result
        if pattern.indicators:
            error_text = (entry.error_analysis or "").lower()
            result_text = (entry.result.error_message or "").lower() if entry.result else ""
            combined_text = error_text + " " + result_text
            
            if any(indicator.lower() in combined_text for indicator in pattern.indicators):
                return True
        
        return False
    
    async def _try_create_new_pattern(self, entry: MemoryEntry) -> Optional[ErrorPattern]:
        """Try to create a new error pattern based on this entry."""
        
        if not entry.error_category or not entry.error_analysis:
            return None
        
        # Look for similar recent entries to establish a pattern
        similar_entries = [
            e for e in self.recent_entries[-20:]  # Check last 20 entries
            if (e.error_category == entry.error_category and
                e.memory_type in [MemoryType.ERROR, MemoryType.FAILURE] and
                e != entry)
        ]
        
        if len(similar_entries) < 2:  # Need at least 2 similar entries to establish pattern
            return None
        
        # Extract common elements
        triggers = self._extract_common_triggers([entry] + similar_entries)
        indicators = self._extract_common_indicators([entry] + similar_entries)
        
        if not triggers and not indicators:
            return None
        
        # Create new pattern
        pattern = ErrorPattern(
            category=entry.error_category,
            description=f"Pattern for {entry.error_category.value} errors",
            frequency=len(similar_entries) + 1,
            triggers=triggers,
            indicators=indicators,
            solutions=[],  # Will be populated as we learn
            first_seen=min(e.timestamp for e in [entry] + similar_entries),
            last_seen=entry.timestamp,
            contexts=[entry.context_used] if entry.context_used else []
        )
        
        return pattern
    
    def _extract_common_triggers(self, entries: List[MemoryEntry]) -> List[str]:
        """Extract common trigger words from a set of entries."""
        
        # Get all query words
        all_words = []
        for entry in entries:
            if entry.request:
                words = entry.request.query.lower().split()
                all_words.extend(words)
        
        # Find words that appear in multiple entries
        word_counts = Counter(all_words)
        common_words = [word for word, count in word_counts.items() 
                      if count >= len(entries) // 2 and len(word) > 3]
        
        return common_words[:5]  # Return top 5
    
    def _extract_common_indicators(self, entries: List[MemoryEntry]) -> List[str]:
        """Extract common indicator phrases from error messages."""
        
        # Get all error text
        all_text = []
        for entry in entries:
            if entry.error_analysis:
                all_text.append(entry.error_analysis.lower())
            if entry.result and entry.result.error_message:
                all_text.append(entry.result.error_message.lower())
        
        if not all_text:
            return []
        
        # Find common phrases (simple approach)
        combined_text = " ".join(all_text)
        words = combined_text.split()
        
        # Look for repeated phrases of 2-3 words
        indicators = []
        for i in range(len(words) - 1):
            phrase = " ".join(words[i:i+2])
            if combined_text.count(phrase) >= len(entries) // 2:
                indicators.append(phrase)
        
        return list(set(indicators))[:5]  # Return top 5 unique
    
    async def _persist_error_pattern(self, pattern: ErrorPattern) -> None:
        """Persist error pattern to database."""
        
        with sqlite3.connect(self.memory_db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO error_patterns (
                    pattern_id, category, description, frequency, triggers,
                    indicators, solutions, first_seen, last_seen, contexts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern.pattern_id,
                pattern.category.value,
                pattern.description,
                pattern.frequency,
                json.dumps(pattern.triggers),
                json.dumps(pattern.indicators),
                json.dumps(pattern.solutions),
                pattern.first_seen.isoformat(),
                pattern.last_seen.isoformat(),
                json.dumps(pattern.contexts)
            ))
            
            conn.commit()
    
    async def _update_insights(self, entry: MemoryEntry) -> None:
        """Update insights based on new memory entry."""
        
        # Strategy effectiveness insights
        if entry.strategy_used:
            await self._update_strategy_insights(entry)
        
        # Error pattern insights
        if entry.identified_patterns:
            await self._update_error_insights(entry)
        
        # General performance insights
        await self._update_performance_insights(entry)
    
    async def _update_strategy_insights(self, entry: MemoryEntry) -> None:
        """Update insights about strategy effectiveness."""
        
        strategy = entry.strategy_used
        if not strategy:
            return
        
        insight_id = f"strategy_{strategy.value}_effectiveness"
        
        if insight_id in self.insights:
            insight = self.insights[insight_id]
            insight.supporting_entries.append(entry.entry_id)
            insight.usage_count += 1
            
            # Recalculate success rate
            strategy_entries = [e for e in self.recent_entries if e.strategy_used == strategy]
            successful = len([e for e in strategy_entries if e.memory_type == MemoryType.SUCCESS])
            insight.success_rate = successful / len(strategy_entries) if strategy_entries else 0.0
            
        else:
            # Create new strategy insight
            insight = ReflexionInsight(
                insight_id=insight_id,
                insight_type="strategy-specific",
                description=f"Effectiveness tracking for {strategy.value} strategy",
                confidence=0.5,
                supporting_entries=[entry.entry_id],
                applicable_strategies=[strategy],
                usage_count=1,
                success_rate=1.0 if entry.memory_type == MemoryType.SUCCESS else 0.0
            )
            self.insights[insight_id] = insight
    
    async def _update_error_insights(self, entry: MemoryEntry) -> None:
        """Update insights about error patterns."""
        
        for pattern_id in entry.identified_patterns:
            if pattern_id in self.error_patterns:
                pattern = self.error_patterns[pattern_id]
                
                insight_id = f"error_pattern_{pattern_id}"
                description = f"Error pattern: {pattern.description}"
                
                if insight_id in self.insights:
                    insight = self.insights[insight_id]
                    insight.supporting_entries.append(entry.entry_id)
                else:
                    insight = ReflexionInsight(
                        insight_id=insight_id,
                        insight_type="error-pattern",
                        description=description,
                        confidence=min(pattern.frequency / 10, 1.0),  # Higher frequency = higher confidence
                        supporting_entries=[entry.entry_id],
                        triggers=pattern.triggers.copy()
                    )
                    self.insights[insight_id] = insight
    
    async def _update_performance_insights(self, entry: MemoryEntry) -> None:
        """Update general performance insights."""
        
        # Cost-effectiveness insight
        if entry.cost_incurred > 0 and entry.confidence_achieved > 0:
            efficiency_ratio = entry.confidence_achieved / entry.cost_incurred
            
            insight_id = "cost_effectiveness"
            if insight_id in self.insights:
                self.insights[insight_id].supporting_entries.append(entry.entry_id)
            else:
                insight = ReflexionInsight(
                    insight_id=insight_id,
                    insight_type="performance",
                    description="Cost-effectiveness patterns in reasoning",
                    supporting_entries=[entry.entry_id]
                )
                self.insights[insight_id] = insight
    
    def retrieve_memories(
        self,
        memory_type: Optional[MemoryType] = None,
        strategy: Optional[ReasoningStrategy] = None,
        error_category: Optional[ErrorCategory] = None,
        tags: Optional[List[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[MemoryEntry]:
        """Retrieve memories based on criteria."""
        
        with sqlite3.connect(self.memory_db_path) as conn:
            cursor = conn.cursor()
            
            # Build query
            query = "SELECT * FROM memory_entries WHERE 1=1"
            params = []
            
            if memory_type:
                query += " AND memory_type = ?"
                params.append(memory_type.value)
            
            if strategy:
                query += " AND strategy_used = ?"
                params.append(strategy.value)
            
            if error_category:
                query += " AND error_category = ?"
                params.append(error_category.value)
            
            if since:
                query += " AND timestamp >= ?"
                params.append(since.isoformat())
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to MemoryEntry objects
            memories = []
            for row in rows:
                # Reconstruct MemoryEntry from database row
                # Row format: entry_id, memory_type, timestamp, request_data, result_data, 
                # summary, key_insights, error_analysis, error_category, strategy_used,
                # confidence_achieved, cost_incurred, time_taken, context_used, learning_weight, tags, identified_patterns
                
                # Parse strategy_used
                strategy_used = None
                if row[9]:  # strategy_used column
                    try:
                        strategy_used = ReasoningStrategy(row[9])
                    except ValueError:
                        strategy_used = None
                
                # Parse error_category
                error_category = None
                if row[8]:  # error_category column
                    try:
                        error_category = ErrorCategory(row[8])
                    except ValueError:
                        error_category = None
                
                # Parse key_insights
                key_insights = []
                if row[6]:  # key_insights column
                    try:
                        key_insights = json.loads(row[6])
                    except (json.JSONDecodeError, TypeError):
                        key_insights = []
                
                # Parse tags
                tags = []
                if row[15]:  # tags column
                    try:
                        tags = json.loads(row[15])
                    except (json.JSONDecodeError, TypeError):
                        tags = []
                
                memory = MemoryEntry(
                    entry_id=row[0],
                    memory_type=MemoryType(row[1]),
                    timestamp=datetime.fromisoformat(row[2]),
                    summary=row[5] or "",
                    key_insights=key_insights,
                    error_analysis=row[7],  # error_analysis column
                    error_category=error_category,
                    strategy_used=strategy_used,
                    confidence_achieved=row[10] or 0.0,
                    cost_incurred=row[11] or 0.0,
                    time_taken=row[12] or 0.0,
                    context_used=row[13] or "",
                    learning_weight=row[14] or 1.0,
                    tags=tags
                )
                memories.append(memory)
            
            return memories
    
    def get_error_patterns(
        self,
        category: Optional[ErrorCategory] = None,
        min_frequency: int = 2
    ) -> List[ErrorPattern]:
        """Get error patterns, optionally filtered by category."""
        
        patterns = list(self.error_patterns.values())
        
        if category:
            patterns = [p for p in patterns if p.category == category]
        
        if min_frequency:
            patterns = [p for p in patterns if p.frequency >= min_frequency]
        
        return sorted(patterns, key=lambda p: p.frequency, reverse=True)
    
    def get_insights(
        self,
        insight_type: Optional[str] = None,
        min_confidence: float = 0.5
    ) -> List[ReflexionInsight]:
        """Get insights, optionally filtered by type and confidence."""
        
        insights = list(self.insights.values())
        
        if insight_type:
            insights = [i for i in insights if i.insight_type == insight_type]
        
        insights = [i for i in insights if i.confidence >= min_confidence]
        
        return sorted(insights, key=lambda i: i.confidence, reverse=True)
    
    def get_strategy_performance(self) -> Dict[ReasoningStrategy, Dict[str, float]]:
        """Get performance statistics for each strategy."""
        
        return dict(self.strategy_performance)
    
    def _load_error_patterns(self) -> Dict[str, ErrorPattern]:
        """Load error patterns from database."""
        
        patterns = {}
        
        with sqlite3.connect(self.memory_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM error_patterns")
            rows = cursor.fetchall()
            
            for row in rows:
                pattern = ErrorPattern(
                    pattern_id=row[0],
                    category=ErrorCategory(row[1]),
                    description=row[2] or "",
                    frequency=row[3] or 0,
                    triggers=json.loads(row[4] or "[]"),
                    indicators=json.loads(row[5] or "[]"),
                    solutions=json.loads(row[6] or "[]"),
                    first_seen=datetime.fromisoformat(row[7]) if row[7] else datetime.now(),
                    last_seen=datetime.fromisoformat(row[8]) if row[8] else datetime.now(),
                    contexts=json.loads(row[9] or "[]")
                )
                patterns[pattern.pattern_id] = pattern
        
        return patterns
    
    def _load_insights(self) -> Dict[str, ReflexionInsight]:
        """Load insights from database."""
        
        insights = {}
        
        with sqlite3.connect(self.memory_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM insights")
            rows = cursor.fetchall()
            
            for row in rows:
                insight = ReflexionInsight(
                    insight_id=row[0],
                    insight_type=row[1] or "general",
                    description=row[2] or "",
                    confidence=row[3] or 0.0,
                    supporting_entries=json.loads(row[4] or "[]"),
                    evidence_strength=row[5] or 0.0,
                    applicable_strategies=[ReasoningStrategy(s) for s in json.loads(row[6] or "[]") if s],
                    applicable_domains=json.loads(row[7] or "[]"),
                    triggers=json.loads(row[8] or "[]"),
                    created_at=datetime.fromisoformat(row[9]) if row[9] else datetime.now(),
                    last_validated=datetime.fromisoformat(row[10]) if row[10] else datetime.now(),
                    usage_count=row[11] or 0,
                    success_rate=row[12] or 0.0
                )
                insights[insight.insight_id] = insight
        
        return insights
    
    def _update_performance_stats(self) -> None:
        """Update performance statistics from recent memories."""
        
        # Reset stats
        self.strategy_performance.clear()
        self.error_frequency.clear()
        
        # Calculate from recent entries
        for entry in self.recent_entries:
            if entry.strategy_used:
                strategy = entry.strategy_used
                self.strategy_performance[strategy]["total_uses"] += 1
                self.strategy_performance[strategy]["total_confidence"] += entry.confidence_achieved
                self.strategy_performance[strategy]["total_cost"] += entry.cost_incurred
                
                if entry.memory_type == MemoryType.SUCCESS:
                    self.strategy_performance[strategy]["successes"] += 1
            
            if entry.error_category:
                self.error_frequency[entry.error_category] += 1
        
        # Calculate averages
        for strategy, stats in self.strategy_performance.items():
            if stats["total_uses"] > 0:
                stats["avg_confidence"] = stats["total_confidence"] / stats["total_uses"]
                stats["avg_cost"] = stats["total_cost"] / stats["total_uses"]
                stats["success_rate"] = stats["successes"] / stats["total_uses"]
    
    async def _cleanup_old_entries(self) -> None:
        """Clean up old memory entries to maintain performance."""
        
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        with sqlite3.connect(self.memory_db_path) as conn:
            cursor = conn.cursor()
            
            # Remove old entries
            cursor.execute(
                "DELETE FROM memory_entries WHERE timestamp < ?",
                (cutoff_date.isoformat(),)
            )
            
            # Also limit total entries
            cursor.execute("""
                DELETE FROM memory_entries 
                WHERE entry_id NOT IN (
                    SELECT entry_id FROM memory_entries 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                )
            """, (self.max_memory_entries,))
            
            conn.commit()
            
            deleted_count = cursor.rowcount
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old memory entries")
    
    async def close(self) -> None:
        """Close the memory system and persist any pending data."""
        
        # Save all patterns and insights
        for pattern in self.error_patterns.values():
            await self._persist_error_pattern(pattern)
        
        # In a full implementation, you'd also persist insights
        
        logger.info("Reflexion memory system closed")


# Convenience functions for memory management
async def create_memory_system(
    db_path: str = "reflexion_memory.db",
    **kwargs
) -> ReflexionMemorySystem:
    """Create and initialize a new Reflexion memory system."""
    return ReflexionMemorySystem(memory_db_path=db_path, **kwargs)


def analyze_error_trends(
    memory_system: ReflexionMemorySystem,
    days: int = 30
) -> Dict[str, Any]:
    """Analyze error trends over a specified time period."""
    
    since = datetime.now() - timedelta(days=days)
    error_memories = memory_system.retrieve_memories(
        memory_type=MemoryType.ERROR,
        since=since
    )
    
    # Analyze trends
    error_counts_by_day = defaultdict(int)
    error_counts_by_category = defaultdict(int)
    
    for memory in error_memories:
        day = memory.timestamp.date()
        error_counts_by_day[day] += 1
        
        if memory.error_category:
            error_counts_by_category[memory.error_category] += 1
    
    return {
        "total_errors": len(error_memories),
        "errors_by_day": dict(error_counts_by_day),
        "errors_by_category": dict(error_counts_by_category),
        "most_common_error": max(error_counts_by_category.items(), key=lambda x: x[1]) if error_counts_by_category else None
    }