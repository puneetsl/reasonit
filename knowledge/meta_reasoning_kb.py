"""
Core Meta-Reasoning Knowledge Base.

This module provides the central knowledge base that combines pattern recognition
and strategic templates to guide reasoning for tricky problems.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import pickle

from .pattern_classifier import ProblemPatternClassifier, ClassificationResult, ProblemType
from .strategy_templates import StrategyTemplateManager, ReasoningFramework, StrategyTemplate


@dataclass
class MetaReasoningGuidance:
    """Complete meta-reasoning guidance for a specific problem."""
    query: str
    classification: ClassificationResult
    recommended_frameworks: List[ReasoningFramework]
    primary_template: Optional[StrategyTemplate]
    guided_prompt: str
    meta_strategies: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    processing_notes: List[str] = field(default_factory=list)


@dataclass
class KnowledgeStats:
    """Statistics about knowledge base usage and performance."""
    total_queries_processed: int = 0
    pattern_distribution: Dict[str, int] = field(default_factory=dict)
    framework_usage: Dict[str, int] = field(default_factory=dict)
    accuracy_feedback: List[float] = field(default_factory=list)
    avg_confidence: float = 0.0
    last_updated: Optional[str] = None


class MetaReasoningKnowledgeBase:
    """
    Central knowledge base for meta-reasoning guidance.
    
    Combines pattern recognition and strategic templates to provide
    comprehensive guidance for handling tricky reasoning problems.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.pattern_classifier = ProblemPatternClassifier()
        self.template_manager = StrategyTemplateManager()
        
        # Data storage
        self.data_dir = Path(data_dir) if data_dir else Path("knowledge_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Statistics and learning
        self.stats = KnowledgeStats()
        self._load_stats()
        
        # Caching for performance
        self._guidance_cache: Dict[str, MetaReasoningGuidance] = {}
        self._cache_max_size = 100
        
        # Learning from feedback
        self._feedback_history: List[Dict[str, Any]] = []
        
        self.logger.info("Meta-reasoning knowledge base initialized")
    
    def analyze_problem(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> MetaReasoningGuidance:
        """
        Analyze a problem and provide comprehensive meta-reasoning guidance.
        
        Args:
            query: The problem or question to analyze
            context: Optional context information
            use_cache: Whether to use cached results
        
        Returns:
            Complete meta-reasoning guidance
        """
        # Check cache first
        cache_key = self._create_cache_key(query, context)
        if use_cache and cache_key in self._guidance_cache:
            self.logger.debug(f"Using cached guidance for query: {query[:50]}...")
            return self._guidance_cache[cache_key]
        
        self.logger.info(f"Analyzing problem: {query[:100]}...")
        
        # Step 1: Classify the problem pattern
        classification = self.pattern_classifier.classify_problem(query)
        
        # Step 2: Get recommended frameworks
        indicators = [indicator for pattern in [classification.primary_pattern] + classification.secondary_patterns 
                     for indicator in pattern.indicators]
        recommended_frameworks = self.template_manager.recommend_frameworks_for_problem(indicators)
        
        # Step 3: Select primary template
        primary_framework = recommended_frameworks[0] if recommended_frameworks else ReasoningFramework.STANDARD
        primary_template = self.template_manager.get_template(primary_framework)
        
        # Step 4: Generate guided prompt
        guided_prompt = self.template_manager.generate_guided_prompt(
            primary_framework, 
            query, 
            include_examples=True
        )
        
        # Step 5: Compile meta-strategies
        meta_strategies = self._compile_meta_strategies(classification, context)
        
        # Step 6: Calculate overall confidence
        confidence_score = self._calculate_guidance_confidence(classification, recommended_frameworks)
        
        # Step 7: Add processing notes
        processing_notes = self._generate_processing_notes(classification, recommended_frameworks)
        
        # Create guidance object
        guidance = MetaReasoningGuidance(
            query=query,
            classification=classification,
            recommended_frameworks=recommended_frameworks,
            primary_template=primary_template,
            guided_prompt=guided_prompt,
            meta_strategies=meta_strategies,
            confidence_score=confidence_score,
            processing_notes=processing_notes
        )
        
        # Cache the result
        self._cache_guidance(cache_key, guidance)
        
        # Update statistics
        self._update_stats(classification, primary_framework)
        
        self.logger.info(f"Analysis complete. Primary pattern: {classification.primary_pattern.problem_type.value}, "
                        f"Framework: {primary_framework.value}, Confidence: {confidence_score:.2f}")
        
        return guidance
    
    def get_pattern_examples(self, problem_type: ProblemType) -> List[str]:
        """Get example queries for a specific problem type."""
        return self.pattern_classifier.get_pattern_examples(problem_type)
    
    def get_framework_summary(self, framework: ReasoningFramework) -> Optional[str]:
        """Get a summary of a specific reasoning framework."""
        return self.template_manager.get_template_summary(framework)
    
    def get_all_frameworks(self) -> Dict[ReasoningFramework, StrategyTemplate]:
        """Get all available reasoning frameworks."""
        return self.template_manager.get_all_templates()
    
    def provide_feedback(
        self, 
        query: str, 
        guidance: MetaReasoningGuidance, 
        success_rating: float,
        notes: Optional[str] = None
    ):
        """
        Provide feedback on guidance effectiveness for learning.
        
        Args:
            query: Original query
            guidance: The guidance that was provided
            success_rating: Rating 0-1 of how well the guidance worked
            notes: Optional notes about the experience
        """
        feedback = {
            "query": query,
            "primary_pattern": guidance.classification.primary_pattern.problem_type.value,
            "framework_used": guidance.recommended_frameworks[0].value if guidance.recommended_frameworks else "none",
            "success_rating": success_rating,
            "guidance_confidence": guidance.confidence_score,
            "notes": notes,
            "timestamp": str(Path.cwd())  # Placeholder for timestamp
        }
        
        self._feedback_history.append(feedback)
        self.stats.accuracy_feedback.append(success_rating)
        
        # Save feedback for learning
        self._save_feedback()
        
        self.logger.info(f"Feedback recorded: {success_rating:.2f} for {guidance.classification.primary_pattern.problem_type.value}")
    
    def get_statistics(self) -> KnowledgeStats:
        """Get knowledge base usage statistics."""
        # Update average confidence
        if self.stats.accuracy_feedback:
            self.stats.avg_confidence = sum(self.stats.accuracy_feedback) / len(self.stats.accuracy_feedback)
        
        return self.stats
    
    def export_knowledge(self, filepath: str) -> None:
        """Export knowledge base data for backup or analysis."""
        export_data = {
            "stats": asdict(self.stats),
            "feedback_history": self._feedback_history,
            "pattern_examples": {
                pattern_type.value: self.pattern_classifier.get_pattern_examples(pattern_type)
                for pattern_type in ProblemType
            },
            "framework_summaries": {
                framework.value: self.template_manager.get_template_summary(framework)
                for framework in ReasoningFramework
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Knowledge base exported to {filepath}")
    
    def _create_cache_key(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Create a cache key for a query and context."""
        context_str = json.dumps(context, sort_keys=True) if context else ""
        return f"{hash(query + context_str)}"
    
    def _cache_guidance(self, key: str, guidance: MetaReasoningGuidance):
        """Cache guidance result with size management."""
        if len(self._guidance_cache) >= self._cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._guidance_cache))
            del self._guidance_cache[oldest_key]
        
        self._guidance_cache[key] = guidance
    
    def _compile_meta_strategies(
        self, 
        classification: ClassificationResult, 
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Compile meta-strategies based on classification and context."""
        strategies = []
        
        # Add pattern-specific meta-guidance
        strategies.extend(classification.meta_guidance)
        
        # Add complexity-based strategies
        if classification.overall_complexity.value in ["high", "extreme"]:
            strategies.append("Break down into smaller sub-problems")
            strategies.append("Consider multiple solution approaches")
        
        # Add multi-pattern strategies
        if len(classification.secondary_patterns) > 0:
            strategies.append("Consider hybrid reasoning approach combining multiple frameworks")
        
        # Add context-specific strategies
        if context:
            if context.get("time_sensitive"):
                strategies.append("Prioritize speed over exhaustive analysis")
            if context.get("high_stakes"):
                strategies.append("Apply extra verification and double-checking")
            if context.get("collaborative"):
                strategies.append("Consider seeking additional perspectives")
        
        # Add confidence-based strategies
        if classification.confidence_score < 0.5:
            strategies.append("Pattern recognition confidence is low - consider multiple approaches")
        
        return strategies
    
    def _calculate_guidance_confidence(
        self, 
        classification: ClassificationResult, 
        frameworks: List[ReasoningFramework]
    ) -> float:
        """Calculate overall confidence in the guidance provided."""
        # Start with pattern classification confidence
        base_confidence = classification.confidence_score
        
        # Boost confidence if we have a strong framework recommendation
        if frameworks and frameworks[0] != ReasoningFramework.STANDARD:
            base_confidence += 0.1
        
        # Reduce confidence for extreme complexity
        if classification.overall_complexity.value == "extreme":
            base_confidence -= 0.2
        
        # Boost confidence for well-known patterns
        well_known_patterns = [
            ProblemType.LOGICAL_PARADOX,
            ProblemType.STATISTICAL_REASONING,
            ProblemType.MATHEMATICAL_PROOF
        ]
        if classification.primary_pattern.problem_type in well_known_patterns:
            base_confidence += 0.1
        
        # Ensure confidence stays in valid range
        return max(0.0, min(1.0, base_confidence))
    
    def _generate_processing_notes(
        self, 
        classification: ClassificationResult, 
        frameworks: List[ReasoningFramework]
    ) -> List[str]:
        """Generate processing notes for debugging and transparency."""
        notes = []
        
        notes.append(f"Primary pattern: {classification.primary_pattern.problem_type.value} "
                    f"(confidence: {classification.primary_pattern.confidence:.2f})")
        
        if classification.secondary_patterns:
            secondary_types = [p.problem_type.value for p in classification.secondary_patterns]
            notes.append(f"Secondary patterns: {', '.join(secondary_types)}")
        
        notes.append(f"Overall complexity: {classification.overall_complexity.value}")
        
        if frameworks:
            framework_names = [f.value for f in frameworks]
            notes.append(f"Recommended frameworks: {', '.join(framework_names)}")
        
        # Add specific pattern indicators
        if classification.primary_pattern.indicators:
            indicators_str = ', '.join(classification.primary_pattern.indicators[:3])
            notes.append(f"Key indicators: {indicators_str}")
        
        return notes
    
    def _update_stats(self, classification: ClassificationResult, framework: ReasoningFramework):
        """Update usage statistics."""
        self.stats.total_queries_processed += 1
        
        # Update pattern distribution
        pattern_type = classification.primary_pattern.problem_type.value
        self.stats.pattern_distribution[pattern_type] = \
            self.stats.pattern_distribution.get(pattern_type, 0) + 1
        
        # Update framework usage
        framework_name = framework.value
        self.stats.framework_usage[framework_name] = \
            self.stats.framework_usage.get(framework_name, 0) + 1
        
        # Save updated stats
        self._save_stats()
    
    def _load_stats(self):
        """Load statistics from disk."""
        stats_file = self.data_dir / "kb_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file, 'r') as f:
                    stats_data = json.load(f)
                    self.stats = KnowledgeStats(**stats_data)
                self.logger.debug("Statistics loaded from disk")
            except Exception as e:
                self.logger.warning(f"Failed to load statistics: {e}")
    
    def _save_stats(self):
        """Save statistics to disk."""
        stats_file = self.data_dir / "kb_stats.json"
        try:
            with open(stats_file, 'w') as f:
                json.dump(asdict(self.stats), f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save statistics: {e}")
    
    def _save_feedback(self):
        """Save feedback history to disk."""
        feedback_file = self.data_dir / "feedback_history.json"
        try:
            with open(feedback_file, 'w') as f:
                json.dump(self._feedback_history, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save feedback: {e}")
    
    def _load_feedback(self):
        """Load feedback history from disk."""
        feedback_file = self.data_dir / "feedback_history.json"
        if feedback_file.exists():
            try:
                with open(feedback_file, 'r') as f:
                    self._feedback_history = json.load(f)
                self.logger.debug("Feedback history loaded from disk")
            except Exception as e:
                self.logger.warning(f"Failed to load feedback history: {e}")
    
    def analyze_patterns_performance(self) -> Dict[str, Any]:
        """Analyze which patterns and frameworks perform best."""
        if not self._feedback_history:
            return {"message": "No feedback data available for analysis"}
        
        # Group feedback by pattern and framework
        pattern_performance = {}
        framework_performance = {}
        
        for feedback in self._feedback_history:
            pattern = feedback["primary_pattern"]
            framework = feedback["framework_used"]
            rating = feedback["success_rating"]
            
            # Pattern performance
            if pattern not in pattern_performance:
                pattern_performance[pattern] = []
            pattern_performance[pattern].append(rating)
            
            # Framework performance
            if framework not in framework_performance:
                framework_performance[framework] = []
            framework_performance[framework].append(rating)
        
        # Calculate averages
        pattern_averages = {
            pattern: sum(ratings) / len(ratings)
            for pattern, ratings in pattern_performance.items()
        }
        
        framework_averages = {
            framework: sum(ratings) / len(ratings)
            for framework, ratings in framework_performance.items()
        }
        
        return {
            "total_feedback_entries": len(self._feedback_history),
            "pattern_performance": pattern_averages,
            "framework_performance": framework_averages,
            "best_pattern": max(pattern_averages, key=pattern_averages.get) if pattern_averages else None,
            "best_framework": max(framework_averages, key=framework_averages.get) if framework_averages else None,
            "overall_average": sum(f["success_rating"] for f in self._feedback_history) / len(self._feedback_history)
        }
    
    def suggest_improvements(self) -> List[str]:
        """Suggest improvements based on performance analysis."""
        suggestions = []
        
        performance = self.analyze_patterns_performance()
        
        if performance.get("overall_average", 0) < 0.7:
            suggestions.append("Overall performance is below 70% - consider refining pattern recognition")
        
        # Identify underperforming patterns
        pattern_perf = performance.get("pattern_performance", {})
        for pattern, avg_rating in pattern_perf.items():
            if avg_rating < 0.6:
                suggestions.append(f"Pattern '{pattern}' is underperforming (avg: {avg_rating:.2f}) - review classification rules")
        
        # Identify underperforming frameworks
        framework_perf = performance.get("framework_performance", {})
        for framework, avg_rating in framework_perf.items():
            if avg_rating < 0.6:
                suggestions.append(f"Framework '{framework}' is underperforming (avg: {avg_rating:.2f}) - review template steps")
        
        if not suggestions:
            suggestions.append("Performance looks good! Continue monitoring and collecting feedback.")
        
        return suggestions
    
    def reset_cache(self):
        """Clear the guidance cache."""
        self._guidance_cache.clear()
        self.logger.info("Guidance cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache usage statistics."""
        return {
            "cache_size": len(self._guidance_cache),
            "max_cache_size": self._cache_max_size,
            "cache_utilization": len(self._guidance_cache) / self._cache_max_size
        }