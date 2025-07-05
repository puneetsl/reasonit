"""
Guidance Injection System for Meta-Reasoning Integration.

This module integrates meta-reasoning guidance into the existing reasoning
pipeline by injecting strategic frameworks and guidance into agent prompts.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .meta_reasoning_kb import MetaReasoningKnowledgeBase, MetaReasoningGuidance
from .pattern_classifier import ProblemType, ComplexityLevel
from .strategy_templates import ReasoningFramework


class InjectionMode(Enum):
    """Modes for injecting guidance into reasoning prompts."""
    FULL = "full"  # Include complete framework guidance
    SUMMARY = "summary"  # Include brief guidance summary
    MINIMAL = "minimal"  # Include only key principles
    ADAPTIVE = "adaptive"  # Choose mode based on complexity


@dataclass
class InjectionConfig:
    """Configuration for guidance injection."""
    mode: InjectionMode = InjectionMode.ADAPTIVE
    include_examples: bool = True
    include_warnings: bool = True
    include_meta_strategies: bool = True
    max_guidance_length: int = 2000
    complexity_threshold: ComplexityLevel = ComplexityLevel.MEDIUM


class GuidanceInjector:
    """
    Injects meta-reasoning guidance into reasoning agent prompts.
    
    Integrates with the existing reasoning pipeline to enhance prompts
    with strategic reasoning frameworks and guidance.
    """
    
    def __init__(self, knowledge_base: Optional[MetaReasoningKnowledgeBase] = None):
        self.logger = logging.getLogger(__name__)
        self.kb = knowledge_base or MetaReasoningKnowledgeBase()
        self.injection_stats = {
            "total_injections": 0,
            "mode_usage": {},
            "framework_usage": {},
            "avg_guidance_length": 0
        }
    
    def enhance_reasoning_prompt(
        self,
        base_prompt: str,
        query: str,
        strategy: str,
        config: Optional[InjectionConfig] = None
    ) -> Tuple[str, MetaReasoningGuidance]:
        """
        Enhance a reasoning prompt with meta-reasoning guidance.
        
        Args:
            base_prompt: The original reasoning prompt
            query: The user's query/problem
            strategy: The reasoning strategy being used
            config: Injection configuration
        
        Returns:
            Tuple of (enhanced_prompt, guidance_used)
        """
        config = config or InjectionConfig()
        
        # Analyze the problem for meta-reasoning guidance
        guidance = self.kb.analyze_problem(query)
        
        # Determine injection mode
        effective_mode = self._determine_injection_mode(guidance, config)
        
        # Generate guidance content
        guidance_content = self._generate_guidance_content(guidance, effective_mode, config)
        
        # Inject guidance into prompt
        enhanced_prompt = self._inject_guidance(base_prompt, guidance_content, strategy)
        
        # Update statistics
        self._update_injection_stats(effective_mode, guidance)
        
        self.logger.info(f"Enhanced prompt with {effective_mode.value} guidance for {guidance.classification.primary_pattern.problem_type.value}")
        
        return enhanced_prompt, guidance
    
    def create_meta_reasoning_section(
        self,
        guidance: MetaReasoningGuidance,
        mode: InjectionMode = InjectionMode.FULL
    ) -> str:
        """Create a standalone meta-reasoning section."""
        sections = []
        
        sections.append("🧠 **META-REASONING GUIDANCE**")
        sections.append("=" * 50)
        
        # Problem classification
        pattern = guidance.classification.primary_pattern
        sections.append(f"**Problem Type**: {pattern.problem_type.value.title()}")
        sections.append(f"**Complexity**: {guidance.classification.overall_complexity.value.title()}")
        sections.append(f"**Confidence**: {guidance.confidence_score:.1%}")
        
        if guidance.classification.secondary_patterns:
            secondary = [p.problem_type.value for p in guidance.classification.secondary_patterns]
            sections.append(f"**Secondary Patterns**: {', '.join(secondary)}")
        
        sections.append("")
        
        # Framework recommendation
        if guidance.recommended_frameworks:
            primary_framework = guidance.recommended_frameworks[0]
            sections.append(f"**Recommended Framework**: {primary_framework.value.replace('_', ' ').title()}")
            
            if guidance.primary_template:
                sections.append(f"**When to Use**: {guidance.primary_template.when_to_use}")
            
            sections.append("")
        
        # Key principles (always include these)
        if guidance.primary_template and guidance.primary_template.key_principles:
            sections.append("**Key Principles**:")
            for principle in guidance.primary_template.key_principles:
                sections.append(f"• {principle}")
            sections.append("")
        
        # Meta-strategies
        if guidance.meta_strategies and mode != InjectionMode.MINIMAL:
            sections.append("**Meta-Strategies**:")
            for strategy in guidance.meta_strategies:
                sections.append(f"• {strategy}")
            sections.append("")
        
        # Include detailed steps for FULL mode
        if mode == InjectionMode.FULL and guidance.primary_template:
            sections.append("**Strategic Steps**:")
            for step in guidance.primary_template.steps:
                sections.append(f"{step.step_number}. **{step.instruction}**")
                sections.append(f"   → {step.purpose}")
                if step.warning:
                    sections.append(f"   ⚠️ {step.warning}")
                sections.append("")
        
        # Common pitfalls
        if guidance.primary_template and guidance.primary_template.common_pitfalls and mode != InjectionMode.MINIMAL:
            sections.append("**Common Pitfalls to Avoid**:")
            for pitfall in guidance.primary_template.common_pitfalls:
                sections.append(f"• {pitfall}")
            sections.append("")
        
        sections.append("🎯 **Apply this guidance while reasoning through the problem systematically.**")
        
        return "\n".join(sections)
    
    def inject_contextual_hints(
        self,
        prompt: str,
        guidance: MetaReasoningGuidance,
        step_context: Optional[str] = None
    ) -> str:
        """Inject contextual hints at specific reasoning steps."""
        if not step_context:
            return prompt
        
        hints = []
        
        # Pattern-specific hints
        pattern_type = guidance.classification.primary_pattern.problem_type
        
        if pattern_type == ProblemType.LOGICAL_PARADOX and "contradiction" in step_context.lower():
            hints.append("💡 Hint: This appears to involve a logical paradox. Consider questioning the logical framework itself rather than forcing a true/false answer.")
        
        elif pattern_type == ProblemType.COUNTERINTUITIVE and "seems" in step_context.lower():
            hints.append("💡 Hint: Your intuition might be misleading here. Carefully examine your assumptions.")
        
        elif pattern_type == ProblemType.AMBIGUOUS_QUERY and "unclear" in step_context.lower():
            hints.append("💡 Hint: This question has multiple interpretations. Consider stating your assumptions explicitly.")
        
        elif pattern_type == ProblemType.STATISTICAL_REASONING and "probability" in step_context.lower():
            hints.append("💡 Hint: Consider base rates and potential biases in this statistical reasoning.")
        
        # Complexity-based hints
        if guidance.classification.overall_complexity == ComplexityLevel.HIGH:
            hints.append("🔍 Hint: This is a complex problem. Consider breaking it down into smaller sub-problems.")
        
        # Add hints to prompt
        if hints:
            hint_section = "\n" + "\n".join(hints) + "\n"
            return prompt + hint_section
        
        return prompt
    
    def create_adaptive_guidance(
        self,
        query: str,
        current_confidence: float,
        reasoning_history: List[str]
    ) -> str:
        """Create adaptive guidance based on current reasoning state."""
        guidance = self.kb.analyze_problem(query)
        
        adaptive_hints = []
        
        # If confidence is low, provide more specific guidance
        if current_confidence < 0.5:
            adaptive_hints.append("🤔 **Low Confidence Detected**")
            adaptive_hints.append("Consider these approaches:")
            
            for framework in guidance.recommended_frameworks[:2]:
                template = self.kb.template_manager.get_template(framework)
                if template:
                    adaptive_hints.append(f"• {template.name}: {template.description}")
        
        # If reasoning seems stuck (repeated similar steps)
        if len(reasoning_history) > 2:
            recent_steps = reasoning_history[-3:]
            if len(set(recent_steps)) < 2:  # Similar steps repeated
                adaptive_hints.append("🔄 **Potential Reasoning Loop Detected**")
                adaptive_hints.append("Try a different approach:")
                
                # Suggest alternative frameworks
                if len(guidance.recommended_frameworks) > 1:
                    alt_framework = guidance.recommended_frameworks[1]
                    template = self.kb.template_manager.get_template(alt_framework)
                    if template:
                        adaptive_hints.append(f"• Alternative: {template.name}")
        
        # Problem-specific adaptive hints
        pattern_type = guidance.classification.primary_pattern.problem_type
        
        if pattern_type == ProblemType.MULTI_CONSTRAINT:
            adaptive_hints.append("⚖️ **Multi-Constraint Problem**")
            adaptive_hints.append("Remember to explicitly identify all constraints and trade-offs.")
        
        elif pattern_type == ProblemType.CAUSAL_CHAIN:
            adaptive_hints.append("🔗 **Causal Chain Analysis**")
            adaptive_hints.append("Trace both upstream causes and downstream effects.")
        
        return "\n".join(adaptive_hints) if adaptive_hints else ""
    
    def _determine_injection_mode(
        self,
        guidance: MetaReasoningGuidance,
        config: InjectionConfig
    ) -> InjectionMode:
        """Determine the appropriate injection mode."""
        if config.mode != InjectionMode.ADAPTIVE:
            return config.mode
        
        # Adaptive mode logic
        complexity = guidance.classification.overall_complexity
        confidence = guidance.confidence_score
        
        # Use FULL for high complexity or low confidence
        if complexity in [ComplexityLevel.HIGH, ComplexityLevel.EXTREME] or confidence < 0.5:
            return InjectionMode.FULL
        
        # Use SUMMARY for medium complexity
        elif complexity == ComplexityLevel.MEDIUM:
            return InjectionMode.SUMMARY
        
        # Use MINIMAL for simple problems
        else:
            return InjectionMode.MINIMAL
    
    def _generate_guidance_content(
        self,
        guidance: MetaReasoningGuidance,
        mode: InjectionMode,
        config: InjectionConfig
    ) -> str:
        """Generate guidance content based on mode and config."""
        if mode == InjectionMode.FULL:
            content = self.create_meta_reasoning_section(guidance, InjectionMode.FULL)
        
        elif mode == InjectionMode.SUMMARY:
            content = self.create_meta_reasoning_section(guidance, InjectionMode.SUMMARY)
        
        elif mode == InjectionMode.MINIMAL:
            # Just key principles and meta-strategies
            parts = []
            
            if guidance.primary_template and guidance.primary_template.key_principles:
                parts.append("🎯 **Key Principles**: " + "; ".join(guidance.primary_template.key_principles[:3]))
            
            if guidance.meta_strategies:
                parts.append("💡 **Strategy**: " + guidance.meta_strategies[0])
            
            content = "\n".join(parts)
        
        else:
            content = ""
        
        # Truncate if too long
        if len(content) > config.max_guidance_length:
            content = content[:config.max_guidance_length] + "\n... [guidance truncated]"
        
        return content
    
    def _inject_guidance(
        self,
        base_prompt: str,
        guidance_content: str,
        strategy: str
    ) -> str:
        """Inject guidance content into the base prompt."""
        if not guidance_content:
            return base_prompt
        
        # Find the best injection point
        injection_markers = [
            "Now solve the problem step by step",
            "Apply the reasoning strategy",
            "Begin reasoning:",
            "Start your analysis",
            "Solve this problem:"
        ]
        
        injection_point = -1
        for marker in injection_markers:
            point = base_prompt.lower().find(marker.lower())
            if point != -1:
                injection_point = point
                break
        
        if injection_point != -1:
            # Inject before the reasoning begins
            return (base_prompt[:injection_point] + 
                   guidance_content + "\n\n" + 
                   base_prompt[injection_point:])
        else:
            # Append to the end
            return base_prompt + "\n\n" + guidance_content
    
    def _update_injection_stats(
        self,
        mode: InjectionMode,
        guidance: MetaReasoningGuidance
    ):
        """Update injection statistics."""
        self.injection_stats["total_injections"] += 1
        
        # Mode usage
        mode_name = mode.value
        self.injection_stats["mode_usage"][mode_name] = \
            self.injection_stats["mode_usage"].get(mode_name, 0) + 1
        
        # Framework usage
        if guidance.recommended_frameworks:
            framework_name = guidance.recommended_frameworks[0].value
            self.injection_stats["framework_usage"][framework_name] = \
                self.injection_stats["framework_usage"].get(framework_name, 0) + 1
    
    def get_injection_stats(self) -> Dict[str, Any]:
        """Get injection usage statistics."""
        stats = self.injection_stats.copy()
        
        # Calculate averages
        if stats["total_injections"] > 0:
            # Most used mode
            if stats["mode_usage"]:
                most_used_mode = max(stats["mode_usage"], key=stats["mode_usage"].get)
                stats["most_used_mode"] = most_used_mode
            
            # Most used framework
            if stats["framework_usage"]:
                most_used_framework = max(stats["framework_usage"], key=stats["framework_usage"].get)
                stats["most_used_framework"] = most_used_framework
        
        return stats
    
    def create_debugging_report(
        self,
        query: str,
        enhanced_prompt: str,
        guidance: MetaReasoningGuidance
    ) -> str:
        """Create a debugging report for guidance injection."""
        report_parts = [
            "🔍 **GUIDANCE INJECTION REPORT**",
            "=" * 40,
            f"**Query**: {query[:100]}...",
            f"**Pattern Detected**: {guidance.classification.primary_pattern.problem_type.value}",
            f"**Complexity**: {guidance.classification.overall_complexity.value}",
            f"**Confidence**: {guidance.confidence_score:.2f}",
            f"**Framework Used**: {guidance.recommended_frameworks[0].value if guidance.recommended_frameworks else 'None'}",
            "",
            "**Processing Notes**:",
        ]
        
        for note in guidance.processing_notes:
            report_parts.append(f"• {note}")
        
        report_parts.extend([
            "",
            f"**Enhanced Prompt Length**: {len(enhanced_prompt)} characters",
            f"**Guidance Content Length**: {len(enhanced_prompt) - len(query)} characters (estimated)",
            "",
            "**Injection Statistics**:",
            f"• Total Injections: {self.injection_stats['total_injections']}",
            f"• Mode Usage: {self.injection_stats['mode_usage']}",
            f"• Framework Usage: {self.injection_stats['framework_usage']}"
        ])
        
        return "\n".join(report_parts)


# Convenience functions for easy integration

def enhance_prompt_with_guidance(
    base_prompt: str,
    query: str,
    strategy: str = "chain_of_thought",
    kb: Optional[MetaReasoningKnowledgeBase] = None
) -> Tuple[str, MetaReasoningGuidance]:
    """
    Convenience function to enhance a prompt with meta-reasoning guidance.
    
    Args:
        base_prompt: The original reasoning prompt
        query: The user's query/problem  
        strategy: The reasoning strategy being used
        kb: Optional knowledge base instance
    
    Returns:
        Tuple of (enhanced_prompt, guidance_used)
    """
    injector = GuidanceInjector(kb)
    return injector.enhance_reasoning_prompt(base_prompt, query, strategy)


def get_quick_guidance(
    query: str,
    kb: Optional[MetaReasoningKnowledgeBase] = None
) -> str:
    """
    Get quick guidance summary for a query.
    
    Args:
        query: The problem to analyze
        kb: Optional knowledge base instance
    
    Returns:
        Brief guidance summary
    """
    kb = kb or MetaReasoningKnowledgeBase()
    guidance = kb.analyze_problem(query)
    
    injector = GuidanceInjector(kb)
    return injector.create_meta_reasoning_section(guidance, InjectionMode.SUMMARY)