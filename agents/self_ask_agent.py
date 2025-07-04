"""
Self-Ask reasoning agent implementation.

This module implements the Self-Ask reasoning strategy that decomposes complex
questions into simpler follow-up questions, answering them iteratively to
build towards the final solution.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from pydantic_ai import RunContext

from models import (
    ReasoningRequest,
    ReasoningResult,
    ReasoningStrategy,
    ContextVariant,
    OutcomeType,
    ToolResult,
    LLMGenerationError,
    ConfidenceThresholdError,
)
from context import ContextGenerator, build_cot_prompt
from agents.base_agent import BaseReasoningAgent, AgentDependencies
from tools import get_tool, execute_tool

logger = logging.getLogger(__name__)


class QuestionType(Enum):
    """Types of questions in Self-Ask decomposition."""
    MAIN_QUESTION = "main_question"
    FOLLOW_UP = "follow_up"
    INTERMEDIATE = "intermediate"
    VERIFICATION = "verification"
    CLARIFICATION = "clarification"


@dataclass
class SelfAskQuestion:
    """Represents a question in the Self-Ask reasoning process."""
    
    # Question identification
    id: str = ""
    question_type: QuestionType = QuestionType.FOLLOW_UP
    depth_level: int = 0
    
    # Question content
    question_text: str = ""
    context: str = ""
    dependencies: List[str] = field(default_factory=list)  # IDs of prerequisite questions
    
    # Answer information
    answer: str = ""
    confidence: float = 0.0
    is_answered: bool = False
    answer_source: str = "reasoning"  # "reasoning", "tool", "search", "given"
    
    # Processing metadata
    attempts: int = 0
    cost: float = 0.0
    tools_used: List[ToolResult] = field(default_factory=list)
    reasoning_trace: str = ""
    
    # Relationships
    parent_question_id: Optional[str] = None
    child_question_ids: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.id:
            self.id = f"q_{id(self)}"


@dataclass
class SelfAskState:
    """Current state of the Self-Ask reasoning process."""
    
    # Question management
    questions: Dict[str, SelfAskQuestion] = field(default_factory=dict)
    main_question_id: Optional[str] = None
    current_question_id: Optional[str] = None
    
    # Processing queue
    unanswered_questions: List[str] = field(default_factory=list)
    ready_to_answer: List[str] = field(default_factory=list)
    
    # Statistics
    total_questions: int = 0
    answered_questions: int = 0
    max_depth: int = 0
    total_cost: float = 0.0
    
    # Configuration
    max_depth_limit: int = 8
    max_questions_limit: int = 20
    confidence_threshold: float = 0.7
    decomposition_threshold: int = 3  # Max sub-questions per question
    
    def add_question(self, question: SelfAskQuestion) -> None:
        """Add a question to the state."""
        self.questions[question.id] = question
        self.total_questions += 1
        self.total_cost += question.cost
        
        if question.depth_level > self.max_depth:
            self.max_depth = question.depth_level
        
        if not question.is_answered:
            self.unanswered_questions.append(question.id)
            self._update_ready_queue()
    
    def answer_question(self, question_id: str, answer: str, confidence: float = 0.8) -> None:
        """Mark a question as answered."""
        if question_id in self.questions:
            question = self.questions[question_id]
            question.answer = answer
            question.confidence = confidence
            question.is_answered = True
            
            if question_id in self.unanswered_questions:
                self.unanswered_questions.remove(question_id)
            
            self.answered_questions += 1
            self._update_ready_queue()
    
    def _update_ready_queue(self) -> None:
        """Update the queue of questions that are ready to be answered."""
        self.ready_to_answer = []
        
        for question_id in self.unanswered_questions:
            question = self.questions[question_id]
            
            # Check if all dependencies are satisfied
            dependencies_satisfied = all(
                dep_id in self.questions and self.questions[dep_id].is_answered
                for dep_id in question.dependencies
            )
            
            if dependencies_satisfied:
                self.ready_to_answer.append(question_id)
    
    def get_next_question(self) -> Optional[SelfAskQuestion]:
        """Get the next question to process."""
        if not self.ready_to_answer:
            return None
        
        # Prioritize by depth (shallower first) and question type
        def priority_key(q_id):
            q = self.questions[q_id]
            type_priority = {
                QuestionType.MAIN_QUESTION: 0,
                QuestionType.FOLLOW_UP: 1,
                QuestionType.INTERMEDIATE: 2,
                QuestionType.VERIFICATION: 3,
                QuestionType.CLARIFICATION: 4
            }
            return (q.depth_level, type_priority.get(q.question_type, 5))
        
        next_id = min(self.ready_to_answer, key=priority_key)
        return self.questions[next_id]
    
    def is_complete(self) -> bool:
        """Check if the reasoning process is complete."""
        return (self.main_question_id and 
                self.main_question_id in self.questions and
                self.questions[self.main_question_id].is_answered)
    
    def get_dependency_chain(self, question_id: str) -> List[SelfAskQuestion]:
        """Get the chain of questions leading to this question."""
        chain = []
        current_id = question_id
        
        while current_id and current_id in self.questions:
            question = self.questions[current_id]
            chain.append(question)
            current_id = question.parent_question_id
        
        return list(reversed(chain))


class SelfAskAgent(BaseReasoningAgent):
    """
    Self-Ask reasoning agent with question decomposition.
    
    This agent implements the Self-Ask strategy by:
    1. Decomposing complex questions into simpler follow-up questions
    2. Answering sub-questions iteratively using available information
    3. Using tools and search when needed for factual information
    4. Building up to the final answer by combining sub-answers
    5. Verifying the solution through additional questions if needed
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        max_depth: int = 6,
        max_questions: int = 15,
        decomposition_threshold: int = 3,
        use_search: bool = True,
        **kwargs
    ):
        super().__init__(
            strategy=ReasoningStrategy.SELF_ASK,
            model_name=model_name,
            **kwargs
        )
        
        self.context_generator = ContextGenerator()
        self.max_depth = max_depth
        self.max_questions = max_questions
        self.decomposition_threshold = decomposition_threshold
        self.use_search = use_search
        
        # Self-Ask specific parameters
        self.verification_threshold = 0.8  # When to add verification questions
        self.search_confidence_boost = 0.2  # Confidence boost for search results
        self.tool_confidence_boost = 0.15  # Confidence boost for tool results
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for Self-Ask reasoning."""
        return """You are an expert reasoning assistant that excels at breaking down complex questions into simpler, answerable sub-questions.

Your approach:
1. **Decompose systematically** - Break complex questions into logical sub-questions
2. **Answer incrementally** - Solve simpler questions first, then build up
3. **Use available information** - Leverage previous answers and available tools
4. **Search when needed** - Use search tools for factual information you don't know
5. **Verify your reasoning** - Check your logic and answers for consistency
6. **Synthesize carefully** - Combine sub-answers into a coherent final answer

Guidelines for Self-Ask:
- Ask "What do I need to know to answer this?" 
- Break questions into factual, logical, and computational sub-parts
- Use "Follow up: [question]" format for sub-questions
- Answer each sub-question before moving to the next
- Use tools and search for information you cannot deduce
- Check your work with verification questions
- Be explicit about your reasoning chain"""
    
    async def _execute_reasoning(
        self,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> ReasoningResult:
        """Execute Self-Ask reasoning with question decomposition."""
        
        logger.info(f"Starting Self-Ask reasoning for: {request.query[:100]}...")
        
        # Initialize Self-Ask state
        self_ask_state = SelfAskState(
            max_depth_limit=self.max_depth,
            max_questions_limit=self.max_questions,
            confidence_threshold=request.confidence_threshold,
            decomposition_threshold=self.decomposition_threshold
        )
        
        # Generate enhanced prompt
        enhanced_prompt = await self._generate_enhanced_prompt(request, context)
        
        # Create main question
        main_question = SelfAskQuestion(
            id="main_q",
            question_type=QuestionType.MAIN_QUESTION,
            depth_level=0,
            question_text=request.query,
            context=enhanced_prompt
        )
        
        self_ask_state.add_question(main_question)
        self_ask_state.main_question_id = main_question.id
        
        # Execute Self-Ask process
        final_answer = await self._execute_self_ask_process(
            self_ask_state, enhanced_prompt, request, context
        )
        
        # Create final result
        if final_answer:
            final_result = await self._create_final_result(
                final_answer, self_ask_state, request, context
            )
        else:
            final_result = ReasoningResult(
                request=request,
                final_answer="Unable to determine answer through Self-Ask decomposition",
                reasoning_trace=self.reasoning_trace.copy(),
                total_cost=self_ask_state.total_cost,
                total_time=0.0,
                confidence_score=0.0,
                strategies_used=[self.strategy],
                outcome=OutcomeType.NO_SOLUTION,
                error_message="Self-Ask process failed to find satisfactory answer",
                timestamp=datetime.now(),
                metadata={
                    "questions_generated": self_ask_state.total_questions,
                    "questions_answered": self_ask_state.answered_questions,
                    "max_depth_reached": self_ask_state.max_depth
                }
            )
        
        logger.info(
            f"Self-Ask reasoning completed: {final_result.outcome}, "
            f"questions={self_ask_state.total_questions}, answered={self_ask_state.answered_questions}"
        )
        
        return final_result
    
    async def _generate_enhanced_prompt(
        self,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> str:
        """Generate an enhanced prompt for Self-Ask reasoning."""
        
        try:
            # Use standard or enriched context for Self-Ask
            context_variant = request.context_variant
            if context_variant == ContextVariant.STANDARD:
                context_variant = ContextVariant.ENRICHED
            
            enhanced = await self.context_generator.generate_context(
                request.query,
                context_variant,
                ReasoningStrategy.SELF_ASK
            )
            
            # Add Self-Ask specific framing
            self_ask_prompt = self._build_self_ask_prompt(enhanced)
            
            self.add_reasoning_step(
                content=f"Generated Self-Ask prompt using {context_variant} context",
                confidence=0.9,
                cost=0.0,
                metadata={"context_variant": context_variant, "prompt_length": len(self_ask_prompt)}
            )
            
            return self_ask_prompt
            
        except Exception as e:
            logger.warning(f"Context generation failed, using basic prompt: {e}")
            return self._build_self_ask_prompt(request.query)
    
    def _build_self_ask_prompt(self, base_prompt: str) -> str:
        """Build Self-Ask specific prompt."""
        return f"""SELF-ASK REASONING:

{base_prompt}

Use the Self-Ask approach to solve this step by step:
1. Break the main question into simpler sub-questions
2. Answer each sub-question using available information or tools
3. Build up to the final answer using the sub-answers
4. Verify your reasoning with follow-up questions if needed

Format your reasoning as:
Question: [sub-question]
Answer: [answer to sub-question]
Follow up: [next question if needed]
...
So the final answer is: [complete answer]"""
    
    async def _execute_self_ask_process(
        self,
        self_ask_state: SelfAskState,
        prompt: str,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> Optional[str]:
        """Execute the main Self-Ask reasoning process."""
        
        iteration = 0
        max_iterations = self_ask_state.max_questions_limit * 2  # Safety limit
        
        while (not self_ask_state.is_complete() and 
               iteration < max_iterations and
               self_ask_state.total_questions < self_ask_state.max_questions_limit):
            
            iteration += 1
            
            # Get next question to process
            current_question = self_ask_state.get_next_question()
            if not current_question:
                # No questions ready - try decomposing unanswered questions
                await self._decompose_complex_questions(self_ask_state, request, context)
                current_question = self_ask_state.get_next_question()
                
                if not current_question:
                    logger.warning("No questions ready to process and no decomposition possible")
                    break
            
            self_ask_state.current_question_id = current_question.id
            
            # Process the current question
            try:
                await self._process_question(
                    current_question, self_ask_state, request, context
                )
                
                # Log progress
                self.add_reasoning_step(
                    content=f"Processed question: {current_question.question_text[:100]}...",
                    confidence=current_question.confidence,
                    cost=current_question.cost,
                    intermediate_result=current_question.answer,
                    metadata={
                        "question_id": current_question.id,
                        "question_type": current_question.question_type.value,
                        "depth": current_question.depth_level,
                        "answered": current_question.is_answered
                    }
                )
                
            except Exception as e:
                logger.error(f"Failed to process question {current_question.id}: {e}")
                # Mark question as answered with low confidence to avoid infinite loops
                self_ask_state.answer_question(
                    current_question.id,
                    f"Unable to answer due to error: {e}",
                    confidence=0.1
                )
        
        # Return the main question's answer if complete
        if self_ask_state.is_complete():
            main_question = self_ask_state.questions[self_ask_state.main_question_id]
            return main_question.answer
        
        return None
    
    async def _process_question(
        self,
        question: SelfAskQuestion,
        self_ask_state: SelfAskState,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> None:
        """Process a single question through decomposition or direct answering."""
        
        question.attempts += 1
        
        # Build context from dependency answers
        question_context = await self._build_question_context(question, self_ask_state)
        
        # Try to answer the question directly first
        direct_answer = await self._attempt_direct_answer(
            question, question_context, request, context
        )
        
        if direct_answer and direct_answer["confidence"] >= self_ask_state.confidence_threshold:
            # Direct answer was successful
            self_ask_state.answer_question(
                question.id,
                direct_answer["answer"],
                direct_answer["confidence"]
            )
            question.answer_source = direct_answer["source"]
            question.cost += direct_answer["cost"]
            question.reasoning_trace = direct_answer["reasoning"]
            
        elif (question.depth_level < self_ask_state.max_depth_limit and 
              self_ask_state.total_questions < self_ask_state.max_questions_limit):
            # Decompose into sub-questions
            sub_questions = await self._decompose_question(
                question, question_context, request, context
            )
            
            if sub_questions:
                # Add sub-questions as dependencies
                for sub_q in sub_questions:
                    sub_q.parent_question_id = question.id
                    sub_q.depth_level = question.depth_level + 1
                    self_ask_state.add_question(sub_q)
                    question.child_question_ids.append(sub_q.id)
                
                # Update dependencies
                question.dependencies.extend([sq.id for sq in sub_questions])
                
            else:
                # Decomposition failed, use best available answer
                if direct_answer:
                    self_ask_state.answer_question(
                        question.id,
                        direct_answer["answer"],
                        direct_answer["confidence"]
                    )
                else:
                    self_ask_state.answer_question(
                        question.id,
                        "Unable to determine answer",
                        confidence=0.1
                    )
        
        else:
            # At depth/question limit, use best available answer
            if direct_answer:
                self_ask_state.answer_question(
                    question.id,
                    direct_answer["answer"],
                    direct_answer["confidence"]
                )
            else:
                self_ask_state.answer_question(
                    question.id,
                    "Reached complexity limit",
                    confidence=0.2
                )
    
    async def _build_question_context(
        self,
        question: SelfAskQuestion,
        self_ask_state: SelfAskState
    ) -> str:
        """Build context for a question from answered dependencies."""
        
        context_parts = [question.context]
        
        # Add answers from dependency questions
        if question.dependencies:
            context_parts.append("\nPrevious answers:")
            for dep_id in question.dependencies:
                if dep_id in self_ask_state.questions:
                    dep_question = self_ask_state.questions[dep_id]
                    if dep_question.is_answered:
                        context_parts.append(f"Q: {dep_question.question_text}")
                        context_parts.append(f"A: {dep_question.answer}")
        
        # Add parent context if available
        if question.parent_question_id and question.parent_question_id in self_ask_state.questions:
            parent = self_ask_state.questions[question.parent_question_id]
            context_parts.append(f"\nMain question context: {parent.question_text}")
        
        return "\n".join(context_parts)
    
    async def _attempt_direct_answer(
        self,
        question: SelfAskQuestion,
        question_context: str,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> Optional[Dict[str, Any]]:
        """Attempt to answer a question directly."""
        
        # Build prompt for direct answering
        direct_prompt = f"""Answer the following question using the available context and your knowledge.

Context:
{question_context}

Question: {question.question_text}

If you can answer confidently, provide:
ANSWER: [your answer]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation of your reasoning]

If you cannot answer confidently, respond with:
CANNOT_ANSWER: [explanation of why]"""
        
        try:
            response = await self.generate_with_context(direct_prompt, context)
            parsed = self._parse_direct_answer(response)
            parsed["cost"] = 0.01  # Estimated cost
            
            # Try tools if direct answer failed and tools are available
            if (not parsed["answer"] or parsed["confidence"] < 0.6) and request.use_tools:
                tool_result = await self._try_tools_for_question(question, context)
                if tool_result and tool_result["confidence"] > parsed["confidence"]:
                    parsed = tool_result
            
            return parsed
            
        except Exception as e:
            logger.warning(f"Direct answer attempt failed: {e}")
            return None
    
    def _parse_direct_answer(self, response: str) -> Dict[str, Any]:
        """Parse a direct answer response."""
        
        import re
        
        # Look for structured answer format
        answer_match = re.search(r"ANSWER[:=]\s*(.+?)(?=CONFIDENCE|REASONING|$)", response, re.IGNORECASE | re.DOTALL)
        confidence_match = re.search(r"CONFIDENCE[:=]\s*([0-9.]+)", response, re.IGNORECASE)
        reasoning_match = re.search(r"REASONING[:=]\s*(.+?)(?=ANSWER|CONFIDENCE|$)", response, re.IGNORECASE | re.DOTALL)
        cannot_answer_match = re.search(r"CANNOT_ANSWER[:=]\s*(.+)", response, re.IGNORECASE)
        
        if cannot_answer_match:
            return {
                "answer": "",
                "confidence": 0.1,
                "reasoning": cannot_answer_match.group(1).strip(),
                "source": "reasoning"
            }
        
        answer = answer_match.group(1).strip() if answer_match else ""
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "Direct reasoning"
        
        # If no structured format, treat entire response as answer
        if not answer and not cannot_answer_match:
            answer = response.strip()
            confidence = 0.6  # Moderate confidence for unstructured answers
        
        return {
            "answer": answer,
            "confidence": max(0.0, min(1.0, confidence)),
            "reasoning": reasoning,
            "source": "reasoning"
        }
    
    async def _try_tools_for_question(
        self,
        question: SelfAskQuestion,
        context: RunContext[AgentDependencies]
    ) -> Optional[Dict[str, Any]]:
        """Try to answer a question using available tools."""
        
        question_lower = question.question_text.lower()
        
        # Try calculation tools for math questions
        if any(indicator in question_lower for indicator in ['calculate', 'compute', 'what is', '+', '-', '*', '/', '=']):
            calc_tool = get_tool("calculate")
            if calc_tool:
                try:
                    # Extract mathematical expression
                    math_expr = self._extract_math_expression(question.question_text)
                    if math_expr:
                        result = await calc_tool.execute(expression=math_expr)
                        if result.success:
                            return {
                                "answer": str(result.output_data),
                                "confidence": 0.9,
                                "reasoning": f"Calculated using tool: {math_expr}",
                                "source": "tool"
                            }
                except Exception as e:
                    logger.warning(f"Calculation tool failed: {e}")
        
        # Try Python executor for complex computations
        if any(indicator in question_lower for indicator in ['code', 'program', 'algorithm', 'compute']):
            python_tool = get_tool("execute_python")
            if python_tool:
                try:
                    code = self._generate_python_code_for_question(question.question_text)
                    if code:
                        result = await python_tool.execute(code=code)
                        if result.success:
                            output = result.output_data.get('stdout', '').strip()
                            if output:
                                return {
                                    "answer": output,
                                    "confidence": 0.8,
                                    "reasoning": f"Computed using Python: {code}",
                                    "source": "tool"
                                }
                except Exception as e:
                    logger.warning(f"Python tool failed: {e}")
        
        return None
    
    def _extract_math_expression(self, question: str) -> Optional[str]:
        """Extract mathematical expression from question text."""
        
        import re
        
        # Look for expressions like "what is 2 + 3", "calculate 5 * 7", etc.
        patterns = [
            r"(?:what is|calculate|compute)\s+([+\-*/()0-9.\s]+)",
            r"([0-9.]+\s*[+\-*/]\s*[0-9.]+(?:\s*[+\-*/]\s*[0-9.]+)*)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                expr = match.group(1).strip()
                # Simple validation
                if re.match(r"^[0-9+\-*/.() \s]+$", expr):
                    return expr
        
        return None
    
    def _generate_python_code_for_question(self, question: str) -> Optional[str]:
        """Generate Python code to answer computational questions."""
        
        question_lower = question.lower()
        
        # Simple patterns for common computational questions
        if "factorial" in question_lower:
            # Extract number for factorial
            import re
            num_match = re.search(r"factorial of (\d+)", question_lower)
            if num_match:
                num = num_match.group(1)
                return f"""
import math
result = math.factorial({num})
print(f"Factorial of {num} is: {{result}}")
"""
        
        if "fibonacci" in question_lower:
            # Extract position for Fibonacci
            import re
            num_match = re.search(r"(\d+)(?:th|st|nd|rd)?\s+fibonacci", question_lower)
            if num_match:
                num = num_match.group(1)
                return f"""
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

result = fibonacci({num})
print(f"The {num}th Fibonacci number is: {{result}}")
"""
        
        return None
    
    async def _decompose_question(
        self,
        question: SelfAskQuestion,
        question_context: str,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> List[SelfAskQuestion]:
        """Decompose a question into simpler sub-questions."""
        
        decomposition_prompt = f"""Break down the following question into {self.decomposition_threshold} or fewer simpler sub-questions that need to be answered first.

Context:
{question_context}

Main question: {question.question_text}

Generate sub-questions that:
1. Are simpler than the main question
2. Can be answered with available information or tools
3. Build towards answering the main question
4. Are logically ordered

Format your response as:
SUB-QUESTION 1: [question]
SUB-QUESTION 2: [question]
...

If the question cannot be meaningfully decomposed, respond with:
NO_DECOMPOSITION: [explanation]"""
        
        try:
            response = await self.generate_with_context(decomposition_prompt, context)
            sub_questions = self._parse_decomposition_response(response)
            
            question.cost += 0.015  # Estimated cost for decomposition
            
            return sub_questions
            
        except Exception as e:
            logger.warning(f"Question decomposition failed: {e}")
            return []
    
    def _parse_decomposition_response(self, response: str) -> List[SelfAskQuestion]:
        """Parse decomposition response into sub-questions."""
        
        import re
        
        # Check for no decomposition
        if re.search(r"NO_DECOMPOSITION", response, re.IGNORECASE):
            return []
        
        sub_questions = []
        
        # Extract sub-questions
        patterns = [
            r"SUB-QUESTION\s*\d+[:]\s*(.+?)(?=SUB-QUESTION\s*\d+|$)",
            r"(\d+)\.?\s*(.+?)(?=\d+\.|$)",
            r"Follow up[:]\s*(.+?)(?=Follow up|$)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if isinstance(match, tuple):
                    question_text = match[-1].strip()
                else:
                    question_text = match.strip()
                
                if question_text and len(question_text) > 5:  # Filter out too short questions
                    sub_q = SelfAskQuestion(
                        question_type=QuestionType.FOLLOW_UP,
                        question_text=question_text[:200]  # Limit length
                    )
                    sub_questions.append(sub_q)
            
            if sub_questions:
                break
        
        return sub_questions[:self.decomposition_threshold]  # Limit number
    
    async def _decompose_complex_questions(
        self,
        self_ask_state: SelfAskState,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> None:
        """Try to decompose remaining complex questions."""
        
        # Find unanswered questions that haven't been decomposed yet
        candidates = []
        for q_id in self_ask_state.unanswered_questions:
            question = self_ask_state.questions[q_id]
            if (not question.child_question_ids and 
                question.depth_level < self_ask_state.max_depth_limit and
                question.attempts < 2):  # Limit decomposition attempts
                candidates.append(question)
        
        # Try decomposing the most promising candidate
        if candidates:
            # Sort by depth (prefer shallower) and question type priority
            candidate = min(candidates, key=lambda q: (q.depth_level, q.attempts))
            question_context = await self._build_question_context(candidate, self_ask_state)
            
            sub_questions = await self._decompose_question(
                candidate, question_context, request, context
            )
            
            if sub_questions:
                for sub_q in sub_questions:
                    sub_q.parent_question_id = candidate.id
                    sub_q.depth_level = candidate.depth_level + 1
                    self_ask_state.add_question(sub_q)
                    candidate.child_question_ids.append(sub_q.id)
                
                candidate.dependencies.extend([sq.id for sq in sub_questions])
    
    async def _create_final_result(
        self,
        final_answer: str,
        self_ask_state: SelfAskState,
        request: ReasoningRequest,
        context: RunContext[AgentDependencies]
    ) -> ReasoningResult:
        """Create the final reasoning result."""
        
        # Build reasoning trace from question chain
        reasoning_summary = self._build_reasoning_summary(self_ask_state)
        
        # Calculate overall confidence
        main_question = self_ask_state.questions[self_ask_state.main_question_id]
        overall_confidence = main_question.confidence
        
        # Weight by sub-question confidences if available
        if main_question.child_question_ids:
            child_confidences = []
            for child_id in main_question.child_question_ids:
                if child_id in self_ask_state.questions:
                    child_q = self_ask_state.questions[child_id]
                    if child_q.is_answered:
                        child_confidences.append(child_q.confidence)
            
            if child_confidences:
                avg_child_confidence = sum(child_confidences) / len(child_confidences)
                overall_confidence = (overall_confidence + avg_child_confidence) / 2
        
        # Add final reasoning step
        self.add_reasoning_step(
            content=f"Completed Self-Ask with {self_ask_state.answered_questions}/{self_ask_state.total_questions} questions answered",
            confidence=overall_confidence,
            cost=0.0,
            intermediate_result=final_answer,
            metadata={
                "total_questions": self_ask_state.total_questions,
                "answered_questions": self_ask_state.answered_questions,
                "max_depth": self_ask_state.max_depth
            }
        )
        
        return ReasoningResult(
            request=request,
            final_answer=final_answer,
            reasoning_trace=self.reasoning_trace.copy(),
            total_cost=self_ask_state.total_cost,
            total_time=0.0,  # Will be set by base class
            confidence_score=overall_confidence,
            strategies_used=[self.strategy],
            outcome=OutcomeType.SUCCESS,
            reflection=self._generate_self_ask_reflection(self_ask_state),
            timestamp=datetime.now(),
            metadata={
                "questions_generated": self_ask_state.total_questions,
                "questions_answered": self_ask_state.answered_questions,
                "max_depth_reached": self_ask_state.max_depth,
                "decomposition_used": any(q.child_question_ids for q in self_ask_state.questions.values()),
                "reasoning_summary": reasoning_summary
            }
        )
    
    def _build_reasoning_summary(self, self_ask_state: SelfAskState) -> str:
        """Build a summary of the reasoning process."""
        
        summary_parts = []
        
        # Get the main question chain
        if self_ask_state.main_question_id:
            main_q = self_ask_state.questions[self_ask_state.main_question_id]
            chain = self_ask_state.get_dependency_chain(main_q.id)
            
            for i, question in enumerate(chain):
                summary_parts.append(f"Q{i+1}: {question.question_text}")
                if question.is_answered:
                    summary_parts.append(f"A{i+1}: {question.answer}")
                    summary_parts.append(f"Confidence: {question.confidence:.3f}")
                summary_parts.append("")
        
        return "\n".join(summary_parts)
    
    def _generate_self_ask_reflection(self, self_ask_state: SelfAskState) -> str:
        """Generate reflection on the Self-Ask reasoning process."""
        
        reflection_parts = [
            f"Generated {self_ask_state.total_questions} questions and answered {self_ask_state.answered_questions} of them.",
            f"Reached maximum depth of {self_ask_state.max_depth} levels."
        ]
        
        # Add insights about the decomposition process
        decomposed_questions = sum(1 for q in self_ask_state.questions.values() if q.child_question_ids)
        if decomposed_questions > 0:
            reflection_parts.append(f"Successfully decomposed {decomposed_questions} complex questions into simpler sub-questions.")
        
        # Add confidence insights
        if self_ask_state.main_question_id:
            main_q = self_ask_state.questions[self_ask_state.main_question_id]
            if main_q.confidence > 0.8:
                reflection_parts.append("High confidence in the final answer through systematic decomposition.")
            elif main_q.confidence < 0.6:
                reflection_parts.append("Lower confidence suggests some sub-questions may need further investigation.")
        
        return " ".join(reflection_parts)
    
    def _get_capabilities(self) -> List[str]:
        """Get list of capabilities for this agent."""
        return [
            "question_decomposition",
            "iterative_answering",
            "dependency_resolution",
            "follow_up_questions",
            "tool_integration",
            "search_integration",
            "incremental_reasoning",
            "verification_questions",
            "context_building",
            "confidence_assessment"
        ]