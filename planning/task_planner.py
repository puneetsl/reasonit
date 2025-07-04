"""
Advanced task planning system with hierarchical decomposition.

This module implements a sophisticated planning system that can break down
complex reasoning tasks into manageable subtasks, handle dependencies,
and coordinate execution across multiple reasoning strategies.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import uuid

from models import (
    ReasoningRequest,
    ReasoningResult,
    ReasoningStrategy,
    ContextVariant,
    OutcomeType,
    SystemConfiguration
)

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks in the planning system."""
    ATOMIC = "atomic"                    # Indivisible task
    COMPOSITE = "composite"              # Can be decomposed
    SEQUENTIAL = "sequential"            # Must execute in order
    PARALLEL = "parallel"                # Can execute simultaneously
    CONDITIONAL = "conditional"          # Execution depends on conditions
    LOOP = "loop"                       # Repeating task
    VERIFICATION = "verification"        # Quality check task


class TaskStatus(Enum):
    """Status of tasks in the planning system."""
    PENDING = "pending"                  # Not yet started
    READY = "ready"                     # Ready to execute
    RUNNING = "running"                 # Currently executing
    COMPLETED = "completed"             # Successfully finished
    FAILED = "failed"                   # Execution failed
    CANCELLED = "cancelled"             # Cancelled by user or system
    BLOCKED = "blocked"                 # Waiting for dependencies
    SKIPPED = "skipped"                 # Skipped due to conditions


class TaskPriority(Enum):
    """Priority levels for task execution."""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    MINIMAL = 1


class DecompositionStrategy(Enum):
    """Strategies for task decomposition."""
    DIVIDE_AND_CONQUER = "divide_and_conquer"
    SEQUENTIAL_STEPS = "sequential_steps"
    PARALLEL_BRANCHES = "parallel_branches"
    ITERATIVE_REFINEMENT = "iterative_refinement"
    HIERARCHICAL = "hierarchical"
    DEPENDENCY_BASED = "dependency_based"


@dataclass
class TaskConstraint:
    """Constraints for task execution."""
    
    max_time: Optional[float] = None
    max_cost: Optional[float] = None
    max_memory: Optional[int] = None
    required_capabilities: List[str] = field(default_factory=list)
    excluded_strategies: List[ReasoningStrategy] = field(default_factory=list)
    minimum_confidence: Optional[float] = None


@dataclass
class TaskDependency:
    """Dependency relationship between tasks."""
    
    task_id: str
    dependency_type: str = "completion"  # completion, success, output
    condition: Optional[str] = None      # Optional condition to check
    timeout: Optional[float] = None      # Max wait time


@dataclass
class Task:
    """A single task in the planning system."""
    
    id: str
    name: str
    description: str
    task_type: TaskType
    priority: TaskPriority = TaskPriority.MEDIUM
    
    # Core task definition
    request: Optional[ReasoningRequest] = None
    query: Optional[str] = None
    expected_output: Optional[str] = None
    
    # Execution control
    status: TaskStatus = TaskStatus.PENDING
    assigned_strategy: Optional[ReasoningStrategy] = None
    context_variant: ContextVariant = ContextVariant.STANDARD
    
    # Dependencies and relationships
    dependencies: List[TaskDependency] = field(default_factory=list)
    children: List[str] = field(default_factory=list)  # Subtask IDs
    parent: Optional[str] = None
    
    # Constraints and requirements
    constraints: Optional[TaskConstraint] = None
    
    # Execution state
    result: Optional[ReasoningResult] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 2
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)


@dataclass
class Plan:
    """A complete execution plan containing multiple tasks."""
    
    id: str
    name: str
    description: str
    
    # Tasks and structure
    tasks: Dict[str, Task] = field(default_factory=dict)
    root_task_ids: List[str] = field(default_factory=list)
    
    # Execution state
    status: TaskStatus = TaskStatus.PENDING
    current_executing: Set[str] = field(default_factory=set)
    
    # Results and metrics
    total_cost: float = 0.0
    total_time: float = 0.0
    success_rate: float = 0.0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanningMetrics:
    """Metrics for the planning system."""
    
    total_plans: int = 0
    successful_plans: int = 0
    failed_plans: int = 0
    
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    
    avg_plan_execution_time: float = 0.0
    avg_task_execution_time: float = 0.0
    avg_decomposition_depth: float = 0.0
    
    strategy_usage: Dict[str, int] = field(default_factory=dict)
    decomposition_strategy_usage: Dict[str, int] = field(default_factory=dict)


class TaskPlanner:
    """
    Advanced task planning system with hierarchical decomposition.
    
    This system can break down complex reasoning tasks into manageable subtasks,
    handle dependencies, and coordinate execution across multiple strategies.
    """
    
    def __init__(
        self,
        config: Optional[SystemConfiguration] = None,
        max_decomposition_depth: int = 5,
        max_concurrent_tasks: int = 10,
        enable_adaptive_planning: bool = True
    ):
        self.config = config or SystemConfiguration()
        self.max_decomposition_depth = max_decomposition_depth
        self.max_concurrent_tasks = max_concurrent_tasks
        self.enable_adaptive_planning = enable_adaptive_planning
        
        # Active plans and state
        self.active_plans: Dict[str, Plan] = {}
        self.completed_plans: List[Plan] = []
        
        # Execution state
        self.execution_queue: deque = deque()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
        # Planning strategies
        self.decomposition_strategies = self._initialize_decomposition_strategies()
        
        # Metrics
        self.metrics = PlanningMetrics()
        
        # Callbacks for integration
        self.task_executors: Dict[ReasoningStrategy, Callable] = {}
        self.progress_callbacks: List[Callable] = []
        
        logger.info(f"Initialized TaskPlanner with max depth {max_decomposition_depth}")
    
    def _initialize_decomposition_strategies(self) -> Dict[DecompositionStrategy, Callable]:
        """Initialize decomposition strategy handlers."""
        
        return {
            DecompositionStrategy.DIVIDE_AND_CONQUER: self._decompose_divide_and_conquer,
            DecompositionStrategy.SEQUENTIAL_STEPS: self._decompose_sequential_steps,
            DecompositionStrategy.PARALLEL_BRANCHES: self._decompose_parallel_branches,
            DecompositionStrategy.ITERATIVE_REFINEMENT: self._decompose_iterative_refinement,
            DecompositionStrategy.HIERARCHICAL: self._decompose_hierarchical,
            DecompositionStrategy.DEPENDENCY_BASED: self._decompose_dependency_based
        }
    
    async def create_plan(
        self,
        request: ReasoningRequest,
        decomposition_strategy: Optional[DecompositionStrategy] = None,
        max_depth: Optional[int] = None
    ) -> Plan:
        """
        Create a complete execution plan for a reasoning request.
        
        Args:
            request: The reasoning request to plan for
            decomposition_strategy: Strategy for task decomposition
            max_depth: Maximum decomposition depth
            
        Returns:
            Complete execution plan
        """
        
        plan_id = str(uuid.uuid4())
        max_depth = max_depth or self.max_decomposition_depth
        
        # Select decomposition strategy
        if decomposition_strategy is None:
            decomposition_strategy = self._select_decomposition_strategy(request)
        
        logger.info(f"Creating plan {plan_id} with strategy {decomposition_strategy.value}")
        
        # Create root task
        root_task = Task(
            id=str(uuid.uuid4()),
            name=f"Root: {request.query[:50]}...",
            description=f"Root task for: {request.query}",
            task_type=TaskType.COMPOSITE,
            request=request,
            query=request.query,
            priority=TaskPriority.HIGH
        )
        
        # Create plan
        plan = Plan(
            id=plan_id,
            name=f"Plan for: {request.query[:30]}...",
            description=f"Execution plan for reasoning request: {request.query}",
            root_task_ids=[root_task.id]
        )
        
        plan.tasks[root_task.id] = root_task
        
        # Decompose the root task
        await self._decompose_task(plan, root_task.id, decomposition_strategy, 0, max_depth)
        
        # Validate and optimize plan
        self._validate_plan(plan)
        self._optimize_plan(plan)
        
        # Store plan
        self.active_plans[plan_id] = plan
        self.metrics.total_plans += 1
        
        logger.info(f"Created plan {plan_id} with {len(plan.tasks)} tasks")
        
        return plan
    
    def _select_decomposition_strategy(self, request: ReasoningRequest) -> DecompositionStrategy:
        """Select appropriate decomposition strategy based on request characteristics."""
        
        query_lower = request.query.lower()
        
        # Check for mathematical problems
        math_keywords = ["calculate", "solve", "equation", "formula", "prove", "derivative"]
        if any(kw in query_lower for kw in math_keywords):
            return DecompositionStrategy.SEQUENTIAL_STEPS
        
        # Check for comparison tasks
        comparison_keywords = ["compare", "contrast", "versus", "vs", "difference", "similar"]
        if any(kw in query_lower for kw in comparison_keywords):
            return DecompositionStrategy.PARALLEL_BRANCHES
        
        # Check for research tasks
        research_keywords = ["research", "analyze", "investigate", "study", "examine"]
        if any(kw in query_lower for kw in research_keywords):
            return DecompositionStrategy.HIERARCHICAL
        
        # Check for step-by-step tasks
        step_keywords = ["steps", "process", "procedure", "how to", "guide", "instructions"]
        if any(kw in query_lower for kw in step_keywords):
            return DecompositionStrategy.SEQUENTIAL_STEPS
        
        # Check for complex reasoning tasks
        reasoning_keywords = ["because", "therefore", "thus", "consequently", "reason", "why"]
        if any(kw in query_lower for kw in reasoning_keywords):
            return DecompositionStrategy.DIVIDE_AND_CONQUER
        
        # Default strategy
        return DecompositionStrategy.DIVIDE_AND_CONQUER
    
    async def _decompose_task(
        self,
        plan: Plan,
        task_id: str,
        strategy: DecompositionStrategy,
        current_depth: int,
        max_depth: int
    ) -> None:
        """Decompose a task using the specified strategy."""
        
        if current_depth >= max_depth:
            logger.warning(f"Max decomposition depth reached for task {task_id}")
            return
        
        task = plan.tasks[task_id]
        
        # Don't decompose atomic tasks
        if task.task_type == TaskType.ATOMIC:
            return
        
        # Get decomposition handler
        decomposition_handler = self.decomposition_strategies.get(strategy)
        if not decomposition_handler:
            logger.error(f"Unknown decomposition strategy: {strategy}")
            return
        
        try:
            # Decompose the task
            subtasks = await decomposition_handler(task, plan)
            
            # Add subtasks to plan
            for subtask in subtasks:
                plan.tasks[subtask.id] = subtask
                task.children.append(subtask.id)
                subtask.parent = task_id
                
                # Recursively decompose complex subtasks
                if subtask.task_type in [TaskType.COMPOSITE, TaskType.SEQUENTIAL, TaskType.PARALLEL]:
                    await self._decompose_task(plan, subtask.id, strategy, current_depth + 1, max_depth)
            
            logger.debug(f"Decomposed task {task_id} into {len(subtasks)} subtasks")
            
        except Exception as e:
            logger.error(f"Failed to decompose task {task_id}: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
    
    async def _decompose_divide_and_conquer(self, task: Task, plan: Plan) -> List[Task]:
        """Decompose task using divide and conquer strategy."""
        
        query = task.query or task.request.query if task.request else ""
        
        # Simple divide and conquer - split query into logical parts
        subtasks = []
        
        # Check if query contains multiple questions or parts
        if "and" in query.lower() or ";" in query:
            parts = self._split_query_parts(query)
            
            for i, part in enumerate(parts):
                subtask = Task(
                    id=str(uuid.uuid4()),
                    name=f"Part {i+1}: {part[:30]}...",
                    description=f"Solve part: {part}",
                    task_type=TaskType.ATOMIC,
                    query=part.strip(),
                    priority=task.priority,
                    assigned_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT
                )
                subtasks.append(subtask)
            
            # Add synthesis task
            synthesis_task = Task(
                id=str(uuid.uuid4()),
                name="Synthesis",
                description="Combine results from all parts",
                task_type=TaskType.ATOMIC,
                query=f"Combine and synthesize the following results for: {query}",
                priority=task.priority,
                assigned_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
                dependencies=[TaskDependency(task_id=st.id) for st in subtasks]
            )
            subtasks.append(synthesis_task)
        
        else:
            # Break into analysis and conclusion
            analysis_task = Task(
                id=str(uuid.uuid4()),
                name="Analysis",
                description="Analyze the problem",
                task_type=TaskType.ATOMIC,
                query=f"Analyze this problem: {query}",
                priority=task.priority,
                assigned_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT
            )
            
            conclusion_task = Task(
                id=str(uuid.uuid4()),
                name="Conclusion",
                description="Draw conclusions",
                task_type=TaskType.ATOMIC,
                query=f"Based on the analysis, provide conclusions for: {query}",
                priority=task.priority,
                assigned_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
                dependencies=[TaskDependency(task_id=analysis_task.id)]
            )
            
            subtasks = [analysis_task, conclusion_task]
        
        return subtasks
    
    async def _decompose_sequential_steps(self, task: Task, plan: Plan) -> List[Task]:
        """Decompose task into sequential steps."""
        
        query = task.query or task.request.query if task.request else ""
        
        # Identify sequential steps based on query type
        steps = self._identify_sequential_steps(query)
        
        subtasks = []
        previous_task_id = None
        
        for i, step in enumerate(steps):
            subtask = Task(
                id=str(uuid.uuid4()),
                name=f"Step {i+1}",
                description=step,
                task_type=TaskType.ATOMIC,
                query=f"Execute step {i+1}: {step}",
                priority=task.priority,
                assigned_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT
            )
            
            # Add dependency on previous step
            if previous_task_id:
                subtask.dependencies.append(TaskDependency(task_id=previous_task_id))
            
            subtasks.append(subtask)
            previous_task_id = subtask.id
        
        return subtasks
    
    async def _decompose_parallel_branches(self, task: Task, plan: Plan) -> List[Task]:
        """Decompose task into parallel branches."""
        
        query = task.query or task.request.query if task.request else ""
        
        # Identify parallel aspects
        branches = self._identify_parallel_branches(query)
        
        subtasks = []
        branch_tasks = []
        
        # Create parallel branch tasks
        for i, branch in enumerate(branches):
            branch_task = Task(
                id=str(uuid.uuid4()),
                name=f"Branch {i+1}",
                description=branch,
                task_type=TaskType.ATOMIC,
                query=f"Analyze branch {i+1}: {branch}",
                priority=task.priority,
                assigned_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT
            )
            subtasks.append(branch_task)
            branch_tasks.append(branch_task)
        
        # Add comparison/synthesis task
        if len(branch_tasks) > 1:
            comparison_task = Task(
                id=str(uuid.uuid4()),
                name="Compare and Synthesize",
                description="Compare results from all branches",
                task_type=TaskType.ATOMIC,
                query=f"Compare and synthesize results for: {query}",
                priority=task.priority,
                assigned_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
                dependencies=[TaskDependency(task_id=bt.id) for bt in branch_tasks]
            )
            subtasks.append(comparison_task)
        
        return subtasks
    
    async def _decompose_iterative_refinement(self, task: Task, plan: Plan) -> List[Task]:
        """Decompose task into iterative refinement steps."""
        
        query = task.query or task.request.query if task.request else ""
        
        subtasks = []
        
        # Initial attempt
        initial_task = Task(
            id=str(uuid.uuid4()),
            name="Initial Attempt",
            description="First attempt at solving the problem",
            task_type=TaskType.ATOMIC,
            query=f"Initial attempt: {query}",
            priority=task.priority,
            assigned_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT
        )
        subtasks.append(initial_task)
        
        # Review and refine
        review_task = Task(
            id=str(uuid.uuid4()),
            name="Review",
            description="Review and identify improvements",
            task_type=TaskType.ATOMIC,
            query=f"Review the initial attempt and identify improvements for: {query}",
            priority=task.priority,
            assigned_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            dependencies=[TaskDependency(task_id=initial_task.id)]
        )
        subtasks.append(review_task)
        
        # Final refinement
        final_task = Task(
            id=str(uuid.uuid4()),
            name="Final Refinement",
            description="Apply improvements and finalize",
            task_type=TaskType.ATOMIC,
            query=f"Apply improvements and provide final answer for: {query}",
            priority=task.priority,
            assigned_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            dependencies=[TaskDependency(task_id=review_task.id)]
        )
        subtasks.append(final_task)
        
        return subtasks
    
    async def _decompose_hierarchical(self, task: Task, plan: Plan) -> List[Task]:
        """Decompose task hierarchically."""
        
        query = task.query or task.request.query if task.request else ""
        
        subtasks = []
        
        # Information gathering
        gather_task = Task(
            id=str(uuid.uuid4()),
            name="Information Gathering",
            description="Gather relevant information",
            task_type=TaskType.ATOMIC,
            query=f"Gather relevant information for: {query}",
            priority=task.priority,
            assigned_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT
        )
        subtasks.append(gather_task)
        
        # Analysis
        analysis_task = Task(
            id=str(uuid.uuid4()),
            name="Analysis",
            description="Analyze gathered information",
            task_type=TaskType.ATOMIC,
            query=f"Analyze the gathered information for: {query}",
            priority=task.priority,
            assigned_strategy=ReasoningStrategy.TREE_OF_THOUGHTS,
            dependencies=[TaskDependency(task_id=gather_task.id)]
        )
        subtasks.append(analysis_task)
        
        # Synthesis
        synthesis_task = Task(
            id=str(uuid.uuid4()),
            name="Synthesis",
            description="Synthesize final answer",
            task_type=TaskType.ATOMIC,
            query=f"Synthesize final answer for: {query}",
            priority=task.priority,
            assigned_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            dependencies=[TaskDependency(task_id=analysis_task.id)]
        )
        subtasks.append(synthesis_task)
        
        return subtasks
    
    async def _decompose_dependency_based(self, task: Task, plan: Plan) -> List[Task]:
        """Decompose task based on dependencies."""
        
        query = task.query or task.request.query if task.request else ""
        
        # Identify dependencies in the query
        dependencies = self._identify_dependencies(query)
        
        subtasks = []
        dependency_tasks = {}
        
        # Create tasks for each dependency
        for dep in dependencies:
            dep_task = Task(
                id=str(uuid.uuid4()),
                name=f"Dependency: {dep}",
                description=f"Resolve dependency: {dep}",
                task_type=TaskType.ATOMIC,
                query=f"Resolve: {dep}",
                priority=task.priority,
                assigned_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT
            )
            subtasks.append(dep_task)
            dependency_tasks[dep] = dep_task.id
        
        # Main resolution task
        if dependency_tasks:
            main_task = Task(
                id=str(uuid.uuid4()),
                name="Main Resolution",
                description="Resolve main query using dependencies",
                task_type=TaskType.ATOMIC,
                query=f"Using resolved dependencies, solve: {query}",
                priority=task.priority,
                assigned_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
                dependencies=[TaskDependency(task_id=tid) for tid in dependency_tasks.values()]
            )
            subtasks.append(main_task)
        
        return subtasks
    
    def _split_query_parts(self, query: str) -> List[str]:
        """Split query into logical parts."""
        
        # Split on common separators
        parts = []
        
        if " and " in query.lower():
            parts = [p.strip() for p in query.split(" and ")]
        elif ";" in query:
            parts = [p.strip() for p in query.split(";")]
        elif "?" in query and query.count("?") > 1:
            parts = [p.strip() + "?" for p in query.split("?") if p.strip()]
        else:
            # Fallback: split into sentences
            parts = [s.strip() for s in query.split(".") if s.strip()]
        
        return [p for p in parts if len(p) > 5]  # Filter out very short parts
    
    def _identify_sequential_steps(self, query: str) -> List[str]:
        """Identify sequential steps from query."""
        
        query_lower = query.lower()
        
        # Math problem steps
        if any(kw in query_lower for kw in ["solve", "equation", "calculate"]):
            return [
                "Identify the problem type and relevant formulas",
                "Set up the equation or calculation",
                "Solve step by step",
                "Verify the solution"
            ]
        
        # Process/procedure steps
        if any(kw in query_lower for kw in ["how to", "steps", "process", "procedure"]):
            return [
                "Understand the requirements",
                "Plan the approach",
                "Execute the main steps",
                "Review and finalize"
            ]
        
        # Research steps
        if any(kw in query_lower for kw in ["research", "analyze", "study"]):
            return [
                "Define the research scope",
                "Gather relevant information",
                "Analyze the data",
                "Draw conclusions"
            ]
        
        # Default steps
        return [
            "Understand the problem",
            "Develop solution approach",
            "Implement solution",
            "Validate results"
        ]
    
    def _identify_parallel_branches(self, query: str) -> List[str]:
        """Identify parallel branches from query."""
        
        query_lower = query.lower()
        
        # Comparison tasks
        if any(kw in query_lower for kw in ["compare", "contrast", "versus", "vs"]):
            # Try to extract entities being compared
            if " vs " in query_lower:
                parts = query.split(" vs ")
                return [f"Analyze {part.strip()}" for part in parts]
            elif " versus " in query_lower:
                parts = query.split(" versus ")
                return [f"Analyze {part.strip()}" for part in parts]
            elif "compare" in query_lower and " and " in query_lower:
                # Extract what's being compared
                return ["Analyze first option", "Analyze second option", "Identify similarities", "Identify differences"]
        
        # Multiple aspects
        if any(kw in query_lower for kw in ["aspects", "factors", "elements", "components"]):
            return [
                "Technical aspects",
                "Economic aspects", 
                "Social aspects",
                "Environmental aspects"
            ]
        
        # Default branches
        return [
            "Primary analysis",
            "Alternative perspective",
            "Supporting evidence"
        ]
    
    def _identify_dependencies(self, query: str) -> List[str]:
        """Identify dependencies from query."""
        
        query_lower = query.lower()
        dependencies = []
        
        # Look for conditional phrases
        if "if" in query_lower:
            dependencies.append("Verify conditions")
        
        if "given that" in query_lower or "assuming" in query_lower:
            dependencies.append("Validate assumptions")
        
        if "first" in query_lower or "before" in query_lower:
            dependencies.append("Complete prerequisite steps")
        
        if "depends on" in query_lower or "requires" in query_lower:
            dependencies.append("Resolve requirements")
        
        # Default dependencies for complex queries
        if len(query) > 100:
            dependencies.extend([
                "Clarify scope and objectives",
                "Gather necessary information"
            ])
        
        return dependencies
    
    def _validate_plan(self, plan: Plan) -> None:
        """Validate plan structure and dependencies."""
        
        # Check for circular dependencies
        for task_id, task in plan.tasks.items():
            if self._has_circular_dependency(plan, task_id, set()):
                logger.error(f"Circular dependency detected for task {task_id}")
                raise ValueError(f"Circular dependency in task {task_id}")
        
        # Check that all dependencies exist
        for task_id, task in plan.tasks.items():
            for dep in task.dependencies:
                if dep.task_id not in plan.tasks:
                    logger.error(f"Missing dependency {dep.task_id} for task {task_id}")
                    raise ValueError(f"Missing dependency {dep.task_id}")
        
        logger.debug(f"Plan {plan.id} validation passed")
    
    def _has_circular_dependency(self, plan: Plan, task_id: str, visited: Set[str]) -> bool:
        """Check for circular dependencies."""
        
        if task_id in visited:
            return True
        
        visited.add(task_id)
        
        task = plan.tasks.get(task_id)
        if not task:
            return False
        
        for dep in task.dependencies:
            if self._has_circular_dependency(plan, dep.task_id, visited.copy()):
                return True
        
        return False
    
    def _optimize_plan(self, plan: Plan) -> None:
        """Optimize plan for better execution."""
        
        # Identify tasks that can run in parallel
        self._identify_parallel_opportunities(plan)
        
        # Optimize task priorities
        self._optimize_task_priorities(plan)
        
        # Set appropriate strategies
        self._optimize_task_strategies(plan)
        
        logger.debug(f"Plan {plan.id} optimization completed")
    
    def _identify_parallel_opportunities(self, plan: Plan) -> None:
        """Identify tasks that can be executed in parallel."""
        
        for task_id, task in plan.tasks.items():
            # Tasks with no dependencies can potentially run in parallel
            if not task.dependencies and task.task_type == TaskType.ATOMIC:
                # Check if siblings can run in parallel
                if task.parent:
                    parent = plan.tasks[task.parent]
                    sibling_ids = [tid for tid in parent.children if tid != task_id]
                    
                    # If siblings also have no dependencies, mark as parallel
                    parallel_siblings = []
                    for sibling_id in sibling_ids:
                        sibling = plan.tasks[sibling_id]
                        if not sibling.dependencies and sibling.task_type == TaskType.ATOMIC:
                            parallel_siblings.append(sibling_id)
                    
                    if len(parallel_siblings) > 0:
                        # Update parent to parallel type if appropriate
                        if parent.task_type == TaskType.COMPOSITE:
                            parent.task_type = TaskType.PARALLEL
    
    def _optimize_task_priorities(self, plan: Plan) -> None:
        """Optimize task priorities based on dependencies."""
        
        # Tasks with more dependents should have higher priority
        dependent_counts = defaultdict(int)
        
        for task in plan.tasks.values():
            for dep in task.dependencies:
                dependent_counts[dep.task_id] += 1
        
        for task_id, count in dependent_counts.items():
            task = plan.tasks[task_id]
            if count > 2:
                task.priority = TaskPriority.HIGH
            elif count > 1:
                task.priority = TaskPriority.MEDIUM
    
    def _optimize_task_strategies(self, plan: Plan) -> None:
        """Optimize reasoning strategies for tasks."""
        
        for task in plan.tasks.values():
            if task.assigned_strategy:
                continue
                
            query = task.query or ""
            query_lower = query.lower()
            
            # Assign strategy based on task characteristics
            if any(kw in query_lower for kw in ["analyze", "compare", "evaluate"]):
                task.assigned_strategy = ReasoningStrategy.TREE_OF_THOUGHTS
            elif any(kw in query_lower for kw in ["step", "solve", "calculate"]):
                task.assigned_strategy = ReasoningStrategy.CHAIN_OF_THOUGHT
            elif any(kw in query_lower for kw in ["explore", "brainstorm", "generate"]):
                task.assigned_strategy = ReasoningStrategy.MONTE_CARLO_TREE_SEARCH
            elif "?" in query_lower and len(query) < 50:
                task.assigned_strategy = ReasoningStrategy.SELF_ASK
            else:
                task.assigned_strategy = ReasoningStrategy.CHAIN_OF_THOUGHT
    
    async def execute_plan(self, plan_id: str) -> Plan:
        """Execute a complete plan."""
        
        plan = self.active_plans.get(plan_id)
        if not plan:
            raise ValueError(f"Plan {plan_id} not found")
        
        logger.info(f"Starting execution of plan {plan_id}")
        
        plan.status = TaskStatus.RUNNING
        start_time = datetime.now()
        
        try:
            # Execute root tasks
            for root_task_id in plan.root_task_ids:
                await self._execute_task_tree(plan, root_task_id)
            
            # Wait for all tasks to complete
            while self.running_tasks:
                await asyncio.sleep(0.1)
            
            # Calculate final metrics
            self._calculate_plan_metrics(plan)
            
            plan.status = TaskStatus.COMPLETED
            plan.total_time = (datetime.now() - start_time).total_seconds()
            
            # Move to completed plans
            self.completed_plans.append(plan)
            del self.active_plans[plan_id]
            
            self.metrics.successful_plans += 1
            
            logger.info(f"Plan {plan_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Plan {plan_id} execution failed: {e}")
            plan.status = TaskStatus.FAILED
            plan.total_time = (datetime.now() - start_time).total_seconds()
            
            self.metrics.failed_plans += 1
        
        return plan
    
    async def _execute_task_tree(self, plan: Plan, task_id: str) -> None:
        """Execute a task and its subtasks."""
        
        task = plan.tasks[task_id]
        
        # Check if dependencies are satisfied
        if not await self._dependencies_satisfied(plan, task_id):
            task.status = TaskStatus.BLOCKED
            return
        
        # Execute based on task type
        if task.task_type == TaskType.ATOMIC:
            await self._execute_atomic_task(plan, task_id)
        elif task.task_type == TaskType.PARALLEL:
            await self._execute_parallel_tasks(plan, task_id)
        elif task.task_type == TaskType.SEQUENTIAL:
            await self._execute_sequential_tasks(plan, task_id)
        else:
            # For composite tasks, execute children
            for child_id in task.children:
                await self._execute_task_tree(plan, child_id)
    
    async def _dependencies_satisfied(self, plan: Plan, task_id: str) -> bool:
        """Check if task dependencies are satisfied."""
        
        task = plan.tasks[task_id]
        
        for dep in task.dependencies:
            dep_task = plan.tasks.get(dep.task_id)
            if not dep_task:
                return False
            
            if dep.dependency_type == "completion":
                if dep_task.status not in [TaskStatus.COMPLETED, TaskStatus.SKIPPED]:
                    return False
            elif dep.dependency_type == "success":
                if dep_task.status != TaskStatus.COMPLETED:
                    return False
        
        return True
    
    async def _execute_atomic_task(self, plan: Plan, task_id: str) -> None:
        """Execute a single atomic task."""
        
        task = plan.tasks[task_id]
        task.status = TaskStatus.RUNNING
        task.start_time = datetime.now()
        
        try:
            # Create reasoning request
            if not task.request and task.query:
                task.request = ReasoningRequest(
                    query=task.query,
                    strategy=task.assigned_strategy or ReasoningStrategy.CHAIN_OF_THOUGHT,
                    context_variant=task.context_variant,
                    confidence_threshold=0.8,
                    session_id=plan.id
                )
            
            # Execute using appropriate executor
            if task.assigned_strategy in self.task_executors:
                executor = self.task_executors[task.assigned_strategy]
                task.result = await executor(task.request)
            else:
                # Default execution (placeholder)
                task.result = self._create_placeholder_result(task.request)
            
            task.status = TaskStatus.COMPLETED
            task.end_time = datetime.now()
            
            # Update metrics
            self.metrics.successful_tasks += 1
            
            # Notify progress callbacks
            for callback in self.progress_callbacks:
                try:
                    await callback(plan, task)
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")
            
        except Exception as e:
            logger.error(f"Task {task_id} execution failed: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.end_time = datetime.now()
            
            self.metrics.failed_tasks += 1
            
            # Retry if possible
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                logger.info(f"Retrying task {task_id} (attempt {task.retry_count})")
                await asyncio.sleep(1)  # Brief delay before retry
                await self._execute_atomic_task(plan, task_id)
    
    async def _execute_parallel_tasks(self, plan: Plan, task_id: str) -> None:
        """Execute child tasks in parallel."""
        
        task = plan.tasks[task_id]
        
        # Start all child tasks concurrently
        child_tasks = []
        for child_id in task.children:
            child_task = asyncio.create_task(self._execute_task_tree(plan, child_id))
            child_tasks.append(child_task)
        
        # Wait for all to complete
        if child_tasks:
            await asyncio.gather(*child_tasks, return_exceptions=True)
        
        task.status = TaskStatus.COMPLETED
    
    async def _execute_sequential_tasks(self, plan: Plan, task_id: str) -> None:
        """Execute child tasks sequentially."""
        
        task = plan.tasks[task_id]
        
        # Execute child tasks one by one
        for child_id in task.children:
            await self._execute_task_tree(plan, child_id)
            
            # Check if child failed and should stop sequence
            child_task = plan.tasks[child_id]
            if child_task.status == TaskStatus.FAILED:
                logger.warning(f"Sequential execution stopped due to failed child {child_id}")
                task.status = TaskStatus.FAILED
                return
        
        task.status = TaskStatus.COMPLETED
    
    def _create_placeholder_result(self, request: ReasoningRequest) -> ReasoningResult:
        """Create a placeholder result for testing."""
        
        from models.types import ReasoningStep
        
        return ReasoningResult(
            request=request,
            final_answer=f"Placeholder answer for: {request.query}",
            reasoning_trace=[
                ReasoningStep(
                    step_number=1,
                    strategy=request.strategy,
                    content=f"Processing: {request.query}",
                    confidence=0.8,
                    cost=0.001
                )
            ],
            total_cost=0.001,
            total_time=0.1,
            confidence_score=0.8,
            strategies_used=[request.strategy],
            outcome=OutcomeType.SUCCESS,
            timestamp=datetime.now()
        )
    
    def _calculate_plan_metrics(self, plan: Plan) -> None:
        """Calculate final metrics for completed plan."""
        
        total_tasks = len(plan.tasks)
        completed_tasks = sum(1 for t in plan.tasks.values() if t.status == TaskStatus.COMPLETED)
        
        plan.success_rate = completed_tasks / total_tasks if total_tasks > 0 else 0.0
        plan.total_cost = sum(
            t.result.total_cost for t in plan.tasks.values() 
            if t.result and hasattr(t.result, 'total_cost')
        )
        
        # Update global metrics
        self.metrics.total_tasks += total_tasks
        self.metrics.successful_tasks += completed_tasks
        
        # Update averages
        if self.metrics.total_plans > 0:
            self.metrics.avg_plan_execution_time = (
                self.metrics.avg_plan_execution_time * 0.9 + plan.total_time * 0.1
            )
    
    def register_task_executor(self, strategy: ReasoningStrategy, executor: Callable) -> None:
        """Register an executor for a specific reasoning strategy."""
        self.task_executors[strategy] = executor
        logger.info(f"Registered executor for strategy {strategy.value}")
    
    def add_progress_callback(self, callback: Callable) -> None:
        """Add a progress callback function."""
        self.progress_callbacks.append(callback)
    
    def get_plan_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a plan."""
        
        plan = self.active_plans.get(plan_id) or next(
            (p for p in self.completed_plans if p.id == plan_id), None
        )
        
        if not plan:
            return None
        
        task_counts = defaultdict(int)
        for task in plan.tasks.values():
            task_counts[task.status.value] += 1
        
        return {
            "plan_id": plan.id,
            "status": plan.status.value,
            "total_tasks": len(plan.tasks),
            "task_status_counts": dict(task_counts),
            "success_rate": plan.success_rate,
            "total_cost": plan.total_cost,
            "total_time": plan.total_time,
            "created_at": plan.created_at.isoformat(),
            "updated_at": plan.updated_at.isoformat()
        }
    
    def get_planning_metrics(self) -> Dict[str, Any]:
        """Get comprehensive planning metrics."""
        
        return {
            "total_plans": self.metrics.total_plans,
            "successful_plans": self.metrics.successful_plans,
            "failed_plans": self.metrics.failed_plans,
            "plan_success_rate": self.metrics.successful_plans / max(self.metrics.total_plans, 1),
            
            "total_tasks": self.metrics.total_tasks,
            "successful_tasks": self.metrics.successful_tasks,
            "failed_tasks": self.metrics.failed_tasks,
            "task_success_rate": self.metrics.successful_tasks / max(self.metrics.total_tasks, 1),
            
            "avg_plan_execution_time": self.metrics.avg_plan_execution_time,
            "avg_task_execution_time": self.metrics.avg_task_execution_time,
            "avg_decomposition_depth": self.metrics.avg_decomposition_depth,
            
            "active_plans": len(self.active_plans),
            "completed_plans": len(self.completed_plans),
            
            "strategy_usage": dict(self.metrics.strategy_usage),
            "decomposition_strategy_usage": dict(self.metrics.decomposition_strategy_usage)
        }
    
    async def cancel_plan(self, plan_id: str) -> bool:
        """Cancel an active plan."""
        
        plan = self.active_plans.get(plan_id)
        if not plan:
            return False
        
        plan.status = TaskStatus.CANCELLED
        
        # Cancel running tasks
        for task in plan.tasks.values():
            if task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.CANCELLED
        
        # Cancel any asyncio tasks
        for task_id, async_task in list(self.running_tasks.items()):
            if task_id in plan.tasks:
                async_task.cancel()
                del self.running_tasks[task_id]
        
        logger.info(f"Cancelled plan {plan_id}")
        return True
    
    async def close(self) -> None:
        """Clean up resources."""
        
        # Cancel all running tasks
        for async_task in self.running_tasks.values():
            async_task.cancel()
        
        self.running_tasks.clear()
        
        logger.info("TaskPlanner closed")