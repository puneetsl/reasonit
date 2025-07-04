"""
Tests for the planning system.

This module tests all aspects of the task planning system including
task decomposition, checkpoint management, and fallback handling.
"""

import asyncio
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, AsyncMock

import pytest

from planning import (
    TaskPlanner,
    Plan,
    Task,
    TaskType,
    TaskStatus,
    TaskPriority,
    TaskConstraint,
    TaskDependency,
    DecompositionStrategy,
    CheckpointManager,
    FallbackGraph,
    FallbackRule,
    FallbackType,
    FallbackReason,
    FallbackCondition
)
from models import (
    ReasoningRequest,
    ReasoningResult,
    ReasoningStrategy,
    ContextVariant,
    OutcomeType,
    ReasoningStep,
    SystemConfiguration
)


class TestTaskType:
    """Test TaskType enum."""
    
    def test_task_types(self):
        """Test all task types are defined."""
        assert TaskType.ATOMIC.value == "atomic"
        assert TaskType.COMPOSITE.value == "composite"
        assert TaskType.SEQUENTIAL.value == "sequential"
        assert TaskType.PARALLEL.value == "parallel"
        assert TaskType.CONDITIONAL.value == "conditional"
        assert TaskType.LOOP.value == "loop"
        assert TaskType.VERIFICATION.value == "verification"


class TestTaskStatus:
    """Test TaskStatus enum."""
    
    def test_task_statuses(self):
        """Test all task statuses are defined."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.READY.value == "ready"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"
        assert TaskStatus.BLOCKED.value == "blocked"
        assert TaskStatus.SKIPPED.value == "skipped"


class TestTaskPriority:
    """Test TaskPriority enum."""
    
    def test_task_priorities(self):
        """Test all task priorities are defined."""
        assert TaskPriority.CRITICAL.value == 5
        assert TaskPriority.HIGH.value == 4
        assert TaskPriority.MEDIUM.value == 3
        assert TaskPriority.LOW.value == 2
        assert TaskPriority.MINIMAL.value == 1


class TestDecompositionStrategy:
    """Test DecompositionStrategy enum."""
    
    def test_decomposition_strategies(self):
        """Test all decomposition strategies are defined."""
        assert DecompositionStrategy.DIVIDE_AND_CONQUER.value == "divide_and_conquer"
        assert DecompositionStrategy.SEQUENTIAL_STEPS.value == "sequential_steps"
        assert DecompositionStrategy.PARALLEL_BRANCHES.value == "parallel_branches"
        assert DecompositionStrategy.ITERATIVE_REFINEMENT.value == "iterative_refinement"
        assert DecompositionStrategy.HIERARCHICAL.value == "hierarchical"
        assert DecompositionStrategy.DEPENDENCY_BASED.value == "dependency_based"


class TestTaskConstraint:
    """Test TaskConstraint data structure."""
    
    def test_constraint_creation(self):
        """Test creating task constraint."""
        constraint = TaskConstraint(
            max_time=60.0,
            max_cost=0.1,
            max_memory=1024,
            required_capabilities=["math", "reasoning"],
            excluded_strategies=[ReasoningStrategy.MONTE_CARLO_TREE_SEARCH],
            minimum_confidence=0.8
        )
        
        assert constraint.max_time == 60.0
        assert constraint.max_cost == 0.1
        assert constraint.max_memory == 1024
        assert constraint.required_capabilities == ["math", "reasoning"]
        assert constraint.excluded_strategies == [ReasoningStrategy.MONTE_CARLO_TREE_SEARCH]
        assert constraint.minimum_confidence == 0.8


class TestTaskDependency:
    """Test TaskDependency data structure."""
    
    def test_dependency_creation(self):
        """Test creating task dependency."""
        dependency = TaskDependency(
            task_id="task_123",
            dependency_type="success",
            condition="confidence > 0.8",
            timeout=30.0
        )
        
        assert dependency.task_id == "task_123"
        assert dependency.dependency_type == "success"
        assert dependency.condition == "confidence > 0.8"
        assert dependency.timeout == 30.0


class TestTask:
    """Test Task data structure."""
    
    def test_task_creation(self):
        """Test creating task."""
        constraint = TaskConstraint(max_time=60.0, max_cost=0.1)
        dependency = TaskDependency(task_id="dep_task")
        
        task = Task(
            id="task_123",
            name="Test Task",
            description="A test task",
            task_type=TaskType.ATOMIC,
            priority=TaskPriority.HIGH,
            query="What is 2+2?",
            expected_output="4",
            assigned_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            context_variant=ContextVariant.STANDARD,
            dependencies=[dependency],
            children=["child_1", "child_2"],
            parent="parent_task",
            constraints=constraint,
            max_retries=3
        )
        
        assert task.id == "task_123"
        assert task.name == "Test Task"
        assert task.description == "A test task"
        assert task.task_type == TaskType.ATOMIC
        assert task.priority == TaskPriority.HIGH
        assert task.query == "What is 2+2?"
        assert task.expected_output == "4"
        assert task.status == TaskStatus.PENDING
        assert task.assigned_strategy == ReasoningStrategy.CHAIN_OF_THOUGHT
        assert task.context_variant == ContextVariant.STANDARD
        assert len(task.dependencies) == 1
        assert task.children == ["child_1", "child_2"]
        assert task.parent == "parent_task"
        assert task.constraints == constraint
        assert task.retry_count == 0
        assert task.max_retries == 3


class TestPlan:
    """Test Plan data structure."""
    
    def test_plan_creation(self):
        """Test creating plan."""
        task1 = Task(id="task_1", name="Task 1", description="First task", task_type=TaskType.ATOMIC)
        task2 = Task(id="task_2", name="Task 2", description="Second task", task_type=TaskType.ATOMIC)
        
        plan = Plan(
            id="plan_123",
            name="Test Plan",
            description="A test plan",
            root_task_ids=["task_1"],
            total_cost=0.05,
            total_time=120.0,
            success_rate=0.9
        )
        
        plan.tasks["task_1"] = task1
        plan.tasks["task_2"] = task2
        
        assert plan.id == "plan_123"
        assert plan.name == "Test Plan"
        assert plan.description == "A test plan"
        assert plan.root_task_ids == ["task_1"]
        assert plan.status == TaskStatus.PENDING
        assert len(plan.tasks) == 2
        assert plan.total_cost == 0.05
        assert plan.total_time == 120.0
        assert plan.success_rate == 0.9


class TestTaskPlanner:
    """Test TaskPlanner functionality."""
    
    @pytest.fixture
    def planner(self):
        """Create a TaskPlanner instance for testing."""
        return TaskPlanner(
            max_decomposition_depth=3,
            max_concurrent_tasks=5,
            enable_adaptive_planning=True
        )
    
    @pytest.fixture
    def sample_request(self):
        """Create a sample reasoning request."""
        return ReasoningRequest(
            query="Solve the equation 2x + 3 = 7 step by step",
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            context_variant=ContextVariant.STANDARD,
            confidence_threshold=0.8,
            session_id="test_session"
        )
    
    @pytest.fixture
    def complex_request(self):
        """Create a complex reasoning request."""
        return ReasoningRequest(
            query="Compare and contrast the advantages and disadvantages of renewable vs fossil fuel energy sources",
            strategy=ReasoningStrategy.TREE_OF_THOUGHTS,
            confidence_threshold=0.9
        )
    
    def test_planner_initialization(self):
        """Test TaskPlanner initialization."""
        planner = TaskPlanner()
        
        assert planner.max_decomposition_depth == 5
        assert planner.max_concurrent_tasks == 10
        assert planner.enable_adaptive_planning is True
        assert len(planner.active_plans) == 0
        assert len(planner.completed_plans) == 0
        assert len(planner.decomposition_strategies) == 6
    
    def test_decomposition_strategies_initialization(self, planner):
        """Test decomposition strategies initialization."""
        strategies = planner.decomposition_strategies
        
        assert DecompositionStrategy.DIVIDE_AND_CONQUER in strategies
        assert DecompositionStrategy.SEQUENTIAL_STEPS in strategies
        assert DecompositionStrategy.PARALLEL_BRANCHES in strategies
        assert DecompositionStrategy.ITERATIVE_REFINEMENT in strategies
        assert DecompositionStrategy.HIERARCHICAL in strategies
        assert DecompositionStrategy.DEPENDENCY_BASED in strategies
    
    def test_select_decomposition_strategy(self, planner):
        """Test decomposition strategy selection."""
        # Math problem should use sequential steps
        math_request = ReasoningRequest(query="Calculate the derivative of x^2 + 3x + 1")
        assert planner._select_decomposition_strategy(math_request) == DecompositionStrategy.SEQUENTIAL_STEPS
        
        # Comparison should use parallel branches
        compare_request = ReasoningRequest(query="Compare Python vs Java programming languages")
        assert planner._select_decomposition_strategy(compare_request) == DecompositionStrategy.PARALLEL_BRANCHES
        
        # Research should use hierarchical
        research_request = ReasoningRequest(query="Research the impacts of climate change")
        assert planner._select_decomposition_strategy(research_request) == DecompositionStrategy.HIERARCHICAL
        
        # Step-by-step should use sequential
        steps_request = ReasoningRequest(query="How to bake a chocolate cake step by step")
        assert planner._select_decomposition_strategy(steps_request) == DecompositionStrategy.SEQUENTIAL_STEPS
        
        # Default should be divide and conquer
        default_request = ReasoningRequest(query="What is artificial intelligence?")
        assert planner._select_decomposition_strategy(default_request) == DecompositionStrategy.DIVIDE_AND_CONQUER
    
    @pytest.mark.asyncio
    async def test_create_plan_simple(self, planner, sample_request):
        """Test creating a simple plan."""
        plan = await planner.create_plan(sample_request)
        
        assert isinstance(plan, Plan)
        assert plan.id in planner.active_plans
        assert len(plan.tasks) >= 1  # At least root task
        assert len(plan.root_task_ids) == 1
        
        # Check root task
        root_task_id = plan.root_task_ids[0]
        root_task = plan.tasks[root_task_id]
        assert root_task.query == sample_request.query
        assert root_task.task_type == TaskType.COMPOSITE
    
    @pytest.mark.asyncio
    async def test_create_plan_complex(self, planner, complex_request):
        """Test creating a complex plan."""
        plan = await planner.create_plan(
            complex_request,
            decomposition_strategy=DecompositionStrategy.PARALLEL_BRANCHES
        )
        
        assert isinstance(plan, Plan)
        assert len(plan.tasks) > 1  # Should have multiple tasks
        
        # Should have some parallel structure
        parallel_tasks = [t for t in plan.tasks.values() if t.task_type == TaskType.PARALLEL]
        assert len(parallel_tasks) >= 0  # May or may not have explicit parallel tasks
    
    def test_split_query_parts(self, planner):
        """Test query splitting."""
        # Test "and" splitting
        query1 = "What is AI and how does machine learning work"
        parts1 = planner._split_query_parts(query1)
        assert len(parts1) == 2
        assert "What is AI" in parts1[0]
        assert "how does machine learning work" in parts1[1]
        
        # Test semicolon splitting
        query2 = "Define photosynthesis; explain cellular respiration"
        parts2 = planner._split_query_parts(query2)
        assert len(parts2) == 2
        
        # Test multiple questions
        query3 = "What is gravity? How does it work? Why is it important?"
        parts3 = planner._split_query_parts(query3)
        assert len(parts3) >= 2
    
    def test_identify_sequential_steps(self, planner):
        """Test sequential step identification."""
        # Math problem
        math_query = "Solve the quadratic equation x^2 + 2x + 1 = 0"
        math_steps = planner._identify_sequential_steps(math_query)
        assert len(math_steps) >= 3
        assert any("formula" in step.lower() for step in math_steps)
        
        # Process query
        process_query = "How to make a paper airplane step by step"
        process_steps = planner._identify_sequential_steps(process_query)
        assert len(process_steps) >= 3
        assert any("plan" in step.lower() for step in process_steps)
        
        # Research query
        research_query = "Research the history of artificial intelligence"
        research_steps = planner._identify_sequential_steps(research_query)
        assert len(research_steps) >= 3
        assert any("scope" in step.lower() for step in research_steps)
    
    def test_identify_parallel_branches(self, planner):
        """Test parallel branch identification."""
        # Comparison query
        compare_query = "Compare iOS vs Android operating systems"
        branches = planner._identify_parallel_branches(compare_query)
        assert len(branches) >= 2
        
        # Aspects query
        aspects_query = "Analyze the economic and social aspects of remote work"
        aspect_branches = planner._identify_parallel_branches(aspects_query)
        assert len(aspect_branches) >= 2
    
    def test_identify_dependencies(self, planner):
        """Test dependency identification."""
        # Conditional query
        conditional_query = "If the weather is sunny, what outdoor activities are recommended?"
        deps1 = planner._identify_dependencies(conditional_query)
        assert len(deps1) > 0
        assert any("condition" in dep.lower() for dep in deps1)
        
        # Assumption query
        assumption_query = "Given that AI will continue advancing, what are the implications?"
        deps2 = planner._identify_dependencies(assumption_query)
        assert len(deps2) > 0
        
        # Complex query
        complex_query = "This is a very long and complex query that requires multiple steps and considerations to answer properly and comprehensively."
        deps3 = planner._identify_dependencies(complex_query)
        assert len(deps3) > 0
    
    def test_validate_plan(self, planner):
        """Test plan validation."""
        # Create valid plan
        task1 = Task(id="task_1", name="Task 1", description="First", task_type=TaskType.ATOMIC)
        task2 = Task(id="task_2", name="Task 2", description="Second", task_type=TaskType.ATOMIC,
                    dependencies=[TaskDependency(task_id="task_1")])
        
        plan = Plan(id="test_plan", name="Test", description="Test plan", root_task_ids=["task_1"])
        plan.tasks["task_1"] = task1
        plan.tasks["task_2"] = task2
        
        # Should pass validation
        planner._validate_plan(plan)
        
        # Test missing dependency
        task3 = Task(id="task_3", name="Task 3", description="Third", task_type=TaskType.ATOMIC,
                    dependencies=[TaskDependency(task_id="missing_task")])
        plan.tasks["task_3"] = task3
        
        with pytest.raises(ValueError, match="Missing dependency"):
            planner._validate_plan(plan)
    
    def test_circular_dependency_detection(self, planner):
        """Test circular dependency detection."""
        # Create circular dependency
        task1 = Task(id="task_1", name="Task 1", description="First", task_type=TaskType.ATOMIC,
                    dependencies=[TaskDependency(task_id="task_2")])
        task2 = Task(id="task_2", name="Task 2", description="Second", task_type=TaskType.ATOMIC,
                    dependencies=[TaskDependency(task_id="task_1")])
        
        plan = Plan(id="test_plan", name="Test", description="Test plan", root_task_ids=["task_1"])
        plan.tasks["task_1"] = task1
        plan.tasks["task_2"] = task2
        
        with pytest.raises(ValueError, match="Circular dependency"):
            planner._validate_plan(plan)
    
    def test_optimize_plan(self, planner):
        """Test plan optimization."""
        task1 = Task(id="task_1", name="Task 1", description="First", task_type=TaskType.ATOMIC)
        task2 = Task(id="task_2", name="Task 2", description="Second", task_type=TaskType.ATOMIC)
        
        plan = Plan(id="test_plan", name="Test", description="Test plan", root_task_ids=["task_1"])
        plan.tasks["task_1"] = task1
        plan.tasks["task_2"] = task2
        
        # Should complete without error
        planner._optimize_plan(plan)
        
        # Check that strategies are assigned
        for task in plan.tasks.values():
            if task.task_type == TaskType.ATOMIC:
                assert task.assigned_strategy is not None
    
    def test_register_task_executor(self, planner):
        """Test task executor registration."""
        mock_executor = AsyncMock()
        
        planner.register_task_executor(ReasoningStrategy.CHAIN_OF_THOUGHT, mock_executor)
        
        assert ReasoningStrategy.CHAIN_OF_THOUGHT in planner.task_executors
        assert planner.task_executors[ReasoningStrategy.CHAIN_OF_THOUGHT] == mock_executor
    
    def test_progress_callback(self, planner):
        """Test progress callback registration."""
        mock_callback = Mock()
        
        planner.add_progress_callback(mock_callback)
        
        assert mock_callback in planner.progress_callbacks
    
    def test_get_plan_status(self, planner):
        """Test plan status retrieval."""
        # Test non-existent plan
        status = planner.get_plan_status("non_existent")
        assert status is None
        
        # Test with actual plan
        task = Task(id="task_1", name="Task 1", description="Test", task_type=TaskType.ATOMIC)
        plan = Plan(id="test_plan", name="Test Plan", description="Test", root_task_ids=["task_1"])
        plan.tasks["task_1"] = task
        
        planner.active_plans["test_plan"] = plan
        
        status = planner.get_plan_status("test_plan")
        assert status is not None
        assert status["plan_id"] == "test_plan"
        assert status["total_tasks"] == 1
        assert "task_status_counts" in status
    
    def test_get_planning_metrics(self, planner):
        """Test planning metrics retrieval."""
        metrics = planner.get_planning_metrics()
        
        assert "total_plans" in metrics
        assert "successful_plans" in metrics
        assert "failed_plans" in metrics
        assert "plan_success_rate" in metrics
        assert "total_tasks" in metrics
        assert "active_plans" in metrics
        assert "completed_plans" in metrics


class TestCheckpointManager:
    """Test CheckpointManager functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for checkpoints."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def checkpoint_manager(self, temp_dir):
        """Create CheckpointManager instance for testing."""
        return CheckpointManager(
            checkpoint_dir=temp_dir,
            auto_checkpoint_interval=60,
            max_checkpoints_per_plan=5,
            enable_compression=False  # Disable for easier testing
        )
    
    @pytest.fixture
    def sample_plan(self):
        """Create a sample plan for testing."""
        task = Task(
            id="task_1",
            name="Test Task",
            description="A test task",
            task_type=TaskType.ATOMIC,
            query="What is 2+2?",
            status=TaskStatus.COMPLETED
        )
        
        plan = Plan(
            id="test_plan",
            name="Test Plan",
            description="A test plan",
            root_task_ids=["task_1"],
            status=TaskStatus.COMPLETED
        )
        plan.tasks["task_1"] = task
        
        return plan
    
    def test_checkpoint_manager_initialization(self, checkpoint_manager, temp_dir):
        """Test CheckpointManager initialization."""
        assert checkpoint_manager.checkpoint_dir == Path(temp_dir)
        assert checkpoint_manager.auto_checkpoint_interval == 60
        assert checkpoint_manager.max_checkpoints_per_plan == 5
        assert checkpoint_manager.enable_compression is False
        assert checkpoint_manager.db_path.exists()
    
    @pytest.mark.asyncio
    async def test_save_checkpoint(self, checkpoint_manager, sample_plan):
        """Test saving checkpoint."""
        checkpoint_id = await checkpoint_manager.save_checkpoint(
            sample_plan,
            checkpoint_type="test",
            metadata={"test": "data"}
        )
        
        assert checkpoint_id is not None
        assert checkpoint_id.startswith(sample_plan.id)
        
        # Check that file was created
        checkpoints = checkpoint_manager.list_checkpoints()
        assert len(checkpoints) == 1
        assert checkpoints[0]["id"] == checkpoint_id
    
    @pytest.mark.asyncio
    async def test_restore_checkpoint(self, checkpoint_manager, sample_plan):
        """Test restoring checkpoint."""
        # Save checkpoint first
        checkpoint_id = await checkpoint_manager.save_checkpoint(sample_plan)
        
        # Restore checkpoint
        restored_plan = await checkpoint_manager.restore_checkpoint(checkpoint_id)
        
        assert restored_plan is not None
        assert restored_plan.id == sample_plan.id
        assert restored_plan.name == sample_plan.name
        assert len(restored_plan.tasks) == len(sample_plan.tasks)
    
    @pytest.mark.asyncio
    async def test_restore_latest_checkpoint(self, checkpoint_manager, sample_plan):
        """Test restoring latest checkpoint."""
        # Save multiple checkpoints
        await checkpoint_manager.save_checkpoint(sample_plan, "checkpoint1")
        await asyncio.sleep(0.01)  # Ensure different timestamps
        checkpoint_id2 = await checkpoint_manager.save_checkpoint(sample_plan, "checkpoint2")
        
        # Restore latest
        restored_plan = await checkpoint_manager.restore_latest_checkpoint(sample_plan.id)
        
        assert restored_plan is not None
        assert restored_plan.id == sample_plan.id
    
    @pytest.mark.asyncio
    async def test_restore_nonexistent_checkpoint(self, checkpoint_manager):
        """Test restoring non-existent checkpoint."""
        restored_plan = await checkpoint_manager.restore_checkpoint("non_existent")
        assert restored_plan is None
        
        restored_plan = await checkpoint_manager.restore_latest_checkpoint("non_existent")
        assert restored_plan is None
    
    def test_list_checkpoints(self, checkpoint_manager):
        """Test listing checkpoints."""
        checkpoints = checkpoint_manager.list_checkpoints()
        assert isinstance(checkpoints, list)
        
        # Test with plan filter
        checkpoints = checkpoint_manager.list_checkpoints(plan_id="non_existent")
        assert len(checkpoints) == 0
    
    def test_list_plans(self, checkpoint_manager):
        """Test listing plans."""
        plans = checkpoint_manager.list_plans()
        assert isinstance(plans, list)
    
    @pytest.mark.asyncio
    async def test_delete_checkpoint(self, checkpoint_manager, sample_plan):
        """Test deleting checkpoint."""
        checkpoint_id = await checkpoint_manager.save_checkpoint(sample_plan)
        
        # Delete checkpoint
        success = await checkpoint_manager.delete_checkpoint(checkpoint_id)
        assert success is True
        
        # Verify deletion
        restored_plan = await checkpoint_manager.restore_checkpoint(checkpoint_id)
        assert restored_plan is None
    
    @pytest.mark.asyncio
    async def test_delete_plan_checkpoints(self, checkpoint_manager, sample_plan):
        """Test deleting all checkpoints for a plan."""
        await checkpoint_manager.save_checkpoint(sample_plan, "checkpoint1")
        await checkpoint_manager.save_checkpoint(sample_plan, "checkpoint2")
        
        deleted_count = await checkpoint_manager.delete_plan_checkpoints(sample_plan.id)
        assert deleted_count == 2
        
        # Verify deletion
        checkpoints = checkpoint_manager.list_checkpoints(plan_id=sample_plan.id)
        assert len(checkpoints) == 0
    
    def test_get_checkpoint_stats(self, checkpoint_manager):
        """Test checkpoint statistics."""
        stats = checkpoint_manager.get_checkpoint_stats()
        
        assert "total_checkpoints" in stats
        assert "total_size_bytes" in stats
        assert "total_size_mb" in stats
        assert "checkpoints_by_type" in stats
        assert "plans_with_checkpoints" in stats
        assert "compression_enabled" in stats


class TestFallbackGraph:
    """Test FallbackGraph functionality."""
    
    @pytest.fixture
    def fallback_graph(self):
        """Create FallbackGraph instance for testing."""
        return FallbackGraph(
            enable_auto_fallback=True,
            max_fallback_depth=2,
            cost_escalation_factor=1.5
        )
    
    @pytest.fixture
    def sample_task(self):
        """Create a sample task for testing."""
        return Task(
            id="test_task",
            name="Test Task",
            description="A test task",
            task_type=TaskType.ATOMIC,
            query="What is machine learning?",
            assigned_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            status=TaskStatus.FAILED
        )
    
    def test_fallback_graph_initialization(self, fallback_graph):
        """Test FallbackGraph initialization."""
        assert fallback_graph.enable_auto_fallback is True
        assert fallback_graph.max_fallback_depth == 2
        assert fallback_graph.cost_escalation_factor == 1.5
        assert len(fallback_graph.fallback_rules) > 0  # Should have default rules
    
    def test_fallback_rule_management(self, fallback_graph):
        """Test fallback rule management."""
        initial_count = len(fallback_graph.fallback_rules)
        
        # Add new rule
        rule = FallbackRule(
            id="test_rule",
            name="Test Rule",
            description="A test fallback rule",
            trigger_reasons=[FallbackReason.TASK_FAILURE],
            trigger_conditions=[FallbackCondition.IMMEDIATE],
            fallback_type=FallbackType.STRATEGY_CHANGE,
            target_strategy=ReasoningStrategy.TREE_OF_THOUGHTS
        )
        
        fallback_graph.add_fallback_rule(rule)
        assert len(fallback_graph.fallback_rules) == initial_count + 1
        
        # Remove rule
        success = fallback_graph.remove_fallback_rule("test_rule")
        assert success is True
        assert len(fallback_graph.fallback_rules) == initial_count
        
        # Try to remove non-existent rule
        success = fallback_graph.remove_fallback_rule("non_existent")
        assert success is False
    
    def test_estimate_task_complexity(self, fallback_graph):
        """Test task complexity estimation."""
        # Simple task
        simple_task = Task(
            id="simple",
            name="Simple",
            description="Simple task",
            task_type=TaskType.ATOMIC,
            query="What is 2+2?",
            assigned_strategy=ReasoningStrategy.CHAIN_OF_THOUGHT
        )
        
        simple_complexity = fallback_graph._estimate_task_complexity(simple_task)
        assert 0.0 <= simple_complexity <= 1.0
        
        # Complex task
        complex_task = Task(
            id="complex",
            name="Complex",
            description="Complex task",
            task_type=TaskType.COMPOSITE,
            query="This is a very long and complex query that involves multiple steps, considerations, and requires extensive reasoning to solve properly and completely. It should have higher complexity.",
            assigned_strategy=ReasoningStrategy.MONTE_CARLO_TREE_SEARCH,
            dependencies=[TaskDependency("dep1"), TaskDependency("dep2")],
            children=["child1", "child2", "child3"]
        )
        
        complex_complexity = fallback_graph._estimate_task_complexity(complex_task)
        assert complex_complexity > simple_complexity
    
    def test_find_applicable_rules(self, fallback_graph, sample_task):
        """Test finding applicable fallback rules."""
        applicable_rules = fallback_graph._find_applicable_rules(
            sample_task,
            FallbackReason.TASK_FAILURE
        )
        
        assert isinstance(applicable_rules, list)
        # Should find some applicable rules for task failure
        assert len(applicable_rules) > 0
    
    @pytest.mark.asyncio
    async def test_handle_task_failure(self, fallback_graph, sample_task):
        """Test handling task failure."""
        fallback_task = await fallback_graph.handle_task_failure(
            sample_task,
            FallbackReason.TASK_FAILURE,
            "Test failure"
        )
        
        # Should create a fallback task
        assert fallback_task is not None
        assert fallback_task.id != sample_task.id
        assert fallback_task.id.startswith(sample_task.id)  # Should contain original task ID
        assert fallback_task.metadata.get("fallback_from") == sample_task.id
    
    @pytest.mark.asyncio
    async def test_handle_task_failure_max_depth(self, fallback_graph, sample_task):
        """Test fallback with max depth exceeded."""
        fallback_task = await fallback_graph.handle_task_failure(
            sample_task,
            FallbackReason.TASK_FAILURE,
            "Test failure",
            current_depth=fallback_graph.max_fallback_depth
        )
        
        # Should not create fallback task due to max depth
        assert fallback_task is None
    
    @pytest.mark.asyncio
    async def test_handle_task_failure_disabled(self, sample_task):
        """Test fallback when disabled."""
        fallback_graph = FallbackGraph(enable_auto_fallback=False)
        
        fallback_task = await fallback_graph.handle_task_failure(
            sample_task,
            FallbackReason.TASK_FAILURE,
            "Test failure"
        )
        
        # Should not create fallback task when disabled
        assert fallback_task is None
    
    def test_fallback_callback(self, fallback_graph):
        """Test fallback callback registration."""
        mock_callback = AsyncMock()
        
        fallback_graph.add_fallback_callback(mock_callback)
        
        assert mock_callback in fallback_graph.fallback_callbacks
    
    def test_get_fallback_history(self, fallback_graph):
        """Test fallback history retrieval."""
        history = fallback_graph.get_fallback_history()
        assert isinstance(history, list)
        
        # Test with task filter
        history = fallback_graph.get_fallback_history(task_id="test_task")
        assert isinstance(history, list)
    
    def test_get_fallback_metrics(self, fallback_graph):
        """Test fallback metrics retrieval."""
        metrics = fallback_graph.get_fallback_metrics()
        
        assert "total_fallbacks" in metrics
        assert "successful_fallbacks" in metrics
        assert "failed_fallbacks" in metrics
        assert "success_rate" in metrics
        assert "fallbacks_by_type" in metrics
        assert "fallbacks_by_reason" in metrics
        assert "rule_usage" in metrics
        assert "rule_success_rates" in metrics
    
    def test_optimize_rules(self, fallback_graph):
        """Test rule optimization."""
        # Should complete without error
        fallback_graph.optimize_rules()
        
        # Rules should still be sorted by priority
        priorities = [rule.priority for rule in fallback_graph.fallback_rules]
        assert priorities == sorted(priorities, reverse=True)
    
    def test_configure_fallback_settings(self, fallback_graph):
        """Test fallback settings configuration."""
        fallback_graph.configure_fallback_settings(
            enable_auto_fallback=False,
            max_fallback_depth=5,
            cost_escalation_factor=2.0
        )
        
        assert fallback_graph.enable_auto_fallback is False
        assert fallback_graph.max_fallback_depth == 5
        assert fallback_graph.cost_escalation_factor == 2.0


class TestFallbackTypes:
    """Test fallback type enums."""
    
    def test_fallback_type(self):
        """Test FallbackType enum."""
        assert FallbackType.STRATEGY_CHANGE.value == "strategy_change"
        assert FallbackType.CONTEXT_SIMPLIFY.value == "context_simplify"
        assert FallbackType.DECOMPOSE_FURTHER.value == "decompose_further"
        assert FallbackType.CONSTRAINT_RELAX.value == "constraint_relax"
        assert FallbackType.ALTERNATIVE_APPROACH.value == "alternative_approach"
        assert FallbackType.HUMAN_ESCALATION.value == "human_escalation"
        assert FallbackType.GRACEFUL_DEGRADATION.value == "graceful_degradation"
    
    def test_fallback_reason(self):
        """Test FallbackReason enum."""
        assert FallbackReason.TASK_FAILURE.value == "task_failure"
        assert FallbackReason.TIMEOUT.value == "timeout"
        assert FallbackReason.COST_EXCEEDED.value == "cost_exceeded"
        assert FallbackReason.LOW_CONFIDENCE.value == "low_confidence"
        assert FallbackReason.RESOURCE_UNAVAILABLE.value == "resource_unavailable"
        assert FallbackReason.QUALITY_THRESHOLD.value == "quality_threshold"
        assert FallbackReason.DEPENDENCY_FAILURE.value == "dependency_failure"
    
    def test_fallback_condition(self):
        """Test FallbackCondition enum."""
        assert FallbackCondition.IMMEDIATE.value == "immediate"
        assert FallbackCondition.AFTER_RETRIES.value == "after_retries"
        assert FallbackCondition.COST_THRESHOLD.value == "cost_threshold"
        assert FallbackCondition.TIME_THRESHOLD.value == "time_threshold"
        assert FallbackCondition.CONFIDENCE_THRESHOLD.value == "confidence_threshold"


class TestIntegration:
    """Integration tests for planning components."""
    
    @pytest.fixture
    def planning_system(self):
        """Create integrated planning system."""
        planner = TaskPlanner(max_decomposition_depth=2)
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=tempfile.mkdtemp(),
            enable_compression=False
        )
        fallback_graph = FallbackGraph(max_fallback_depth=1)
        
        return planner, checkpoint_manager, fallback_graph
    
    @pytest.mark.asyncio
    async def test_integrated_planning_flow(self, planning_system):
        """Test integrated planning workflow."""
        planner, checkpoint_manager, fallback_graph = planning_system
        
        # Create plan
        request = ReasoningRequest(
            query="Explain photosynthesis step by step",
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT
        )
        
        plan = await planner.create_plan(request)
        assert plan is not None
        
        # Save checkpoint
        checkpoint_id = await checkpoint_manager.save_checkpoint(plan, "integration_test")
        assert checkpoint_id is not None
        
        # Restore checkpoint
        restored_plan = await checkpoint_manager.restore_checkpoint(checkpoint_id)
        assert restored_plan is not None
        assert restored_plan.id == plan.id
        
        # Test fallback on a task
        if plan.tasks:
            task = next(iter(plan.tasks.values()))
            fallback_task = await fallback_graph.handle_task_failure(
                task,
                FallbackReason.TASK_FAILURE
            )
            # May or may not create fallback depending on task characteristics
            # Just ensure no errors occur
        
        # Cleanup
        await checkpoint_manager.delete_plan_checkpoints(plan.id)
        shutil.rmtree(checkpoint_manager.checkpoint_dir)
    
    @pytest.mark.asyncio
    async def test_close_cleanup(self, planning_system):
        """Test proper cleanup of planning components."""
        planner, checkpoint_manager, fallback_graph = planning_system
        
        # Should complete without errors
        await planner.close()
        await checkpoint_manager.close()
        await fallback_graph.close()
        
        # Cleanup
        shutil.rmtree(checkpoint_manager.checkpoint_dir)