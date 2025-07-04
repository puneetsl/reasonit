name: "LLM-Based Reasoning Architecture: Complete Implementation PRP"
description: |

## Purpose
Build a complete LLM-based reasoning architecture that matches or exceeds GPT-4-level performance using orchestrated smaller models (GPT-4o Mini) through layered meta-reasoning, memory, dynamic planning, cost-awareness, and tool use.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Follow all rules in CLAUDE.md

---

## Goal
Create a production-ready "slow but smarter" LLM reasoning system that demonstrates how orchestrated smaller models can achieve GPT-4-level reasoning performance at a fraction of the cost. The system should implement all components from the Architecture.md diagram with full testing, documentation, and iterative refinement capabilities.

## Why
- **Cost Efficiency**: Achieve GPT-4-level reasoning at 30x lower cost using GPT-4o Mini
- **Research Impact**: Serve as reference implementation for multi-agent reasoning architectures
- **Modularity**: Demonstrate layered design principles for extensible AI systems
- **Performance**: Match or exceed GPT-4 on reasoning benchmarks through orchestration

## What
A complete Python-based reasoning architecture implementing:
- **Adaptive Computation Controller**: Confidence-based routing and cost management
- **Multi-Agent Orchestration**: CoT, ToT, MCTS, Self-Ask reasoning strategies
- **Context Variation Engine**: Minified, Standard, Enriched, Symbolic, Exemplar prompts
- **Tool Orchestra**: Python execution, knowledge base, search, verification
- **Reflexion Memory**: Episodic memory with error pattern learning
- **Smart Coaching System**: Large model hints for small model uncertainty
- **Constitutional Review**: Bias detection and safety validation

### Success Criteria
- [ ] Complete architecture matching Architecture.md diagram
- [ ] All reasoning strategies (CoT, ToT, MCTS, Self-Ask) implemented
- [ ] Adaptive controller routes based on confidence and cost
- [ ] Reflexion memory learns from failures
- [ ] Tool integration working (Python, search, verification)
- [ ] 90%+ test coverage with comprehensive validation
- [ ] Cost tracking and optimization working
- [ ] Documentation complete with examples

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Critical for understanding architecture patterns
- file: CLAUDE.md
  why: Project principles, architecture requirements, testing protocols
  
- file: Architecture.md
  why: Complete system diagram showing all components and connections
  
- file: DeepResearch.md
  why: Latest research on LLM orchestration, CoT, ToT, MCTS, Reflexion patterns
  
- url: https://ai.pydantic.dev/agents/
  why: Pydantic AI agent patterns for implementation
  
- url: https://ai.pydantic.dev/multi-agent-applications/
  why: Multi-agent orchestration patterns
  
- url: https://python.langchain.com/docs/concepts/agents
  why: Agent frameworks and tool integration patterns
  
- url: https://arxiv.org/abs/2201.11903
  why: Chain-of-Thought prompting paper - foundational technique
  
- url: https://arxiv.org/abs/2305.10601
  why: Tree of Thoughts paper - deliberative reasoning
  
- url: https://arxiv.org/abs/2303.11366
  why: Reflexion paper - iterative improvement and memory
  
- url: https://llm-mcts.github.io/
  why: MCTS with LLMs implementation guidance
  
- url: https://github.com/kyegomez/tree-of-thoughts
  why: Tree of Thoughts Python implementation examples
  
- url: https://github.com/noahshinn/reflexion
  why: Reflexion framework implementation
  
- url: https://docs.anthropic.com/claude/docs/constitutional-ai
  why: Constitutional AI for safety and bias detection
```

### Current Codebase tree
```bash
/Users/puneetl/Development/per/reasonit/
├── CLAUDE.md                    # Project instructions and principles
├── Architecture.md             # Complete system architecture diagram
├── DeepResearch.md             # Latest research on LLM orchestration
├── PRPs/                       # Product Requirements Prompts
│   ├── templates/
│   │   └── prp_base.md        # PRP template
│   └── EXAMPLE_multi_agent_prp.md
├── examples/                   # Empty directory for examples
└── README.md                   # Project overview
```

### Desired Codebase tree with files to be added
```bash
/Users/puneetl/Development/per/reasonit/
├── pyproject.toml               # Poetry configuration with dependencies
├── poetry.lock                 # Dependency lock file
├── .env.example                # Environment variables template
├── agents/                     # Reasoning strategy implementations
│   ├── __init__.py
│   ├── base_agent.py          # Base agent class with common functionality
│   ├── cot_agent.py           # Chain of Thought reasoning
│   ├── tot_agent.py           # Tree of Thoughts reasoning
│   ├── mcts_agent.py          # Monte Carlo Tree Search reasoning
│   ├── self_ask_agent.py      # Self-Ask reasoning
│   └── reflexion_agent.py     # Reflexion-based iterative improvement
├── controllers/               # Meta-reasoning and orchestration
│   ├── __init__.py
│   ├── adaptive_controller.py # Main orchestration controller
│   ├── cost_manager.py        # Cost tracking and optimization
│   ├── confidence_monitor.py  # Confidence assessment and routing
│   └── cascade_router.py      # Decision engine for model selection
├── context/                   # Prompt generators and context variations
│   ├── __init__.py
│   ├── context_generator.py   # Base context generation
│   ├── variants.py            # Minified, Standard, Enriched, Symbolic, Exemplar
│   └── prompt_templates.py    # Reusable prompt templates
├── models/                    # LLM API wrappers and configurations
│   ├── __init__.py
│   ├── base_model.py          # Base LLM interface
│   ├── openai_wrapper.py      # OpenAI GPT-4o Mini wrapper
│   ├── anthropic_wrapper.py   # Claude wrapper for comparison
│   └── model_ensemble.py      # Multi-model coordination
├── tools/                     # External tool integrations
│   ├── __init__.py
│   ├── python_executor.py     # Python code execution
│   ├── search_tool.py         # Web search integration
│   ├── knowledge_base.py      # Knowledge base queries
│   ├── calculator.py          # Mathematical calculations
│   └── verifier.py            # Solution verification
├── reflection/                # Reflexion memory and learning
│   ├── __init__.py
│   ├── episodic_memory.py     # Memory storage and retrieval
│   ├── error_analyzer.py      # Error pattern detection
│   ├── success_patterns.py    # Success strategy extraction
│   └── lesson_learner.py      # Insight generation from experience
├── proofs/                    # Formal verification and certificates
│   ├── __init__.py
│   ├── proof_generator.py     # Generate formal proofs
│   ├── verifier.py            # Verify proof correctness
│   └── certificate_manager.py # Manage proof certificates
├── planning/                  # Task planning and checkpoint management
│   ├── __init__.py
│   ├── master_planner.py      # High-level task decomposition
│   ├── checkpoint_manager.py  # Save and restore reasoning states
│   └── fallback_strategies.py # Backup reasoning approaches
├── config/                    # Configuration management
│   ├── __init__.py
│   ├── settings.py            # Application settings
│   └── model_configs.py       # Model-specific configurations
├── tests/                     # Comprehensive test suite
│   ├── __init__.py
│   ├── test_agents/           # Agent-specific tests
│   ├── test_controllers/      # Controller tests
│   ├── test_tools/            # Tool integration tests
│   ├── test_reflection/       # Memory and learning tests
│   ├── test_integration/      # End-to-end integration tests
│   └── test_benchmarks/       # Performance benchmark tests
├── examples/                  # Usage examples and demos
│   ├── __init__.py
│   ├── basic_reasoning.py     # Simple reasoning examples
│   ├── complex_math.py        # Mathematical reasoning demo
│   ├── code_generation.py     # Code generation example
│   └── research_task.py       # Research and analysis demo
├── cli.py                     # Command-line interface
├── main.py                    # Main application entry point
└── benchmarks/                # Performance evaluation
    ├── __init__.py
    ├── gsm8k_eval.py          # Math reasoning evaluation
    ├── humaneval_eval.py      # Code generation evaluation
    └── mmlu_eval.py           # General knowledge evaluation
```

### Known Gotchas & Library Quirks
```python
# CRITICAL: OpenAI API rate limits and cost management
# - GPT-4o Mini: 150 req/min, $0.15/1M input tokens, $0.60/1M output tokens
# - Must implement exponential backoff for rate limiting
# - Track token usage for cost optimization

# CRITICAL: Pydantic AI async throughout
# - All agent methods must be async
# - Use RunContext for dependency injection
# - Pass ctx.usage for token tracking in multi-agent calls

# CRITICAL: Memory management for long conversations
# - Implement sliding window with max capacity
# - Use vector embeddings for semantic memory search
# - Consider memory consolidation strategies

# CRITICAL: Tool integration patterns
# - Tools must be registered with @agent.tool decorator
# - Handle tool execution errors gracefully
# - Implement timeouts for external API calls

# CRITICAL: Reflexion memory format
# - Store attempt_id, strategy, outcome, reflection, lessons
# - Use JSON serialization for persistence
# - Implement memory retrieval by similarity

# CRITICAL: Constitutional AI implementation
# - Define clear principles for bias detection
# - Implement critique generation and revision loops
# - Handle principle conflicts gracefully
```

## Implementation Blueprint

### Data models and structure

Core data models for type safety and consistency:
```python
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Literal
from datetime import datetime
from enum import Enum

class ReasoningStrategy(str, Enum):
    CHAIN_OF_THOUGHT = "cot"
    TREE_OF_THOUGHTS = "tot"
    MONTE_CARLO_TREE_SEARCH = "mcts"
    SELF_ASK = "self_ask"
    REFLEXION = "reflexion"

class ContextVariant(str, Enum):
    MINIFIED = "minified"
    STANDARD = "standard"
    ENRICHED = "enriched"
    SYMBOLIC = "symbolic"
    EXEMPLAR = "exemplar"

class ReasoningRequest(BaseModel):
    query: str = Field(..., description="The question or task to reason about")
    strategy: Optional[ReasoningStrategy] = None
    context_variant: ContextVariant = ContextVariant.STANDARD
    max_cost: Optional[float] = Field(None, description="Maximum cost in dollars")
    max_time: Optional[int] = Field(None, description="Maximum time in seconds")
    use_tools: bool = True
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)

class ReasoningStep(BaseModel):
    step_id: str
    strategy: ReasoningStrategy
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    cost: float = Field(ge=0.0)
    timestamp: datetime = Field(default_factory=datetime.now)
    tools_used: List[str] = []

class ReasoningResult(BaseModel):
    request: ReasoningRequest
    final_answer: str
    reasoning_trace: List[ReasoningStep]
    total_cost: float
    total_time: float
    confidence_score: float
    strategies_used: List[ReasoningStrategy]
    reflection: Optional[str] = None
    
class MemoryEntry(BaseModel):
    entry_id: str
    query: str
    strategy: ReasoningStrategy
    outcome: Literal["success", "failure", "partial"]
    reflection: str
    lessons: List[str]
    timestamp: datetime = Field(default_factory=datetime.now)
    
class ToolResult(BaseModel):
    tool_name: str
    input_data: Dict[str, Any]
    output_data: Any
    success: bool
    error_message: Optional[str] = None
    execution_time: float
```

### List of tasks to be completed in order

```yaml
Task 1: Project Setup and Configuration
CREATE pyproject.toml:
  - PATTERN: Use Poetry for dependency management per CLAUDE.md
  - DEPENDENCIES: pydantic, pydantic-ai, openai, anthropic, httpx, pytest, ruff, mypy
  - INCLUDE: Development dependencies for testing and linting

CREATE .env.example:
  - INCLUDE: All API keys, model configurations, feature flags
  - FOLLOW: Security best practices from CLAUDE.md

Task 2: Core Models and Base Classes
CREATE models/base_model.py:
  - PATTERN: Abstract base class for all LLM wrappers
  - INCLUDE: Token counting, cost tracking, rate limiting
  - IMPLEMENT: Async methods with proper error handling

CREATE models/openai_wrapper.py:
  - PATTERN: OpenAI API client with exponential backoff
  - IMPLEMENT: GPT-4o Mini integration with cost tracking
  - HANDLE: Rate limits and API errors gracefully

Task 3: Base Agent Framework
CREATE agents/base_agent.py:
  - PATTERN: Pydantic AI agent with RunContext support
  - INCLUDE: Common functionality for all reasoning agents
  - IMPLEMENT: Tool registration, confidence assessment, cost tracking

Task 4: Context Generation System
CREATE context/context_generator.py:
  - PATTERN: Generate 5 context variants (Minified, Standard, Enriched, Symbolic, Exemplar)
  - IMPLEMENT: Prompt transformation algorithms
  - OPTIMIZE: Context length for cost efficiency

CREATE context/prompt_templates.py:
  - PATTERN: Reusable templates for each reasoning strategy
  - INCLUDE: Few-shot examples from DeepResearch.md
  - IMPLEMENT: Template parameterization

Task 5: Tool Integration Framework
CREATE tools/python_executor.py:
  - PATTERN: Safe Python code execution with timeout
  - IMPLEMENT: Sandboxed execution environment
  - HANDLE: Security constraints and error recovery

CREATE tools/search_tool.py:
  - PATTERN: Web search integration (DuckDuckGo or similar)
  - IMPLEMENT: Result processing and summarization
  - HANDLE: Rate limiting and API errors

Task 6: Reasoning Strategy Implementations
CREATE agents/cot_agent.py:
  - PATTERN: Chain-of-Thought with self-consistency from DeepResearch.md
  - IMPLEMENT: Step-by-step reasoning with intermediate verification
  - INCLUDE: Multiple reasoning paths with majority voting

CREATE agents/tot_agent.py:
  - PATTERN: Tree of Thoughts with BFS/DFS exploration
  - IMPLEMENT: Thought generation, evaluation, and backtracking
  - OPTIMIZE: Beam search for efficiency

CREATE agents/mcts_agent.py:
  - PATTERN: Monte Carlo Tree Search from research papers
  - IMPLEMENT: Selection, expansion, simulation, backpropagation
  - INCLUDE: Value estimation and UCB selection

CREATE agents/self_ask_agent.py:
  - PATTERN: Self-Ask with follow-up question decomposition
  - IMPLEMENT: Question generation and tool integration
  - HANDLE: Multi-hop reasoning with external lookups

Task 7: Reflexion Memory System
CREATE reflection/episodic_memory.py:
  - PATTERN: JSON-based memory storage with vector search
  - IMPLEMENT: Memory consolidation and retrieval
  - INCLUDE: Similarity search for relevant experiences

CREATE reflection/error_analyzer.py:
  - PATTERN: Pattern detection in failed attempts
  - IMPLEMENT: Error classification and root cause analysis
  - GENERATE: Actionable insights for improvement

CREATE agents/reflexion_agent.py:
  - PATTERN: Iterative improvement with memory integration
  - IMPLEMENT: Reflection generation and strategy adjustment
  - CYCLE: Attempt -> Evaluate -> Reflect -> Improve

Task 8: Adaptive Controller Implementation
CREATE controllers/adaptive_controller.py:
  - PATTERN: Main orchestration controller from Architecture.md
  - IMPLEMENT: Query complexity analysis and confidence thresholds
  - ROUTE: Dynamic path selection based on cost-benefit analysis

CREATE controllers/cost_manager.py:
  - PATTERN: Token usage tracking and budget management
  - IMPLEMENT: Real-time cost monitoring and optimization
  - ALERT: Budget threshold warnings and hard limits

CREATE controllers/confidence_monitor.py:
  - PATTERN: Confidence assessment across multiple strategies
  - IMPLEMENT: Uncertainty quantification and escalation triggers
  - DECIDE: When to invoke large model hints

Task 9: Smart Coaching System
CREATE controllers/cascade_router.py:
  - PATTERN: SMART coaching from DeepResearch.md
  - IMPLEMENT: Selective large model consultation
  - OPTIMIZE: Hint generation without full problem solving

Task 10: Constitutional Review System
CREATE proofs/proof_generator.py:
  - PATTERN: Constitutional AI principles from research
  - IMPLEMENT: Bias detection and safety validation
  - GENERATE: Formal verification where applicable

Task 11: Planning and Checkpoint System
CREATE planning/master_planner.py:
  - PATTERN: Hierarchical task decomposition
  - IMPLEMENT: Problem breakdown and solution blueprints
  - MANAGE: Checkpoint creation and restoration

Task 12: CLI and Main Application
CREATE cli.py:
  - PATTERN: Rich CLI with streaming responses
  - IMPLEMENT: Interactive reasoning sessions
  - DISPLAY: Tool usage visibility and cost tracking

CREATE main.py:
  - PATTERN: FastAPI application if needed
  - IMPLEMENT: REST API for reasoning requests
  - INCLUDE: Health checks and metrics endpoints

Task 13: Comprehensive Testing
CREATE tests/ structure:
  - PATTERN: PyTest with fixtures and mocks
  - IMPLEMENT: Unit tests for each component
  - INCLUDE: Integration tests for full workflows
  - ACHIEVE: 90%+ test coverage

Task 14: Benchmarking and Evaluation
CREATE benchmarks/:
  - PATTERN: Standard reasoning benchmarks (GSM8K, HumanEval, MMLU)
  - IMPLEMENT: Performance comparison with GPT-4
  - MEASURE: Cost efficiency and accuracy metrics

Task 15: Documentation and Examples
CREATE examples/:
  - PATTERN: End-to-end usage examples
  - IMPLEMENT: Common reasoning scenarios
  - DEMONSTRATE: All system capabilities
```

### Per task pseudocode

```python
# Task 2: Core Models Implementation
class BaseLLMWrapper:
    async def generate(self, prompt: str, **kwargs) -> str:
        # PATTERN: Async generation with token counting
        start_time = time.time()
        
        # CRITICAL: Implement rate limiting
        await self.rate_limiter.acquire()
        
        try:
            # PATTERN: API call with exponential backoff
            response = await self._make_api_call(prompt, **kwargs)
            
            # PATTERN: Track usage and cost
            self.usage_tracker.add_usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                cost=self._calculate_cost(response.usage)
            )
            
            return response.content
            
        except RateLimitError:
            # PATTERN: Exponential backoff retry
            await self._exponential_backoff()
            return await self.generate(prompt, **kwargs)
        
        except Exception as e:
            # PATTERN: Structured error handling
            logger.error(f"LLM generation failed: {e}")
            raise LLMGenerationError(f"Failed to generate response: {e}")

# Task 5: Tool Integration Pattern
@agent.tool
async def execute_python_code(ctx: RunContext, code: str) -> str:
    """Execute Python code safely with timeout and sandboxing."""
    # PATTERN: Sandboxed execution with security constraints
    sandbox = PythonSandbox(
        timeout=30,
        memory_limit_mb=128,
        restricted_imports=['os', 'sys', 'subprocess']
    )
    
    try:
        # PATTERN: Safe execution with capture
        result = await sandbox.execute(code)
        
        # PATTERN: Update context with tool usage
        ctx.usage.tool_calls += 1
        
        return f"Execution successful:\n{result.stdout}"
        
    except TimeoutError:
        return "Code execution timed out after 30 seconds"
    except Exception as e:
        return f"Code execution failed: {e}"

# Task 6: Chain of Thought Implementation
class ChainOfThoughtAgent:
    async def reason(self, query: str, context: RunContext) -> ReasoningResult:
        # PATTERN: Multi-path reasoning with self-consistency
        reasoning_paths = []
        
        for i in range(self.num_paths):
            # PATTERN: Generate independent reasoning chain
            prompt = self._build_cot_prompt(query, context.variant)
            
            response = await self.llm.generate(
                prompt,
                temperature=0.7,  # CRITICAL: Higher temp for diversity
                max_tokens=1000
            )
            
            # PATTERN: Extract reasoning steps
            steps = self._parse_reasoning_steps(response)
            reasoning_paths.append(steps)
        
        # PATTERN: Self-consistency via majority vote
        final_answer = self._majority_vote([path.final_answer for path in reasoning_paths])
        
        # PATTERN: Confidence based on consensus
        confidence = self._calculate_consensus_confidence(reasoning_paths)
        
        return ReasoningResult(
            final_answer=final_answer,
            reasoning_trace=reasoning_paths[0],  # Best path
            confidence_score=confidence,
            strategies_used=[ReasoningStrategy.CHAIN_OF_THOUGHT]
        )

# Task 7: Reflexion Memory Integration
class ReflexionAgent:
    async def reason_with_memory(self, query: str) -> ReasoningResult:
        # PATTERN: Retrieve relevant past experiences
        similar_experiences = await self.memory.retrieve_similar(query, top_k=5)
        
        # PATTERN: Incorporate lessons learned
        enhanced_prompt = self._build_prompt_with_lessons(query, similar_experiences)
        
        # PATTERN: Attempt reasoning
        attempt_result = await self.base_agent.reason(enhanced_prompt)
        
        # PATTERN: Evaluate and reflect
        if attempt_result.confidence_score < self.reflection_threshold:
            # PATTERN: Generate reflection
            reflection = await self._generate_reflection(query, attempt_result)
            
            # PATTERN: Store experience
            memory_entry = MemoryEntry(
                entry_id=str(uuid.uuid4()),
                query=query,
                strategy=ReasoningStrategy.REFLEXION,
                outcome="partial" if attempt_result.confidence_score < 0.5 else "success",
                reflection=reflection,
                lessons=self._extract_lessons(reflection)
            )
            
            await self.memory.store(memory_entry)
            
            # PATTERN: Retry with reflection
            if attempt_result.confidence_score < 0.5:
                return await self.reason_with_memory(query)  # Recursive improvement
        
        return attempt_result

# Task 8: Adaptive Controller Pattern
class AdaptiveController:
    async def route_query(self, request: ReasoningRequest) -> ReasoningResult:
        # PATTERN: Query complexity analysis
        complexity = await self._analyze_complexity(request.query)
        
        # PATTERN: Cost-benefit calculation
        if complexity.estimated_cost > request.max_cost:
            # PATTERN: Fallback to simpler strategy
            strategy = self._select_cost_effective_strategy(complexity)
        else:
            # PATTERN: Optimal strategy selection
            strategy = self._select_optimal_strategy(complexity)
        
        # PATTERN: Dynamic agent selection
        agent = self.agent_registry[strategy]
        
        # PATTERN: Execute with monitoring
        result = await agent.reason(request.query)
        
        # PATTERN: Confidence-based escalation
        if result.confidence_score < request.confidence_threshold:
            # PATTERN: Escalate to more capable strategy
            fallback_strategy = self._select_fallback_strategy(strategy)
            fallback_agent = self.agent_registry[fallback_strategy]
            
            result = await fallback_agent.reason(request.query)
        
        return result
```

### Integration Points
```yaml
ENVIRONMENT:
  - add to: .env
  - vars: |
      # LLM Configuration
      OPENAI_API_KEY=sk-...
      ANTHROPIC_API_KEY=sk-...
      
      # Model Settings
      PRIMARY_MODEL=gpt-4o-mini
      FALLBACK_MODEL=gpt-4
      
      # Cost Management
      MAX_DAILY_COST=10.0
      CONFIDENCE_THRESHOLD=0.7
      
      # Memory Configuration
      MEMORY_MAX_ENTRIES=10000
      MEMORY_CONSOLIDATION_INTERVAL=3600
      
      # Tool Configuration
      ENABLE_PYTHON_EXECUTION=true
      ENABLE_WEB_SEARCH=true
      SEARCH_API_KEY=...

CONFIG:
  - Poetry setup: poetry install && poetry shell
  - Environment: cp .env.example .env && edit API keys
  - Memory: Creates ./memory/ directory for persistence
  
DEPENDENCIES:
  - Core: pydantic, pydantic-ai, openai, anthropic
  - Tools: httpx, requests, python-sandbox
  - Development: pytest, ruff, mypy, black
  - Optional: streamlit for UI, fastapi for API
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
ruff check --fix                 # Auto-fix style issues
ruff format                      # Format code consistently
mypy --strict                    # Type checking with strict mode

# Expected: No errors. If errors, READ and fix methodically.
```

### Level 2: Unit Tests
```python
# test_agents/test_cot_agent.py
async def test_cot_basic_reasoning():
    """Test basic chain of thought reasoning"""
    agent = ChainOfThoughtAgent()
    result = await agent.reason("What is 2 + 2?")
    assert result.final_answer == "4"
    assert result.confidence_score > 0.8
    assert len(result.reasoning_trace) > 0

async def test_cot_self_consistency():
    """Test self-consistency voting mechanism"""
    agent = ChainOfThoughtAgent(num_paths=5)
    result = await agent.reason("Complex math problem")
    assert result.confidence_score > 0.6
    assert len(result.strategies_used) == 1

# test_controllers/test_adaptive_controller.py
async def test_adaptive_routing():
    """Test adaptive controller routing logic"""
    controller = AdaptiveController()
    request = ReasoningRequest(
        query="Simple arithmetic",
        max_cost=0.01,
        confidence_threshold=0.8
    )
    
    result = await controller.route_query(request)
    assert result.total_cost <= request.max_cost
    assert result.confidence_score >= request.confidence_threshold

# test_reflection/test_memory.py
async def test_memory_storage_retrieval():
    """Test memory storage and retrieval"""
    memory = EpisodicMemory()
    entry = MemoryEntry(
        entry_id="test-1",
        query="Test query",
        strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
        outcome="success",
        reflection="Worked well",
        lessons=["Use step-by-step approach"]
    )
    
    await memory.store(entry)
    retrieved = await memory.retrieve_similar("Similar test query", top_k=1)
    assert len(retrieved) == 1
    assert retrieved[0].entry_id == "test-1"

# test_integration/test_full_workflow.py
async def test_complete_reasoning_workflow():
    """Test full end-to-end reasoning workflow"""
    system = ReasoningSystem()
    
    request = ReasoningRequest(
        query="If I have 15 apples and give away 1/3, then buy 8 more, how many do I have?",
        strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
        use_tools=True
    )
    
    result = await system.process_request(request)
    
    assert result.final_answer == "18"
    assert result.confidence_score > 0.8
    assert result.total_cost < 0.05  # Should be very cheap with GPT-4o Mini
    assert len(result.reasoning_trace) > 2
```

```bash
# Run tests iteratively until all pass:
pytest tests/ -v --cov=agents --cov=controllers --cov=tools --cov=reflection --cov-report=term-missing

# Target: 90%+ coverage. If failing, debug specific tests and fix code.
```

### Level 3: Integration Test
```bash
# Test CLI interface
python cli.py

# Expected interaction:
# > What is the capital of France?
# 🤖 [CoT] Let me think through this step by step...
# 🤖 The capital of France is Paris.
# 📊 Cost: $0.001 | Confidence: 0.95 | Strategy: Chain of Thought

# Test complex reasoning
python cli.py --strategy tot --max-cost 0.10

# > Solve this logic puzzle: If all birds can fly, and penguins are birds, but penguins cannot fly, what's wrong with the premise?
# 🤖 [ToT] Exploring multiple reasoning paths...
# 🤖 The premise "all birds can fly" is incorrect. Penguins are birds that cannot fly.
# 📊 Cost: $0.045 | Confidence: 0.92 | Strategy: Tree of Thoughts

# Test benchmarking
python benchmarks/gsm8k_eval.py --num-samples 100

# Expected: 
# GSM8K Evaluation Results:
# - Accuracy: 85.3%
# - Average Cost: $0.012 per problem
# - Average Time: 8.2 seconds
# - GPT-4 Comparison: 82.1% accuracy at $0.30 per problem
```

### Level 4: Performance Validation
```bash
# Run benchmark suite
python benchmarks/run_all_benchmarks.py

# Expected Results:
# GSM8K (Math): 85%+ accuracy, <$0.02 per problem
# HumanEval (Code): 80%+ accuracy, <$0.05 per problem  
# MMLU (General): 75%+ accuracy, <$0.01 per problem
# Overall: Match or exceed GPT-4 performance at 20x lower cost

# Memory efficiency test
python tests/test_memory_efficiency.py

# Expected: Handle 1000+ queries without memory leaks
```

## Final Validation Checklist
- [ ] All tests pass: `pytest tests/ -v`
- [ ] No linting errors: `ruff check`
- [ ] No type errors: `mypy --strict`
- [ ] Benchmark results meet targets (85%+ accuracy, 20x cost reduction)
- [ ] Memory system learns from failures
- [ ] Tool integration works correctly
- [ ] CLI provides rich interactive experience
- [ ] All reasoning strategies implemented and working
- [ ] Adaptive controller routes optimally
- [ ] Cost tracking accurate and enforced
- [ ] Documentation complete with examples
- [ ] Code follows CLAUDE.md principles

---

## Anti-Patterns to Avoid
- ❌ Don't hardcode API keys - use environment variables
- ❌ Don't use sync functions in async agent context
- ❌ Don't skip token counting and cost tracking
- ❌ Don't ignore confidence thresholds in routing
- ❌ Don't forget to implement memory consolidation
- ❌ Don't use blocking operations in async tools
- ❌ Don't skip validation of tool outputs
- ❌ Don't ignore rate limiting for external APIs
- ❌ Don't commit sensitive data or credentials
- ❌ Don't skip error handling in complex workflows

## Research Integration Notes
Based on DeepResearch.md findings:
- **Self-Consistency**: Implement multiple reasoning paths with majority voting
- **SMART Coaching**: Use large model hints only when small model confidence is low
- **Tool Integration**: Emphasize ReAct pattern for tool use
- **Iterative Refinement**: Implement Reflexion-style memory and improvement loops
- **Cost Optimization**: Target 20-30x cost reduction while maintaining accuracy
- **Constitutional AI**: Implement safety and bias detection throughout

## Performance Targets
- **Cost Efficiency**: Achieve GPT-4-level reasoning at 20x lower cost
- **Accuracy**: 85%+ on GSM8K, 80%+ on HumanEval, 75%+ on MMLU
- **Speed**: Average 10-15 seconds per complex reasoning task
- **Memory**: Learn from failures and improve over time
- **Scalability**: Handle 1000+ concurrent reasoning sessions

## Confidence Score: 9/10

High confidence due to:
- Comprehensive research foundation from DeepResearch.md
- Clear architecture specification in Architecture.md
- Proven patterns from recent academic papers
- Established Python frameworks (Pydantic AI, OpenAI)
- Detailed implementation blueprint with specific tasks
- Comprehensive validation gates and benchmarks
- Experience with similar multi-agent systems

Minor uncertainty around:
- Optimal hyperparameter tuning for each reasoning strategy
- Memory consolidation performance at scale
- Integration complexity between all components

The PRP provides sufficient context and structure for one-pass implementation success.