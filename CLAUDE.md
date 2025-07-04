### 🧠 Project Purpose & Vision

We are building an **LLM-based reasoning architecture** that pushes the limits of small models (like GPT-4o Mini) by layering meta-reasoning, memory, dynamic planning, cost-awareness, and tool use. Our goal is to **match or exceed GPT-4-level reasoning performance** using cheaper components orchestrated intelligently.

This is a **research-grade, production-intent project** that:

* Maximizes quality **per dollar/token**
* Encourages deep reasoning over shallow one-shot answers
* Favors modular, extensible architecture built for iteration

If successful, this agent will serve as a reference implementation for a "slow but smarter" LLM reasoning system.

---

### 🧱 Architecture Principles

* **Layered Design**: Clear separation between planning, reasoning, tool use, verification, and final synthesis.
* **Meta-Reasoning First**: Start with a task-type analysis and pick the right strategy (e.g. CoT, ToT, MCTS).
* **Parallelization by Context**: Use Minified, Standard, Enriched, Symbolic, and Exemplar variations in parallel.
* **Tool Use by Default**: Don't guess when you can verify. Call tools (Python, KBs, Search) liberally.
* **Confidence-Aware Routing**: Evaluate outputs and only escalate to more expensive steps when needed.
* **Learning from Experience**: Use Reflexion memory to retain what worked and feed it back into the loop.
* You can look at the whole architecture in `Architechture.md`

---

### 🗂 Code Structure

```
/agents          # Reasoning strategy wrappers (CoT, ToT, SelfAsk, MCTS, etc)
/controllers    # Meta-reasoning, adaptive controller, cost manager
/context         # Prompt generators for each context variation
/models          # API wrappers for GPT-4o Mini, logic model, etc
/tools           # Search, math, code exec, KB calls, etc
/reflection      # Reflexion memory + error pattern learner
/proofs          # Formal verification + certificate logic
/planning        # Task planners, fallback graph, checkpoint engine
/tests           # Pytest suites covering unit + integration
```

Use `pyproject.toml` + `poetry` for dependencies. Follow PEP8 + black formatting + docstrings. Use `pydantic` for schema validation. Prefer `FastAPI` if an API interface is needed.

---

### 🧪 Testing Protocol

* Unit tests **must accompany every new module**. Add to `/tests/` using Pytest.
* Every new logic path must have:

  * ✅ 1 passing test
  * ⚠️ 1 edge case test
  * ❌ 1 failure test
* Tests must be run via poetry based project-specific venv.
* Add test coverage summaries to PRs.

---

### 🧩 Workflow

* All tasks must be reflected in `TASK.md`. If starting a new thread or conversation, reference the task.
* Update `PLANNING.md` with major architectural or strategy changes.
* Use `README.md` only for setup + high-level overview — deep reasoning details go in `PLANNING.md`.
* Use inline comments where the **why** is non-obvious.

---

### ⚙️ Agent Behavior Expectations

* Agents should **prefer slow, verified answers** to fast but shallow responses.
* When confidence is low, escalate to coach model (larger LLM) or reroute.
* Always produce **reasoning traces** and **confidence maps** for downstream auditability.
* Reflect after failure. Save lessons to Reflexion memory.

---

### ✅ Done = Documented

A feature isn’t done unless:

* It's tested
* It's listed in `TASK.md`
* It's referenced in `PLANNING.md`
* The README reflects any changes to how the project is used or deployed

---

### 🔐 AI Behavior & Constraints

* Don’t invent tools or APIs. Only use real packages / verified endpoints.
* Don’t silently skip steps. Fail loudly if unsure.
* Never mutate existing files or logic unless the task explicitly says so.
* Always ask for clarification when context is missing.

---

### 🧠 You Are Here to Think, Not Just Generate

Claude (or any LLM) used in this repo should:

* Prioritize **reasoning over regurgitation**
* Use tools when needed
* Reflect after mistakes
* Keep cost in mind

We are **building intelligence, not autocomplete.**
