```mermaid
graph TB
    subgraph "Adaptive Computation Controller NEW"
        ACC[Adaptive Controller<br/>Monitors confidence & cost]
        ACC --> AC1[Query Complexity Analysis]
        ACC --> AC2[Confidence Threshold Check]
        ACC --> AC3[Cost-Benefit Calculator]
        ACC --> AC4[Dynamic Path Selection]
    end
    
    subgraph "Hierarchical Planning Layer NEW"
        HP[Master Planner<br/>High-level strategy only]
        HP --> HP1[Problem Decomposition]
        HP --> HP2[Solution Blueprint]
        HP --> HP3[Checkpoint Definition]
        HP --> HP4[Fallback Strategies]
    end
    
    subgraph "Context Variation Paths"
        A[Original Prompt] --> CV[Context Variants]
        CV --> C[Minified: Core only]
        CV --> D[Standard: Original]
        CV --> E[Megafied: Enriched]
        CV --> F[Symbolic: Abstract form NEW]
        CV --> G[Exemplar: With examples NEW]
    end
    
    subgraph "Multi-Model Ensemble NEW"
        C --> M1[Mini Model 1<br/>Fast reasoning]
        D --> M2[Mini Model 2<br/>Balanced approach]
        E --> M3[Mini Model 3<br/>Deep analysis]
        F --> M4[Specialized Model<br/>Logic/Math focus]
        G --> M5[Pattern Model<br/>Example-based]
    end
    
    subgraph "SMART Coaching System NEW"
        SC[Confidence Monitor]
        M1 --> SC
        M2 --> SC
        M3 --> SC
        M4 --> SC
        M5 --> SC
        SC --> SH{Need Help?}
        SH -->|Low Confidence| LH[Large Model Hint<br/>Strategic guidance only]
        LH --> M1
        LH --> M2
        LH --> M3
        LH --> M4
        LH --> M5
    end
    
    subgraph "Tool Orchestra NEW"
        TO[Tool Selector]
        TO --> T1[Python Executor<br/>Precise calculations]
        TO --> T2[Knowledge Base<br/>Fact verification]
        TO --> T3[Logic Verifier<br/>Consistency check]
        TO --> T4[Search Engine<br/>Current info]
    end
    
    subgraph "Self-Proving Module NEW"
        SP[Proof Generator]
        SP --> SP1[Generate Certificate]
        SP --> SP2[Formal Verification]
        SP --> SP3[Constraint Check]
        SP --> SP4[Confidence Score]
    end
    
    subgraph "Monte Carlo Tree Search NEW"
        MCTS[MCTS Controller]
        MCTS --> MC1[Branch Generation]
        MCTS --> MC2[Value Estimation]
        MCTS --> MC3[Path Selection]
        MCTS --> MC4[Backpropagation]
    end
    
    subgraph "Reflexion Memory NEW"
        RM[Episodic Memory]
        RM --> RM1[Past Attempts]
        RM --> RM2[Error Patterns]
        RM --> RM3[Success Strategies]
        RM --> RM4[Lesson Extraction]
    end
    
    subgraph "Cascade Decision Engine"
        CDE[Smart Router]
        M1 --> CDE
        M2 --> CDE
        M3 --> CDE
        M4 --> CDE
        M5 --> CDE
        TO --> CDE
        SP --> CDE
        MCTS --> CDE
        RM --> CDE
        
        CDE --> CD1{Confident?}
        CD1 -->|Yes| FF[Fast Finalize]
        CD1 -->|No| CD2{Worth Escalating?}
        CD2 -->|Yes| ESC[Escalate to Large Model]
        CD2 -->|No| REF[Refine with Tools/Memory]
    end
    
    subgraph "Constitutional Review NEW"
        CR[Principle Checker]
        CR --> CR1[Accuracy Review]
        CR --> CR2[Completeness Check]
        CR --> CR3[Bias Detection]
        CR --> CR4[Safety Validation]
    end
    
    subgraph "Final Assembly"
        FF --> FA[Response Assembler]
        ESC --> FA
        REF --> FA
        CR --> FA
        FA --> OUT[Final Output<br/>+ Reasoning Trace<br/>+ Confidence Map<br/>+ Cost Report]
    end
    
    style ACC fill:#ffcdd2
    style HP fill:#e1bee7
    style SC fill:#fff9c4
    style TO fill:#c5cae9
    style SP fill:#b2dfdb
    style MCTS fill:#d7ccc8
    style RM fill:#ffccbc
    style CR fill:#f8bbd0
    style A fill:#e1f5fe
    style OUT fill:#c8e6c9