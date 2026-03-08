# NEXUS-AGI

**NEXUS-AGI** adalah framework AGI (Artificial General Intelligence) modular berbasis Python dengan kemampuan:

- Multi-agent federation dengan orchestrator dan specialist agents
- Recursive Self-Improvement (RSI) loop
- Hierarchical planning dengan MCTS, Tree-of-Thought, Chain-of-Thought
- Episodic, Semantic, dan Working memory dengan vector embeddings
- World model dengan causal reasoning dan prediction engine
- Constitutional AI safety layer
- REST API + WebSocket server (FastAPI)
- CLI interface

## Quick Start

```bash
# Clone
git clone https://github.com/ajul8866/nexus-agi
cd nexus-agi

# Install
pip install -e .

# Configure
cp .env.example .env
# Edit .env with your API keys

# Run CLI
nexus --help
nexus run "Research the latest developments in AI safety"
nexus serve  # Start API server

# Docker
docker-compose up -d
```

## Architecture

```
nexus/
├── kernel.py          # Core event loop & orchestrator
├── agents/            # Multi-agent federation
│   ├── base.py        # AgentBase class
│   ├── orchestrator.py
│   ├── specialist.py
│   └── reflection.py  # Meta-cognition
├── memory/            # Memory systems
│   ├── episodic.py    # Event-based memory
│   ├── semantic.py    # Concept/knowledge memory
│   ├── working.py     # Short-term working memory
│   └── long_term.py   # Persistent storage
├── planning/          # Reasoning & planning
│   ├── hierarchical.py
│   ├── mcts.py        # Monte Carlo Tree Search
│   ├── chain_of_thought.py
│   └── tree_of_thought.py
├── world_model/       # Internal world representation
│   ├── model.py
│   ├── causal.py
│   └── prediction.py
├── tools/             # Tool system
│   ├── registry.py
│   ├── executor.py
│   ├── sandbox.py
│   └── chainer.py
├── rsi/               # Recursive Self-Improvement
│   ├── monitor.py
│   ├── optimizer.py
│   ├── generator.py
│   └── improver.py
├── safety/            # Safety & alignment
│   ├── constitutional.py
│   ├── filter.py
│   ├── alignment.py
│   └── validator.py
├── api/               # REST API server
│   └── server.py
└── cli/               # Command-line interface
    └── main.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| GET | /status | System status |
| POST | /tasks | Submit task |
| GET | /tasks/{id} | Get task status |
| GET | /tasks | List tasks |
| DELETE | /tasks/{id} | Cancel task |
| POST | /memory/query | Query memory |
| GET | /memory/stats | Memory stats |
| GET | /agents | List agents |
| POST | /agents | Create agent |
| WS | /ws | WebSocket stream |

## CLI Commands

```bash
nexus run "task description"     # Run a task
nexus agent list                  # List agents
nexus agent create MyAgent        # Create agent
nexus memory query "search term"  # Query memory
nexus memory stats                # Memory statistics
nexus improve status              # RSI status
nexus improve run --cycles 3      # Run RSI cycles
nexus serve                       # Start API server
nexus version                     # Version info
```

## License

MIT License - Copyright (c) 2024 SULFIKAR
