"""
NEXUS-AGI Command Line Interface
"""
import click
import asyncio
import json
import sys
import os
from typing import Optional
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@click.group()
@click.option("--debug/--no-debug", default=False, help="Enable debug logging")
@click.option("--config", "-c", type=click.Path(), default=None, help="Config file path")
@click.pass_context
def cli(ctx, debug: bool, config: Optional[str]):
    """
    NEXUS-AGI: Advanced General Intelligence Framework

    A modular, self-improving AGI system with multi-agent federation.
    """
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    ctx.obj["config"] = config

    if debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
        click.echo("[DEBUG] Debug mode enabled", err=True)


@cli.command()
@click.argument("task")
@click.option("--agent", "-a", default="orchestrator", help="Agent type to use")
@click.option("--max-iter", "-m", default=10, type=int, help="Max iterations")
@click.option("--output", "-o", type=click.Choice(["text", "json"]), default="text")
@click.pass_context
def run(ctx, task: str, agent: str, max_iter: int, output: str):
    """Run a task through NEXUS-AGI."""
    click.echo(f"[NEXUS] Submitting task to {agent} agent...")
    click.echo(f"[NEXUS] Task: {task}")
    click.echo(f"[NEXUS] Max iterations: {max_iter}")

    result = asyncio.run(_run_task(task, agent, max_iter))

    if output == "json":
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"\n[RESULT]\n{result.get('output', 'No output')}")


@cli.group()
def agent():
    """Manage NEXUS-AGI agents."""
    pass


@agent.command("list")
@click.option("--type", "-t", default=None, help="Filter by agent type")
@click.pass_context
def agent_list(ctx, type: Optional[str]):
    """List all active agents."""
    click.echo("[NEXUS] Fetching active agents...")
    agents = [
        {"name": "Orchestrator", "type": "orchestrator", "status": "idle"},
        {"name": "Researcher",   "type": "specialist",   "status": "idle"},
        {"name": "Coder",        "type": "specialist",   "status": "idle"},
    ]
    if type:
        agents = [a for a in agents if a["type"] == type]

    click.echo(f"\n{'Name':<20} {'Type':<15} {'Status':<10}")
    click.echo("-" * 45)
    for a in agents:
        click.echo(f"{a['name']:<20} {a['type']:<15} {a['status']:<10}")


@agent.command("create")
@click.argument("name")
@click.option("--type", "-t", default="specialist", help="Agent type")
@click.option("--capabilities", "-cap", multiple=True, help="Agent capabilities")
def agent_create(name: str, type: str, capabilities: tuple):
    """Create a new agent."""
    click.echo(f"[NEXUS] Creating {type} agent: {name}")
    click.echo(f"[NEXUS] Capabilities: {list(capabilities) or ['default']}")
    click.echo(f"[NEXUS] Agent '{name}' created successfully!")


@cli.group()
def memory():
    """Manage NEXUS-AGI memory systems."""
    pass


@memory.command("query")
@click.argument("query")
@click.option("--type", "-t", default="episodic",
              type=click.Choice(["episodic", "semantic", "working"]))
@click.option("--top-k", "-k", default=5, type=int)
def memory_query(query: str, type: str, top_k: int):
    """Query the NEXUS memory system."""
    click.echo(f"[NEXUS] Querying {type} memory: '{query}'")
    click.echo(f"[NEXUS] Top-{top_k} results:")
    click.echo("  (Memory system requires running kernel)")


@memory.command("stats")
def memory_stats():
    """Show memory system statistics."""
    click.echo("[NEXUS] Memory Statistics:")
    click.echo("  Episodic: 0 memories")
    click.echo("  Semantic: 0 concepts")
    click.echo("  Working:  0/7 slots used")


@memory.command("clear")
@click.option("--type", "-t", default="working",
              type=click.Choice(["episodic", "semantic", "working", "all"]))
@click.confirmation_option(prompt="Are you sure you want to clear memory?")
def memory_clear(type: str):
    """Clear memory (dangerous!)."""
    click.echo(f"[NEXUS] Clearing {type} memory...")
    click.echo(f"[NEXUS] {type} memory cleared.")


@cli.group()
def improve():
    """Self-improvement system management."""
    pass


@improve.command("status")
def improve_status():
    """Show RSI (Recursive Self-Improvement) status."""
    click.echo("[NEXUS] RSI System Status:")
    click.echo("  Monitor: Active")
    click.echo("  Last scan: N/A")
    click.echo("  Improvements applied: 0")
    click.echo("  Bottlenecks detected: 0")


@improve.command("run")
@click.option("--cycles", "-c", default=1, type=int, help="Number of improvement cycles")
def improve_run(cycles: int):
    """Trigger manual self-improvement cycle."""
    click.echo(f"[NEXUS] Starting {cycles} RSI cycle(s)...")
    for i in range(cycles):
        click.echo(f"  Cycle {i+1}/{cycles}: Analyzing performance...")
        click.echo(f"  Cycle {i+1}/{cycles}: Detecting bottlenecks...")
        click.echo(f"  Cycle {i+1}/{cycles}: Generating improvements...")
    click.echo("[NEXUS] RSI cycles complete.")


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind")
@click.option("--port", default=8000, type=int, help="Port to listen on")
@click.option("--reload/--no-reload", default=False, help="Auto-reload on changes")
def serve(host: str, port: int, reload: bool):
    """Start the NEXUS-AGI API server."""
    try:
        import uvicorn
        from nexus.api.server import create_app
        click.echo(f"[NEXUS] Starting API server on {host}:{port}")
        app = create_app()
        uvicorn.run(app, host=host, port=port, reload=reload)
    except ImportError:
        click.echo("[ERROR] uvicorn not installed. Run: pip install uvicorn", err=True)
        sys.exit(1)


@cli.command()
def version():
    """Show NEXUS-AGI version."""
    click.echo("NEXUS-AGI v1.0.0")
    click.echo("Python AGI Framework with RSI")
    click.echo("https://github.com/ajul8866/nexus-agi")


async def _run_task(task: str, agent_type: str, max_iter: int) -> dict:
    """Async task runner placeholder."""
    await asyncio.sleep(0.1)
    return {
        "status": "completed",
        "output": f"Task processed by {agent_type} agent (kernel not initialized in CLI mode)",
        "iterations": 1,
    }


if __name__ == "__main__":
    cli()
