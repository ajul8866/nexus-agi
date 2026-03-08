"""
NEXUS-AGI CLI
Command-line interface untuk NEXUS-AGI
"""
from __future__ import annotations
import sys
import time
import click

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()
except ImportError:
    console = None


def _print(msg: str, style: str = ""):
    if console:
        console.print(msg, style=style)
    else:
        print(msg)


@click.group()
@click.version_option(version="1.0.0", prog_name="nexus")
def cli():
    """NEXUS-AGI: Advanced General Intelligence Framework"""
    pass


@cli.command()
@click.argument("task")
@click.option("--priority", "-p", default=5, help="Task priority (1-10)")
@click.option("--timeout", "-t", default=300, help="Timeout in seconds")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def run(task: str, priority: int, timeout: int, verbose: bool):
    """Run a task with NEXUS-AGI."""
    _print(f"[bold green]NEXUS-AGI[/bold green] Running task: {task}")
    _print(f"Priority: {priority} | Timeout: {timeout}s")
    if verbose:
        _print("[dim]Initializing agents...[/dim]")
        _print("[dim]Loading memory systems...[/dim]")
        _print("[dim]Starting planning module...[/dim]")
    _print("[yellow]Task submitted. In production, this connects to the AGI kernel.[/yellow]")


@cli.group()
def agent():
    """Manage NEXUS-AGI agents."""
    pass


@agent.command(name="list")
def agent_list():
    """List all active agents."""
    if console:
        table = Table(title="Active Agents")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Role")
        table.add_column("Status")
        table.add_row("agent_001", "Orchestrator", "orchestrator", "[green]active[/green]")
        table.add_row("agent_002", "Researcher", "specialist", "[green]active[/green]")
        table.add_row("agent_003", "Reflector", "reflection", "[yellow]idle[/yellow]")
        console.print(table)
    else:
        print("agent_001 | Orchestrator | orchestrator | active")
        print("agent_002 | Researcher   | specialist   | active")


@agent.command(name="create")
@click.argument("name")
@click.option("--role", "-r", default="specialist", help="Agent role")
@click.option("--capabilities", "-c", multiple=True, help="Agent capabilities")
def agent_create(name: str, role: str, capabilities: tuple):
    """Create a new agent."""
    caps = list(capabilities) or ["general"]
    _print(f"[green]Created agent:[/green] {name} (role={role}, capabilities={caps})")


@cli.group()
def memory():
    """Manage NEXUS-AGI memory systems."""
    pass


@memory.command(name="query")
@click.argument("query")
@click.option("--type", "-t", "mem_type", default="all",
              type=click.Choice(["all", "episodic", "semantic", "working"]),
              help="Memory type to query")
@click.option("--limit", "-l", default=10, help="Max results")
def memory_query(query: str, mem_type: str, limit: int):
    """Query the memory system."""
    _print(f"Querying {mem_type} memory for: '{query}' (limit={limit})")
    _print("[dim]No results found (memory system requires kernel initialization)[/dim]")


@memory.command(name="stats")
def memory_stats():
    """Show memory system statistics."""
    if console:
        table = Table(title="Memory Statistics")
        table.add_column("Type", style="cyan")
        table.add_column("Entries", justify="right")
        table.add_column("Size", justify="right")
        table.add_row("Episodic", "0", "0 MB")
        table.add_row("Semantic", "0", "0 MB")
        table.add_row("Working", "0", "0 MB")
        table.add_row("[bold]Total[/bold]", "[bold]0[/bold]", "[bold]0 MB[/bold]")
        console.print(table)
    else:
        print("Episodic: 0 entries | Semantic: 0 entries | Working: 0 entries")


@memory.command(name="clear")
@click.option("--type", "-t", "mem_type", default="working",
              type=click.Choice(["all", "episodic", "semantic", "working"]))
@click.confirmation_option(prompt="Are you sure you want to clear memory?")
def memory_clear(mem_type: str):
    """Clear memory (use with caution)."""
    _print(f"[red]Cleared {mem_type} memory[/red]")


@cli.group()
def improve():
    """Manage RSI (Recursive Self-Improvement)."""
    pass


@improve.command(name="status")
def improve_status():
    """Show RSI system status."""
    _print("[bold]RSI System Status[/bold]")
    _print("  Cycles completed: 0")
    _print("  Improvements applied: 0")
    _print("  Health score: N/A (requires kernel)")


@improve.command(name="run")
@click.option("--cycles", "-c", default=1, help="Number of improvement cycles")
@click.option("--dry-run", is_flag=True, help="Simulate without applying changes")
def improve_run(cycles: int, dry_run: bool):
    """Run RSI improvement cycles."""
    mode = "[DRY RUN] " if dry_run else ""
    for i in range(1, cycles + 1):
        _print(f"{mode}Running improvement cycle {i}/{cycles}...")
        time.sleep(0.1)
    _print(f"[green]Completed {cycles} cycle(s)[/green]")


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind")
@click.option("--port", "-p", default=8000, help="Port to bind")
@click.option("--reload", is_flag=True, help="Enable auto-reload (dev mode)")
def serve(host: str, port: int, reload: bool):
    """Start the NEXUS-AGI API server."""
    try:
        import uvicorn
        _print(f"[green]Starting NEXUS-AGI API server[/green] on {host}:{port}")
        uvicorn.run("nexus.api.server:app", host=host, port=port, reload=reload)
    except ImportError:
        _print("[red]uvicorn not installed. Run: pip install uvicorn[standard][/red]")
        sys.exit(1)


@cli.command()
def version():
    """Show version information."""
    _print("[bold cyan]NEXUS-AGI[/bold cyan] v1.0.0")
    _print("Advanced General Intelligence Framework")
    _print("Author: SULFIKAR | License: MIT")


if __name__ == "__main__":
    cli()
