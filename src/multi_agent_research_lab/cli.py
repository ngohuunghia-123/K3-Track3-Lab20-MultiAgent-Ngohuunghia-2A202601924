"""Command-line entrypoint for the lab starter."""

import time
from typing import Annotated

import typer
from dotenv import load_dotenv
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient

load_dotenv()  # Nạp các biến môi trường từ file .env vào os.environ

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline using the real LLM."""

    _init()
    settings = get_settings()
    request = _parse_query(query)
    state = ResearchState(request=request)

    system_prompt = (
        "You are a research assistant. Given a research query, write a comprehensive, "
        "well-structured answer of 300-500 words. Include key concepts, mechanisms, "
        "trade-offs, and practical considerations. Be factual and precise."
    )
    user_prompt = f"Research query: {request.query}"

    client = LLMClient(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
    )

    start = time.perf_counter()
    response = client.complete(system_prompt=system_prompt, user_prompt=user_prompt)
    latency = time.perf_counter() - start

    state.final_answer = response.content

    console.print(
        Panel.fit(
            state.final_answer,
            title="[bold green]Single-Agent Baseline[/bold green]",
        )
    )

    # Show metrics table
    table = Table(title="Baseline Metrics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")
    table.add_row("Latency", f"{latency:.2f}s")
    table.add_row("Input tokens", str(response.input_tokens or "N/A"))
    table.add_row("Output tokens", str(response.output_tokens or "N/A"))
    cost_str = f"${response.cost_usd:.6f}" if response.cost_usd else "N/A"
    table.add_row("Estimated cost", cost_str)
    console.print(table)


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow skeleton."""

    _init()
    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
