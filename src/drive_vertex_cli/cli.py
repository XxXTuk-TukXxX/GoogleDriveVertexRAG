from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from drive_vertex_cli.auth_setup import (
    AuthSetupError,
    apply_env_updates,
    complete_auth_setup,
    validate_existing_file,
)
from drive_vertex_cli.client import DriveVertexChatSession, DriveVertexClient
from drive_vertex_cli.config import ConfigurationError, load_settings
from drive_vertex_cli.drive_client import build_drive_service, list_accessible_folders
from drive_vertex_cli.env_file import read_env_values, upsert_env_file

app = typer.Typer(no_args_is_help=True, help="Index Google Drive content with Vertex AI.")

DEFAULT_ENV_FILE = Path(".env")
CONSOLE = Console()
ERROR_CONSOLE = Console(stderr=True)
AI_DEFAULTS = {
    "VERTEX_GEMINI_MODEL": "gemini-2.5-flash",
    "VERTEX_GEMINI_TEMPERATURE": "0.2",
    "VERTEX_EMBEDDING_MODEL": "text-embedding-005",
    "VERTEX_EMBEDDING_DIMENSIONS": "768",
    "DRIVE_VERTEX_DEFAULT_TOP_K": "5",
    "DRIVE_VERTEX_CONVERSATION_MAX_TURNS": "6",
}
AI_LABELS = {
    "VERTEX_GEMINI_MODEL": "Gemini model",
    "VERTEX_GEMINI_TEMPERATURE": "Gemini temperature",
    "VERTEX_EMBEDDING_MODEL": "Embedding model",
    "VERTEX_EMBEDDING_DIMENSIONS": "Embedding dimensions",
    "DRIVE_VERTEX_DEFAULT_TOP_K": "Default retrieval top-k",
    "DRIVE_VERTEX_CONVERSATION_MAX_TURNS": "Conversation history turns",
}


def _prompt_choice(message: str, *, choices: set[str], default: str) -> str:
    while True:
        value = typer.prompt(message, default=default).strip().lower()
        if value in choices:
            return value
        _print_error(f"Choose one of: {', '.join(sorted(choices))}")


def _prompt_non_empty(message: str, *, default: str = "") -> str:
    while True:
        value = typer.prompt(message, default=default).strip()
        if value:
            return value
        _print_error("This value is required.")


def _console_url(path: str, project_id: str | None) -> str:
    if project_id:
        return f"https://console.cloud.google.com{path}?project={project_id}"
    return f"https://console.cloud.google.com{path}"


def _missing_file_guidance(
    *,
    kind: str,
    project_id: str | None,
    target_path: str,
) -> list[str]:
    if kind == "drive_oauth_client":
        return [
            "Required file missing: Google Drive OAuth client JSON.",
            "To create it in Google Cloud Console:",
            f"1. Enable Google Drive API: {_console_url('/apis/library/drive.googleapis.com', project_id)}",
            f"2. Configure the OAuth consent screen: {_console_url('/auth/branding', project_id)}",
            f"3. If the app is External and still in testing, add yourself under Audience > Test users to avoid the 'Access blocked ... has not completed the Google verification process' error: {_console_url('/auth/audience', project_id)}",
            f"4. Create an OAuth client ID, choose Desktop app, and download the JSON: {_console_url('/apis/credentials', project_id)}",
            f"5. Save the downloaded file to: {target_path}",
            "6. Paste that path here again.",
        ]

    service_account_lines = [
        "Required file missing: service account JSON key.",
        "To create it in Google Cloud Console:",
        f"1. Open Service Accounts: {_console_url('/iam-admin/serviceaccounts', project_id)}",
        "2. Create a service account or select an existing one.",
        "3. Open the Keys tab, choose Add key, then Create new key, then JSON.",
        f"4. Save the downloaded file to: {target_path}",
    ]
    if kind == "drive_service_account":
        service_account_lines.append(
            "5. Share the target Google Drive folder with the service account email, then paste that path here again."
        )
    else:
        service_account_lines.append("5. Paste that path here again.")
    return service_account_lines


def _print_guidance(lines: list[str]) -> None:
    ERROR_CONSOLE.print(
        Panel(
            "\n".join(lines),
            title="How To Continue",
            border_style="yellow",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def _print_error(message: str) -> None:
    ERROR_CONSOLE.print(
        Panel(
            message,
            title="Error",
            border_style="red",
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )


def _print_auth_header(env_file: Path) -> None:
    CONSOLE.print(
        Panel(
            (
                f"[bold]{env_file}[/bold]\n\n"
                "Writes project configuration, creates local credentials when possible, "
                "enables required Google APIs, and verifies Drive access."
            ),
            title="drive-vertex auth",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def _print_ai_header(env_file: Path) -> None:
    CONSOLE.print(
        Panel(
            (
                f"[bold]{env_file}[/bold]\n\n"
                "Shows or updates the AI generation and retrieval settings used by the CLI."
            ),
            title="drive-vertex ai",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def _ai_setting_rows(values: dict[str, str]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for key in (
        "VERTEX_GEMINI_MODEL",
        "VERTEX_GEMINI_TEMPERATURE",
        "VERTEX_EMBEDDING_MODEL",
        "VERTEX_EMBEDDING_DIMENSIONS",
        "DRIVE_VERTEX_DEFAULT_TOP_K",
        "DRIVE_VERTEX_CONVERSATION_MAX_TURNS",
    ):
        value = values[key]
        if key == "VERTEX_EMBEDDING_DIMENSIONS" and not value:
            value = "auto"
        rows.append((AI_LABELS[key], value))
    return rows


def _print_ai_settings(env_file: Path, values: dict[str, str], *, title: str) -> None:
    table = Table(
        title=title,
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    table.add_column("Setting", style="bold cyan")
    table.add_column("Value")
    for label, value in _ai_setting_rows(values):
        table.add_row(label, value)

    CONSOLE.print("")
    CONSOLE.print(
        Panel(
            f"Configuration file: [bold]{env_file}[/bold]",
            title="AI Settings",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )
    CONSOLE.print(table)


def _effective_ai_values(existing: dict[str, str]) -> dict[str, str]:
    return {
        key: existing.get(key, default)
        for key, default in AI_DEFAULTS.items()
    }


def _normalize_embedding_dimensions_value(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"", "auto", "default", "model-default"}:
        return ""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise typer.BadParameter(
            "Embedding dimensions must be an integer or `auto`."
        ) from exc
    if parsed <= 0:
        raise typer.BadParameter("Embedding dimensions must be greater than 0.")
    return str(parsed)


def _normalize_float_value(value: str, *, label: str, minimum: float = 0.0) -> str:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise typer.BadParameter(f"{label} must be a number.") from exc
    if parsed < minimum:
        raise typer.BadParameter(f"{label} must be at least {minimum}.")
    return str(parsed)


def _normalize_int_value(value: str, *, label: str, minimum: int) -> str:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise typer.BadParameter(f"{label} must be an integer.") from exc
    if parsed < minimum:
        raise typer.BadParameter(f"{label} must be at least {minimum}.")
    return str(parsed)


def _prompt_validated(
    message: str,
    *,
    default: str,
    normalizer,
) -> str:
    while True:
        value = typer.prompt(message, default=default).strip()
        try:
            return normalizer(value)
        except typer.BadParameter as exc:
            _print_error(str(exc))


def _prompt_ai_updates(existing: dict[str, str]) -> dict[str, str]:
    current = _effective_ai_values(existing)
    updates: dict[str, str] = {}

    updates["VERTEX_GEMINI_MODEL"] = _prompt_non_empty(
        "Gemini model",
        default=current["VERTEX_GEMINI_MODEL"],
    )
    updates["VERTEX_GEMINI_TEMPERATURE"] = _prompt_validated(
        "Gemini temperature",
        default=current["VERTEX_GEMINI_TEMPERATURE"],
        normalizer=lambda value: _normalize_float_value(
            value,
            label="Gemini temperature",
        ),
    )
    updates["VERTEX_EMBEDDING_MODEL"] = _prompt_non_empty(
        "Embedding model",
        default=current["VERTEX_EMBEDDING_MODEL"],
    )
    updates["VERTEX_EMBEDDING_DIMENSIONS"] = _prompt_validated(
        "Embedding dimensions (number or `auto`)",
        default=current["VERTEX_EMBEDDING_DIMENSIONS"] or "auto",
        normalizer=_normalize_embedding_dimensions_value,
    )
    updates["DRIVE_VERTEX_DEFAULT_TOP_K"] = _prompt_validated(
        "Default retrieval top-k",
        default=current["DRIVE_VERTEX_DEFAULT_TOP_K"],
        normalizer=lambda value: _normalize_int_value(
            value,
            label="Default retrieval top-k",
            minimum=1,
        ),
    )
    updates["DRIVE_VERTEX_CONVERSATION_MAX_TURNS"] = _prompt_validated(
        "Conversation history turns",
        default=current["DRIVE_VERTEX_CONVERSATION_MAX_TURNS"],
        normalizer=lambda value: _normalize_int_value(
            value,
            label="Conversation history turns",
            minimum=0,
        ),
    )
    return updates

def _print_auth_summary(
    *,
    env_file: Path,
    summary,
    folder_id: str | None = None,
) -> None:
    steps_table = Table(
        title="Setup Results",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    steps_table.add_column("Status", style="bold", no_wrap=True)
    steps_table.add_column("Details")
    for step in summary.completed_steps:
        steps_table.add_row("done", step)
    for warning in summary.warnings:
        steps_table.add_row("warning", warning)

    next_step = "run [bold]drive-vertex sync[/bold]"
    if not folder_id:
        next_step = (
            "run [bold]drive-vertex sync[/bold] and choose a Drive folder from the list"
        )

    CONSOLE.print("")
    CONSOLE.print(
        Panel(
            f"Saved configuration to [bold]{env_file}[/bold].",
            title="Configuration Saved",
            border_style="green",
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )
    CONSOLE.print(steps_table)
    CONSOLE.print(
        Panel(
            f"Next step: {next_step}.",
            title="Next Step",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )


def _print_sync_header(folder_id: str, index_dir: Path) -> None:
    details = Table.grid(padding=(0, 2))
    details.add_column(style="bold cyan", no_wrap=True)
    details.add_column()
    details.add_row("Folder ID", folder_id)
    details.add_row("Index Dir", str(index_dir))
    CONSOLE.print(
        Panel(
            details,
            title="drive-vertex sync",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def _print_sync_summary(stats, index_dir: Path) -> None:
    summary_table = Table(
        title="Index Summary",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    summary_table.add_column("Metric", style="bold cyan")
    summary_table.add_column("Value", justify="right")
    summary_table.add_row("Indexed Files", str(stats.indexed_file_count))
    summary_table.add_row("Skipped Files", str(stats.skipped_file_count))
    summary_table.add_row("Chunks", str(stats.chunk_count))
    summary_table.add_row("Index Dir", str(index_dir))
    CONSOLE.print("")
    CONSOLE.print(summary_table)

    if stats.skipped_reasons:
        skipped_lines = "\n".join(
            f"- {reason}" for reason in stats.skipped_reasons[:10]
        )
        if len(stats.skipped_reasons) > 10:
            skipped_lines += (
                f"\n... and {len(stats.skipped_reasons) - 10} more skipped entries."
            )
        CONSOLE.print(
            Panel(
                skipped_lines,
                title="Skipped Files",
                border_style="yellow",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )


def _prompt_path(
    message: str,
    default: str,
    *,
    must_exist: bool = False,
    missing_file_kind: str | None = None,
    project_id: str | None = None,
) -> str:
    guidance_shown = False
    while True:
        value = typer.prompt(message, default=default).strip()
        if not must_exist:
            return value
        try:
            return str(validate_existing_file(value, label=message))
        except AuthSetupError as exc:
            _print_error(str(exc))
            if missing_file_kind and not guidance_shown:
                _print_guidance(
                    _missing_file_guidance(
                        kind=missing_file_kind,
                        project_id=project_id,
                        target_path=str(Path(value).expanduser()),
                    )
                )
                guidance_shown = True


def _persist_selected_folder(folder_id: str) -> None:
    upsert_env_file(DEFAULT_ENV_FILE, {"GOOGLE_DRIVE_FOLDER_ID": folder_id})
    apply_env_updates({"GOOGLE_DRIVE_FOLDER_ID": folder_id})


def _choose_drive_folder(settings) -> str:
    service = build_drive_service(settings)
    folders = list_accessible_folders(service)
    if not folders:
        raise RuntimeError(
            "No Google Drive folders were visible to the authenticated account."
        )

    table = Table(
        title="Available Google Drive Folders",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    table.add_column("#", justify="right", style="bold cyan", no_wrap=True)
    table.add_column("Folder", style="bold")
    table.add_column("Default", style="green", no_wrap=True)
    table.add_column("Open")
    current_index: int | None = None
    for index, folder in enumerate(folders, start=1):
        default_marker = ""
        if folder.folder_id == settings.default_folder_id:
            current_index = index
            default_marker = "current"
        table.add_row(
            str(index),
            folder.name,
            default_marker,
            folder.web_view_link or "",
        )

    CONSOLE.print(table)

    default_index = str(current_index or 1)
    while True:
        choice = typer.prompt("Choose folder number", default=default_index).strip()
        if not choice.isdigit():
            _print_error("Enter a folder number from the list.")
            continue

        selected_index = int(choice)
        if 1 <= selected_index <= len(folders):
            selected_folder = folders[selected_index - 1]
            _persist_selected_folder(selected_folder.folder_id)
            CONSOLE.print(
                Panel(
                    (
                        f"Saved [bold]{selected_folder.name}[/bold] as "
                        f"[bold]GOOGLE_DRIVE_FOLDER_ID[/bold] in [bold]{DEFAULT_ENV_FILE}[/bold]."
                    ),
                    title="Folder Selected",
                    border_style="green",
                    box=box.ROUNDED,
                    padding=(0, 1),
                )
            )
            return selected_folder.folder_id
        _print_error("Enter a folder number from the list.")


def _resolve_sync_folder_id(explicit_folder_id: str | None, settings) -> str:
    if explicit_folder_id:
        return explicit_folder_id

    if sys.stdin.isatty():
        return _choose_drive_folder(settings)

    if settings.default_folder_id:
        return settings.default_folder_id

    raise typer.BadParameter(
        "Run `drive-vertex sync` in a terminal to choose a Drive folder, or set GOOGLE_DRIVE_FOLDER_ID."
    )


@app.command()
def auth(
    env_file: Annotated[Path, typer.Option("--env-file")] = Path(".env"),
) -> None:
    """Configure the CLI, create local credentials, enable APIs, and verify Drive access."""

    existing = read_env_values(env_file)

    _print_auth_header(env_file)

    project = _prompt_non_empty(
        "Google Cloud project ID",
        default=existing.get("GOOGLE_CLOUD_PROJECT", ""),
    )
    location = typer.prompt(
        "Vertex AI location",
        default=existing.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
    ).strip()

    vertex_auth_mode = _prompt_choice(
        "Vertex auth mode [adc/service-account]",
        choices={"adc", "service-account"},
        default="service-account"
        if existing.get("GOOGLE_APPLICATION_CREDENTIALS")
        else "adc",
    )

    google_application_credentials = ""
    if vertex_auth_mode == "service-account":
        google_application_credentials = _prompt_path(
            "Path to the Vertex AI service account JSON",
            existing.get("GOOGLE_APPLICATION_CREDENTIALS", ".secrets/vertex-service-account.json"),
            must_exist=True,
            missing_file_kind="vertex_service_account",
            project_id=project,
        )

    drive_auth_mode = _prompt_choice(
        "Drive auth mode [oauth/service-account]",
        choices={"oauth", "service-account"},
        default="service-account"
        if existing.get("GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE")
        else "oauth",
    )

    drive_service_account_file = ""
    drive_oauth_client_secret_file = ""
    drive_token_file = existing.get(
        "GOOGLE_DRIVE_TOKEN_FILE",
        ".secrets/google-drive-token.json",
    )

    if drive_auth_mode == "oauth":
        drive_oauth_client_secret_file = _prompt_path(
            "Path to the Google Drive OAuth client secret JSON",
            existing.get(
                "GOOGLE_DRIVE_OAUTH_CLIENT_SECRET_FILE",
                ".secrets/google-drive-oauth-client.json",
            ),
            must_exist=True,
            missing_file_kind="drive_oauth_client",
            project_id=project,
        )
        drive_token_file = typer.prompt(
            "Path to the cached Google Drive OAuth token file",
            default=drive_token_file,
        ).strip()
    else:
        drive_service_account_file = _prompt_path(
            "Path to the Google Drive service account JSON",
            existing.get(
                "GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE",
                google_application_credentials or ".secrets/drive-service-account.json",
            ),
            must_exist=True,
            missing_file_kind="drive_service_account",
            project_id=project,
        )

    index_dir = typer.prompt(
        "Local index directory",
        default=existing.get("DRIVE_VERTEX_INDEX_DIR", ".cache/drive-vertex-index"),
    ).strip()

    updates = {
        "GOOGLE_CLOUD_PROJECT": project,
        "GOOGLE_CLOUD_LOCATION": location,
        "GOOGLE_APPLICATION_CREDENTIALS": google_application_credentials,
        "GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE": drive_service_account_file,
        "GOOGLE_DRIVE_OAUTH_CLIENT_SECRET_FILE": drive_oauth_client_secret_file,
        "GOOGLE_DRIVE_TOKEN_FILE": drive_token_file,
        "DRIVE_VERTEX_INDEX_DIR": index_dir,
        "VERTEX_GEMINI_MODEL": existing.get("VERTEX_GEMINI_MODEL", "gemini-2.5-flash"),
        "VERTEX_GEMINI_TEMPERATURE": existing.get("VERTEX_GEMINI_TEMPERATURE", "0.2"),
        "VERTEX_EMBEDDING_MODEL": existing.get(
            "VERTEX_EMBEDDING_MODEL", "text-embedding-005"
        ),
        "VERTEX_EMBEDDING_DIMENSIONS": existing.get(
            "VERTEX_EMBEDDING_DIMENSIONS", "768"
        ),
        "DRIVE_VERTEX_DEFAULT_TOP_K": existing.get("DRIVE_VERTEX_DEFAULT_TOP_K", "5"),
        "DRIVE_VERTEX_CONVERSATION_MAX_TURNS": existing.get(
            "DRIVE_VERTEX_CONVERSATION_MAX_TURNS",
            "6",
        ),
    }
    upsert_env_file(env_file, updates)
    apply_env_updates(updates)

    try:
        settings = load_settings()
        summary = complete_auth_setup(
            settings=settings,
            vertex_auth_mode=vertex_auth_mode,
            drive_auth_mode=drive_auth_mode,
        )
    except (AuthSetupError, ConfigurationError, RuntimeError) as exc:
        ERROR_CONSOLE.print("")
        ERROR_CONSOLE.print(
            Panel(
                f"Saved configuration to [bold]{env_file}[/bold].",
                title="Configuration Saved",
                border_style="yellow",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )
        _print_error(f"Setup failed: {exc}")
        raise typer.Exit(code=1) from exc

    _print_auth_summary(env_file=env_file, summary=summary)


@app.command()
def ai(
    env_file: Annotated[Path, typer.Option("--env-file")] = Path(".env"),
    gemini_model: Annotated[str | None, typer.Option("--gemini-model")] = None,
    embedding_model: Annotated[str | None, typer.Option("--embedding-model")] = None,
    embedding_dimensions: Annotated[
        str | None,
        typer.Option("--embedding-dimensions", help="Integer or `auto`."),
    ] = None,
    temperature: Annotated[float | None, typer.Option("--temperature", min=0.0)] = None,
    default_top_k: Annotated[int | None, typer.Option("--default-top-k", min=1)] = None,
    conversation_max_turns: Annotated[
        int | None,
        typer.Option("--conversation-max-turns", min=0),
    ] = None,
    show: Annotated[
        bool,
        typer.Option("--show", help="Show the effective AI settings without changing them."),
    ] = False,
) -> None:
    """Show or update the AI models and retrieval settings used by the CLI."""

    existing = read_env_values(env_file)
    provided_updates = {
        "VERTEX_GEMINI_MODEL": gemini_model,
        "VERTEX_GEMINI_TEMPERATURE": None if temperature is None else str(temperature),
        "VERTEX_EMBEDDING_MODEL": embedding_model,
        "VERTEX_EMBEDDING_DIMENSIONS": embedding_dimensions,
        "DRIVE_VERTEX_DEFAULT_TOP_K": None if default_top_k is None else str(default_top_k),
        "DRIVE_VERTEX_CONVERSATION_MAX_TURNS": (
            None if conversation_max_turns is None else str(conversation_max_turns)
        ),
    }
    non_null_updates = {key: value for key, value in provided_updates.items() if value is not None}

    if show and not non_null_updates:
        _print_ai_settings(
            env_file,
            _effective_ai_values(existing),
            title="Current AI Settings",
        )
        return

    _print_ai_header(env_file)

    if non_null_updates:
        updates = dict(non_null_updates)
        if "VERTEX_EMBEDDING_DIMENSIONS" in updates:
            updates["VERTEX_EMBEDDING_DIMENSIONS"] = _normalize_embedding_dimensions_value(
                updates["VERTEX_EMBEDDING_DIMENSIONS"]
            )
    else:
        updates = _prompt_ai_updates(existing)

    merged_values = _effective_ai_values(existing)
    merged_values.update(updates)

    upsert_env_file(env_file, updates)
    apply_env_updates(updates)

    try:
        load_settings(require_project=False)
    except ConfigurationError as exc:
        _print_error(str(exc))
        raise typer.Exit(code=1) from exc

    _print_ai_settings(env_file, merged_values, title="Saved AI Settings")


@app.command()
def sync(
    folder_id: Annotated[str | None, typer.Option("--folder-id")] = None,
    recursive: Annotated[bool, typer.Option("--recursive/--no-recursive")] = True,
    index_dir: Annotated[Path | None, typer.Option("--index-dir")] = None,
    chunk_size_tokens: Annotated[int, typer.Option("--chunk-size-tokens")] = 350,
    chunk_overlap_tokens: Annotated[int, typer.Option("--chunk-overlap-tokens")] = 60,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 5,
) -> None:
    """Sync a Google Drive folder into the local vector index."""

    try:
        settings = load_settings()
        chosen_folder_id = _resolve_sync_folder_id(folder_id, settings)
        chosen_index_dir = index_dir or settings.index_dir
        client = DriveVertexClient(settings)

        _print_sync_header(chosen_folder_id, chosen_index_dir)
        with CONSOLE.status(
            "[bold cyan]Indexing Google Drive content...[/bold cyan]",
            spinner="dots",
        ):
            stats = client.sync(
                folder_id=chosen_folder_id,
                index_dir=chosen_index_dir,
                recursive=recursive,
                chunk_size_tokens=chunk_size_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
                batch_size=batch_size,
            )
    except ConfigurationError as exc:
        _print_error(str(exc))
        raise typer.Exit(code=1) from exc
    except RuntimeError as exc:
        _print_error(str(exc))
        raise typer.Exit(code=1) from exc

    _print_sync_summary(stats, chosen_index_dir)


@app.command()
def ask(
    question: Annotated[
        str | None,
        typer.Argument(help="Question to ask against the indexed corpus."),
    ] = None,
    top_k: Annotated[int | None, typer.Option("--top-k", min=1)] = None,
    refresh: Annotated[bool, typer.Option("--refresh")] = False,
    folder_id: Annotated[str | None, typer.Option("--folder-id")] = None,
    index_dir: Annotated[Path | None, typer.Option("--index-dir")] = None,
) -> None:
    """Ask Gemini a question, or start an interactive chat when no question is passed."""

    try:
        settings = load_settings()
        chosen_index_dir = index_dir or settings.index_dir
        client = DriveVertexClient(settings)
        chosen_top_k = top_k or settings.default_top_k
    except ConfigurationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    if question is None:
        if refresh:
            typer.echo("Refreshing local index before answering ...")
            chosen_folder_id = _resolve_sync_folder_id(folder_id, settings)
            client.sync(
                folder_id=chosen_folder_id,
                index_dir=chosen_index_dir,
                recursive=True,
                chunk_size_tokens=350,
                chunk_overlap_tokens=60,
                batch_size=5,
            )

        _run_interactive_chat(
            client.open_chat(
                index_dir=chosen_index_dir,
                model=settings.gemini_model,
                top_k=chosen_top_k,
                temperature=settings.gemini_temperature,
                conversation_max_turns=settings.conversation_max_turns,
            )
        )
        return

    try:
        resolved_folder_id = (
            _resolve_sync_folder_id(folder_id, settings) if refresh else folder_id
        )
        if refresh:
            typer.echo("Refreshing local index before answering ...")

        result = client.ask(
            question,
            top_k=chosen_top_k,
            refresh=refresh,
            folder_id=resolved_folder_id,
            index_dir=chosen_index_dir,
            model=settings.gemini_model,
            temperature=settings.gemini_temperature,
            conversation_max_turns=settings.conversation_max_turns,
        )
    except ConfigurationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    _print_answer(result)


def _run_interactive_chat(session: DriveVertexChatSession) -> None:
    typer.echo("Interactive mode. Type your question and press Enter.")
    typer.echo("Type `exit`, `quit`, or `/exit` to leave.")

    while True:
        try:
            prompt = typer.prompt("you", prompt_suffix="> ").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo("\nExiting.")
            return

        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit", "/exit"}:
            typer.echo("Exiting.")
            return

        try:
            result = session.ask(prompt)
        except RuntimeError as exc:
            typer.echo(f"\nai> {exc}")
            typer.echo("")
            continue
        typer.echo(f"\nai> {result.answer}")
        if result.hits:
            typer.echo("sources:")
            seen_paths: set[str] = set()
            for hit in result.hits:
                if hit.record.drive_path in seen_paths:
                    continue
                seen_paths.add(hit.record.drive_path)
                source = f"- {hit.record.drive_path}"
                if hit.record.web_view_link:
                    source += f" ({hit.record.web_view_link})"
                typer.echo(source)
        typer.echo("")


def _print_answer(result) -> None:
    typer.echo(result.answer)
    if result.hits:
        typer.echo("\nSources:")
        seen_paths: set[str] = set()
        for hit in result.hits:
            if hit.record.drive_path in seen_paths:
                continue
            seen_paths.add(hit.record.drive_path)
            source = f"- {hit.record.drive_path}"
            if hit.record.web_view_link:
                source += f" ({hit.record.web_view_link})"
            typer.echo(source)


@app.command()
def status(index_dir: Annotated[Path | None, typer.Option("--index-dir")] = None) -> None:
    """Show local index metadata."""

    try:
        client = DriveVertexClient.from_env(require_project=False)
        status_info = client.status(index_dir=index_dir)
    except ConfigurationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Index directory: {status_info.index_dir}")
    typer.echo(f"Folder ID: {status_info.manifest.folder_id}")
    typer.echo(f"Synced at: {status_info.manifest.synced_at}")
    typer.echo(f"Files: {status_info.manifest.file_count}")
    typer.echo(f"Chunks: {status_info.manifest.chunk_count}")
    typer.echo(f"Gemini model: {status_info.gemini_model}")
    typer.echo(f"Gemini temperature: {status_info.gemini_temperature}")
    typer.echo(f"Default retrieval top-k: {status_info.default_top_k}")
    typer.echo(f"Conversation history turns: {status_info.conversation_max_turns}")
    typer.echo(f"Configured embedding model: {status_info.configured_embedding_model}")
    typer.echo(
        "Configured embedding dimensions: "
        f"{status_info.configured_embedding_dimensions if status_info.configured_embedding_dimensions is not None else 'auto'}"
    )
    typer.echo(f"Indexed embedding model: {status_info.manifest.embedding_model}")
    typer.echo(f"Indexed embedding dimensions: {status_info.manifest.embedding_dimensions}")
