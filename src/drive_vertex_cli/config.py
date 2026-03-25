from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class ConfigurationError(RuntimeError):
    """Raised when environment-backed configuration is missing or invalid."""

    pass


@dataclass(slots=True)
class Settings:
    """Runtime configuration loaded from environment variables."""

    google_cloud_project: str
    google_cloud_location: str
    google_application_credentials: Path | None
    gemini_model: str
    gemini_temperature: float
    embedding_model: str
    embedding_dimensions: int | None
    default_top_k: int
    conversation_max_turns: int
    drive_service_account_file: Path | None
    drive_oauth_client_secret_file: Path | None
    drive_token_file: Path
    index_dir: Path
    default_folder_id: str | None


def _read_path(env_name: str) -> Path | None:
    """Return an expanded path from an environment variable, if present."""

    value = os.getenv(env_name)
    if not value:
        return None
    return Path(value).expanduser()


def _read_int(env_name: str, default: int, *, minimum: int | None = None) -> int:
    """Parse an integer setting with optional lower-bound validation."""

    raw_value = os.getenv(env_name)
    if raw_value in {None, ""}:
        value = default
    else:
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ConfigurationError(f"{env_name} must be an integer.") from exc

    if minimum is not None and value < minimum:
        raise ConfigurationError(f"{env_name} must be at least {minimum}.")
    return value


def _read_float(env_name: str, default: float, *, minimum: float | None = None) -> float:
    """Parse a float setting with optional lower-bound validation."""

    raw_value = os.getenv(env_name)
    if raw_value in {None, ""}:
        value = default
    else:
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ConfigurationError(f"{env_name} must be a number.") from exc

    if minimum is not None and value < minimum:
        raise ConfigurationError(f"{env_name} must be at least {minimum}.")
    return value


def load_settings(*, require_project: bool = True) -> Settings:
    """Load the library configuration from the current process environment."""

    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if require_project and not project:
        raise ConfigurationError("GOOGLE_CLOUD_PROJECT is required.")

    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    gemini_model = os.getenv("VERTEX_GEMINI_MODEL", "gemini-2.5-flash")
    gemini_temperature = _read_float("VERTEX_GEMINI_TEMPERATURE", 0.2, minimum=0.0)
    embedding_model = os.getenv("VERTEX_EMBEDDING_MODEL", "text-embedding-005")
    default_top_k = _read_int("DRIVE_VERTEX_DEFAULT_TOP_K", 5, minimum=1)
    conversation_max_turns = _read_int(
        "DRIVE_VERTEX_CONVERSATION_MAX_TURNS",
        6,
        minimum=0,
    )

    raw_dimensions = os.getenv("VERTEX_EMBEDDING_DIMENSIONS")
    if raw_dimensions:
        try:
            embedding_dimensions = int(raw_dimensions)
        except ValueError as exc:
            raise ConfigurationError(
                "VERTEX_EMBEDDING_DIMENSIONS must be an integer."
            ) from exc
    else:
        embedding_dimensions = None

    drive_token_file = _read_path("GOOGLE_DRIVE_TOKEN_FILE") or Path(
        ".secrets/google-drive-token.json"
    )
    index_dir = _read_path("DRIVE_VERTEX_INDEX_DIR") or Path(".cache/drive-vertex-index")

    return Settings(
        google_cloud_project=project or "",
        google_cloud_location=location,
        google_application_credentials=_read_path("GOOGLE_APPLICATION_CREDENTIALS"),
        gemini_model=gemini_model,
        gemini_temperature=gemini_temperature,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        default_top_k=default_top_k,
        conversation_max_turns=conversation_max_turns,
        drive_service_account_file=_read_path("GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE"),
        drive_oauth_client_secret_file=_read_path(
            "GOOGLE_DRIVE_OAUTH_CLIENT_SECRET_FILE"
        ),
        drive_token_file=drive_token_file,
        index_dir=index_dir,
        default_folder_id=os.getenv("GOOGLE_DRIVE_FOLDER_ID"),
    )
