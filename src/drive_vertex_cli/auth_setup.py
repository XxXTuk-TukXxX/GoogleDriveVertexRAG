from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials as ServiceAccountCredentials

from drive_vertex_cli.config import Settings
from drive_vertex_cli.drive_client import build_drive_service, get_folder_status

REQUIRED_SERVICES = (
    "aiplatform.googleapis.com",
    "drive.googleapis.com",
)
GOOGLE_CLOUD_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
EMPTY_PATH_ENV_VARS = (
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE",
    "GOOGLE_DRIVE_OAUTH_CLIENT_SECRET_FILE",
    "GOOGLE_DRIVE_TOKEN_FILE",
    "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE",
)


class AuthSetupError(RuntimeError):
    """Raised when the interactive auth setup cannot complete safely."""

    pass


@dataclass(slots=True)
class AuthSetupSummary:
    """Human-readable outcome of the auth bootstrap flow."""

    completed_steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def apply_env_updates(updates: Mapping[str, str]) -> None:
    """Apply `.env` updates to the current process so follow-up steps see them."""

    for key, value in updates.items():
        if value == "":
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def validate_existing_file(path_value: str, *, label: str) -> Path:
    """Validate that a user-supplied credential file exists and is a regular file."""

    path = Path(path_value).expanduser()
    if not path.exists():
        raise AuthSetupError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise AuthSetupError(f"{label} is not a file: {path}")
    return path


def complete_auth_setup(
    *,
    settings: Settings,
    vertex_auth_mode: str,
    drive_auth_mode: str,
) -> AuthSetupSummary:
    """Create or verify local auth state, enable APIs, and check Drive access."""

    summary = AuthSetupSummary()

    if vertex_auth_mode == "adc":
        _ensure_adc(settings.google_cloud_project, summary)
        access_token = _get_adc_access_token()
    else:
        access_token = _get_service_account_access_token(settings)
        summary.completed_steps.append(
            f"Verified Vertex AI service account file at {settings.google_application_credentials}."
        )

    _enable_required_services(
        project_id=settings.google_cloud_project,
        access_token=access_token,
        services=REQUIRED_SERVICES,
    )
    summary.completed_steps.append(
        "Enabled required Google Cloud APIs: "
        + ", ".join(REQUIRED_SERVICES)
        + "."
    )

    token_existed = settings.drive_token_file.exists()
    service = build_drive_service(settings)
    if drive_auth_mode == "oauth":
        if settings.drive_token_file.exists() and not token_existed:
            summary.completed_steps.append(
                f"Created Google Drive OAuth token at {settings.drive_token_file}."
            )
        else:
            summary.completed_steps.append(
                f"Verified Google Drive OAuth token at {settings.drive_token_file}."
            )
    else:
        summary.completed_steps.append(
            f"Verified Google Drive service account file at {settings.drive_service_account_file}."
        )

    if settings.default_folder_id:
        folder_status = get_folder_status(service, settings.default_folder_id)
        summary.completed_steps.append(
            f"Verified Google Drive folder '{folder_status.name}' is accessible."
        )
        if folder_status.visible_child_count == 0:
            summary.warnings.append(
                "The configured Drive folder is reachable but currently has no visible child items."
            )
    else:
        summary.warnings.append(
            "No default Drive folder ID was provided, so folder accessibility was not verified."
        )

    return summary


def _ensure_adc(project_id: str, summary: AuthSetupSummary) -> None:
    """Ensure Application Default Credentials exist for local Vertex AI usage."""

    _require_gcloud()

    if _adc_available():
        summary.completed_steps.append("Verified existing Application Default Credentials.")
    else:
        _run_command(
            [
                "gcloud",
                "auth",
                "application-default",
                "login",
                f"--project={project_id}",
            ]
        )
        summary.completed_steps.append("Created Application Default Credentials.")

    try:
        _run_command(
            [
                "gcloud",
                "auth",
                "application-default",
                "set-quota-project",
                project_id,
            ]
        )
        summary.completed_steps.append(
            f"Set the ADC quota project to {project_id}."
        )
    except AuthSetupError as exc:
        summary.warnings.append(
            "Could not set the ADC quota project automatically. "
            f"You may need to run `gcloud auth application-default set-quota-project {project_id}` yourself. "
            f"Details: {exc}"
        )


def _adc_available() -> bool:
    """Return whether `gcloud` can currently mint an ADC access token."""

    try:
        result = _run_command(
            ["gcloud", "auth", "application-default", "print-access-token"],
            capture_output=True,
        )
        return bool(result.stdout.strip())
    except AuthSetupError:
        return False


def _get_adc_access_token() -> str:
    """Read a short-lived access token from Application Default Credentials."""

    result = _run_command(
        ["gcloud", "auth", "application-default", "print-access-token"],
        capture_output=True,
    )
    token = result.stdout.strip()
    if not token:
        raise AuthSetupError("Application Default Credentials did not return an access token.")
    return token


def _get_service_account_access_token(settings: Settings) -> str:
    """Mint a Cloud API access token from a configured service account key."""

    if not settings.google_application_credentials:
        raise AuthSetupError(
            "GOOGLE_APPLICATION_CREDENTIALS must point to a service account file for service-account mode."
        )

    credentials = ServiceAccountCredentials.from_service_account_file(
        str(settings.google_application_credentials),
        scopes=[GOOGLE_CLOUD_SCOPE],
    )
    credentials.refresh(Request())
    if not credentials.token:
        raise AuthSetupError("Could not mint an access token from the service account file.")
    return credentials.token


def _enable_required_services(
    *,
    project_id: str,
    access_token: str,
    services: Sequence[str],
) -> None:
    """Enable the Google APIs required by the library for the chosen project."""

    with tempfile.NamedTemporaryFile("w", delete=False) as handle:
        # `gcloud services enable` accepts an access-token file, not a raw token value.
        handle.write(access_token)
        token_path = handle.name

    try:
        _run_command(
            [
                "gcloud",
                "services",
                "enable",
                *services,
                f"--project={project_id}",
                f"--access-token-file={token_path}",
                "--quiet",
            ]
        )
    finally:
        Path(token_path).unlink(missing_ok=True)


def _require_gcloud() -> None:
    """Fail fast when the Google Cloud CLI is unavailable."""

    _run_command(["gcloud", "--version"], capture_output=True)


def _run_command(
    command: Sequence[str],
    *,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and convert common failures into `AuthSetupError`."""

    try:
        return subprocess.run(
            list(command),
            check=True,
            text=True,
            capture_output=capture_output,
            env=_clean_subprocess_env(),
        )
    except FileNotFoundError as exc:
        raise AuthSetupError(f"Required command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        suffix = f": {details}" if details else "."
        raise AuthSetupError(f"Command failed: {' '.join(command)}{suffix}") from exc


def _clean_subprocess_env() -> dict[str, str]:
    """Drop empty credential-path variables that break downstream Google CLIs."""

    env = os.environ.copy()
    for key in EMPTY_PATH_ENV_VARS:
        if env.get(key) == "":
            env.pop(key, None)
    return env
