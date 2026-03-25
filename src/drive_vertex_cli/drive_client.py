from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from google.auth.credentials import Credentials as BaseCredentials
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from googleapiclient.discovery import Resource, build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow

from drive_vertex_cli.config import Settings

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
SHORTCUT_MIME_TYPE = "application/vnd.google-apps.shortcut"
GOOGLE_APPS_EXPORTS = {
    "application/vnd.google-apps.document": (
        "text/plain",
        ".txt",
    ),
    "application/vnd.google-apps.presentation": (
        "application/pdf",
        ".pdf",
    ),
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsx",
    ),
}


@dataclass(slots=True)
class DriveDocument:
    """A non-folder Drive file discovered during folder traversal."""

    file_id: str
    name: str
    mime_type: str
    modified_time: str | None
    web_view_link: str | None
    drive_path: str


@dataclass(slots=True)
class DriveFolderStatus:
    """Metadata used to verify that a configured folder is reachable."""

    folder_id: str
    name: str
    web_view_link: str | None
    visible_child_count: int


@dataclass(slots=True)
class DriveFolderOption:
    """A folder that can be shown to the user for interactive selection."""

    folder_id: str
    name: str
    web_view_link: str | None


def build_drive_service(settings: Settings) -> Resource:
    """Build an authenticated Google Drive API client from the active settings."""

    credentials = _load_drive_credentials(settings)
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def _load_drive_credentials(settings: Settings) -> BaseCredentials:
    """Load Drive credentials from either a service account or local OAuth cache."""

    if settings.drive_service_account_file:
        return ServiceAccountCredentials.from_service_account_file(
            str(settings.drive_service_account_file), scopes=DRIVE_SCOPES
        )

    if not settings.drive_oauth_client_secret_file:
        raise RuntimeError(
            "Set GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE or GOOGLE_DRIVE_OAUTH_CLIENT_SECRET_FILE."
        )

    token_file = settings.drive_token_file
    token_file.parent.mkdir(parents=True, exist_ok=True)

    credentials = None
    if token_file.exists():
        credentials = Credentials.from_authorized_user_file(str(token_file), DRIVE_SCOPES)

    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())

    if not credentials or not credentials.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            str(settings.drive_oauth_client_secret_file), DRIVE_SCOPES
        )
        credentials = flow.run_local_server(port=0)
        token_file.write_text(credentials.to_json())

    return credentials


def list_documents(
    service: Resource,
    folder_id: str,
    *,
    recursive: bool = True,
) -> list[DriveDocument]:
    """List supported Drive files inside a folder, optionally recursing into subfolders."""

    documents: list[DriveDocument] = []
    _walk_folder(service, folder_id, "", documents, recursive=recursive)
    documents.sort(key=lambda document: document.drive_path.lower())
    return documents


def _walk_folder(
    service: Resource,
    folder_id: str,
    prefix: str,
    documents: list[DriveDocument],
    *,
    recursive: bool,
) -> None:
    page_token: str | None = None
    while True:
        response = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                spaces="drive",
                fields=(
                    "nextPageToken, files("
                    "id, name, mimeType, modifiedTime, webViewLink, "
                    "shortcutDetails(targetId,targetMimeType)"
                    ")"
                ),
                orderBy="folder,name",
                pageToken=page_token,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
            )
            .execute()
        )

        for item in response.get("files", []):
            item_name = item["name"]
            item_path = f"{prefix}/{item_name}" if prefix else item_name
            mime_type = item["mimeType"]
            item_id = item["id"]

            if mime_type == SHORTCUT_MIME_TYPE:
                # Shortcuts are resolved to their target files so the index contains the
                # real document metadata rather than the lightweight pointer object.
                target_id = item.get("shortcutDetails", {}).get("targetId")
                if not target_id:
                    continue
                target = _get_file_metadata(service, target_id)
                mime_type = target["mimeType"]
                item_id = target["id"]
                item = {**target, "name": item_name}

            if mime_type == FOLDER_MIME_TYPE:
                if recursive:
                    _walk_folder(
                        service,
                        item_id,
                        item_path,
                        documents,
                        recursive=recursive,
                    )
                continue

            documents.append(
                DriveDocument(
                    file_id=item_id,
                    name=item_name,
                    mime_type=mime_type,
                    modified_time=item.get("modifiedTime"),
                    web_view_link=item.get("webViewLink"),
                    drive_path=item_path,
                )
            )

        page_token = response.get("nextPageToken")
        if not page_token:
            break


def _get_file_metadata(service: Resource, file_id: str) -> dict[str, Any]:
    """Fetch a small metadata view for a Drive file or shortcut target."""

    return (
        service.files()
        .get(
            fileId=file_id,
            fields="id, name, mimeType, modifiedTime, webViewLink",
            supportsAllDrives=True,
        )
        .execute()
    )


def download_document(service: Resource, document: DriveDocument) -> tuple[str, str, bytes]:
    """Download or export a Drive file into bytes ready for text extraction."""

    export_definition = GOOGLE_APPS_EXPORTS.get(document.mime_type)
    if export_definition:
        # Native Google Workspace documents must be exported before extraction.
        export_mime_type, extension = export_definition
        request = service.files().export_media(
            fileId=document.file_id,
            mimeType=export_mime_type,
        )
        name = _ensure_suffix(document.name, extension)
        mime_type = export_mime_type
    else:
        request = service.files().get_media(
            fileId=document.file_id,
            supportsAllDrives=True,
        )
        name = document.name
        mime_type = document.mime_type

    buffer = BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return name, mime_type, buffer.getvalue()


def _ensure_suffix(name: str, suffix: str) -> str:
    """Append a file suffix when a Drive export did not preserve one."""

    return name if Path(name).suffix else f"{name}{suffix}"


def get_folder_status(service: Resource, folder_id: str) -> DriveFolderStatus:
    """Return metadata for a folder and count the items visible inside it."""

    metadata = (
        service.files()
        .get(
            fileId=folder_id,
            fields="id, name, mimeType, webViewLink",
            supportsAllDrives=True,
        )
        .execute()
    )
    if metadata["mimeType"] != FOLDER_MIME_TYPE:
        raise RuntimeError(f"The provided Drive ID is not a folder: {folder_id}")

    return DriveFolderStatus(
        folder_id=metadata["id"],
        name=metadata["name"],
        web_view_link=metadata.get("webViewLink"),
        visible_child_count=_count_visible_children(service, folder_id),
    )


def _count_visible_children(service: Resource, folder_id: str) -> int:
    """Count children visible to the authenticated principal for a folder."""

    total = 0
    page_token: str | None = None
    while True:
        response = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                spaces="drive",
                fields="nextPageToken, files(id)",
                pageSize=1000,
                pageToken=page_token,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
            )
            .execute()
        )
        total += len(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            return total


def list_accessible_folders(service: Resource) -> list[DriveFolderOption]:
    """List folders visible to the authenticated account for interactive sync selection."""

    folders: list[DriveFolderOption] = []
    seen_ids: set[str] = set()
    page_token: str | None = None

    while True:
        response = (
            service.files()
            .list(
                q=f"mimeType = '{FOLDER_MIME_TYPE}' and trashed = false",
                spaces="drive",
                fields="nextPageToken, files(id,name,webViewLink)",
                pageSize=1000,
                pageToken=page_token,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
            )
            .execute()
        )
        for item in response.get("files", []):
            folder_id = item["id"]
            if folder_id in seen_ids:
                continue
            seen_ids.add(folder_id)
            folders.append(
                DriveFolderOption(
                    folder_id=folder_id,
                    name=item["name"],
                    web_view_link=item.get("webViewLink"),
                )
            )

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    folders.sort(key=lambda folder: (folder.name.lower(), folder.folder_id))
    return folders
