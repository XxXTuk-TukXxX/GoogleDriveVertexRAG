"""Public library exports for Drive Vertex."""

from drive_vertex_cli.auth_setup import AuthSetupError, AuthSetupSummary
from drive_vertex_cli.client import DriveVertexChatSession, DriveVertexClient, DriveVertexStatus
from drive_vertex_cli.config import ConfigurationError, Settings
from drive_vertex_cli.drive_client import DriveFolderOption, DriveFolderStatus
from drive_vertex_cli.index_store import ChunkRecord, IndexManifest, LocalIndex, SearchHit
from drive_vertex_cli.indexer import SyncStats
from drive_vertex_cli.retrieval import RetrievalAnswer

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AuthSetupError",
    "AuthSetupSummary",
    "ChunkRecord",
    "ConfigurationError",
    "DriveFolderOption",
    "DriveFolderStatus",
    "DriveVertexChatSession",
    "DriveVertexClient",
    "DriveVertexStatus",
    "IndexManifest",
    "LocalIndex",
    "RetrievalAnswer",
    "SearchHit",
    "Settings",
    "SyncStats",
]
