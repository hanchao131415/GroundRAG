"""Filesystem-backed document registry for the demo knowledge base."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path


ALLOWED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt", ".xlsx"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    source: str
    department: str
    size: int
    modified_at: float

    def as_dict(self) -> dict:
        return {
            "id": self.document_id,
            "source": self.source,
            "department": self.department,
            "size": self.size,
            "modified_at": self.modified_at,
            "status": "ready",
        }


def _document_id(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def _safe_name(filename: str) -> str:
    name = Path(filename or "").name
    if not name or name in {".", ".."}:
        raise ValueError("filename is required")
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"unsupported file type: {suffix or 'none'}")
    return name


def _department_for(source: str) -> str:
    parts = Path(source).parts
    return parts[0] if len(parts) > 1 else "公共"


def list_documents(root: str, allowed_departments: set[str] | None = None) -> list[DocumentRecord]:
    base = Path(root).resolve()
    if not base.exists():
        return []
    records = []
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        source = path.relative_to(base).as_posix()
        department = _department_for(source)
        if allowed_departments is not None and "*" not in allowed_departments and department not in allowed_departments:
            continue
        stat = path.stat()
        records.append(DocumentRecord(_document_id(source), source, department, stat.st_size, stat.st_mtime))
    return records


def resolve_document(root: str, document_id: str) -> tuple[Path, DocumentRecord] | None:
    for record in list_documents(root):
        if record.document_id != document_id:
            continue
        return Path(root).resolve() / Path(record.source), record
    return None


def save_upload(root: str, filename: str, content: bytes, department: str) -> DocumentRecord:
    name = _safe_name(filename)
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("file exceeds 20 MB limit")
    if not content:
        raise ValueError("file is empty")
    if (not department or any(char in department for char in '<>:"/\\|?*')
            or department in {".", ".."}):
        raise ValueError("invalid department")
    base = Path(root).resolve()
    target_dir = base / department
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / name
    if target.exists():
        raise FileExistsError(name)
    temp = target.with_name(f".{target.name}.uploading")
    temp.write_bytes(content)
    os.replace(temp, target)
    source = target.relative_to(base).as_posix()
    stat = target.stat()
    return DocumentRecord(_document_id(source), source, department, stat.st_size, stat.st_mtime)


def delete_document(root: str, document_id: str) -> DocumentRecord:
    resolved = resolve_document(root, document_id)
    if resolved is None:
        raise FileNotFoundError(document_id)
    path, record = resolved
    path.unlink()
    return record
