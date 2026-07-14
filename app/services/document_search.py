"""User-scoped search over uploaded document metadata and extracted text."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, literal, or_

from app.models.database.database import SessionLocal
from app.models.database.file_upload import FileUpload


class DocumentSearchService:
    @staticmethod
    def search(user_id: int, query: str = "", limit: int = 10) -> list[dict[str, Any]]:
        normalized_query = query.strip().lower()
        limit = max(1, min(limit, 1000))

        with SessionLocal() as session:
            extracted_text_column = (
                FileUpload.extracted_text
                if normalized_query
                else literal("").label("extracted_text")
            )
            filters = [FileUpload.user_id == user_id]
            if normalized_query:
                filters.append(
                    or_(
                        func.lower(FileUpload.filename).contains(normalized_query, autoescape=True),
                        func.lower(FileUpload.original_filename).contains(
                            normalized_query, autoescape=True
                        ),
                        func.lower(FileUpload.extracted_text).contains(
                            normalized_query, autoescape=True
                        ),
                    )
                )
            candidate_limit = min(1000, max(100, limit * 10))
            files = (
                session.query(
                    FileUpload.file_id,
                    FileUpload.filename,
                    FileUpload.original_filename,
                    extracted_text_column,
                    FileUpload.mime_type,
                    FileUpload.file_size,
                    FileUpload.upload_date,
                    FileUpload.is_processed,
                )
                .filter(*filters)
                .order_by(FileUpload.upload_date.desc())
                .limit(candidate_limit)
                .all()
            )

            matches: list[dict[str, Any]] = []
            for file in files:
                filename = file.filename or ""
                original_filename = file.original_filename or ""
                extracted_text = file.extracted_text or ""
                searchable = "\n".join((filename, original_filename, extracted_text)).lower()
                if normalized_query and normalized_query not in searchable:
                    continue

                score = 1
                match_types: list[str] = []
                if normalized_query:
                    if normalized_query in filename.lower():
                        score += 2
                        match_types.append("filename")
                    if normalized_query in original_filename.lower():
                        score += 2
                        match_types.append("original_filename")
                    if normalized_query in extracted_text.lower():
                        score += 1
                        match_types.append("content")

                matches.append(
                    {
                        "file_id": file.file_id,
                        "filename": filename,
                        "original_filename": original_filename,
                        "mime_type": file.mime_type,
                        "file_size": file.file_size,
                        "upload_date": file.upload_date.isoformat() if file.upload_date else None,
                        "is_processed": file.is_processed,
                        "match_score": score,
                        "match_type": match_types,
                        "excerpt": DocumentSearchService._excerpt(extracted_text, normalized_query),
                    }
                )

        matches.sort(
            key=lambda item: (item["match_score"], item["upload_date"] or ""),
            reverse=True,
        )
        return matches[:limit]

    @staticmethod
    def _excerpt(text: str, query: str, radius: int = 220) -> str:
        if not text:
            return ""
        if not query:
            return text[: radius * 2].strip()
        index = text.lower().find(query)
        if index < 0:
            return text[: radius * 2].strip()
        start = max(0, index - radius)
        end = min(len(text), index + len(query) + radius)
        prefix = "…" if start else ""
        suffix = "…" if end < len(text) else ""
        return f"{prefix}{text[start:end].strip()}{suffix}"
