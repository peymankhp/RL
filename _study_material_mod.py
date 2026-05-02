"""
Study material workspace for the RL Learning Portal.

Books are discovered from filesystem folders at render time. Links are stored in
JSON so edits survive app reloads and restarts.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import json
import re
from urllib.parse import urlparse

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
PRIMARY_BOOKS_DIR = BASE_DIR / "StudyMaterial" / "Books"
LEGACY_BOOKS_DIR = BASE_DIR / "Study Material" / "Books"
DATA_DIR = BASE_DIR / "portal_data"
LINKS_FILE = DATA_DIR / "study_links.json"
STATUS_OPTIONS = ["Not start", "In progress", "Done"]


def _ensure_storage() -> None:
    PRIMARY_BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not LINKS_FILE.exists():
        LINKS_FILE.write_text("[]", encoding="utf-8")


def _read_json(path: Path, fallback):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return fallback


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _is_valid_url(value: str) -> bool:
    if not value:
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _normalise_link_rows(rows: list[dict]) -> tuple[list[dict], list[str]]:
    cleaned = []
    errors = []
    for idx, row in enumerate(rows, start=1):
        name = str(row.get("Name", "") or "").strip()
        link_type = str(row.get("Type", "") or "").strip()
        status = str(row.get("Status", "") or "Not start").strip()
        author = str(row.get("Author", "") or "").strip()
        url = str(row.get("Link", "") or "").strip()
        started = row.get("Started", "")

        if not any([name, link_type, author, url, str(started or "").strip()]):
            continue

        if not name:
            errors.append(f"Row {idx}: Name is required.")
        if status not in STATUS_OPTIONS:
            errors.append(f"Row {idx}: Status must be one of {', '.join(STATUS_OPTIONS)}.")

        try:
            score = int(row.get("Score", 1) or 1)
        except (TypeError, ValueError):
            score = 0
        if score < 1 or score > 5:
            errors.append(f"Row {idx}: Score must be between 1 and 5.")

        started_text = ""
        if pd.notna(started) and str(started).strip():
            if isinstance(started, (datetime, date, pd.Timestamp)):
                started_text = started.date().isoformat() if hasattr(started, "date") else started.isoformat()
            else:
                try:
                    started_text = pd.to_datetime(started).date().isoformat()
                except (TypeError, ValueError):
                    errors.append(f"Row {idx}: Started must be a valid date.")

        if not _is_valid_url(url):
            errors.append(f"Row {idx}: Link must be a valid http(s) URL.")

        cleaned.append(
            {
                "Name": name,
                "Type": link_type,
                "Status": status,
                "Score": score,
                "Author": author,
                "Started": started_text,
                "Link": url,
            }
        )

    return cleaned, errors


def _load_links_dataframe() -> pd.DataFrame:
    records = _read_json(LINKS_FILE, [])
    columns = ["Name", "Type", "Status", "Score", "Author", "Started", "Link"]
    df = pd.DataFrame(records, columns=columns)
    for col in columns:
        if col not in df:
            df[col] = ""
    if df.empty:
        df = pd.DataFrame([{col: "" for col in columns}])
        df.loc[0, "Status"] = "Not start"
        df.loc[0, "Score"] = 1
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce").fillna(1).astype(int)
    df["Started"] = pd.to_datetime(df["Started"], errors="coerce").dt.date
    return df[columns]


def _book_title(path: Path) -> str:
    title = path.stem.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", title).strip().title()


def _discover_books() -> list[dict]:
    folders = [PRIMARY_BOOKS_DIR]
    if LEGACY_BOOKS_DIR != PRIMARY_BOOKS_DIR:
        folders.append(LEGACY_BOOKS_DIR)

    books = []
    seen = set()
    for folder in folders:
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
            if not path.is_file() or path.name.startswith("."):
                continue
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            stat = path.stat()
            books.append(
                {
                    "title": _book_title(path),
                    "filename": path.name,
                    "type": path.suffix.lstrip(".").upper() or "FILE",
                    "size_mb": stat.st_size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d"),
                    "path": path,
                    "folder": folder.name,
                }
            )
    return books


def _render_books() -> None:
    books = _discover_books()
    st.markdown("### Books")
    st.caption(f"Drop files into `{PRIMARY_BOOKS_DIR}` and refresh the app to list them automatically.")

    if not books:
        st.info("No books found yet.")
        return

    summary = pd.DataFrame(
        [
            {
                "Title": b["title"],
                "Filename": b["filename"],
                "Type": b["type"],
                "Size": f"{b['size_mb']:.1f} MB",
                "Modified": b["modified"],
                "Folder": b["folder"],
            }
            for b in books
        ]
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

    cols = st.columns(3)
    for idx, book in enumerate(books):
        with cols[idx % 3]:
            with st.container(border=True):
                st.markdown(f"**{book['title']}**")
                st.caption(f"{book['filename']} · {book['type']} · {book['size_mb']:.1f} MB")
                st.caption(f"Modified {book['modified']}")
                st.download_button(
                    "Download",
                    data=book["path"].read_bytes(),
                    file_name=book["filename"],
                    key=f"book_download_{idx}_{book['filename']}",
                    use_container_width=True,
                )


def _render_links() -> None:
    st.markdown("### Links")
    st.caption("Edit the table, add rows with the plus control, delete rows from the table menu, then press Save.")

    if "study_links_df" not in st.session_state:
        st.session_state.study_links_df = _load_links_dataframe()

    edited = st.data_editor(
        st.session_state.study_links_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Name": st.column_config.TextColumn("Name", required=True),
            "Type": st.column_config.TextColumn("Type"),
            "Status": st.column_config.SelectboxColumn("Status", options=STATUS_OPTIONS, required=True),
            "Score": st.column_config.NumberColumn("Score", min_value=1, max_value=5, step=1, format="%d"),
            "Author": st.column_config.TextColumn("Author"),
            "Started": st.column_config.DateColumn("Started"),
            "Link": st.column_config.LinkColumn("Link", validate=r"^https?://.+", required=True),
        },
        key="study_links_editor",
    )

    preview_rows, preview_errors = _normalise_link_rows(edited.to_dict("records"))
    if preview_rows:
        stars = pd.DataFrame(preview_rows)
        stars["Score"] = stars["Score"].apply(lambda value: "★" * int(value) + "☆" * (5 - int(value)))
        st.dataframe(stars, use_container_width=True, hide_index=True)

    col_save, col_reload = st.columns([1, 1])
    with col_save:
        if st.button("Save links", type="primary", use_container_width=True):
            rows, errors = _normalise_link_rows(edited.to_dict("records"))
            if errors:
                for error in errors:
                    st.error(error)
            else:
                _write_json(LINKS_FILE, rows)
                st.session_state.study_links_df = pd.DataFrame(rows) if rows else _load_links_dataframe()
                st.success("Links saved.")
    with col_reload:
        if st.button("Reload saved links", use_container_width=True):
            st.session_state.study_links_df = _load_links_dataframe()
            st.rerun()

    if preview_errors:
        st.warning("Some rows need fixes before saving.")


def main_study_material() -> None:
    _ensure_storage()
    st.markdown(
        """
        <div style="background:#12121f;border:1px solid #2a2a3e;border-radius:14px;
                    padding:1.4rem 1.6rem;margin-bottom:1rem">
            <h2 style="color:white;margin:0;font-size:1.6rem">📚 Study Material</h2>
            <p style="color:#9e9ebb;margin:.4rem 0 0">
                Books are filesystem-driven. Links are editable and saved locally.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_books, tab_links = st.tabs(["Books", "Links"])
    with tab_books:
        _render_books()
    with tab_links:
        _render_links()
