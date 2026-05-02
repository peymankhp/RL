"""
Reusable persistent notes panel for the RL Learning Portal.

Each section or tab gets:
  section_notes/<section_slug>/note.md
  section_notes/<section_slug>/assets/

The note file is Markdown. Users can write text, LaTeX with $...$ or $$...$$,
and attach images that are inserted as Markdown image references.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
NOTES_ROOT = BASE_DIR / "section_notes"
IMAGE_TYPES = ["png", "jpg", "jpeg", "gif", "webp", "bmp", "svg"]


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "section"


def _unique_path(directory: Path, filename: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._") or "image.png"
    candidate = directory / safe_name
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    idx = 2
    while True:
        candidate = directory / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def _read_note(note_file: Path) -> str:
    try:
        return note_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _reload_note(note_file: Path, draft_key: str) -> None:
    st.session_state[draft_key] = _read_note(note_file)


def _append_to_draft(draft_key: str, content: str) -> None:
    current = st.session_state.get(draft_key, "").rstrip()
    st.session_state[draft_key] = f"{current}\n\n{content.strip()}\n" if current else f"{content.strip()}\n"


def _render_markdown_with_local_images(markdown_text: str, note_dir: Path) -> None:
    """Render Markdown while showing local image references through st.image."""
    image_line = re.compile(r"^\s*!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)\s*$")
    buffer: list[str] = []

    def flush_buffer() -> None:
        if buffer:
            st.markdown("\n".join(buffer))
            buffer.clear()

    for line in markdown_text.splitlines():
        match = image_line.match(line)
        if not match:
            buffer.append(line)
            continue

        flush_buffer()
        raw_path = match.group("path").strip()
        image_path = Path(raw_path)
        if not image_path.is_absolute():
            image_path = note_dir / image_path

        if image_path.exists():
            st.image(str(image_path), caption=match.group("alt") or image_path.name)
        else:
            st.warning(f"Missing image: {raw_path}")

    flush_buffer()


def render_notes(section_title: str, section_slug: str | None = None) -> None:
    """Show the persistent note editor at the end of a section page."""
    slug = section_slug or _slug(section_title)
    note_dir = NOTES_ROOT / slug
    assets_dir = note_dir / "assets"
    note_file = note_dir / "note.md"
    note_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    note_file.touch(exist_ok=True)

    draft_key = f"note_draft_{slug}"
    saved_key = f"note_saved_{slug}"

    if draft_key not in st.session_state:
        st.session_state[draft_key] = _read_note(note_file)

    st.divider()
    st.markdown(
        f"""
        <div style="background:#12121f;border:1px solid #2a2a3e;border-radius:12px;
                    padding:1rem 1.2rem;margin:.5rem 0 1rem">
            <h3 style="color:white;margin:0;font-size:1.2rem">📝 Notes for {section_title}</h3>
            <p style="color:#9e9ebb;margin:.35rem 0 0;font-size:.9rem;line-height:1.6">
            Add text, LaTeX formulas, and images. Saved notes reload automatically next time.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    existing_note = _read_note(note_file)
    with st.expander("Saved note preview", expanded=bool(existing_note.strip())):
        if existing_note.strip():
            _render_markdown_with_local_images(existing_note, note_dir)
        else:
            st.info("No saved note yet.")

    content_type = st.radio(
        "Add content",
        ["Text", "LaTeX formula", "Image"],
        horizontal=True,
        key=f"note_content_type_{slug}",
    )

    if content_type == "Text":
        text_key = f"note_text_piece_{slug}"
        st.text_area(
            "Text",
            key=text_key,
            height=120,
            placeholder="Write a short explanation, definition, or reminder.",
        )
        if st.button("Append text", key=f"note_append_text_{slug}"):
            text = st.session_state.get(text_key, "").strip()
            if text:
                _append_to_draft(draft_key, text)
                st.rerun()
            else:
                st.warning("Write some text before appending.")

    elif content_type == "LaTeX formula":
        latex_key = f"note_latex_piece_{slug}"
        st.text_area(
            "LaTeX formula",
            key=latex_key,
            height=120,
            placeholder=r"V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]",
        )
        if st.button("Append formula", key=f"note_append_latex_{slug}"):
            formula = st.session_state.get(latex_key, "").strip()
            if formula:
                formula = formula.strip("$")
                _append_to_draft(draft_key, f"$${formula}$$")
                st.rerun()
            else:
                st.warning("Write a formula before appending.")

    else:
        uploaded_files = st.file_uploader(
            "Image",
            type=IMAGE_TYPES,
            accept_multiple_files=True,
            key=f"note_upload_{slug}",
        )
        if uploaded_files and st.button("Insert image", key=f"note_insert_images_{slug}"):
            inserted = []
            for uploaded in uploaded_files:
                target = _unique_path(assets_dir, uploaded.name)
                target.write_bytes(uploaded.getbuffer())
                inserted.append(f"![{target.name}](assets/{target.name})")

            _append_to_draft(draft_key, "\n\n".join(inserted))
            st.rerun()

    st.text_area(
        "Write your note",
        key=draft_key,
        height=260,
        placeholder=(
            "Example:\n"
            "Key idea: Bellman optimality backs up the best next action.\n\n"
            "$$V^*(s)=\\max_a \\sum_{s',r}p(s',r|s,a)[r+\\gamma V^*(s')]$$\n"
        ),
    )

    col_save, col_reload, col_path = st.columns([1, 1, 4])
    with col_save:
        if st.button("Save note", type="primary", key=f"note_save_{slug}"):
            note_file.write_text(st.session_state[draft_key], encoding="utf-8")
            st.session_state[saved_key] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.success("Note saved.")
    with col_reload:
        st.button(
            "Reload saved",
            key=f"note_reload_{slug}",
            on_click=_reload_note,
            args=(note_file, draft_key),
        )
    with col_path:
        st.caption(f"Saved to: {note_file}")

    if saved_key in st.session_state:
        st.caption(f"Last saved in this session: {st.session_state[saved_key]}")
