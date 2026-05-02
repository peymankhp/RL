"""
Persistent discussion board for the RL Learning Portal.

Posts are Markdown documents stored in JSON. Uploaded images are saved as files
and inserted into the post body with local Markdown image references.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import re
import uuid

import streamlit as st

from _notes_mod import IMAGE_TYPES, _render_markdown_with_local_images, _unique_path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "portal_data"
POSTS_FILE = DATA_DIR / "discussion_posts.json"
ASSETS_DIR = DATA_DIR / "discussion_assets"


def _ensure_storage() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    if not POSTS_FILE.exists():
        POSTS_FILE.write_text("[]", encoding="utf-8")


def _read_posts() -> list[dict]:
    try:
        records = json.loads(POSTS_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    return records if isinstance(records, list) else []


def _write_posts(posts: list[dict]) -> None:
    POSTS_FILE.write_text(json.dumps(posts, indent=2, ensure_ascii=False), encoding="utf-8")


def _empty_draft() -> dict:
    return {"id": "", "title": "", "body": "", "created_at": "", "updated_at": ""}


def _set_discussion_draft(post: dict | None = None) -> None:
    st.session_state.discussion_draft = dict(post or _empty_draft())


def _append_to_body(text: str) -> None:
    draft = st.session_state.get("discussion_draft", _empty_draft())
    current = draft.get("body", "").rstrip()
    draft["body"] = f"{current}\n\n{text.strip()}\n" if current else f"{text.strip()}\n"
    st.session_state.discussion_draft = draft


def _relative_asset(path: Path) -> str:
    return path.relative_to(DATA_DIR).as_posix()


def _post_summary(body: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "[image]", body or "")
    text = re.sub(r"\$+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:220] + ("..." if len(text) > 220 else "")


def _render_editor(posts: list[dict]) -> None:
    if "discussion_draft" not in st.session_state:
        _set_discussion_draft()

    post_options = {"New post": None}
    post_options.update({f"{post.get('title') or 'Untitled'} · {post.get('updated_at', '')}": post for post in posts})

    selected_label = st.selectbox("Post", list(post_options.keys()), key="discussion_selected_label")
    selected_post = post_options[selected_label]

    col_load, col_new = st.columns([1, 1])
    with col_load:
        if st.button("Load selected", use_container_width=True):
            _set_discussion_draft(selected_post)
            st.rerun()
    with col_new:
        if st.button("New blank post", use_container_width=True):
            _set_discussion_draft()
            st.rerun()

    draft = st.session_state.discussion_draft
    with st.form("discussion_editor_form", clear_on_submit=False):
        title = st.text_input("Title", value=draft.get("title", ""), placeholder="Post title")
        body = st.text_area(
            "Rich text / LaTeX body",
            value=draft.get("body", ""),
            height=300,
            placeholder=(
                "Use Markdown for rich text and LaTeX for formulas.\n\n"
                "**Idea:** policy gradients optimize expected return.\n\n"
                "$$\\nabla_\\theta J(\\theta)=\\mathbb{E}[\\nabla_\\theta \\log \\pi_\\theta(a|s)G_t]$$"
            ),
        )
        saved = st.form_submit_button("Save post", type="primary", use_container_width=True)

    st.session_state.discussion_draft["title"] = title
    st.session_state.discussion_draft["body"] = body

    if saved:
        clean_title = title.strip()
        clean_body = body.strip()
        if not clean_title:
            st.error("Title is required.")
        elif not clean_body:
            st.error("Post body is required.")
        else:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            draft_id = draft.get("id") or uuid.uuid4().hex
            updated = False
            next_posts = []
            for post in posts:
                if post.get("id") == draft_id:
                    next_posts.append(
                        {
                            **post,
                            "title": clean_title,
                            "body": clean_body,
                            "updated_at": now,
                        }
                    )
                    updated = True
                else:
                    next_posts.append(post)
            if not updated:
                next_posts.insert(
                    0,
                    {
                        "id": draft_id,
                        "title": clean_title,
                        "body": clean_body,
                        "created_at": now,
                        "updated_at": now,
                    },
                )
            _write_posts(next_posts)
            _set_discussion_draft(
                {
                    "id": draft_id,
                    "title": clean_title,
                    "body": clean_body,
                    "created_at": draft.get("created_at") or now,
                    "updated_at": now,
                }
            )
            st.success("Post saved.")
            st.rerun()

    uploaded = st.file_uploader(
        "Upload images for this post",
        type=IMAGE_TYPES,
        accept_multiple_files=True,
        key="discussion_images",
    )
    if uploaded and st.button("Insert uploaded images", use_container_width=True):
        inserted = []
        for image in uploaded:
            target = _unique_path(ASSETS_DIR, image.name)
            target.write_bytes(image.getbuffer())
            inserted.append(f"![{target.name}]({_relative_asset(target)})")
        _append_to_body("\n\n".join(inserted))
        st.rerun()

    with st.expander("Preview", expanded=True):
        if st.session_state.discussion_draft.get("body", "").strip():
            _render_markdown_with_local_images(st.session_state.discussion_draft["body"], DATA_DIR)
        else:
            st.info("Write a post to preview rich text, images, and LaTeX.")

    if draft.get("id"):
        if st.button("Delete this post", type="secondary"):
            _write_posts([post for post in posts if post.get("id") != draft["id"]])
            _set_discussion_draft()
            st.success("Post deleted.")
            st.rerun()


def _render_posts(posts: list[dict]) -> None:
    st.markdown("### Saved posts")
    if not posts:
        st.info("No discussion posts yet.")
        return

    for post in posts:
        with st.expander(f"{post.get('title') or 'Untitled'} · updated {post.get('updated_at', '')}", expanded=False):
            st.caption(_post_summary(post.get("body", "")))
            _render_markdown_with_local_images(post.get("body", ""), DATA_DIR)


def main_discussion_board() -> None:
    _ensure_storage()
    posts = sorted(_read_posts(), key=lambda p: p.get("updated_at", ""), reverse=True)

    st.markdown(
        """
        <div style="background:#12121f;border:1px solid #2a2a3e;border-radius:14px;
                    padding:1.4rem 1.6rem;margin-bottom:1rem">
            <h2 style="color:white;margin:0;font-size:1.6rem">💬 Discussion Board</h2>
            <p style="color:#9e9ebb;margin:.4rem 0 0">
                Save Markdown-rich posts with images and LaTeX formulas.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_write, tab_posts = st.tabs(["Write / Edit", "Saved Posts"])
    with tab_write:
        _render_editor(posts)
    with tab_posts:
        _render_posts(posts)
