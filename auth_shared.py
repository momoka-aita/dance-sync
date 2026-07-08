# -*- coding: utf-8 -*-
"""
auth_shared.py — Dance Sync 共通ユーティリティ（完全版）

機能:
- データルート/ユーザDBの管理
- パスワードの安全保存（scrypt）
- ログイン状態ユーティリティ: is_logged_in(), current_username(), require_login()
- ユーザディレクトリ・セッション一覧: user_dir(), list_sessions()
- タブ風トップナビ: render_top_nav()
- ページ共通枠: page_scaffold(...)

使い方:
    from auth_shared import page_scaffold, render_top_nav
    # メイン(home.py)はログイン画面を出したいので任意。サブページでは:
    with page_scaffold(title="📝 記録", page_title="記録 - Dance Sync", active_tab_label="記録", layout="centered"):
        ... ページ本体 ...
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import json
import secrets
import hashlib

import streamlit as st

# ====== ストレージ（既存互換） =====================================================

DATA_ROOT = Path.home() / ".dance_sync_data"
USERS_DB = DATA_ROOT / "users.json"


def _ensure_private_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)
    try:
        p.chmod(0o700)
    except Exception:
        # Windows 等で chmod が効かない場合は無視
        pass


_ensure_private_dir(DATA_ROOT)


def load_users() -> Dict[str, Any]:
    if USERS_DB.exists():
        return json.loads(USERS_DB.read_text("utf-8"))
    return {"users": {}}


def save_users(db: dict) -> None:
    USERS_DB.write_text(json.dumps(db, ensure_ascii=False, indent=2), "utf-8")
    try:
        USERS_DB.chmod(0o600)
    except Exception:
        pass


def user_dir(username: str) -> Path:
    p = DATA_ROOT / "users" / username
    _ensure_private_dir(p)
    return p


def list_sessions(username: str) -> List[Path]:
    """
    ユーザのセッションディレクトリを新しい順で列挙
    セッションは 'session-*' というディレクトリ名で保存されている前提
    """
    udir = user_dir(username)
    if not udir.exists():
        return []
    return sorted([p for p in udir.glob("session-*") if p.is_dir()], reverse=True)


# ====== パスワード（scrypt） =======================================================


def hash_password(pw: str, salt: bytes | None = None) -> tuple[str, str]:
    """入力パスワードを scrypt でハッシュ化して (salt_hex, hash_hex) を返す"""
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.scrypt(pw.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=64)
    return salt.hex(), dk.hex()


def verify_password(pw: str, salt_hex: str, hash_hex: str) -> bool:
    """scrypt で検証"""
    salt = bytes.fromhex(salt_hex)
    dk = hashlib.scrypt(pw.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=64)
    return secrets.compare_digest(dk.hex(), hash_hex)


# ====== ログイン状態ユーティリティ =================================================


def is_logged_in() -> bool:
    """現在セッションに auth_user が入っているか"""
    return bool(st.session_state.get("auth_user"))


def current_username() -> Optional[str]:
    """ログイン中のユーザ名（未ログインなら None）"""
    return st.session_state.get("auth_user")


def require_login() -> None:
    """
    未ログインなら案内して停止。
    ※ 通常は page_scaffold 内で呼ばれる（home.py は任意の実装でOK）。
    """
    if not is_logged_in():
        st.info("メインページでログインしてください。")
        st.stop()


# ====== 共通ナビ（タブ風） =========================================================
# アプリ内で表示するページをここに定義（表示順）
# path は「project/home.py」から見た相対パス。先頭スラッシュ禁止・実在ファイル名に一致させる。
PAGES: List[Dict[str, str]] = [
    {"path": "home.py", "label": "ホーム", "icon": "🏠"},
    {"path": "pages/1_動画音声記録.py", "label": "記録", "icon": "🎥"},
    {"path": "pages/2_テキスト記録.py", "label": "テキスト", "icon": "📝"},
    {"path": "pages/3_動画音声履歴.py", "label": "履歴", "icon": "📜"},
]


def render_top_nav(active_label: Optional[str] = None) -> None:
    """
    画面上部に“タブ風”のページリンクを横並び表示。
    active_label と一致するラベルに ● を付ける（簡易アクティブ表示）。
    """
    cols = st.columns(len(PAGES))
    for i, p in enumerate(PAGES):
        with cols[i]:
            label = f'{p["icon"]} {p["label"]}'
            if active_label and p["label"] == active_label:
                label += "  ●"
            st.page_link(
                page=p["path"],
                label=label,
                icon=None,
                use_container_width=True,
            )


# ====== ページ共通の外枠 ===========================================================
@contextmanager
def page_scaffold(
    title: str,
    page_title: Optional[str] = None,
    active_tab_label: Optional[str] = None,
    layout: str = "wide",
):
    """
    各ページの“型”。set_page_config → 上部ナビ → （未ログインなら案内して停止） → タイトル。
    """
    # set_page_config はページごとに最初期に呼ぶ必要がある
    st.set_page_config(page_title=page_title or title, layout=layout)

    # 上部ナビは未ログインでも必ず表示
    render_top_nav(active_label=active_tab_label)

    # 未ログインなら案内だけ出して本文は描画せず停止
    if not is_logged_in():
        st.info("ホームページでログインしてください。")
        st.stop()

    # ログイン済みならタイトルを出して本文へ
    st.title(title)

    try:
        yield
    finally:
        pass
