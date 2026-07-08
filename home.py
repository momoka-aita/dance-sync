# home.py — ホーム（メイン）
import streamlit as st
from auth_shared import (
    load_users,
    save_users,
    is_logged_in,
    current_username,
    render_top_nav,
    user_dir,
    hash_password,
    verify_password,
)

st.set_page_config(page_title="Dance Sync", layout="wide")
render_top_nav(active_label="ホーム")

st.title("🏠 ホーム")

if not is_logged_in():
    st.subheader("ログイン / 新規登録")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("ユーザ名")
        pw = st.text_input("パスワード", type="password")
        mode = st.radio("操作", ["ログイン", "新規登録"], horizontal=True)
        submit = st.form_submit_button("OK")

    if submit:
        if not username or not pw:
            st.error("ユーザ名とパスワードを入力してください。")
            st.stop()
        db = load_users()
        users = db.get("users", {})
        if mode == "新規登録":
            if username in users:
                st.error("そのユーザは既に存在します。")
                st.stop()
            salt_hex, hash_hex = hash_password(pw)
            users[username] = {"salt": salt_hex, "hash": hash_hex}
            db["users"] = users
            save_users(db)
            st.success("新規登録が完了しました。続けてログインしました。")
            st.session_state["auth_user"] = username
            st.rerun()
        else:
            u = users.get(username)
            if not u:
                st.error("ユーザ名またはパスワードが違います。")
                st.stop()
            ok_login = verify_password(pw, u["salt"], u["hash"])
            if not ok_login:
                st.error("ユーザ名またはパスワードが違います。")
                st.stop()
            st.session_state["auth_user"] = username
            st.success(f"{username} としてログインしました。")
            st.rerun()

    st.caption("※ ログイン後は上部ナビから各ページに移動できます。")
    st.stop()

# ログイン後
user = current_username()
st.success(f"ログイン中: {user}")
st.caption(f"ユーザデータ: {user_dir(user)}")

# ボタン行（ログアウトのみ）
if st.button("🔒 ログアウト", use_container_width=True):
    st.session_state.pop("auth_user", None)
    st.rerun()
