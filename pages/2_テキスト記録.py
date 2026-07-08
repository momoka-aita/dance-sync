# 2_テキスト記録.py — 共通ログイン対応版（全置き換え）
# テキスト記録（比較手法）: ログイン済みユーザのみに表示
import time, json, hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
import streamlit as st
import pandas as pd

# 共通認証ユーティリティ（auth_shared.py）
from auth_shared import page_scaffold, current_username, user_dir

# ====== タイムゾーン ======
JST = timezone(timedelta(hours=9))


# ====== ユーティリティ（保存場所は user_dir()/textlog 配下） ======
def _textlog_dir(user_id: str) -> Path:
    p = user_dir(user_id) / "textlog"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _records_path(user_id: str) -> Path:
    return _textlog_dir(user_id) / "records.json"


def load_records(user_id: str) -> list[dict]:
    p = _records_path(user_id)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_records(user_id: str, records: list[dict]) -> None:
    p = _records_path(user_id)
    p.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def add_record(user_id: str, text: str, elapsed_sec: float):
    records = load_records(user_id)
    now = datetime.now(JST)
    item = {
        "id": hashlib.sha1(f"{now.timestamp()}-{text}".encode("utf-8")).hexdigest()[
            :10
        ],
        "saved_at": now.isoformat(timespec="seconds"),
        "elapsed_sec": round(float(elapsed_sec), 3),
        "text": text.strip(),
    }
    records.insert(0, item)  # 新しい順
    save_records(user_id, records)


def delete_record(user_id: str, rec_id: str):
    records = load_records(user_id)
    records = [r for r in records if r["id"] != rec_id]
    save_records(user_id, records)


# ====== セッション状態（計測用だけ保持） ======
if "timer_running" not in st.session_state:
    st.session_state.timer_running = False
if "timer_start" not in st.session_state:
    st.session_state.timer_start = 0.0
if "current_text" not in st.session_state:
    st.session_state.current_text = ""


# ====== ページ本体 ======
def record_view():
    user = current_username()
    st.caption(f"ログイン中: {user}")

    # ① 記録エリア
    st.subheader("① 記録エリア")
    st.write(
        "「記録開始」を押すとストップウォッチが動き、同時に入力欄と「記録終了」ボタンが表示されます。"
    )

    col_main, _spacer = st.columns([0.7, 0.3])
    with col_main:
        # タイマーが止まっているときは「開始」だけ表示
        if not st.session_state.timer_running:
            btn_start = st.button("▶ 記録開始", use_container_width=True)
            if btn_start:
                st.session_state.timer_running = True
                st.session_state.timer_start = time.time()
                st.session_state.current_text = ""  # 新規入力を想定して空に
                st.rerun()

        # タイマー起動中は「入力欄」と「終了」ボタンを表示
        else:
            st.session_state.current_text = st.text_area(
                "テキスト（メモ）",
                value=st.session_state.current_text,
                height=280,
                placeholder="ここに振付メモやアイデアを入力…",
            )
            btn_stop = st.button("■ 記録終了", use_container_width=True, type="primary")

            if btn_stop:
                elapsed = time.time() - st.session_state.timer_start
                text = st.session_state.current_text.strip()
                if text == "":
                    st.warning("テキストが空です。内容を入力してください。")
                else:
                    add_record(user, text, elapsed)
                    st.session_state.current_text = ""  # 入力欄リセット
                    st.session_state.timer_running = False
                    st.session_state.timer_start = 0.0
                    st.success("保存しました。")
                    st.rerun()

    # ② ストップウォッチ
    st.subheader("② ストップウォッチ")
    elapsed = (
        (time.time() - st.session_state.timer_start)
        if st.session_state.timer_running
        else 0.0
    )
    st.metric("経過時間", f"{elapsed:0.2f} 秒")
    st.caption("開始中は値が自動更新されます。")

    st.divider()

    # ③ 履歴
    st.subheader("③ 履歴")
    records = load_records(user)
    if not records:
        st.info("まだ保存された記録はありません。")
    else:
        df = pd.DataFrame(records)[["saved_at", "elapsed_sec", "text", "id"]]
        df = df.rename(
            columns={
                "saved_at": "保存時刻(JST)",
                "elapsed_sec": "記録時間(秒)",
                "text": "テキスト",
                "id": "ID",
            }
        )
        st.dataframe(df, use_container_width=True, height=260)

        with st.expander("個別操作（閲覧/削除）"):
            for rec in records:
                st.markdown("---")
                st.markdown(
                    f"**保存時刻**: {rec['saved_at']} / **記録時間**: {rec['elapsed_sec']} 秒"
                )
                st.text_area(
                    "テキスト（閲覧専用）",
                    value=rec["text"],
                    height=140,
                    key=f"view_{rec['id']}",
                    disabled=True,
                )
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("この記録を削除", key=f"del_{rec['id']}"):
                        delete_record(user, rec["id"])
                        st.rerun()
                with col2:
                    st.code(rec["id"], language="text")

    st.divider()
    colx, coly = st.columns([1, 1])
    with colx:
        st.caption("ログアウトは上部ヘッダーのボタンからできます。")
    with coly:
        if records:
            df = pd.DataFrame(records)
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇ 履歴をCSVで保存", csv, file_name="text_records.csv", mime="text/csv"
            )


# ====== レイアウト（共通ログイン付き） ======
with page_scaffold(
    title="📝 テキスト記録",
    page_title="テキスト記録",
    active_tab_label="テキスト",  # ← PAGES のラベルと一致
    layout="centered",  # ← スクショの表示に寄せて中央寄せ
):
    record_view()
