# 1_動画音声記録.py — アカウント別保存・セッション履歴（再生/削除/原本復元）付き
# タイムライン2本・▲クリックで原本再生・アスペクト比保持（上下=幅揃え）

import streamlit as st
from auth_shared import page_scaffold, render_top_nav


st.set_page_config(page_title="Dance Sync Project", layout="centered")

render_top_nav(active_label="記録")

# ここにメインの既存UI（ログインUI含む）
# ログイン完了で ss["auth_user"] をセットする想定


from moviepy.editor import AudioFileClip, CompositeAudioClip
import os
import json
import tempfile
import time
import hashlib
import secrets
from datetime import datetime
from pathlib import Path
import unicodedata
from typing import List, Tuple
import shutil

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, filtfilt, find_peaks, spectrogram
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from moviepy.editor import (
    VideoFileClip,
    ImageSequenceClip,
    clips_array,
    CompositeAudioClip,
)
from moviepy.audio.AudioClip import AudioClip, AudioArrayClip

from moviepy.editor import CompositeVideoClip  # 縦積みで使う
from moviepy.video.fx.all import colorx  # 明るさ調整（vfx.colorx の中身）

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.ticker import MultipleLocator

import base64
import streamlit.components.v1 as components

# ====== 手動マーク用カスタムコンポーネント（タップ=カウント / 長押し=補足） ======
_COMPONENT_DIR = Path(__file__).resolve().parent.parent / "components" / "dance_marker"
_DANCE_MARKER_AVAILABLE = False
try:
    _dance_marker = components.declare_component("dance_marker", path=str(_COMPONENT_DIR))
    _DANCE_MARKER_AVAILABLE = True
except Exception:
    _dance_marker = None


def dance_marker(
    *,
    label: str,
    video_data_url: str | None = None,
    mode: str | None = None,
    max_beats: int = 8,
    key: str | None = None,
):
    """手動マーク用コンポーネントを描画し、保存された結果を dict で返す。

    返り値（保存ボタンが押されるまで None）:
      {"label", "mode", "beats":[秒...], "supps":[[a,b]...],
       "video_b64":str|None, "video_mime":str|None, "ts":int}
    """
    if not _DANCE_MARKER_AVAILABLE:
        st.info("手動マーク機能はクラウド版では利用できません。ローカル環境またはサーバー版をご利用ください。")
        return None
    return _dance_marker(
        label=label,
        video_data_url=video_data_url,
        mode=mode,
        max_beats=max_beats,
        key=key,
        default=None,
    )


# --- 日本語フォント設定 ---
# Macならヒラギノ / Noto Sans JP が安定
jp_font_candidates = [
    "Hiragino Sans",
    "Hiragino Kaku Gothic ProN",
    "Noto Sans CJK JP",
    "IPAexGothic",
    "IPAPGothic",
]

for font in jp_font_candidates:
    if any(font in f.name for f in font_manager.fontManager.ttflist):
        rcParams["font.family"] = font
        break

rcParams["axes.unicode_minus"] = False  # − が文字化けするのを防ぐ

# ===== フォントサイズ一括設定 =====
rcParams["font.size"] = 16  # 基本サイズ（まずは16〜18がおすすめ）
rcParams["axes.titlesize"] = 18  # タイトル
rcParams["axes.labelsize"] = 16  # 軸ラベル
rcParams["xtick.labelsize"] = 14  # x軸目盛
rcParams["ytick.labelsize"] = 14  # y軸目盛
rcParams["legend.fontsize"] = 14
rcParams["figure.titlesize"] = 18


def _rerun():
    """Streamlit の rerun をバージョン差を吸収して呼ぶ"""
    fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if fn:
        fn()
    else:
        st.warning(
            "このStreamlitバージョンには rerun API が見つかりません。`pip install -U streamlit` を試してください。"
        )


# ★ 追加：終端を踏まないための安全マージン
EPS = 1e-3  # 必要なら 2e-3 〜 3e-3 に上げてもOK


def _media_url(path: Path) -> str:
    """
    ローカルファイルを data: URL（base64）にして返す。
    Streamlit の media_file_manager に依存しない安全版。
    """
    import base64

    p = Path(path)
    ext = p.suffix.lower()

    # MIME の振り分け（m4a/mp4a→audio/mp4, mp3→audio/mpeg）
    if ext == ".mp4":
        mime = "video/mp4"
    elif ext in (".wav",):
        mime = "audio/wav"
    elif ext in (".m4a", ".mp4a", ".mp3"):
        mime = "audio/mp4" if ext in (".m4a", ".mp4a") else "audio/mpeg"
    else:
        mime = "application/octet-stream"

    b = p.read_bytes()
    b64 = base64.b64encode(b).decode("ascii")
    return f"data:{mime};base64,{b64}"


# ===== クリップ末尾の安全マージン（秒） =====
# これを少し大きめにとることで、
# 「14.74秒しかないのに 14.76秒を読みに行く」ような事故を防ぐ
SAFE_TAIL_MARGIN = 0.25  # 末尾 0.25 秒は使わない
# 元動画（.MOVなど）の音声を切り出すとき専用のマージン
SRC_TAIL_MARGIN = 0.03  # 末尾 0.03 秒は使わない（補足用の安全マージン）


# ====== 書き出し時の長さを安全に丸めるユーティリティ ======
def _safe_dur(T, clip=None, eps: float = 1e-3):
    """
    MoviePy が末尾を越えてフレームを取りに行かないように、
    ・クリップの実長さ clip.duration
    ・指定された目標長 T
    のうち短い方から、さらに eps 秒だけマイナスした長さを返す。
    """
    # まずクリップ側の実長さを取得
    base = 0.0
    if clip is not None:
        try:
            base = float(getattr(clip, "duration", 0.0) or 0.0)
        except Exception:
            base = 0.0

    # T が None のとき
    if T is None:
        if base > 0:
            # クリップの実長さに合わせて少しだけ短く
            return max(0.0, base - eps)
        # クリップ長も分からないならそのまま None を返す（set_duration しない想定）
        return None

    # T が指定されていて、クリップ長も分かるとき
    if base > 0:
        return max(0.0, min(float(T), base) - eps)

    # クリップ長が取れないときは従来通り T - eps だけにしておく
    return max(0.0, float(T) - eps)


# ====== オプション ======
ENABLE_HOVER_PREVIEW = False  # クリック優先（必要なら True）

USE_PLOTLY = True
try:
    import plotly.graph_objects as go
    from streamlit_plotly_events import plotly_events
except Exception:
    USE_PLOTLY = False

# ffmpeg の場所（あれば）
try:
    import imageio_ffmpeg as _i

    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", _i.get_ffmpeg_exe())
except Exception:
    pass
os.environ["IMAGEIO_FFMPEG_LOGLEVEL"] = "info"

# ====== セキュア保存先（VPSでも他人から見えにくい隠しディレクトリ）======
DATA_ROOT = Path.home() / ".dance_sync_data"  # 隠しフォルダ
USERS_DB = DATA_ROOT / "users.json"  # 認証DB


def _ensure_private_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    try:
        p.chmod(0o700)  # 所有者のみ
    except Exception:
        pass


_ensure_private_dir(DATA_ROOT)


# ====== メイン：ログイン画面 ======
def login_main_view():
    st.title("Dance Sync Project")
    st.subheader("ログイン")
    st.caption("ユーザー名とパスワードを入力してログインしてください。")

    # --- ログインフォーム ---
    with st.form("login_form", clear_on_submit=False):
        li_user = st.text_input("ユーザー名", placeholder="momoka_a など")
        li_pass = st.text_input(
            "パスワード", type="password", placeholder="8文字以上を推奨"
        )
        col = st.columns([1, 1, 2])
        with col[0]:
            do_login = st.form_submit_button("ログイン", use_container_width=True)
        with col[1]:
            do_reset = st.form_submit_button(
                "🔄 全リセット（セッションのみ）", use_container_width=True
            )

    if do_reset:
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.success("セッション状態をリセットしました。")
        _rerun()

    if do_login:
        db = _load_users()
        u = db["users"].get(li_user)
        if not li_user or not li_pass:
            st.error("ユーザー名とパスワードを入力してください。")
        elif not u:
            st.error("ユーザーが存在しません。下の『新規登録』から作成できます。")
        elif _verify_password(li_pass, u["salt"], u["hash"]):
            ss.auth_user = li_user
            ss.session_dir = str(_new_session_dir(li_user))
            st.success(f"ログインしました：{li_user}")
            _rerun()
        else:
            st.error("パスワードが違います。")

    # --- 新規登録 ---
    with st.expander("新規登録", expanded=False):
        with st.form("signup_form", clear_on_submit=False):
            su_user = st.text_input("新しいユーザー名", key="su_user_main")
            su_pass = st.text_input(
                "新しいパスワード", type="password", key="su_pass_main"
            )
            submitted = st.form_submit_button("登録（重複不可）")
        if submitted:
            if not su_user or not su_pass:
                st.error("ユーザー名とパスワードを入力してください。")
            else:
                db = _load_users()
                if su_user in db["users"]:
                    st.error("そのユーザー名は使用できません（重複）")
                else:
                    salt_hex, hash_hex = _hash_password(su_pass)
                    db["users"][su_user] = {"salt": salt_hex, "hash": hash_hex}
                    _save_users(db)
                    _ensure_private_dir(_user_dir(su_user))
                    st.success("登録しました。上のフォームからログインしてください。")


def _load_users():
    if USERS_DB.exists():
        return json.loads(USERS_DB.read_text("utf-8"))
    return {"users": {}}


def _save_users(db):
    USERS_DB.write_text(json.dumps(db, ensure_ascii=False, indent=2), "utf-8")
    try:
        USERS_DB.chmod(0o600)  # 所有者のみ
    except Exception:
        pass


# scrypt で安全にハッシュ化（ソルト付き）


def _hash_password(pw: str, salt: bytes | None = None):
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.scrypt(pw.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=64)
    return salt.hex(), dk.hex()


def _verify_password(pw: str, salt_hex: str, hash_hex: str) -> bool:
    salt = bytes.fromhex(salt_hex)
    dk = hashlib.scrypt(pw.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=64)
    return secrets.compare_digest(dk.hex(), hash_hex)


def _user_dir(username: str) -> Path:
    p = DATA_ROOT / "users" / username
    _ensure_private_dir(p)
    return p


def _new_session_dir(username: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    p = _user_dir(username) / f"session-{ts}"
    _ensure_private_dir(p)
    (p / "uploads").mkdir(exist_ok=True)  # 原本保存用
    return p


def _list_sessions(username: str) -> list[Path]:
    udir = _user_dir(username)
    if not udir.exists():
        return []
    return sorted([p for p in udir.glob("session-*") if p.is_dir()], reverse=True)


# ====== Session ======
ss = st.session_state
_defaults = {
    "auth_user": None,
    "session_dir": None,
    "upper_path": None,
    "lower_path": None,
    "wav_u": None,
    "wav_l": None,
    "supp_u": [],
    "supp_l": [],
    "upper_beats": [],
    "lower_beats": [],
    "target_end": 0.0,
    "preview_path": None,
    "processed": False,
    "pre_u": [],
    "pre_l": [],
    "last_hover_key": None,
    "last_hover_ts": 0.0,
    # ==== 手動マーク（タップ=カウント / 長押し=補足）====
    "manual_beats_u": [],
    "manual_beats_l": [],
    "manual_supp_u": [],
    "manual_supp_l": [],
    "manual_video_u": None,  # カメラ撮影した場合の保存先パス
    "manual_video_l": None,
    "manual_ts_u": 0,
    "manual_ts_l": 0,
}
for k, v in _defaults.items():
    ss.setdefault(k, v)

# ====== サイドバー：ログイン / 新規登録 / リセット ======

# --- メインページでのログイン必須化（このページからはログインUIを出さない）
if not ss.get("auth_user"):
    st.info("ホームページでログインしてください。")
    st.stop()

# 現在のセッションディレクトリ
SESSION_DIR = Path(ss.session_dir) if ss.get("session_dir") else None
if SESSION_DIR is None:
    # ありえないが保険
    ss.session_dir = str(_new_session_dir(ss.auth_user))
    SESSION_DIR = Path(ss.session_dir)

# ---- アカウント表示＆ログアウトボタン（メインページ上部） ----
col_user, col_sp, col_logout = st.columns([1, 6, 1])
with col_user:
    st.caption(f"ログイン中：**{ss.auth_user}**")
with col_logout:
    if st.button("ログアウト", use_container_width=True):
        ss.auth_user = None
        ss.session_dir = None
        for k in [
            "upper_path",
            "lower_path",
            "processed",
            "preview_path",
            "pre_u",
            "pre_l",
        ]:
            ss[k] = None if k.endswith("_path") else []
        _rerun()
st.divider()


# ====== 入力 ======
st.subheader("アップロード")
upper_video = st.file_uploader(
    "上半身動画をアップロード", type=["mp4", "mov", "m4v"], key="up_v"
)
lower_video = st.file_uploader(
    "下半身動画をアップロード", type=["mp4", "mov", "m4v"], key="lo_v"
)

target_bpm = st.slider("Target BPM", 40, 240, 60, 1)


# ====== 手動マーク（タップ=カウント / 長押し=補足） ======
def _save_recorded_video(video_b64: str, video_mime: str, tag: str) -> str:
    """コンポーネントから送られた base64 動画をセッションフォルダへ保存しパスを返す。"""
    ext = ".mp4" if "mp4" in (video_mime or "") else ".webm"
    up_dir = SESSION_DIR / "uploads"
    up_dir.mkdir(parents=True, exist_ok=True)
    out = up_dir / f"{tag}_rec{ext}"
    i = 1
    while out.exists():
        out = up_dir / f"{tag}_rec_{i}{ext}"
        i += 1
    out.write_bytes(base64.b64decode(video_b64))
    return str(out)


def _handle_marker_result(res: dict | None, track: str):
    """コンポーネントの保存結果を session_state に反映する（track: 'u' or 'l'）。"""
    if not res:
        return
    ts = int(res.get("ts", 0))
    if ts <= int(ss.get(f"manual_ts_{track}", 0)):
        return  # 同じ保存結果の重複反映を防ぐ
    ss[f"manual_ts_{track}"] = ts
    ss[f"manual_beats_{track}"] = [float(t) for t in res.get("beats", [])]
    ss[f"manual_supp_{track}"] = [
        (float(a), float(b)) for a, b in res.get("supps", [])
    ]
    if res.get("mode") == "record" and res.get("video_b64"):
        tag = "upper" if track == "u" else "lower"
        path = _save_recorded_video(res["video_b64"], res.get("video_mime", ""), tag)
        ss[f"manual_video_{track}"] = path
        # 撮影動画を同期処理の入力として採用
        ss["upper_path" if track == "u" else "lower_path"] = path
    st.toast(
        f"{'上半身' if track=='u' else '下半身'}：カウント{len(ss[f'manual_beats_{track}'])}件 "
        f"/ 補足{len(ss[f'manual_supp_{track}'])}件を保存しました",
    )


with st.expander("✋ 手動マーク（タップ=カウント / 長押し=補足）", expanded=False):
    st.caption(
        "アップロードした動画を再生しながら、または『カメラで撮影』しながら、"
        "ボタンを **タップでカウント拍**、**長押しで補足区間** をマークできます。"
        "保存すると自動検出（Whisper）より優先して同期に使われます。"
    )

    def _video_data_url_for(video_file, track: str) -> str | None:
        """アップロードファイル or 既存パスから data URL を作る。"""
        # 撮影済みならそのパスを優先
        rec_path = ss.get(f"manual_video_{track}")
        if rec_path and os.path.exists(rec_path):
            return _media_url(Path(rec_path))
        if video_file is not None and hasattr(video_file, "getvalue"):
            data = video_file.getvalue()
            b64 = base64.b64encode(data).decode("ascii")
            mime = "video/mp4"
            name = getattr(video_file, "name", "").lower()
            if name.endswith(".mov") or name.endswith(".m4v"):
                mime = "video/quicktime"
            return f"data:{mime};base64,{b64}"
        return None

    mk_u, mk_l = st.tabs(["上半身", "下半身"])
    with mk_u:
        res_u = dance_marker(
            label="上半身",
            video_data_url=_video_data_url_for(upper_video, "u"),
            max_beats=8,
            key="marker_u",
        )
        _handle_marker_result(res_u, "u")
        if ss.get("manual_beats_u"):
            st.success(
                f"保存済み — カウント {len(ss['manual_beats_u'])} 件 / "
                f"補足 {len(ss['manual_supp_u'])} 件"
            )
    with mk_l:
        res_l = dance_marker(
            label="下半身",
            video_data_url=_video_data_url_for(lower_video, "l"),
            max_beats=8,
            key="marker_l",
        )
        _handle_marker_result(res_l, "l")
        if ss.get("manual_beats_l"):
            st.success(
                f"保存済み — カウント {len(ss['manual_beats_l'])} 件 / "
                f"補足 {len(ss['manual_supp_l'])} 件"
            )

    if ss.get("manual_beats_u") or ss.get("manual_beats_l"):
        if st.button("手動マークをクリア", key="clear_manual"):
            for t in ("u", "l"):
                ss[f"manual_beats_{t}"] = []
                ss[f"manual_supp_{t}"] = []
                ss[f"manual_video_{t}"] = None
                ss[f"manual_ts_{t}"] = 0
            _rerun()


# ====== 詳細設定（折りたたみ） ======
with st.expander("詳細設定", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        offset_sec = st.number_input("Offset (秒)", value=0.0, step=0.05, min_value=0.0)
    with c2:
        out_fps = st.slider("出力FPS", 15, 60, 30, 1)

    norm_audio = st.checkbox("音声正規化（dynaudnorm）", value=True)

    cp1, cp2, cp3 = st.columns(3)
    with cp1:
        min_interval = st.number_input(
            "最小間隔（秒）", value=0.25, step=0.05, min_value=0.05
        )
    with cp2:
        sens = st.slider("感度（相対しきい値）", 0.05, 0.95, 0.20, 0.05)
    with cp3:
        phase_shift = st.selectbox(
            "位相", ["1から開始", "2から開始", "3から開始", "4から開始"], index=0
        )

    layout = "上下（縦）"
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        gain_upper = st.slider("上半身の音量", 0.0, 1.5, 0.7, 0.05)
    with mc2:
        gain_lower = st.slider("下半身の音量", 0.0, 1.5, 0.7, 0.05)
    with mc3:
        gain_click = st.slider("クリック音量", 0.0, 1.5, 0.9, 0.05)

    pc1, pc2 = st.columns(2)

    gap_col = st.columns(1)[0]
    with gap_col:
        unify_gap = st.number_input(
            "補足の『小さな隙間』を同一扱いするしきい値 [s]",
            value=0.25,
            step=0.05,
            min_value=0.0,
            help="ここ以下の隙間は強制的に同じ補足として結合します（最終仕上げ）。",
        )

    with pc1:
        protect_supplement = st.checkbox("補足（低声）をワープから保護", value=True)
    with pc2:
        supp_th = st.slider("補足検出しきい値（低中域）", 0.10, 0.80, 0.20, 0.05)

    whisper_model_size = st.selectbox(
        "Whisperモデルサイズ（小さいほど速いが精度低）",
        ["tiny", "base", "small", "medium", "large"],
        index=1,
        help="初回は自動ダウンロード。base推奨（速度と精度のバランス）。",
    )

    st.markdown("#### 先頭/末尾の余白")
    head_pad = st.number_input(
        "先頭余白 [s]", value=0.20, step=0.05, min_value=0.0, key="head_pad"
    )
    tail_pad = st.number_input(
        "末尾余白 [s]", value=0.30, step=0.05, min_value=0.0, key="tail_pad"
    )

    dim_factor = st.slider(
        "ミュート側の明るさ（暗くする度合い）",
        0.3,
        1.0,
        0.20,
        0.05,
        help="1.0で暗くしない。0.6〜0.7が少し暗い目安。",
        key="dim_factor",
    )


# ====== ヘルパ ======
def _ensure_local_path(
    video, into_dir: Path | None = None, tag: str = "upper"
) -> tuple[str, bool]:
    """
    - すでにローカルパスの場合はそのまま返す
    - アップロードされた場合は into_dir/uploads に {tag}.* の名前で保存して返す
    """
    if isinstance(video, str) and os.path.exists(video):
        return video, False
    if hasattr(video, "read"):
        ext = Path(video.name).suffix or ".mp4"
        up_dir = (into_dir or Path(".")).joinpath("uploads")
        up_dir.mkdir(parents=True, exist_ok=True)

        tmp_path = up_dir / f"{tag}{ext}"
        i = 1
        while tmp_path.exists():
            tmp_path = up_dir / f"{tag}_{i}{ext}"
            i += 1

        video.seek(0)
        with open(tmp_path, "wb") as f:
            f.write(video.read())
        try:
            tmp_path.chmod(0o600)
        except Exception:
            pass
        return str(tmp_path), True
    raise ValueError("Unsupported video input")


# --- subclip 安全化ヘルパ ---
def _cap_range(
    a: float, b: float, dur: float, eps: float = 1e-3
) -> tuple[float, float]:
    """
    a,b を [0, dur-eps] に収め、a < b を保証して返す。端の丸めも入れる。
    """
    if dur is None or dur <= 0:
        return (0.0, eps)
    a2 = max(0.0, min(float(a), max(0.0, dur - eps)))
    b2 = max(a2 + eps, min(float(b), max(0.0, dur - eps)))
    return (round(a2, 3), round(b2, 3))


def build_target_beats_uniform(k: int, bpm: float, offset: float = 0.0):
    step = 60.0 / bpm
    return np.array([offset + step * i for i in range(k)], dtype=float)


def time_map_from_markers(src_beats: np.ndarray, tgt_beats: np.ndarray):
    f = PchipInterpolator(src_beats, tgt_beats, extrapolate=True)
    g = PchipInterpolator(tgt_beats, src_beats, extrapolate=True)
    return f, g


def build_warped_frames(video, g_map, target_end: float, fps_out: int):
    path, _ = _ensure_local_path(video, into_dir=SESSION_DIR)  # セッション内に確保
    clip = None
    try:
        clip = VideoFileClip(path)
        t_tgt = np.arange(0.0, max(target_end, 1e-6), 1.0 / fps_out)
        t_src = np.clip(g_map(t_tgt), 0.0, clip.duration - 1e-3)
        frames = [clip.get_frame(float(ts)) for ts in t_src]
        return ImageSequenceClip(frames, fps=fps_out)
    finally:
        try:
            if clip is not None:
                clip.close()
        except Exception:
            pass


def build_warped_audio(video, g_map, target_end: float):
    path, _ = _ensure_local_path(video, into_dir=SESSION_DIR)
    clip = VideoFileClip(path)  # keepalive（←ここでは閉じない）

    ac = clip.audio
    if ac is None:
        try:
            clip.close()
        except Exception:
            pass
        return None, None, False, path  # ← clip も None を返す

    src_dur = ac.duration
    src_fps = getattr(ac, "fps", 44100)

    def make_frame(t):
        tt = np.array(t, dtype=float)
        ts = np.clip(g_map(tt), 0.0, src_dur - 1e-3)
        if np.ndim(ts) == 0:
            return ac.get_frame(float(ts))
        return np.vstack([ac.get_frame(float(x)) for x in ts])

    warped = AudioClip(make_frame, duration=target_end, fps=src_fps)
    return warped, clip, False, path  # ← clip を返す


# ====== 音声系 ======
def _bandpass_envelope(y, sr, low=300, high=3000, ma_ms=20):
    b, a = butter(4, [low / (sr * 0.5), high / (sr * 0.5)], btype="band")
    yb = filtfilt(b, a, y)
    env = np.abs(yb)
    win = max(1, int(sr * ma_ms / 1000))
    k = np.ones(win) / win
    env_ma = np.convolve(env, k, mode="same")
    m = env_ma.max() or 1.0
    return env_ma / m


def _band_envelope(y, sr, low, high, ma_ms=20):
    b, a = butter(4, [low / (sr * 0.5), high / (sr * 0.5)], btype="band")
    yb = filtfilt(b, a, y)
    env = np.abs(yb)
    win = max(1, int(sr * ma_ms / 1000))
    k = np.ones(win) / win
    return np.convolve(env, k, mode="same")


def _sine_click(sr=44100, freq=1000, dur=0.045, ramp=0.005, gain=0.8):
    n = int(sr * dur)
    t = np.arange(n) / sr
    x = np.sin(2 * np.pi * freq * t).astype(np.float32)
    r = int(sr * ramp)
    if r > 0:
        w = np.linspace(0, 1, r)
        x[:r] *= w
        x[-r:] *= w[::-1]
    return (gain * x).reshape(-1, 1)


def _render_click_track(beats_sec, sr=44100, length=0.0, base_freq=900):
    if beats_sec is None or len(beats_sec) == 0:
        return None
    total = int(sr * max(length, beats_sec[-1] + 0.5))
    y = np.zeros((total, 1), dtype=np.float32)
    for i, t in enumerate(beats_sec):
        freq = base_freq * 1.2 if (i % 4) == 0 else base_freq
        amp = 1.0 if (i % 4) == 0 else 0.7
        click = _sine_click(sr, freq, 0.045, 0.004, amp)
        p = int(sr * t)
        q = min(total, p + len(click))
        if p < total:
            y[p:q, 0] += click[: q - p, 0]
    y /= (np.max(np.abs(y)) or 1.0) * 1.05
    return y, sr


def _ndarray_to_audioclip(y, sr):
    return AudioArrayClip(y, fps=sr)


def _mute_regions_on_clip(
    aclip: AudioClip, spans: list[tuple[float, float]]
) -> AudioClip:
    """
    AudioClip の一部区間を無音化するヘルパー。
    spans はターゲット時間軸（ワープ後）の [(start, end), ...]。
    """
    if aclip is None:
        return aclip

    spans = [(float(a), float(b)) for a, b in spans if b > a]
    if not spans:
        return aclip

    spans_arr = np.array(spans, dtype=float)

    def make_frame(t):
        tt = np.array(t, dtype=float)
        y = aclip.get_frame(tt)

        # スカラ時刻
        if np.ndim(tt) == 0:
            t0 = float(tt)
            for a, b in spans_arr:
                if a <= t0 <= b:
                    return np.zeros_like(y)
            return y

        # ベクトル時刻
        mask = np.zeros_like(tt, dtype=bool)
        for a, b in spans_arr:
            mask |= (tt >= a) & (tt <= b)

        y = np.array(y, copy=True)
        if y.ndim == 1:
            y[mask] = 0.0
        else:
            y[mask, :] = 0.0
        return y

    return AudioClip(
        make_frame,
        duration=float(aclip.duration or 0.0),
        fps=int(getattr(aclip, "fps", 44100)),
    )


def _detect_beats_by_peaks(
    wav_path, min_interval=0.25, sens=0.20, phase_label="1から開始"
):
    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    env = _bandpass_envelope(data, sr, 300, 3000, 20)
    distance = int(sr * min_interval)
    peaks, _ = find_peaks(env, height=sens, distance=distance)
    peaks_sec = peaks / sr
    if len(peaks_sec) >= 4:
        iv = np.diff(peaks_sec)
        med = np.median(iv)
        keep = [peaks_sec[0]]
        for t in peaks_sec[1:]:
            if t - keep[-1] >= max(0.6 * med, 0.15):
                keep.append(t)
        peaks_sec = np.array(keep, dtype=float)
    return np.array(peaks_sec, dtype=float)


def _merge_small_gaps(
    segs: list[tuple[float, float]], max_gap: float
) -> list[tuple[float, float]]:
    """隣り合う区間の隙間が max_gap 以下なら強制的に結合する最終仕上げ。"""
    if not segs:
        return []
    segs = sorted((float(a), float(b)) for a, b in segs)
    out = [[segs[0][0], segs[0][1]]]
    for a, b in segs[1:]:
        if a - out[-1][1] <= float(max_gap):
            out[-1][1] = max(out[-1][1], float(b))
        else:
            out.append([float(a), float(b)])
    return [(float(a), float(b)) for a, b in out]


def _detect_supplement_segments_auto(
    wav_path,
    bands=[(80, 400), (120, 700), (150, 1000)],
    base_th=0.20,
    min_len=0.30,
    merge_gap=0.25,
    ma_ms=30,
    hole_close=0.12,  # 補足マスク中の"穴"をこの秒数以下なら埋める
    edge_pad=0.04,  # 検出区間の端を少し広げる
    # ▼ 追加：ここが"短い隙間をひとつにする"肝
    min_silence_to_split=0.20,  # これ未満の沈黙では区間を分割しない
    th_hyst=0.06,  # ON/OFF 閾の差（ヒステリシス）
    small_gap_unify=0.25,
):
    """
    複数帯域のエンベロープを合成し、データ駆動の自動しきい値で補足を検出。
    - 短い無音(hole)の埋め、近接区間の結合、端の拡張に加えて
    - ヒステリシス(th_on/th_off)と「短い沈黙は無視」の状態機械で分割を防ぐ
    """
    import numpy as np
    import soundfile as sf
    from scipy.signal import butter, filtfilt

    y, sr = sf.read(wav_path, dtype="float32")
    y = y.mean(axis=1) if y.ndim > 1 else y
    y = y / (np.max(np.abs(y)) or 1.0)

    # 複数帯域の移動平均エンベロープ（max合成）
    envs = []
    for low, high in bands:
        b, a = butter(4, [low / (sr * 0.5), high / (sr * 0.5)], btype="band")
        yb = filtfilt(b, a, y)
        env = np.abs(yb)
        win = max(1, int(sr * ma_ms / 1000))
        k = np.ones(win) / win
        envs.append(np.convolve(env, k, mode="same"))
    env = np.max(np.vstack(envs), axis=0)
    env = env / (env.max() or 1.0)

    # 自動しきい値
    p70 = float(np.percentile(env, 70))
    p80 = float(np.percentile(env, 80))
    auto_th = max(base_th, min(0.85, max(0.15, (p70 + p80) / 2.0)))

    # --- ヒステリシス＆短い沈黙無視 ---
    th_on = float(min(0.99, auto_th + th_hyst / 2))
    th_off = float(max(0.01, auto_th - th_hyst / 2))

    i, n = 0, len(env)
    segs = []
    in_seg = False
    seg_start = None

    while i < n:
        v = env[i]
        if not in_seg:
            if v >= th_on:  # OFF→ON
                in_seg = True
                seg_start = i
            i += 1
        else:
            if v < th_off:  # ON中に閾下へ→沈黙の長さを測る
                j = i + 1
                while j < n and env[j] < th_off:
                    j += 1
                silence_sec = (j - i) / sr
                if silence_sec < float(min_silence_to_split):
                    i = j  # 短い沈黙は無視（区間継続）
                    continue
                else:
                    segs.append((seg_start / sr, i / sr))  # 分割
                    in_seg = False
                    seg_start = None
                    i = j
            else:
                i += 1

    if in_seg and seg_start is not None:
        segs.append((seg_start / sr, n / sr))

    # 短すぎるのは除外
    segs = [(float(a), float(b)) for (a, b) in segs if (b - a) >= float(min_len)]

    # 近いセグメント結合
    merged = []
    for a, b in segs:
        if not merged or (a - merged[-1][1]) > merge_gap:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)

    # マスク中の小さな穴を埋める（仕上げの保険）
    if hole_close and hole_close > 0:
        # 既に結合されているので、ここでは更に"ちょい隙間"も飲み込む
        gap = float(hole_close)
        merged2 = []
        for a, b in merged:
            if not merged2 or (a - merged2[-1][1]) > gap:
                merged2.append([a, b])
            else:
                merged2[-1][1] = max(merged2[-1][1], b)
        merged = merged2

    if small_gap_unify and small_gap_unify > 0:
        merged = _merge_small_gaps(merged, float(small_gap_unify))

    # 端を少し広げる（見た目の途切れ防止）
    if edge_pad and edge_pad > 0:
        pad = float(edge_pad)
        merged = [[max(0.0, a - pad), b + pad] for (a, b) in merged]

    return [(float(a), float(b)) for a, b in merged]


# ====== 音声認識（Whisper）によるビート・補足検出 ======

# カウント語の基本形（長音符ーを除去した後で照合する）
# ※ "ワーン"→"ワン"、"いーち"→"いち" のように正規化してから比較
_COUNT_WORDS_BASE = {
    # アラビア数字・全角数字・漢数字
    "1", "2", "3", "4", "5", "6", "7", "8",
    "１", "２", "３", "４", "５", "６", "７", "８",
    "一", "二", "三", "四", "五", "六", "七", "八",
    # ひらがな
    "いち", "に", "さん", "し", "よん", "ご", "ろく", "なな", "しち", "はち",
    # 英語カウント（カタカナ）：「ワーン」→「ワン」のように正規化後に一致
    "ワン", "ツ", "ツー", "スリ", "スリー", "フォ", "フォー",
    "ファイブ", "シクス", "シックス", "セブン", "エイト",
    # Whisperが英字で書き起こすケース
    "one", "two", "three", "four", "five", "six", "seven", "eight",
}


def _normalize_count_word(w: str) -> str:
    """長音符（ー・〜・～）を除去して小文字化した正規形を返す。"""
    return w.replace("ー", "").replace("〜", "").replace("～", "").lower().strip()


# 正規化済みカウント語セット（照合用）
_COUNT_WORDS_NORMALIZED = {_normalize_count_word(w) for w in _COUNT_WORDS_BASE}


def _is_count_word(w: str) -> bool:
    """元の形または正規化後の形がカウント語かどうか判定する。"""
    if w in _COUNT_WORDS_BASE:
        return True
    return _normalize_count_word(w) in _COUNT_WORDS_NORMALIZED


@st.cache_resource
def _load_whisper_model(model_size: str):
    try:
        import whisper
        return whisper.load_model(model_size)
    except ImportError:
        st.error("openai-whisper が未インストールです。`pip install openai-whisper` を実行してください。")
        return None


def _transcribe_words(wav_path: str, model) -> list[dict]:
    """Whisperで単語ごとのタイムスタンプ付き書き起こしを返す。"""
    import whisper
    result = model.transcribe(
        wav_path,
        language="ja",
        word_timestamps=True,
        fp16=False,
    )
    words = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            # 空白・句読点・記号を除去してから格納
            raw = w["word"].strip()
            for ch in " 　、。！？!?,.:;…":
                raw = raw.replace(ch, "")
            words.append({
                "word": raw,
                "start": float(w["start"]),
                "end": float(w["end"]),
            })
    return words


def _count_word_to_number(w: str) -> int | None:
    """カウント語を対応する数字(1-8)に変換する。カウント語でなければNoneを返す。"""
    normalized = _normalize_count_word(w)
    _MAP = {
        "1": 1, "１": 1, "一": 1, "いち": 1, "ワン": 1, "one": 1,
        # ツー → 正規化後「ツ」、two
        "2": 2, "２": 2, "二": 2, "に": 2, "ツ": 2, "two": 2,
        # スリー → 正規化後「スリ」、three
        "3": 3, "３": 3, "三": 3, "さん": 3, "スリ": 3, "three": 3,
        # フォー → 正規化後「フォ」、four
        "4": 4, "４": 4, "四": 4, "し": 4, "よん": 4, "フォ": 4, "four": 4,
        # ファイブ（長音なし）、five
        "5": 5, "５": 5, "五": 5, "ご": 5, "ファイブ": 5, "five": 5,
        # シックス → 正規化後「シックス」（ーなし）、シクス、six
        "6": 6, "６": 6, "六": 6, "ろく": 6, "シクス": 6, "シックス": 6, "six": 6,
        # セブン（長音なし）、seven
        "7": 7, "７": 7, "七": 7, "なな": 7, "しち": 7, "セブン": 7, "seven": 7,
        # エイト（長音なし）、eight
        "8": 8, "８": 8, "八": 8, "はち": 8, "エイト": 8, "eight": 8,
    }
    return _MAP.get(normalized)


def _classify_words(words: list[dict]) -> list[dict]:
    """各単語に is_beat フラグを付けて返す。
    同じカウント番号の2回目以降は is_beat=False（補足扱い）にする。
    例：「ワン」が2回出たら1回目だけ is_beat=True。
    """
    seen: set[int] = set()
    result = []
    for w in words:
        num = _count_word_to_number(w["word"])
        if num is not None and num not in seen:
            seen.add(num)
            result.append({**w, "is_beat": True})
        else:
            result.append({**w, "is_beat": False})
    return result


def _detect_beats_by_speech(wav_path: str, model) -> np.ndarray:
    """カウント語（1〜8）の発話タイムスタンプをビートとして返す。
    - 伸ばして発音（ワーン・いーち等）も検出。
    - 同じ数字が2回以上出た場合は最初の1回のみカウントとして扱う。
    - 最大8カウントまで。"""
    words = _transcribe_words(wav_path, model)
    classified = _classify_words(words)
    beats = [w["start"] for w in classified if w["is_beat"]]
    return np.array(sorted(beats)[:8], dtype=float)


def _detect_supplement_segments_speech(
    wav_path: str,
    model,
    merge_gap: float = 0.40,
    min_len: float = 0.25,
) -> list[tuple[float, float]]:
    """カウント語以外（2回目以降の同じカウント語も含む）の発話区間を補足として返す。"""
    words = _transcribe_words(wav_path, model)
    classified = _classify_words(words)
    supp_words = [
        (float(w["start"]), float(w["end"]))
        for w in classified
        if not w["is_beat"]
    ]
    if not supp_words:
        return []

    segs: list[tuple[float, float]] = []
    cur_a, cur_b = supp_words[0]
    for a, b in supp_words[1:]:
        if a - cur_b <= merge_gap:
            cur_b = max(cur_b, b)
        else:
            if cur_b - cur_a >= min_len:
                segs.append((cur_a, cur_b))
            cur_a, cur_b = a, b
    if cur_b - cur_a >= min_len:
        segs.append((cur_a, cur_b))
    return segs


def _merge_supp_no_beat_between(
    segs: list[tuple[float, float]],
    beats: np.ndarray,
) -> list[tuple[float, float]]:
    """連続する補足セグメント間にカウントビートが存在しない場合は一つに結合する。"""
    if len(segs) <= 1:
        return segs

    beats = np.asarray(beats, dtype=float)
    merged = [list(segs[0])]
    for a, b in segs[1:]:
        prev_end = merged[-1][1]
        # 前の補足の終わりと今の補足の始まりの間にビートがあるか
        beats_between = beats[(beats > prev_end) & (beats < a)]
        if len(beats_between) == 0:
            # ビートなし → 同じ補足として結合
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(float(a), float(b)) for a, b in merged]


# ====== 音声特徴量マッチング（上下カウントの対応付け） ======

def _compute_mfcc_scipy(y: np.ndarray, sr: int, n_mfcc: int = 13, n_fft: int = 512, hop: int = 128, n_mels: int = 40) -> np.ndarray:
    """scipy のみで MFCC 平均特徴ベクトルを計算する。"""
    from scipy.signal import spectrogram as _spectrogram
    from scipy.fft import dct

    f, t, Sxx = _spectrogram(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, mode="magnitude")

    # メルフィルタバンク
    f_min, f_max = 80.0, min(8000.0, sr / 2.0)
    mel_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + f_max / 700.0)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    bin_pts = np.clip(bin_pts, 0, Sxx.shape[0] - 1)

    fb = np.zeros((n_mels, Sxx.shape[0]))
    for m in range(1, n_mels + 1):
        fl, fc, fr = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        for k in range(fl, fc):
            if fc > fl:
                fb[m - 1, k] = (k - fl) / (fc - fl)
        for k in range(fc, fr):
            if fr > fc:
                fb[m - 1, k] = (fr - k) / (fr - fc)

    mel_energy = fb @ Sxx  # (n_mels, T)
    mel_log = np.log(mel_energy + 1e-10)
    mfcc = dct(mel_log, axis=0, norm="ortho")[:n_mfcc]
    return mfcc.mean(axis=1)  # (n_mfcc,)


def _extract_beat_features(wav_path: str, beats: np.ndarray, window: float = 0.35) -> list:
    """各ビート時刻周辺の音声からMFCC特徴量を抽出する。取れなければ None。"""
    y, sr = sf.read(wav_path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    n = len(y)
    feats = []
    for t in beats:
        a = max(0, int((float(t) - 0.05) * sr))
        b = min(n, int((float(t) + window) * sr))
        seg = y[a:b]
        if len(seg) < int(sr * 0.05):
            feats.append(None)
            continue
        try:
            feats.append(_compute_mfcc_scipy(seg, sr))
        except Exception:
            feats.append(None)
    return feats


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _match_beats_by_features(
    beats_u: np.ndarray,
    feats_u: list,
    beats_l: np.ndarray,
    feats_l: list,
    threshold: float = 0.70,
):
    """
    MFCC特徴量のコサイン類似度＋ハンガリアン法で上下カウントを対応付ける。
    平均類似度が threshold 未満なら (None, score) を返す → 呼び出し側でフォールバック。
    """
    from scipy.optimize import linear_sum_assignment

    Nu, Nl = len(beats_u), len(beats_l)
    if Nu == 0 or Nl == 0:
        return None, 0.0

    # 類似度行列 (Nu × Nl)
    sim = np.zeros((Nu, Nl), dtype=float)
    for i, fi in enumerate(feats_u):
        for j, fj in enumerate(feats_l):
            if fi is not None and fj is not None:
                sim[i, j] = _cosine_sim(fi, fj)

    # ハンガリアン法（コスト = 1 - 類似度）
    row_idx, col_idx = linear_sum_assignment(1.0 - sim)
    avg_sim = float(sim[row_idx, col_idx].mean()) if len(row_idx) > 0 else 0.0

    if avg_sim < threshold:
        return None, avg_sim

    matched_u = np.array([beats_u[i] for i in row_idx], dtype=float)
    matched_l = np.array([beats_l[j] for j in col_idx], dtype=float)

    # 上半身の時刻順に並べ直す
    order = np.argsort(matched_u)
    matched_u = matched_u[order]
    matched_l = matched_l[order]

    # PchipInterpolator は両系列が strictly increasing である必要がある
    # マッチングで lower が非単調になった場合はフォールバックへ
    if len(matched_u) > 1 and (np.any(np.diff(matched_u) <= 0) or np.any(np.diff(matched_l) <= 0)):
        return None, avg_sim

    return (matched_u, matched_l), avg_sim


def _exclude_peaks_near_segments(peaks, segs, radius=0.15):
    out = []
    for t in peaks:
        if any((a - radius) <= t <= (b + radius) for a, b in segs):
            continue
        out.append(t)
    return np.array(out, dtype=float)


def _filter_supp_segments_by_beats(
    segs: list[tuple[float, float]],
    beats: np.ndarray | list[float],
    near: float = 0.18,
    min_overlap_ratio: float = 0.35,
):
    if segs is None or len(segs) == 0 or beats is None or len(beats) == 0:
        return segs

    beats = np.asarray(beats, dtype=float)
    keep = []
    for a, b in segs:
        L = float(a)
        R = float(b)
        seg_len = max(1e-6, R - L)
        overlap = 0.0
        for t in beats:
            x0 = max(L, t - near)
            x1 = min(R, t + near)
            if x1 > x0:
                overlap += x1 - x0
        if (overlap / seg_len) < float(min_overlap_ratio):
            keep.append((float(L), float(R)))
    return keep


def _make_beep(sr=16000, freq=1000, dur=0.08, ramp=0.004, gain=0.9):
    n = int(sr * dur)
    t = np.arange(n) / sr
    x = np.sin(2 * np.pi * freq * t).astype(np.float32)
    r = int(sr * ramp)
    if r > 0:
        w = np.linspace(0, 1, r)
        x[:r] *= w
        x[-r:] *= w[::-1]
    return (gain * x).astype(np.float32)


def _write_segment_wav(src_wav, seg: Tuple[float, float], out_path: str, add_beep=True):
    y, sr = sf.read(src_wav, dtype="float32")
    y = y.mean(axis=1) if y.ndim > 1 else y
    a, b = seg
    a_i = max(0, int(a * sr))
    b_i = max(a_i, int(b * sr))
    seg_y = y[a_i:b_i]
    parts = []
    if add_beep:
        parts += [_make_beep(sr), np.zeros(int(sr * 0.03), dtype=np.float32)]
    parts.append(seg_y.astype(np.float32))
    out = np.concatenate(parts) if parts else np.zeros(int(sr * 0.25), dtype=np.float32)
    out = (out / ((np.max(np.abs(out)) or 1.0) * 1.01)).astype(np.float32)
    sf.write(out_path, out, sr)


def _write_segment_video(src_vid, seg: Tuple[float, float], out_path: str, fps=30):
    a, b = float(seg[0]), float(seg[1])
    base = VideoFileClip(src_vid)
    try:
        dur = float(base.duration or 0.0)
        a, b = _cap_range(a, b, dur)  # ★ 追加：区間を動画内にクリップ
        clip = base.subclip(a, b)
        clip.write_videofile(out_path, fps=int(fps), codec="libx264", audio_codec="aac")
        clip.close()
    finally:
        base.close()


def _write_segment_video_proxy(
    src_vid,
    seg: Tuple[float, float],
    out_path: str,
    fps=30,
    max_h=360,
    v_bitrate="1200k",
    a_bitrate="64k",
):
    """▲クリック用の軽量プレビュー。ここも無音にする。"""
    a, b = float(seg[0]), float(seg[1])
    base = VideoFileClip(src_vid)
    try:
        dur = float(base.duration or 0.0)
        a, b = _cap_range(a, b, dur)
        clip = base.subclip(a, b)

        # ★無音化
        clip = clip.without_audio()

        vf = f"scale=-2:{int(max_h)}"
        clip.write_videofile(
            out_path,
            fps=int(fps),
            codec="libx264",
            audio=False,  # ← 音声を出力しない
            bitrate=v_bitrate,
            ffmpeg_params=["-vf", vf, "-movflags", "+faststart", "-pix_fmt", "yuv420p"],
            preset="veryfast",
            logger=None,
        )
        clip.close()
    finally:
        base.close()


def _make_range_video_for_supp(
    src_vid, beats_src: np.ndarray, i0: int, i1: int, out_path: str, fps=30, max_h=360
):
    """補足が属する直前カウント範囲 [i0, i1] の"元動画"を軽量MP4で切り出す（※無音にする）。"""
    base = VideoFileClip(src_vid)
    try:
        dur = float(base.duration or 0.0)
        start = float(beats_src[i0]) if 0 <= i0 < len(beats_src) else 0.0
        end = float(beats_src[i1 + 1]) if (i1 + 1) < len(beats_src) else dur
        a, b = _cap_range(start, end, dur)
        clip = base.subclip(a, b)

        # ★ここが重要：動画側は音声を完全にミュート
        clip = clip.without_audio()

        vf = f"scale=-2:{int(max_h)}"
        clip.write_videofile(
            out_path,
            fps=int(fps),
            codec="libx264",
            audio=False,  # ← 音声トラックを出力しない
            ffmpeg_params=["-vf", vf, "-movflags", "+faststart", "-pix_fmt", "yuv420p"],
            preset="veryfast",
            logger=None,
        )
        clip.close()
    finally:
        base.close()


# === 補足（原速）を"直前のカウント範囲"へ貼る ===


def _last_beat_index_before(t0: float, beats_src: np.ndarray) -> int | None:
    if beats_src is None or len(beats_src) == 0:
        return None
    idx = np.searchsorted(beats_src, t0, side="right") - 1
    return int(idx) if idx >= 0 else None


def _last_beat_index_before(t0: float, beats_src: np.ndarray) -> int | None:
    if beats_src is None or len(beats_src) == 0:
        return None
    idx = np.searchsorted(beats_src, t0, side="right") - 1
    return int(idx) if idx >= 0 else None


def _pair_supp_with_count_ranges(
    supp_segments: list[tuple[float, float]], beats_src: np.ndarray
) -> list[tuple[tuple[float, float], tuple[int, int]]]:
    """
    各補足 (a,b) を「直前のカウント範囲 [i_start, i_end]」に割り当てる。
    - 連続する補足が同じ j（a の直前のビート index）を持つ場合は、
      "同じ [i_start, i_end]" を再利用して複数の補足を同一範囲に載せる。
    """
    if beats_src is None or len(beats_src) == 0 or not supp_segments:
        return []

    # 開始時刻で安定ソート
    segs = sorted([(float(a), float(b)) for a, b in supp_segments], key=lambda x: x[0])

    out: list[tuple[tuple[float, float], tuple[int, int]]] = []
    last_j: int | None = None
    last_i_start: int = 0
    last_i_end: int = -1

    for a, b in segs:
        # a の直前のビート index
        j = int(np.searchsorted(beats_src, a, side="right") - 1)
        if j < 0:
            # 最初のビートより前の補足は割り当て不可
            continue

        if last_j is None:
            # 1本目：0 から j まで
            i_start = 0
            i_end = j
        elif j > last_j:
            # j が進んだ：前回終点の次から新しい j までを消費
            i_start = last_i_end + 1
            i_end = j
        else:
            # j が同じ：同じ範囲を再利用（複数補足を同一範囲に載せる）
            i_start = last_i_start
            i_end = last_i_end

        # 安全ガード
        if i_start > i_end or i_end < 0:
            i_start = max(0, min(j, len(beats_src) - 1))
            i_end = j

        out.append(((a, b), (i_start, i_end)))
        last_j, last_i_start, last_i_end = j, i_start, i_end

    return out


def build_supplement_audio_clip(
    original_audio_path: str,
    supp_segments: list[tuple[float, float]],
    beats_src: np.ndarray,
    target_beats: np.ndarray,
    total_duration: float,
    trim_to_span: bool = True,
    gap_between_segs: float = 0.02,  # 使わないけど互換のため残す
):
    """
    補足を"原速のまま"直前のカウント範囲に貼る（シンプル版）。

    - original_audio_path: 一度 ffmpeg で抽出した WAV
      (tmp_upper.wav / tmp_lower.wav) を渡す想定。
    - 各補足 (a, b) について：
        * 「a の直前のビート」が属するカウント範囲 [i_start, i_end] を求める
        * そのカウント範囲の「先頭ビート」 target_beats[i_start] から
          元の波形 (a, b) をそのまま貼る
    - 同じ範囲に複数の補足がある場合も、「範囲の頭」から全部重ね貼りするだけ。
      （時間方向に詰めて並べたりはしない）
    """

    if not supp_segments or total_duration <= 0.0 or not original_audio_path:
        return None

    # 元オーディオ（WAV）を読み込み
    try:
        y, sr = sf.read(original_audio_path, dtype="float32")
    except Exception:
        return None

    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    n_src = len(y)
    if n_src <= 0:
        return None

    # 元オーディオの長さと安全終端（末尾ちょい手前までしか使わない）
    src_dur = n_src / float(sr)
    src_safe_end = max(0.0, src_dur - float(SRC_TAIL_MARGIN))

    beats_src = np.asarray(beats_src, dtype=float)
    target_beats = np.asarray(target_beats, dtype=float)
    if beats_src.size == 0 or target_beats.size == 0:
        return None

    # 出力全体の長さ（末尾を踏まないようにほんの少しだけ短く）
    T = max(float(total_duration) - 1e-3, 0.0)
    n_out = int(T * sr)
    if n_out <= 0:
        return None
    out = np.zeros(n_out, dtype=np.float32)

    # どの補足がどのカウント範囲に属するか
    pairs = _pair_supp_with_count_ranges(supp_segments, beats_src)
    if not pairs:
        return None

    eps = 1e-3

    for (a, b), (i_start, i_end) in pairs:
        A0 = float(a)
        B0 = float(b)
        if B0 <= A0:
            continue

        # この補足の直前カウント範囲の「先頭ビート」に貼る
        if i_start < 0 or i_start >= len(target_beats):
            continue
        t_dest = float(target_beats[i_start])
        if t_dest >= T - eps:
            continue

        # 元オーディオ側：末尾マージンを考慮した安全な長さ
        if A0 >= src_safe_end - eps:
            continue  # ほぼ末尾なら諦める
        seg_src_len = max(eps, min(B0, src_safe_end) - A0)

        # 出力タイムライン側：この位置から T までの残り時間
        max_len_here = max(eps, T - t_dest)

        # 実際に使う長さ（元の安全長さとタイムラインの残りの min）
        seg_len = min(seg_src_len, max_len_here)
        if seg_len <= eps:
            continue

        # 元オーディオ側インデックス
        sa = int(round(A0 * sr))
        sb = int(round((A0 + seg_len) * sr))
        sa = max(0, min(sa, n_src))
        sb = max(sa, min(sb, n_src))
        if sb <= sa:
            continue

        # 出力側インデックス（範囲の頭から貼る）
        da = int(round(t_dest * sr))
        db = int(round((t_dest + seg_len) * sr))
        da = max(0, min(da, n_out))
        db = max(da, min(db, n_out))
        L = db - da
        if L <= 0:
            continue

        # 元の波形をそのまま加算貼り付け（時間伸縮なし）
        out[da : da + L] += y[sa : sa + L]

    # (N,1) にして AudioArrayClip にラップ
    data = out.reshape(-1, 1)
    try:
        clip = AudioArrayClip(data, fps=sr).audio_fadein(0.01).audio_fadeout(0.01)
    except Exception:
        clip = AudioArrayClip(data, fps=sr)

    return clip.set_duration(T)


def build_supplement_audio_clip_from_video(
    video_path: str,
    supp_segments: list[tuple[float, float]],
    beats_src: np.ndarray,
    target_beats: np.ndarray,
    total_duration: float,
    trim_to_span: bool = True,
):
    """
    元動画の音声から補足区間だけをそのまま切り出し、
    BPMワープを一切かけず、前のカウント範囲に貼り付けていく。
    """

    # 元動画の音声をそのままロード（正規化もワープも無し）
    y, sr = _load_audio_array_from_video_path(
        video_path,
        target_sr=44100,
        normalize=False,
        tmp_dir=SESSION_DIR,
    )

    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    n_src = len(y)

    beats_src = np.asarray(beats_src)
    target_beats = np.asarray(target_beats)

    total_T = float(max(total_duration - 1e-3, 0))
    n_out = int(total_T * sr)

    out = np.zeros(n_out, dtype=np.float32)

    for a, b in supp_segments:
        if b <= a:
            continue

        # この補足区間が属するビート index
        j = int(np.searchsorted(beats_src, a, "right") - 1)
        if j < 0 or j >= len(target_beats):
            continue

        # 貼り付け先の開始時間（ターゲット側）
        t_dest = target_beats[j]
        da = int(t_dest * sr)
        if da >= n_out:
            continue

        # 元動画側の切り出し
        sa = max(0, int(a * sr))
        sb = min(n_src, int(b * sr))
        seg = y[sa:sb]

        # 出力へ貼り付け
        db = min(da + len(seg), n_out)
        out[da:db] += seg[: db - da]

    # AudioArrayClip化
    return AudioArrayClip(out.reshape(-1, 1), fps=sr)


# ====== WAV抽出 ======
def _extract_wav_from_path(video_path, out_path, target_sr=44100, normalize=False):
    clip = None
    try:
        clip = VideoFileClip(video_path)
        audio = clip.audio
        if audio is None:
            raise RuntimeError("音声が含まれていません")
        ff = ["-ac", "1"]
        ff += ["-af", "dynaudnorm=f=75:g=15"] if normalize else []
        audio.write_audiofile(
            out_path, fps=target_sr, nbytes=2, codec="pcm_s16le", ffmpeg_params=ff
        )
    finally:
        if clip is not None:
            clip.close()


# ====== 可視化（スペクトログラム） ======


def _load_audio_array_from_video_path(
    video_path, target_sr=44100, normalize=False, tmp_dir: Path | None = None
):
    tmp_dir = Path(tmp_dir or ".")
    _ensure_private_dir(tmp_dir)
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".wav", dir=str(tmp_dir)
    ).name
    try:
        _extract_wav_from_path(video_path, tmp, target_sr, normalize)
        y, sr = sf.read(tmp, dtype="float32")
        y = y.mean(axis=1) if y.ndim > 1 else y
        return y, sr
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


def _compute_spectrogram(y, sr, nperseg=1024, noverlap=768):
    f, t, Sxx = spectrogram(
        y,
        fs=sr,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        mode="magnitude",
    )
    return f, t, 20.0 * np.log10(Sxx + 1e-12)


def _nearest_idx(array, value):
    idx = np.searchsorted(array, value)
    idx = int(np.clip(idx, 1, len(array) - 1))
    return idx if abs(array[idx] - value) < abs(array[idx - 1] - value) else idx - 1


def _plot_alignment_spectrograms_path(
    upper_path,
    lower_path,
    upper_beats,
    lower_beats,
    normalize_audio=True,
    fmin=50,
    fmax=8000,
    supp_u=None,  # ★ 追加：上半身の補足 [(a,b), ...]
    supp_l=None,  # ★ 追加：下半身の補足 [(a,b), ...]
):
    y_u, sr_u = _load_audio_array_from_video_path(
        upper_path, 16000, normalize_audio, tmp_dir=SESSION_DIR
    )
    y_l, sr_l = _load_audio_array_from_video_path(
        lower_path, 16000, normalize_audio, tmp_dir=SESSION_DIR
    )
    fu, tu, Su = _compute_spectrogram(y_u, sr_u)
    fl, tl, Sl = _compute_spectrogram(y_l, sr_l)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    fig.suptitle("音声スペクトログラム（上：上半身，下：下半身）", fontsize=22)

    # # --- 上半身 ---
    # m1 = ax1.pcolormesh(tu, fu, Su, shading="gouraud")
    # ax1.set_title("Upper Body: Audio Spectrogram")
    # ax1.set_ylabel("Frequency [Hz]")
    # ax1.set_yscale("log")
    # ax1.set_ylim(max(fmin, 1), min(fmax, sr_u / 2.0))
    # fig.colorbar(m1, ax=ax1).set_label("Amplitude [dB]")

    # # --- 下半身 ---
    # m2 = ax2.pcolormesh(tl, fl, Sl, shading="gouraud")
    # ax2.set_title("Lower Body: Audio Spectrogram")
    # ax2.set_xlabel("Time [s]")
    # ax2.set_ylabel("Frequency [Hz]")
    # ax2.set_yscale("log")
    # ax2.set_ylim(max(fmin, 1), min(fmax, sr_l / 2.0))
    # fig.colorbar(m2, ax=ax2).set_label("Amplitude [dB]")

    # # --- 上半身 ---
    # m1 = ax1.pcolormesh(tu, fu, Su, shading="gouraud")
    # ax1.set_title("上半身：音声スペクトログラム")
    # ax1.set_ylabel("周波数 [Hz]")
    # ax1.set_yscale("log")
    # ax1.set_ylim(max(fmin, 1), min(fmax, sr_u / 2.0))
    # cbar1 = fig.colorbar(m1, ax=ax1)
    # cbar1.set_label("振幅 [dB]")

    # # --- 下半身 ---
    # m2 = ax2.pcolormesh(tl, fl, Sl, shading="gouraud")
    # ax2.set_title("下半身：音声スペクトログラム")
    # ax2.set_xlabel("時間 [s]")
    # ax2.set_ylabel("周波数 [Hz]")
    # ax2.set_yscale("log")
    # ax2.set_ylim(max(fmin, 1), min(fmax, sr_l / 2.0))
    # cbar2 = fig.colorbar(m2, ax=ax2)
    # cbar2.set_label("振幅 [dB]")

    # --- 上半身 ---
    m1 = ax1.pcolormesh(tu, fu, Su, shading="gouraud")
    ax1.set_ylabel("周波数 [Hz]", fontsize=18)
    ax1.set_yscale("log")
    ax1.set_ylim(max(fmin, 1), min(fmax, sr_u / 2.0))

    ax1.xaxis.set_major_locator(MultipleLocator(5))

    cbar1 = fig.colorbar(m1, ax=ax1)
    cbar1.set_label("振幅 [dB]", fontsize=18)
    cbar1.ax.tick_params(labelsize=14)

    # --- 下半身 ---
    m2 = ax2.pcolormesh(tl, fl, Sl, shading="gouraud")
    ax2.set_xlabel("時間 [秒]", fontsize=18)
    ax2.set_ylabel("周波数 [Hz]", fontsize=18)
    ax2.set_yscale("log")
    ax2.set_ylim(max(fmin, 1), min(fmax, sr_l / 2.0))

    ax2.xaxis.set_major_locator(MultipleLocator(5))

    cbar2 = fig.colorbar(m2, ax=ax2)
    cbar2.set_label("振幅 [dB]", fontsize=18)
    cbar2.ax.tick_params(labelsize=14)

    # --- 拍（赤丸）＆上下対応線（シアン） ---
    k = min(len(upper_beats), len(lower_beats))
    u_pts = []
    l_pts = []
    for i in range(k):
        tu_i = _nearest_idx(tu, upper_beats[i])
        tl_i = _nearest_idx(tl, lower_beats[i])
        fu_mask = (fu >= fmin) & (fu <= fmax)
        fl_mask = (fl >= fmin) & (fl <= fmax)
        u_col = Su[:, tu_i]
        l_col = Sl[:, tl_i]
        u_freq = (
            float(fu[fu_mask][np.argmax(u_col[fu_mask])])
            if fu_mask.sum() > 0
            else float(np.sqrt(fmin * (fmax if fmax > fmin else sr_u / 2.0)))
        )
        l_freq = (
            float(fl[fl_mask][np.argmax(l_col[fl_mask])])
            if fl_mask.sum() > 0
            else float(np.sqrt(fmin * (fmax if fmax > fmin else sr_l / 2.0)))
        )

        # 拍の赤丸
        ax1.plot(upper_beats[i], u_freq, "o", ms=8, mfc="none", mec="red", mew=2.5)
        ax2.plot(lower_beats[i], l_freq, "o", ms=8, mfc="none", mec="red", mew=2.5)

        u_pts.append((upper_beats[i], u_freq))
        l_pts.append((lower_beats[i], l_freq))

    # 上下対応線（紫）
    for (xu, yu), (xl, yl) in zip(u_pts, l_pts):
        fig.add_artist(
            ConnectionPatch(
                xyA=(xu, yu),
                coordsA=ax1.transData,
                xyB=(xl, yl),
                coordsB=ax2.transData,
                color="purple",
                lw=2.5,
                alpha=0.9,
            )
        )

    # --- ★補足区間：横線＋青丸（始点・終点）を描画 ---
    # 補足のラインを置く高さ（スペクトログラムの下の方に揃える）
    line_y_u = min(fmax, max(fmin * 1.2, fmin + 1.0))
    line_y_l = min(fmax, max(fmin * 1.2, fmin + 1.0))

    # 上半身の補足
    if supp_u:
        for a, b in supp_u:
            # 横線
            ax1.plot([a, b], [line_y_u, line_y_u], "-", color="blue", lw=3, alpha=0.9)
            # 始点・終点の青丸
            ax1.plot(a, line_y_u, "o", ms=6, mfc="white", mec="blue", mew=2)
            ax1.plot(b, line_y_u, "o", ms=6, mfc="white", mec="blue", mew=2)

    # 下半身の補足
    if supp_l:
        for a, b in supp_l:
            ax2.plot([a, b], [line_y_l, line_y_l], "-", color="blue", lw=3, alpha=0.9)
            ax2.plot(a, line_y_l, "o", ms=6, mfc="white", mec="blue", mew=2)
            ax2.plot(b, line_y_l, "o", ms=6, mfc="white", mec="blue", mew=2)

    st.pyplot(fig)
    plt.close(fig)


# ====== 補足アンカーで g(t) 修正 ======


def time_map_with_anchors(f_map, g_map, anchors_src):
    mapped = []
    for a, b in sorted(anchors_src):
        A = float(f_map(a))
        B = float(f_map(b))
        if B > A:
            mapped.append((a, b, A, B))
    merged = []
    for a, b, A, B in sorted(mapped, key=lambda x: x[2]):
        if not merged or A > merged[-1][3]:
            merged.append([a, b, A, B])
        else:
            merged[-1][3] = max(merged[-1][3], B)
    pieces = []
    t_cur = 0.0
    shift = 0.0
    for a, b, A, B in merged:
        if A > t_cur:
            pieces.append(("warp", t_cur, A, shift))
        pieces.append(("anchor", A, B, (A - a)))
        t_cur = B
        shift = b - float(g_map(B))
    pieces.append(("warp", t_cur, float("inf"), shift))

    def g_anchored(t):
        t = np.array(t, dtype=float)
        out = np.empty_like(t)
        for kind, L, R, param in pieces:
            m = (t >= L) & (t < R)
            if not np.any(m):
                continue
            out[m] = (
                t[m] - float(param) if kind == "anchor" else g_map(t[m]) + float(param)
            )
        return out

    return g_anchored


def _audioclip_from_wav_stereo(
    path: str, out_sr: int = 48000, fade: float = 0.01
) -> AudioClip:
    y, sr = sf.read(path, dtype="float32")
    if y.ndim == 1:
        y = np.stack([y, y], axis=1)  # mono→stereo
    elif y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)

    peak = float(np.max(np.abs(y)) or 1.0)
    y = (y / (peak * 1.01)).astype(np.float32)

    # ★ここがポイント：元の sr のまま AudioArrayClip を作る（set_fpsしない）
    clip = AudioArrayClip(y, fps=sr)

    try:
        clip = clip.audio_fadein(fade).audio_fadeout(fade)
    except Exception:
        pass
    return clip


def _mux_range_with_supp_audio(range_mp4: str, wav_path: str, out_path: str):
    from moviepy.editor import VideoFileClip

    v = a = merged = None
    try:
        v = VideoFileClip(range_mp4)  # without_audio 済み
        V = float(v.duration or 0.0)
        if V <= 0:
            raise RuntimeError("range video duration is zero")

        # 48kHz/stereo の AudioClip を"元 sr のまま"作ってから合成（①の修正を利用）
        a = (
            _audioclip_from_wav_stereo(wav_path, out_sr=48000, fade=0.01)
            .set_start(0.0)
            .set_duration(max(0.0, V - EPS))
        )

        merged = v.set_audio(a).set_duration(max(0.0, V - EPS))

        # ★ audio=True とビットレートを明示。端末差の"無音化"を回避
        merged.write_videofile(
            out_path,
            fps=int(v.fps or 30),
            codec="libx264",
            audio=True,  # ← 明示
            audio_codec="aac",
            audio_bitrate="160k",  # ← 明示
            ffmpeg_params=[
                "-ac",
                "2",  # Stereo
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-shortest",
            ],
            logger=None,
        )
    finally:
        for c in (merged, a, v):
            try:
                if c is not None:
                    c.close()
            except Exception:
                pass


def _precut_segments(
    track_name,
    segs,
    src_vid,
    src_wav,
    out_fps=30,
    out_dir: Path | None = None,
    beats_src: np.ndarray | None = None,
):
    """
    ▲クリックで再生するためのプリカットを作る。
    - item['wav'] は補足の"音声だけ"（元動画のその部分）
    - item['range_mp4'] はその補足が属する「直前カウント範囲」の"元動画（無音）"
    - item['range_with_supp'] は range_mp4 + 補足音声を多重化した MP4
    """
    out = []
    out_dir = Path(out_dir or ".")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 直前カウント範囲を紐づけ
    pairs = _pair_supp_with_count_ranges(
        segs, np.asarray(beats_src) if beats_src is not None else np.array([])
    )

    for i, ((a, b), ie) in enumerate(pairs):
        key = f"{'U' if track_name=='上半身' else 'L'}-{i+1}"
        base = out_dir / f"orig_{key}_{a:.3f}_{b:.3f}"

        # 補足の音声（元動画のその部分だけ／ビープ無し）
        wav = str(base) + ".wav"
        if not os.path.exists(wav):
            _write_segment_wav(src_wav, (a, b), wav, add_beep=False)

        # 直前カウント範囲の"無音"ビデオ
        range_mp4 = str(base) + ".range.mp4"
        if beats_src is not None and len(beats_src) >= 2 and ie is not None:
            i_start, i_end = ie
            try:
                _make_range_video_for_supp(
                    src_vid,
                    np.asarray(beats_src),
                    int(i_start),
                    int(i_end),
                    range_mp4,
                    fps=int(out_fps),
                    max_h=360,
                )
            except Exception:
                # フォールバック：補足区間そのもの
                _write_segment_video_proxy(
                    src_vid, (a, b), range_mp4, fps=int(out_fps), max_h=360
                )
        else:
            _write_segment_video_proxy(
                src_vid, (a, b), range_mp4, fps=int(out_fps), max_h=360
            )
            i_start, i_end = 0, 0

        # ★ 無音ビデオ + 補足音声 を合成したプレビュー用MP4
        range_with_supp = str(base) + ".range_with_supp.mp4"
        try:
            if os.path.exists(range_with_supp):
                os.remove(range_with_supp)
            _mux_range_with_supp_audio(range_mp4, wav, range_with_supp)
        except Exception as e:
            # 失敗したら無音の range_mp4 を使う（警告だけ出す）
            range_with_supp = range_mp4
            try:
                st.warning(
                    f"補足音声の合成に失敗したため無音プレビューを表示します: {os.path.basename(range_mp4)} / err={e}"
                )
            except Exception:
                pass

        out.append(
            {
                "a": float(a),
                "b": float(b),
                "wav": wav,
                "range_mp4": range_mp4,
                "range_with_supp": range_with_supp,
                "key": key,
                "i_start": int(i_start),
                "i_end": int(i_end),
            }
        )

    return out


# ====== アスペクト比を保ったままサイズ合わせ =====


def _match_sizes(cu, cl):
    """
    上下（縦）専用: 幅を揃えてアスペクト比維持
    """
    target_h = min(cu.size[1], cl.size[1], 480)  # 高さを基準に
    return cu.resize(height=target_h), cl.resize(height=target_h)


def _stack_vertical_preserve_aspect(
    clip_top, clip_bottom, single_height=None, bg=(0, 0, 0)
):
    """
    縦動画2本を上下に積む。各クリップは「高さ」を基準に等倍縮小し、アスペクト比はMoviePyに任せて維持。
    - single_height: 各クリップの仕上げ高さ（Noneなら小さい方に合わせる）
    """

    w_t, h_t = clip_top.size
    w_b, h_b = clip_bottom.size

    if single_height is None:
        single_height = min(h_t, h_b)  # 小さい方に合わせる（歪み防止）

    top_r = clip_top.resize(height=int(single_height))
    bot_r = clip_bottom.resize(height=int(single_height))

    # 幅はそれぞれの比率のまま。最終キャンバスは「大きい方の幅」に合わせる
    W = int(max(top_r.w, bot_r.w))
    H = int(top_r.h + bot_r.h)

    # 左右中央に寄せて上下配置（横方向は余白で埋める）
    from moviepy.editor import CompositeVideoClip

    stacked = CompositeVideoClip(
        [
            top_r.set_position(("center", 0)),
            bot_r.set_position(("center", top_r.h)),
        ],
        size=(W, H),
        bg_color=bg,
    )
    return stacked


def _stack_vertical_noresize(clip_top, clip_bottom, bg=(0, 0, 0)):

    W = int(max(clip_top.w, clip_bottom.w))
    H = int(clip_top.h + clip_bottom.h)

    from moviepy.editor import CompositeVideoClip

    stacked = CompositeVideoClip(
        [
            clip_top.set_position(("center", 0)),
            clip_bottom.set_position(("center", clip_top.h)),
        ],
        size=(W, H),
        bg_color=bg,
    )
    return stacked


from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
import traceback


def _safe_write_videofile(clip, out_path: Path, fps: int, vf: str):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_audio = out_path.with_suffix(".tmp.m4a")
    try:
        # 前回失敗の残骸があるとI/Oで落ちることがあるので先に消す
        if out_path.exists():
            try:
                out_path.unlink()
            except Exception:
                pass

        # --- 書き出し直前の最終ガード（丸め誤差で終端を踏まない） ---
        dur_fix = float(clip.duration or 0.0)
        if dur_fix > 0:
            clip = clip.set_duration(max(0.0, dur_fix - 1e-3))

        clip.write_videofile(
            str(out_path),
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            ffmpeg_params=[
                "-vf",
                vf,
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-shortest",
            ],
            temp_audiofile=str(tmp_audio),  # 一時音声の保存先を明示
            remove_temp=True,
            threads="auto",
            logger=None,  # Streamlitへの大量ログ回避（必要なら外す）
        )
        try:
            out_path.chmod(0o600)
        except Exception:
            pass
    except Exception as e:
        st.error(f"動画の書き出しに失敗しました: {out_path.name}")
        st.exception(e)  # 具体原因を表示
        st.code("".join(traceback.format_exc()))
        raise
    finally:
        # 一時音声の掃除（例外時でも）
        try:
            if tmp_audio.exists():
                tmp_audio.unlink()
        except Exception:
            pass


def _run_sync_pipeline(
    upper_beats: np.ndarray,
    lower_beats: np.ndarray,
    supp_u: list[tuple[float, float]],
    supp_l: list[tuple[float, float]],
    wav_u: str,
    wav_l: str,
):
    """
    与えられた upper_beats / lower_beats / supp_u / supp_l をそのまま使って
    同期処理〜プレビュー生成までを行う共通処理。
    - 最初の自動検出
    - 手動編集後の再同期
    の両方から呼び出す。
    """

    # ===== 検証 =====
    if min(len(upper_beats), len(lower_beats)) < 2:
        st.error("ビート（カウント）が不足しています。カウント時刻を調整してください。")
        st.stop()

    # numpy 化しておく
    upper_beats = np.array(upper_beats, dtype=float)
    lower_beats = np.array(lower_beats, dtype=float)

    # ===== 拍列＆ターゲット長 =====
    k = min(len(upper_beats), len(lower_beats))
    upper_beats = upper_beats[:k]
    lower_beats = lower_beats[:k]
    target_beats = build_target_beats_uniform(k, target_bpm, offset_sec)
    target_end = float(target_beats[-1])

    # 余白を考慮して最終尺を延長
    head_pad_val = st.session_state.get("head_pad", 0.20)
    tail_pad_val = st.session_state.get("tail_pad", 0.30)
    target_end_padded = float(target_end) + float(head_pad_val) + float(tail_pad_val)

    # 以降の set_duration / クリック長さ などは T を使うと安全
    T = target_end_padded

    # ===== 時間写像 =====
    f_u, g_u_raw = time_map_from_markers(upper_beats, target_beats)
    f_l, g_l_raw = time_map_from_markers(lower_beats, target_beats)
    g_u = time_map_with_anchors(f_u, g_u_raw, supp_u) if protect_supplement else g_u_raw
    g_l = time_map_with_anchors(f_l, g_l_raw, supp_l) if protect_supplement else g_l_raw

    # ===== ベース音声側でミュートする補足区間（ターゲット時間軸） =====
    mute_u = []
    if supp_u:
        for a, b in supp_u:
            # 元動画時間 (a,b) → ターゲット時間軸へ f_u で写像
            Au = float(f_u(float(a)))
            Bu = float(f_u(float(b)))
            if Bu > Au:
                mute_u.append((Au, Bu))

    mute_l = []
    if supp_l:
        for a, b in supp_l:
            Al = float(f_l(float(a)))
            Bl = float(f_l(float(b)))
            if Bl > Al:
                mute_l.append((Al, Bl))

    # ===== 映像ワープ =====
    st.info("動画を同期処理中（映像）…")
    warped_upper = build_warped_frames(ss.upper_path, g_u, T, out_fps)
    warped_lower = build_warped_frames(ss.lower_path, g_l, T, out_fps)

    # ===== 音声ワープ =====
    st.info("音声を同期処理中…")
    keep_u_clip = None
    keep_l_clip = None
    audio_u, keep_u_clip, _, keep_u_path = build_warped_audio(ss.upper_path, g_u, T)
    audio_l, keep_l_clip, _, keep_l_path = build_warped_audio(ss.lower_path, g_l, T)
    # ベース音声から補足区間を無音化（補足は別トラック su/sl からだけ鳴らす）
    if audio_u is not None and mute_u:
        audio_u = _mute_regions_on_clip(audio_u, mute_u)
    if audio_l is not None and mute_l:
        audio_l = _mute_regions_on_clip(audio_l, mute_l)

    # ===== クリックトラック（終端安全化） =====
    click_y_sr = _render_click_track(
        target_beats, sr=44100, length=_safe_dur(T), base_freq=900
    )
    click_clip = None
    if click_y_sr is not None:
        click_y, click_sr = click_y_sr
        click_clip = _ndarray_to_audioclip(click_y, click_sr).volumex(gain_click)
        click_clip = click_clip.set_duration(_safe_dur(T))

    # サイズ表示（デバッグ用）
    st.caption(
        f"upper size: {warped_upper.w}x{warped_upper.h} / "
        f"lower size: {warped_lower.w}x{warped_lower.h}"
    )

    # ===== 合成（無音ベース動画） =====
    st.info("出力を合成中…（無音の合成動画を1本）")

    T = float(target_end_padded if "target_end_padded" in locals() else target_end)

    wu = warped_upper.set_duration(T)
    wl = warped_lower.set_duration(T)

    base_clip = _stack_vertical_noresize(wu, wl, bg=(0, 0, 0)).set_duration(T)
    V = _safe_dur(base_clip.duration or T)
    base_clip = base_clip.set_duration(V)

    out_base = SESSION_DIR / "preview_base_silent.mp4"

    # 540p固定スケール
    vf = "scale=-2:540,setsar=1"

    _safe_write_videofile(base_clip.without_audio(), out_base, fps=out_fps, vf=vf)
    try:
        base_clip.close()
    except Exception:
        pass

    # ===== 音声ミックスを書き出し（上/下/両方） =====
    def _write_audiofile_from_clip(
        aclip: AudioClip, path_: str, T=None, fps: int = 44100
    ) -> str:
        """
        CompositeAudioClip などを安全にファイル出力するヘルパ。
        T が指定されている場合でも、実際の aclip.duration を超えないように
        _safe_dur で丸めてから write_audiofile する。
        """
        # 目標長さをクリップの実長さに合わせて安全に丸める
        safe_T = _safe_dur(T, aclip)

        if safe_T is not None:
            aclip = aclip.set_duration(safe_T)

        aclip.write_audiofile(
            path_,
            fps=fps,
            codec="aac",
            ffmpeg_params=["-ac", "2"],
        )
        return path_

    # クリック（全モードに常に含める想定）
    _click = click_clip
    if _click is None:
        y_sr = _render_click_track(
            target_beats, sr=44100, length=_safe_dur(T), base_freq=900
        )
        if y_sr is not None:
            cy, csr = y_sr
            _click = _ndarray_to_audioclip(cy, csr).volumex(gain_click)

    if _click is not None:
        _click = _click.set_duration(_safe_dur(T)).audio_fadeout(0.02)

    # ==== 補足（原速）トラックを WAV から生成 ====
    def _supp_clip(wav_path, supp_segments, beats_src):
        if not (wav_path and supp_segments):
            return None
        return build_supplement_audio_clip(
            original_audio_path=wav_path,
            supp_segments=supp_segments,
            beats_src=np.array(beats_src, dtype=float),
            target_beats=target_beats,
            total_duration=_safe_dur(T),
            trim_to_span=True,
        )

    # 上半身の補足（原速）
    su = _supp_clip(wav_u, supp_u, upper_beats)
    # 下半身の補足（原速）
    sl = _supp_clip(wav_l, supp_l, lower_beats)

    # ==== ベースとなる BPM ワープ後の原音 ====
    base_u = audio_u.volumex(gain_upper) if audio_u is not None else None
    base_l = audio_l.volumex(gain_lower) if audio_l is not None else None

    # 補足にも同じゲインを掛ける
    if su is not None:
        su = su.volumex(gain_upper)
    if sl is not None:
        sl = sl.volumex(gain_lower)

    def _mix(*clips):
        clips = [c for c in clips if c is not None]
        if not clips:
            return None
        return CompositeAudioClip(clips).set_duration(_safe_dur(T))

    # ===== 最終ミックス =====
    # 上モード：上半身のBPMワープ原音 + 上補足 + クリック
    aud_upper = _mix(base_u, su, _click)

    # 下モード：下半身のBPMワープ原音 + 下補足 + クリック
    aud_lower = _mix(base_l, sl, _click)

    # 両方モード：上下のBPMワープ原音 + 上下補足 + クリック
    aud_both = _mix(base_u, base_l, su, sl, _click)

    p_aud_upper = SESSION_DIR / "audio_upper.m4a"
    p_aud_lower = SESSION_DIR / "audio_lower.m4a"
    p_aud_both = SESSION_DIR / "audio_both.m4a"

    for clip_, path_ in [
        (aud_upper, p_aud_upper),
        (aud_lower, p_aud_lower),
        (aud_both, p_aud_both),
    ]:
        if clip_ is None:
            continue
        try:
            path_written = _write_audiofile_from_clip(clip_, path_, T=T)
            if path_written is not None and path_ != path_written:
                path_ = path_written
        finally:
            try:
                clip_.close()
            except Exception:
                pass

    # 互換用エイリアス
    try:
        import shutil as _sh

        _sh.copy2(str(out_base), str(SESSION_DIR / "preview_synced.mp4"))
    except Exception:
        pass

    # ==== セッション変数に保存 ====
    ss.wav_u, ss.wav_l = wav_u, wav_l
    ss.supp_u, ss.supp_l = supp_u, supp_l
    ss.upper_beats, ss.lower_beats = upper_beats.tolist(), lower_beats.tolist()
    ss.target_end = float(target_end)
    ss.preview_path = str(SESSION_DIR / "preview_synced.mp4")
    ss.processed = True
    ss.target_beats = target_beats.tolist()
    ss.pre_u = _precut_segments(
        "上半身",
        supp_u,
        ss.upper_path,
        wav_u,
        out_fps,
        out_dir=SESSION_DIR,
        beats_src=np.array(upper_beats, dtype=float),
    )
    ss.pre_l = _precut_segments(
        "下半身",
        supp_l,
        ss.lower_path,
        wav_l,
        out_fps,
        out_dir=SESSION_DIR,
        beats_src=np.array(lower_beats, dtype=float),
    )

    # メタJSON
    try:
        meta = {
            "version": 2,
            "upper_path": ss.upper_path,
            "lower_path": ss.lower_path,
            "target_bpm": float(target_bpm),
            "offset_sec": float(offset_sec),
            "target_end": float(ss.target_end),
            "head_pad": float(head_pad_val),
            "tail_pad": float(tail_pad_val),
            "upper_beats": [float(x) for x in ss.upper_beats],
            "lower_beats": [float(x) for x in ss.lower_beats],
            "target_beats": [float(x) for x in ss.target_beats],
            "supp_u": [(float(a), float(b)) for (a, b) in ss.supp_u],
            "supp_l": [(float(a), float(b)) for (a, b) in ss.supp_l],
            "files": {
                "base": "preview_base_silent.mp4",
                "aud_upper": "audio_upper.m4a",
                "aud_lower": "audio_lower.m4a",
                "aud_both": "audio_both.m4a",
            },
        }
        (SESSION_DIR / "session.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), "utf-8"
        )
        try:
            (SESSION_DIR / "session.json").chmod(0o600)
        except Exception:
            pass
    except Exception as e:
        st.warning(f"メタ保存(session.json)に失敗: {e}")

    st.success("同期完了。タイムラインで元動画をクリック再生できます。")


def _resync_with_manual():
    """
    ss.* に保持されているカウント（upper_beats / lower_beats）と
    補足（supp_u / supp_l）をそのまま使って再同期する。
    （音声認識がうまくいかなかったときの「手動修正版で同期」用）
    """
    if not (ss.upper_path and ss.lower_path and ss.wav_u and ss.wav_l):
        st.error(
            "原本または一時WAVが見つかりません。先に一度『同期して書き出し』を実行してください。"
        )
        return

    upper_beats = np.array(ss.upper_beats, dtype=float)
    lower_beats = np.array(ss.lower_beats, dtype=float)
    supp_u = [(float(a), float(b)) for (a, b) in ss.supp_u]
    supp_l = [(float(a), float(b)) for (a, b) in ss.supp_l]

    _run_sync_pipeline(upper_beats, lower_beats, supp_u, supp_l, ss.wav_u, ss.wav_l)

    # =========================================


# 自動検出固定。CSV変数は参照エラー防止のためNoneで初期化
csv_count_ue = csv_count_sita = csv_hosoku_ue = csv_hosoku_sita = None
sync_method = "自動検出"

# ==== 👇 CSV読み込みモードの場合のみ処理 ====
if sync_method == "CSV読み込み":
    csv_count_ue.seek(0)
    csv_count_sita.seek(0)
    csv_hosoku_ue.seek(0)
    csv_hosoku_sita.seek(0)
    if csv_count_ue and csv_count_sita and csv_hosoku_ue and csv_hosoku_sita:
        df_count_ue = pd.read_csv(csv_count_ue).dropna()
        df_count_lo = pd.read_csv(csv_count_sita).dropna()
        df_supp_ue = pd.read_csv(csv_hosoku_ue).dropna(how="any")
        df_supp_lo = pd.read_csv(csv_hosoku_sita).dropna(how="any")

        upper_beats = df_count_ue.iloc[:, 0].dropna().astype(float).tolist()
        lower_beats = df_count_lo.iloc[:, 0].dropna().astype(float).tolist()
        supp_u = list(df_supp_ue.astype(float).itertuples(index=False, name=None))
        supp_l = list(df_supp_lo.astype(float).itertuples(index=False, name=None))

        # df_count_ue = pd.read_csv(csv_count_ue)
        # df_count_sita = pd.read_csv(csv_count_sita)
        # df_hosoku_ue = pd.read_csv(csv_hosoku_ue)
        # df_hosoku_sita = pd.read_csv(csv_hosoku_sita)

        # # リストに変換
        # upper_beats = df_count_ue["time"].dropna().astype(float).tolist()
        # lower_beats = df_count_sita["time"].dropna().astype(float).tolist()
        # supp_u = list(
        #     df_hosoku_ue.dropna().astype(float).itertuples(index=False, name=None)
        # )
        # supp_l = list(
        #     df_hosoku_sita.dropna().astype(float).itertuples(index=False, name=None)
        # )

        st.success("✅ アップロードされたCSVを読み込んで同期データを取得しました。")
    else:
        st.warning("⚠️ すべてのCSVファイルをアップロードしてください。")


# 実行ボタン：同期して書き出し
# =========================================
if st.button("同期して書き出し", type="primary"):
    # 入力の保存（原本をセッションフォルダに保存）
    if upper_video:
        ss.upper_path, _ = _ensure_local_path(
            upper_video, into_dir=SESSION_DIR, tag="upper"
        )
    if lower_video:
        ss.lower_path, _ = _ensure_local_path(
            lower_video, into_dir=SESSION_DIR, tag="lower"
        )

    if not (ss.upper_path and ss.lower_path):
        st.error("動画2本をアップロードしてください。")
        st.stop()

    # 音声抽出
    wav_u = str(SESSION_DIR / "tmp_upper.wav")
    wav_l = str(SESSION_DIR / "tmp_lower.wav")
    _extract_wav_from_path(ss.upper_path, wav_u, normalize=norm_audio)
    _extract_wav_from_path(ss.lower_path, wav_l, normalize=norm_audio)

    # ==== 手動マークの有無を判定 ====
    use_manual_u = len(ss.get("manual_beats_u", [])) >= 2
    use_manual_l = len(ss.get("manual_beats_l", [])) >= 2

    # ==== 👇 自動検出 or CSV で処理切り替え ====
    if sync_method == "自動検出":
        if use_manual_u and use_manual_l:
            # ===== 両方とも手動マーク：Whisperを使わず手動タイムスタンプで同期 =====
            st.success("手動マーク（タップ=カウント / 長押し=補足）を使って同期します。")
            upper_beats = np.array(sorted(ss["manual_beats_u"])[:8], dtype=float)
            lower_beats = np.array(sorted(ss["manual_beats_l"])[:8], dtype=float)
            supp_u = [(float(a), float(b)) for a, b in ss.get("manual_supp_u", [])]
            supp_l = [(float(a), float(b)) for a, b in ss.get("manual_supp_l", [])]
            supp_u = _merge_supp_no_beat_between(supp_u, upper_beats)
            supp_l = _merge_supp_no_beat_between(supp_l, lower_beats)
            # 番号順（撮影順）で対応付け。短い方に合わせる
            k = min(len(upper_beats), len(lower_beats))
            upper_beats = upper_beats[:k]
            lower_beats = lower_beats[:k]
            st.info(f"上半身: {len(upper_beats)}拍 / 下半身: {len(lower_beats)}拍（手動）")
        else:
            st.info(f"Whisper（{whisper_model_size}）でビート・補足を認識中…")

            _wmodel = _load_whisper_model(whisper_model_size)
            if _wmodel is None:
                st.stop()

            # ① カウント検出（手動があればそれを優先）
            if use_manual_u:
                upper_beats = np.array(sorted(ss["manual_beats_u"])[:8], dtype=float)
                st.caption("上半身：手動マークのカウントを使用")
            else:
                upper_beats = _detect_beats_by_speech(wav_u, _wmodel)
            if use_manual_l:
                lower_beats = np.array(sorted(ss["manual_beats_l"])[:8], dtype=float)
                st.caption("下半身：手動マークのカウントを使用")
            else:
                lower_beats = _detect_beats_by_speech(wav_l, _wmodel)

            # ② 補足検出（手動があればそれを優先）
            if use_manual_u and ss.get("manual_supp_u"):
                supp_u = [(float(a), float(b)) for a, b in ss["manual_supp_u"]]
            else:
                supp_u = _detect_supplement_segments_speech(
                    wav_u, _wmodel, merge_gap=float(unify_gap) if unify_gap > 0 else 0.40
                )
            if use_manual_l and ss.get("manual_supp_l"):
                supp_l = [(float(a), float(b)) for a, b in ss["manual_supp_l"]]
            else:
                supp_l = _detect_supplement_segments_speech(
                    wav_l, _wmodel, merge_gap=float(unify_gap) if unify_gap > 0 else 0.40
                )

            # 補足間にカウントがない場合は一つに結合
            supp_u = _merge_supp_no_beat_between(supp_u, upper_beats)
            supp_l = _merge_supp_no_beat_between(supp_l, lower_beats)

            # ③ カウント認識が不十分な場合はピーク検出でフォールバック（上限8件）
            if len(upper_beats) < 2:
                st.warning("上半身：Whisperでカウントを検出できませんでした。ピーク検出にフォールバックします。")
                upper_beats = _detect_beats_by_peaks(
                    wav_u, min_interval=min_interval, sens=sens, phase_label=phase_shift
                )[:8]
            if len(lower_beats) < 2:
                st.warning("下半身：Whisperでカウントを検出できませんでした。ピーク検出にフォールバックします。")
                lower_beats = _detect_beats_by_peaks(
                    wav_l, min_interval=min_interval, sens=sens, phase_label=phase_shift
                )[:8]

            st.info(f"上半身: {len(upper_beats)}カウント検出 / 下半身: {len(lower_beats)}カウント検出")

            # ④ 対応付け：手動マークが絡む場合は番号順、それ以外はMFCC
            if use_manual_u or use_manual_l:
                st.caption("手動マークを含むため番号順で対応付けます。")
                k = min(len(upper_beats), len(lower_beats))
                upper_beats = np.array(upper_beats, dtype=float)[:k]
                lower_beats = np.array(lower_beats, dtype=float)[:k]
            else:
                st.info("音声特徴量で上下カウントを対応付け中…")
                feats_u = _extract_beat_features(wav_u, upper_beats)
                feats_l = _extract_beat_features(wav_l, lower_beats)
                matched, sim_score = _match_beats_by_features(
                    upper_beats, feats_u, lower_beats, feats_l, threshold=0.70
                )
                if matched is not None:
                    upper_beats, lower_beats = matched
                    st.success(f"音声特徴量マッチング成功（平均類似度: {sim_score:.2f}）")
                else:
                    st.warning(
                        f"音声特徴量マッチングの品質が低いため（類似度: {sim_score:.2f}）、"
                        "番号順の対応に切り替えます。"
                    )
                    # フォールバック：短い方に合わせて先頭から順番に対応
                    k = min(len(upper_beats), len(lower_beats))
                    upper_beats = upper_beats[:k]
                    lower_beats = lower_beats[:k]

    else:
        # ==== CSV読み込み ====
        csv_count_ue.seek(0)
        csv_count_sita.seek(0)
        csv_hosoku_ue.seek(0)
        csv_hosoku_sita.seek(0)
        df_count_ue = pd.read_csv(csv_count_ue).dropna()
        df_count_sita = pd.read_csv(csv_count_sita).dropna()
        df_hosoku_ue = pd.read_csv(csv_hosoku_ue).dropna(how="any")
        df_hosoku_sita = pd.read_csv(csv_hosoku_sita).dropna(how="any")

        upper_beats = df_count_ue.iloc[:, 0].dropna().astype(float).tolist()
        lower_beats = df_count_sita.iloc[:, 0].dropna().astype(float).tolist()
        supp_u = list(df_hosoku_ue.astype(float).itertuples(index=False, name=None))
        supp_l = list(df_hosoku_sita.astype(float).itertuples(index=False, name=None))

        # df_count_ue = pd.read_csv(csv_count_ue)
        # df_count_lo = pd.read_csv(csv_count_sita)
        # df_supp_ue = pd.read_csv(csv_hosoku_ue).dropna()
        # df_supp_lo = pd.read_csv(csv_hosoku_sita).dropna()

        # upper_beats = df_count_ue["time"].dropna().astype(float).tolist()
        # lower_beats = df_count_sita["time"].dropna().astype(float).tolist()
        # supp_u = list(
        #     df_hosoku_ue.dropna().astype(float).itertuples(index=False, name=None)
        # )
        # supp_l = list(
        #     df_hosoku_sita.dropna().astype(float).itertuples(index=False, name=None)
        # )

        st.success("CSVからビート・補足を読み込みました！")

    # ==== 👇 最後の共通処理 ====
    _run_sync_pipeline(upper_beats, lower_beats, supp_u, supp_l, wav_u, wav_l)

    # # ビート列（自動検出）
    # st.info("音声ピークからビート抽出中…（ASRなし）")
    # upper_beats = _detect_beats_by_peaks(
    #     wav_u, min_interval=min_interval, sens=sens, phase_label=phase_shift
    # )
    # lower_beats = _detect_beats_by_peaks(
    #     wav_l, min_interval=min_interval, sens=sens, phase_label=phase_shift
    # )

    # # 補足の自動検出
    # supp_u = _detect_supplement_segments_auto(
    #     wav_u,
    #     base_th=float(supp_th),
    #     hole_close=0.20,
    #     merge_gap=0.35,
    #     edge_pad=0.06,
    #     min_silence_to_split=0.22,
    #     th_hyst=0.06,
    #     small_gap_unify=float(unify_gap),
    # )
    # supp_l = _detect_supplement_segments_auto(
    #     wav_l,
    #     base_th=float(supp_th),
    #     hole_close=0.20,
    #     merge_gap=0.35,
    #     edge_pad=0.06,
    #     min_silence_to_split=0.22,
    #     th_hyst=0.06,
    #     small_gap_unify=float(unify_gap),
    # )

    # # ピーク誤検出の除外
    # upper_beats = _exclude_peaks_near_segments(upper_beats, supp_u, radius=0.15)
    # lower_beats = _exclude_peaks_near_segments(lower_beats, supp_l, radius=0.15)

    # # 「補足っぽくない」区間の除外
    # supp_u = _filter_supp_segments_by_beats(
    #     supp_u, upper_beats, near=0.14, min_overlap_ratio=0.55
    # )
    # supp_l = _filter_supp_segments_by_beats(
    #     supp_l, lower_beats, near=0.14, min_overlap_ratio=0.55
    # )

    # _run_sync_pipeline(upper_beats, lower_beats, supp_u, supp_l, wav_u, wav_l)


# ====== 可視化（スペクトログラム） ======
if ss.processed:
    with st.expander("スペクトログラム＆同期対応点（赤丸＋上下接続線）", expanded=True):
        try:
            _plot_alignment_spectrograms_path(
                ss.upper_path,
                ss.lower_path,
                np.array(ss.upper_beats, dtype=float),
                np.array(ss.lower_beats, dtype=float),
                normalize_audio=norm_audio,
                fmin=50,
                fmax=8000,
                supp_u=ss.supp_u,
                supp_l=ss.supp_l,
            )
            st.caption("赤丸：拍/ピークの時刻。シアン：上下の対応点。")
        except Exception as e:
            st.warning(f"スペクトログラム可視化でエラー: {e}")

# ====== カウント＆補足の手動調整 ======
if ss.processed:
    st.subheader("カウント・補足の手動調整")

    tab_beats, tab_supp = st.tabs(["カウント（赤丸）", "補足（青ライン）"])

    # ---- カウント編集 ----
    with tab_beats:
        col_u, col_l = st.columns(2)

        with col_u:
            st.markdown("**上半身：カウント時刻 [s]**")
            df_u = pd.DataFrame({"time": ss.upper_beats})
            df_u_edit = st.data_editor(
                df_u,
                num_rows="dynamic",
                key="edit_upper_beats",
                hide_index=True,
            )

        with col_l:
            st.markdown("**下半身：カウント時刻 [s]**")
            df_l = pd.DataFrame({"time": ss.lower_beats})
            df_l_edit = st.data_editor(
                df_l,
                num_rows="dynamic",
                key="edit_lower_beats",
                hide_index=True,
            )

    # ---- 補足編集 ----
    with tab_supp:
        col_su, col_sl = st.columns(2)

        with col_su:
            st.markdown("**上半身：補足区間 [s]（start / end）**")
            df_su = pd.DataFrame(ss.supp_u, columns=["start", "end"])
            df_su_edit = st.data_editor(
                df_su,
                num_rows="dynamic",
                key="edit_supp_u",
                hide_index=True,
            )

        with col_sl:
            st.markdown("**下半身：補足区間 [s]（start / end）**")
            df_sl = pd.DataFrame(ss.supp_l, columns=["start", "end"])
            df_sl_edit = st.data_editor(
                df_sl,
                num_rows="dynamic",
                key="edit_supp_l",
                hide_index=True,
            )

    # 編集内容を ss.* に反映
    if st.button("上の編集内容をグラフに反映する", key="apply_manual_edits"):

        def _clean_times(series):
            out = []
            for v in series:
                if v is None or str(v) == "":
                    continue
                try:
                    t = float(v)
                except ValueError:
                    continue
                if t >= 0.0:
                    out.append(t)
            return out

        def _clean_segments(df):
            segs = []
            for _, row in df.iterrows():
                try:
                    a = float(row.get("start"))
                    b = float(row.get("end"))
                except (TypeError, ValueError):
                    continue
                if b > a:
                    segs.append((a, b))
            return segs

        ss.upper_beats = _clean_times(df_u_edit["time"])
        ss.lower_beats = _clean_times(df_l_edit["time"])
        ss.supp_u = _clean_segments(df_su_edit)
        ss.supp_l = _clean_segments(df_sl_edit)

        st.success(
            "編集内容を反映しました。スペクトログラムとタイムラインを更新します。"
        )
        _rerun()


# ====== タイムライン（▲クリックで原本を再生） ======


def _make_timeline_fig_with_mapping(
    supp_u,
    supp_l,
    upper_beats,
    lower_beats,
    target_beats,
    target_end,
    total_duration: float | None = None,
):
    _T = float(total_duration) if total_duration is not None else float(target_end)
    fig = go.Figure()

    # 軸ライン
    fig.add_trace(
        go.Scatter(
            x=[0, target_end],
            y=[1, 1],
            mode="lines",
            line=dict(dash="dot"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, target_end],
            y=[0, 0],
            mode="lines",
            line=dict(dash="dot"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # --- 原本の補足（これまで通り：上=青、下=赤） ---
    for a, b in supp_u:
        fig.add_shape(
            type="rect",
            x0=a,
            x1=b,
            y0=1 - 0.25,
            y1=1 + 0.25,
            line=dict(color="#1f77b4"),
            fillcolor="#1f77b4",
            opacity=0.25,
        )
    for a, b in supp_l:
        fig.add_shape(
            type="rect",
            x0=a,
            x1=b,
            y0=0 - 0.25,
            y1=0 + 0.25,
            line=dict(color="#d62728"),
            fillcolor="#d62728",
            opacity=0.25,
        )

    # ▲マーカー（開始点の目印）
    if supp_u:
        xs_u = [a for (a, _) in supp_u]
        ys_u = [1 + 0.28] * len(supp_u)
        cd_u = [
            {"key": f"U-{i+1}", "track_id": "U", "index": i, "start": a, "end": b}
            for i, (a, b) in enumerate(supp_u)
        ]
        fig.add_trace(
            go.Scatter(
                x=xs_u,
                y=ys_u,
                mode="markers",
                marker=dict(
                    symbol="triangle-up", size=12, line=dict(width=1), color="#1f77b4"
                ),
                name="上半身▲",
                customdata=cd_u,
                hovertemplate="上半身 #%{customdata.index}<br>開始: %{customdata.start:.2f}s<br>終了: %{customdata.end:.2f}s<extra></extra>",
            )
        )
    if supp_l:
        xs_l = [a for (a, _) in supp_l]
        ys_l = [0 + 0.28] * len(supp_l)
        cd_l = [
            {"key": f"L-{i+1}", "track_id": "L", "index": i, "start": a, "end": b}
            for i, (a, b) in enumerate(supp_l)
        ]
        fig.add_trace(
            go.Scatter(
                x=xs_l,
                y=ys_l,
                mode="markers",
                marker=dict(
                    symbol="triangle-up", size=12, line=dict(width=1), color="#d62728"
                ),
                name="下半身▲",
                customdata=cd_l,
                hovertemplate="下半身 #%{customdata.index}<br>開始: %{customdata.start:.2f}s<br>終了: %{customdata.end:.2f}s<extra></extra>",
            )
        )

    # --- 補足→直前カウント範囲（ターゲット時間軸）を帯で重ねる ---
    # 上半身は緑、下半身は橙で薄く塗る
    def _draw_mapped_spans(supp_segments, beats_src, lane_y, color):
        pairs = _pair_supp_with_count_ranges(supp_segments, beats_src)
        for n, ((a, b), (i0, i1)) in enumerate(pairs, start=1):
            if i0 >= len(target_beats):
                continue  # 安全のためスキップ
            tp = float(target_beats[i0])
            if (i1 + 1) < len(target_beats):
                te = float(target_beats[i1 + 1])
            else:
                te = float(_T)
            # 細めの帯（laneの上下にやや広げる）
            fig.add_shape(
                type="rect",
                x0=tp,
                x1=te,
                y0=lane_y - 0.40,
                y1=lane_y + 0.40,
                line=dict(color=f"rgba{color[:-1]},0.9)"),  # 濃い枠
                fillcolor=f"rgba{color[:-1]},0.18)",  # 薄い塗り
            )
            # ラベル（"U: 1-4" のように拍範囲を表示・中央に）
            xc = (tp + te) / 2.0
            fig.add_annotation(
                x=xc,
                y=lane_y + 0.48,
                text=f"{'U' if lane_y==1 else 'L'}: {i0+1}-{i1+1}",
                showarrow=False,
                font=dict(size=11),
                align="center",
            )

    _draw_mapped_spans(
        supp_u, np.array(upper_beats, dtype=float), lane_y=1, color="(44,160,44)"
    )  # 緑
    _draw_mapped_spans(
        supp_l, np.array(lower_beats, dtype=float), lane_y=0, color="(255,127,14)"
    )  # 橙

    # x終端（見切れ防止）
    x_end = (
        max(
            target_end or 0.0,
            (supp_u[-1][1] if supp_u else 0.0),
            (supp_l[-1][1] if supp_l else 0.0),
            (float(target_beats[-1]) if len(target_beats) else 0.0),
        )
        + 0.2
    )

    fig.update_layout(
        height=320,
        showlegend=False,
        font=dict(size=18),  # 全体
        xaxis=dict(
            title=dict(
                text="時間[s]",
                font=dict(size=20),  # ← ここが正解
            ),
            range=[0, x_end],
            tickfont=dict(size=18),
            automargin=True,
        ),
        yaxis=dict(
            title="",
            tickmode="array",
            tickvals=[0, 1],
            ticktext=["下半身", "上半身"],
            range=[-0.6, 1.6],
            tickfont=dict(size=18),
            automargin=True,
        ),
        margin=dict(l=40, r=10, t=20, b=40),
    )
    return fig


# def _show_original_by_key(key: str, note, audio_box, video_box):
#     lst = ss.pre_u if key.startswith("U-") else ss.pre_l
#     item = next((x for x in lst if x["key"] == key), None)
#     if not item:
#         note.warning("該当クリップが見つかりません")
#         return
#     note.info(f"原本: {key}  {item['a']:.2f}s→{item['b']:.2f}s")
#     audio_box.audio(item["wav"], format="audio/wav")
#     video_box.video(item["mp4"])


# def _show_original_by_key(key: str, note, audio_box, video_box):
#     lst = ss.pre_u if key.startswith("U-") else ss.pre_l
#     item = next((x for x in lst if x["key"] == key), None)
#     if not item:
#         note.warning("該当クリップが見つかりません")
#         return

#     # 表示文言：補足(a,b) と 直前カウント範囲[i_start,i_end]
#     txt = (
#         f"原本: {key}  補足 {item['a']:.2f}s→{item['b']:.2f}s  "
#         f"／ 直前カウント範囲: {item.get('i_start',0)+1}-{item.get('i_end',0)+1}"
#     )
#     note.info(txt)

#     # 音声は補足の音声（最後まで流れる）
#     audio_box.audio(item["wav"], format="audio/wav")

#     # 映像は"直前カウント範囲"の元動画（範囲で終了）
#     range_mp4 = item.get("range_mp4")
#     if range_mp4 and os.path.exists(range_mp4):
#         video_box.video(range_mp4)
#     else:
#         # フォールバック（まず起きない）：補足区間ビデオがあればそれを出す
#         alt = item.get("mp4") or ""
#         if alt and os.path.exists(alt):
#             video_box.video(alt)
#         else:
#             note.warning("プレビュー用ビデオが見つかりません。")


def _show_original_by_key(key: str, note, audio_box, video_box):
    lst = ss.pre_u if key.startswith("U-") else ss.pre_l
    item = next((x for x in lst if x["key"] == key), None)
    if not item:
        note.warning("該当クリップが見つかりません")
        return

    # 表示文言（今回の要件）
    txt = (
        f"原本: {key}  補足 {item['a']:.2f}s→{item['b']:.2f}s  "
        f"／ 直前カウント範囲: {item.get('i_start',0)+1}-{item.get('i_end',0)+1} "
        f"（映像=直前カウント範囲／音声=補足の説明のみを多重化）"
    )
    note.info(txt)

    # 別の audio は鳴らさない
    try:
        audio_box.empty()
    except Exception:
        pass

    # ★ 合成済みの範囲動画＋補足音声を再生
    merged = item.get("range_with_supp")
    if merged and os.path.exists(merged):
        video_box.video(merged)
    else:
        # フォールバック：無音の範囲動画
        range_mp4 = item.get("range_mp4")
        if range_mp4 and os.path.exists(range_mp4):
            video_box.video(range_mp4)
        else:
            note.warning("プレビュー用ビデオが見つかりません。")


if ss.processed:
    st.subheader("補足セグメント・タイムライン（▲をクリックで元動画を再生）")
    preview_note = st.empty()
    preview_audio = st.empty()
    preview_video = st.empty()

    if USE_PLOTLY:
        fig = _make_timeline_fig_with_mapping(
            ss.supp_u,
            ss.supp_l,
            np.array(ss.upper_beats, dtype=float),
            np.array(ss.lower_beats, dtype=float),
            np.array(ss.target_beats, dtype=float),
            ss.target_end,
            total_duration=float(ss.target_end) + float(head_pad) + float(tail_pad),
        )
        st.caption("▲をクリックで元動画を再生。")
        ev = plotly_events(
            fig,
            click_event=True,
            hover_event=ENABLE_HOVER_PREVIEW,
            select_event=False,
            override_height=320,
            key="timeline_plot",
        )

        if ev:
            e = ev[0]
            cd = e.get("customdata")
            key = None
            start = float(e.get("x", 0.0))
            end = 0.0
            if isinstance(cd, dict) and "key" in cd:
                key = str(cd["key"])
                start = float(cd.get("start", start))
                end = float(cd.get("end", end))
            else:
                cn = int(e.get("curveNumber", -1))
                trace_name = str(e.get("traceName", ""))
                if cn >= 0:
                    if "上" in trace_name or trace_name.startswith("上半身"):
                        track_id = "U"
                    elif "下" in trace_name or trace_name.startswith("下半身"):
                        track_id = "L"
                    else:
                        track_id = "U" if cn == 2 else "L" if cn == 3 else "U"
                else:
                    track_id = "U"
                pi = int(e.get("pointIndex", e.get("pointNumber", 0)))
                key = f"{track_id}-{pi+1}"

            label = "上半身" if key.startswith("U-") else "下半身"
            num = key.split("-", 1)[1] if "-" in key else "?"
            preview_note.markdown(
                f"#### 固定表示（原本）: {label} #{num}  {start:.2f}s→{end:.2f}s"
            )
            _show_original_by_key(key, preview_note, preview_audio, preview_video)

        if ENABLE_HOVER_PREVIEW and ev:
            e = ev[-1]
            cd = e.get("customdata")
            key = None
            if isinstance(cd, dict) and "key" in cd:
                key = str(cd["key"])
            else:
                cn = int(e.get("curveNumber", -1))
                trace_name = str(e.get("traceName", ""))
                if "上" in trace_name or trace_name.startswith("上半身"):
                    track_id = "U"
                elif "下" in trace_name or trace_name.startswith("下半身"):
                    track_id = "L"
                else:
                    track_id = "U" if cn == 2 else "L" if cn == 3 else "U"
                pi = int(e.get("pointIndex", e.get("pointNumber", 0)))
                key = f"{track_id}-{pi+1}"

            now = time.time()
            if key and (key != ss.last_hover_key or (now - ss.last_hover_ts) > 1.0):
                _show_original_by_key(key, preview_note, preview_audio, preview_video)
                ss.last_hover_key, ss.last_hover_ts = key, now
    else:
        st.warning("plotly/streamlit-plotly-events 未導入。ボタンで原本を再生します。")
        with st.expander("上半身（原本）", expanded=True):
            for i, item in enumerate(ss.pre_u):
                if st.button(
                    f"▶︎ 上 #{i+1} {item['a']:.2f}s→{item['b']:.2f}s", key=f"u{i}"
                ):
                    preview_note.info(
                        f"原本: U-{i+1}  {item['a']:.2f}s→{item['b']:.2f}s"
                    )
                    preview_audio.audio(item["wav"], format="audio/wav")
                    preview_video.video(item["mp4"])
        with st.expander("下半身（原本）", expanded=True):
            for i, item in enumerate(ss.pre_l):
                if st.button(
                    f"▶︎ 下 #{i+1} {item['a']:.2f}s→{item['b']:.2f}s", key=f"l{i}"
                ):
                    preview_note.info(
                        f"原本: L-{i+1}  {item['a']:.2f}s→{item['b']:.2f}s"
                    )
                    preview_audio.audio(item["wav"], format="audio/wav")
                    preview_video.video(item["mp4"])

    # # ====== 同期プレビュー（BPMワープ後） ======
    # if ss.preview_path and os.path.exists(ss.preview_path):
    #     st.subheader("プレビュー")
    #     st.video(ss.preview_path)
    #     with open(ss.preview_path, "rb") as f:
    #         st.download_button("動画をダウンロード", f, file_name=Path(ss.preview_path).name, mime="video/mp4")
    #     st.info(f"今回の成果物フォルダ: `{SESSION_DIR}`")

# # ====== 同期プレビュー（BPMワープ後） ======
# if ss.processed:
#     st.subheader("プレビュー（現状：無音ベース＋音声は別再生）")

#     base_path = SESSION_DIR / "preview_base_silent.mp4"
#     if base_path.exists():
#         st.video(str(base_path))
#     else:
#         st.warning("無音のベース動画が見つかりません。")

#     # 生成済みのミックス wav を個別に再生
#     p_aud_upper = SESSION_DIR / "audio_upper.wav"
#     p_aud_lower = SESSION_DIR / "audio_lower.wav"
#     p_aud_both = SESSION_DIR / "audio_both.wav"

#     c1, c2, c3 = st.columns(3)
#     with c1:
#         if p_aud_upper.exists():
#             st.markdown("**上だけ（クリック＋上補足）**")
#             st.audio(str(p_aud_upper))
#     with c2:
#         if p_aud_lower.exists():
#             st.markdown("**下だけ（クリック＋下補足）**")
#             st.audio(str(p_aud_lower))
#     with c3:
#         if p_aud_both.exists():
#             st.markdown("**両方（クリック＋上下補足）**")
#             st.audio(str(p_aud_both))

#     st.info(f"今回の成果物フォルダ: `{SESSION_DIR}`")

if ss.processed:
    st.markdown("---")
    if st.button(
        "⬇ 手動調整したカウント／補足で再同期する", type="primary", key="resync_manual"
    ):
        _resync_with_manual()


# ====== 同期プレビュー ======


def _render_sequential_preview():
    """パターン②：各カウント範囲について『補足音声だけ』→『無音の範囲動画』を
    順番に再生していくプレイヤー。ss.pre_u / ss.pre_l の生成物を再利用する。"""
    # 補足アイテムを収集（上半身→下半身、範囲の先頭ビート順）
    items = []
    for track_label, pre in (("上半身", ss.get("pre_u") or []), ("下半身", ss.get("pre_l") or [])):
        for it in pre:
            wav_p = it.get("wav")
            vid_p = it.get("range_mp4")
            if not (wav_p and vid_p and os.path.exists(wav_p) and os.path.exists(vid_p)):
                continue
            items.append(
                {
                    "track": track_label,
                    "key": it.get("key", ""),
                    "i_start": int(it.get("i_start", 0)),
                    "audio": _media_url(Path(wav_p)),
                    "video": _media_url(Path(vid_p)),
                }
            )
    # 範囲の先頭ビート順に整列（上半身優先）
    items.sort(key=lambda x: (x["i_start"], 0 if x["track"] == "上半身" else 1))

    if not items:
        st.info(
            "補足説明が登録されていないため、このパターンでは再生できません。"
            "（手動マークの長押し、または自動検出で補足が必要です）"
        )
        return

    st.caption(
        f"補足 {len(items)} 件を、各区間ごとに「🔊 補足説明のみ」→「▶ 無音の動画」の順で再生します。"
    )

    import json as _json

    items_json = _json.dumps(items, ensure_ascii=False)

    html = """
<div style="max-width:900px;margin:auto;font-family:system-ui,-apple-system,'Hiragino Sans','Noto Sans JP',sans-serif;color:#eee;">
  <div id="badge" style="padding:8px 12px;border-radius:10px;background:#222;font-size:14px;margin-bottom:8px;">
    準備中…
  </div>
  <div style="position:relative;background:#000;border-radius:10px;overflow:hidden;">
    <video id="demo" playsinline muted preload="auto"
           style="width:100%;max-height:520px;display:block;background:#000;"></video>
    <div id="phaseOverlay" style="position:absolute;left:0;right:0;top:0;bottom:0;
         display:flex;align-items:center;justify-content:center;pointer-events:none;
         font-size:22px;font-weight:700;text-shadow:0 1px 4px #000;"></div>
  </div>
  <audio id="supp" preload="auto"></audio>
  <div style="display:flex;gap:8px;margin-top:10px;flex-wrap:wrap;">
    <button id="startBtn" style="flex:1;min-width:120px;padding:12px;border:none;border-radius:10px;background:#16a34a;color:#fff;font-size:15px;font-weight:700;cursor:pointer;">▶ 最初から再生</button>
    <button id="nextBtn" style="padding:12px 16px;border:1px solid #555;border-radius:10px;background:#2b2b2b;color:#ddd;font-size:14px;cursor:pointer;">⏭ 次の区間へ</button>
  </div>
  <div id="progress" style="margin-top:8px;font-size:13px;color:#9ca3af;"></div>
</div>
<script>
(function(){
  const ITEMS = __ITEMS__;
  const demo = document.getElementById('demo');
  const supp = document.getElementById('supp');
  const badge = document.getElementById('badge');
  const overlay = document.getElementById('phaseOverlay');
  const startBtn = document.getElementById('startBtn');
  const nextBtn = document.getElementById('nextBtn');
  const progress = document.getElementById('progress');

  let idx = 0;
  let phase = 'idle';   // 'audio' | 'video' | 'done'

  function setProgress(){
    progress.textContent = '区間 ' + Math.min(idx+1, ITEMS.length) + ' / ' + ITEMS.length;
  }
  function badgeText(it, ph){
    return (ph==='audio' ? '🔊 補足説明のみ' : '▶ 無音の動画') +
           '　[' + it.track + ' ' + it.key + ']';
  }

  function playAudioPhase(){
    const it = ITEMS[idx];
    phase = 'audio';
    badge.textContent = badgeText(it, 'audio');
    overlay.textContent = '🔊 補足説明';
    // 動画は範囲の先頭フレームを表示したまま静止
    demo.src = it.video;
    demo.muted = true;
    demo.pause();
    try { demo.currentTime = 0; } catch(_){}
    supp.src = it.audio;
    supp.currentTime = 0;
    supp.play().catch(()=>{});
    setProgress();
  }
  function playVideoPhase(){
    const it = ITEMS[idx];
    phase = 'video';
    badge.textContent = badgeText(it, 'video');
    overlay.textContent = '';
    try { supp.pause(); } catch(_){}
    demo.muted = true;
    try { demo.currentTime = 0; } catch(_){}
    demo.play().catch(()=>{});
  }
  function nextItem(){
    idx++;
    if (idx >= ITEMS.length){
      phase = 'done';
      badge.textContent = '✅ すべての区間を再生しました';
      overlay.textContent = '';
      startBtn.textContent = '🔁 もう一度最初から';
      return;
    }
    playAudioPhase();
  }

  supp.addEventListener('ended', () => { if (phase==='audio') playVideoPhase(); });
  demo.addEventListener('ended', () => { if (phase==='video') nextItem(); });

  startBtn.addEventListener('click', () => {
    idx = 0; startBtn.textContent = '▶ 最初から再生'; playAudioPhase();
  });
  nextBtn.addEventListener('click', () => {
    // 現区間をスキップして次へ（音声中でも動画中でも）
    try { supp.pause(); } catch(_){}
    try { demo.pause(); } catch(_){}
    nextItem();
  });

  setProgress();
  badge.textContent = '▶ 「最初から再生」を押してください';
})();
</script>
"""
    html = html.replace("__ITEMS__", items_json)

    import streamlit.components.v1 as components

    components.html(html, height=680, scrolling=False)


if ss.processed:
    st.subheader("プレビュー")
    _pattern = st.radio(
        "再生パターン",
        ["① ホバーで補足音声を切替（通常）", "② 補足音声 → 無音動画 を順番に再生"],
        key="preview_pattern",
        horizontal=True,
    )
    ss["_pattern_is_seq"] = _pattern.startswith("②")

# ---- パターン②：補足音声 → 無音動画 の順次再生 ----
if ss.processed and ss.get("_pattern_is_seq"):
    _render_sequential_preview()

# ---- パターン①：従来のホバー音声切替 ----
if ss.processed and not ss.get("_pattern_is_seq"):
    st.caption("上：上半身のみ／中央：両方／下：下半身のみ（ホバーで切替）")

    base = SESSION_DIR / "preview_base_silent.mp4"
    a_up = SESSION_DIR / "audio_upper.m4a"
    a_lo = SESSION_DIR / "audio_lower.m4a"
    a_bo = SESSION_DIR / "audio_both.m4a"

    if not (base.exists() and a_up.exists() and a_lo.exists() and a_bo.exists()):
        st.warning(
            "プレビュー用のファイルが不足しています。上の同期処理を実行してください。"
        )
        st.stop()

    import streamlit.components.v1 as components  # ← 重複せずココ1行だけ

    html = """
<div style="max-width: 900px;margin:auto;">

  <div id="modeBadge"
       style="position:relative;margin-bottom:8px;font-family:system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;">
    <span id="label"
          style="padding:6px 10px;border-radius:8px;background:#222;color:#fff;font-size:12px;">
      Hover area: BOTH
    </span>
  </div>

  <div id="playerWrap" style="position:relative;display:inline-block;">
    <video id="vid" src="__BASE__"
           style="max-width:100%;border-radius:8px;background:#000;display:block;"
           playsinline controls preload="metadata"></video>

    <!-- ▼ これが抜けていた：ホバー判定用の透明オーバーレイ -->

    <audio id="audUp" src="__AUP__" preload="auto" playsinline crossorigin="anonymous"></audio>
    <audio id="audLo" src="__ALO__" preload="auto" playsinline crossorigin="anonymous"></audio>
    <audio id="audBo" src="__ABO__" preload="auto" playsinline crossorigin="anonymous"></audio>

    </div>

  <div style="margin-top:8px;opacity:0.8;font-size:12px;">
    上：上半身のみ／中央：両方／下：下半身のみ。<br>
    再生/一時停止は動画をクリック。音声は常に動画と同期します。
  </div>
</div>

<script>
(function(){
  const vid   = document.getElementById('vid');
  const up    = document.getElementById('audUp');
  const lo    = document.getElementById('audLo');
  const both  = document.getElementById('audBo');
  const label = document.getElementById('label');
  const wrap  = document.getElementById('playerWrap');

  // --- 1) 初回ジェスチャで音声デバイスを解錠（ネイティブcontrols対応） ---
  let unlocked = false;
  function unlockAllOnce() {
    if (unlocked) return;
    unlocked = true;
    [up, lo, both].forEach(a => {
      try {
        a.muted = true;
        a.play().then(() => { try{ a.pause(); }catch(_){}}).catch(()=>{});
      } catch(_) {}
    });
  }
  // クリック系どこでもOK（controls押下も拾う）
  for (const el of [document, wrap, vid]) {
    el.addEventListener('pointerdown', unlockAllOnce, { once: true, passive: true });
    el.addEventListener('touchstart',  unlockAllOnce, { once: true, passive: true });
    el.addEventListener('keydown',     unlockAllOnce, { once: true });
  }

  // --- 2) 状態 ---
  let mode = 'both'; // 'upper' | 'lower' | 'both'
  let syncing = false;

  function ensureSync(a){
    if (!a) return;
    const dt = Math.abs((a.currentTime||0) - (vid.currentTime||0));
    if (dt > 0.08) a.currentTime = vid.currentTime;
  }

  function applyMode(m){
    mode = m;
    const mu = (m === 'upper'), ml = (m === 'lower'), mb = (m === 'both');

    up.muted   = !mu;
    lo.muted   = !ml;
    both.muted = !mb;

    // 動画再生中なら直ちに反映
    [up, lo, both].forEach(a => {
      try { a.volume = 1.0; } catch(_){}
      ensureSync(a);
      if (!a.muted && !vid.paused) {
        a.play().catch(()=>{});
      }
    });
    label.textContent = 'Hover area: ' + (mu?'UPPER':ml?'LOWER':'BOTH');
  }

  // --- 3) 事前 priming（loadを明示） ---
  function prime(a){
    if (!a) return;
    try { a.load(); } catch(_){}
    try { a.volume = 1.0; } catch(_){}
  }
  [up, lo, both].forEach(prime);

  // 初期は BOTH を鳴らす前提でミュート設定
  up.muted = true; lo.muted = true; both.muted = false;

  // --- 4) ネイティブ再生ボタン対応：play/pause イベントで連動 ---
  vid.addEventListener('play', async () => {
    // 動画が再生されたら選択中の1本だけ鳴らす
    for (const a of [up, lo, both]) {
      try {
        ensureSync(a);
        if (
          (mode === 'upper' && a === up) ||
          (mode === 'lower' && a === lo) ||
          (mode === 'both'  && a === both)
        ) {
          a.muted = false;
          await a.play();
        } else {
          a.pause();
        }
      } catch(_){}
    }
  });

  vid.addEventListener('pause', () => {
    [up, lo, both].forEach(a => { try { a.pause(); } catch(_){} });
  });

  // 既存のクリックトグルも残したい場合（任意）
  vid.addEventListener('click', async () => {
    if (vid.paused) {
      try { await vid.play(); } catch(_){}
    } else {
      vid.pause();
    }
  });

  // --- 5) 再生中の微小ズレ補正 ---
  vid.addEventListener('timeupdate', () => {
    if (syncing) return;
    syncing = true;
    [up, lo, both].forEach(ensureSync);
    syncing = false;
  });
  vid.addEventListener('ended', () => {
    [up, lo, both].forEach(a => { try{ a.pause(); }catch(_){}} );
  });

  // --- 6) hover area 判定：必ず video 自身で拾う（controls の上でもOK） ---
  function handlePointerMove(ev){
    const rect = vid.getBoundingClientRect();
    if (rect.height <= 0) return;
    const y = (ev.clientY - rect.top) / rect.height;
    const newMode = (y < 0.40) ? 'upper' : (y > 0.60) ? 'lower' : 'both';
    if (newMode !== mode) applyMode(newMode);
  }
  vid.addEventListener('pointermove',  handlePointerMove, { passive: true });
  vid.addEventListener('mousemove',    handlePointerMove, { passive: true });
  vid.addEventListener('mouseleave',   () => applyMode('both'));

  // --- 7) ラベル初期表示＆軽い同期 ---
  applyMode('both');
  [up, lo, both].forEach(ensureSync);
})();
</script>

"""

    # ローカルファイルを Streamlit のメディアURLへ
    base_url = _media_url(base)
    a_up_url = _media_url(a_up)
    a_lo_url = _media_url(a_lo)
    a_bo_url = _media_url(a_bo)

    html = (
        html.replace("__BASE__", base_url)
        .replace("__AUP__", a_up_url)
        .replace("__ALO__", a_lo_url)
        .replace("__ABO__", a_bo_url)
    )

    # ← 高さは"ベース動画の実解像度"から決める（warped_* は使わない）
    try:
        from moviepy.editor import VideoFileClip as _V

        with _V(str(base)) as _clip:
            h_px = int(_clip.h)
    except Exception:
        h_px = 480

    # ▼ Safari等でコントロールが隠れないよう余白を増やす
    _SCALE = 0.85  # 表示倍率（従来 0.70 だとやや窮屈）
    _UI_MARGIN = 220  # 再生ボタン/説明テキストぶんの余白（従来 160）
    components.html(
        html,
        height=int(h_px * _SCALE) + _UI_MARGIN,
        scrolling=False,
    )

    st.info(f"今回の成果物フォルダ: `{SESSION_DIR}`")

# ====== 5. 過去セッション履歴（再生・削除・原本復元で再同期） ======
st.subheader("履歴")
sessions = _list_sessions(ss.auth_user)
if not sessions:
    st.caption("まだ履歴はありません。")
else:
    for sdir in sessions:
        with st.expander(f"{sdir.name}", expanded=False):
            vids = sorted(sdir.glob("*.mp4"))
            wavs = sorted(sdir.glob("*.wav"))
            up_dir = sdir / "uploads"
            up_upper = next(iter(sorted(up_dir.glob("upper*"))), None)
            up_lower = next(iter(sorted(up_dir.glob("lower*"))), None)

            colA, colB, colC = st.columns([2, 2, 1])

            # ★履歴でも"単一ホルダ"にだけ描画（大量<video>生成を防ぐ）
            hist_video_holder = st.empty()
            hist_audio_holder = st.empty()

            with colA:
                st.markdown("**動画一覧（クリックで再生）**")
                if vids:
                    for v in vids:
                        if st.button(f"▶ {v.name}", key=f"play_{sdir.name}_{v.name}"):
                            hist_video_holder.video(str(v))
                else:
                    st.caption("（このセッションに mp4 はありません）")

            with colB:
                st.markdown("**音声ファイル**")
                if wavs:
                    for w in wavs:
                        if st.button(
                            f"♪ 再生 {w.name}", key=f"playa_{sdir.name}_{w.name}"
                        ):
                            hist_audio_holder.audio(str(w))
                else:
                    st.caption("（このセッションに wav はありません）")

            with colC:
                st.markdown("**操作**")
                if up_upper and up_lower:
                    if st.button("原本復元→現在入力に設定", key=f"restore_{sdir.name}"):
                        ss.upper_path = str(up_upper)
                        ss.lower_path = str(up_lower)
                        ss.processed = False
                        ss.preview_path = None
                        st.success(
                            "原本を現在の入力に設定しました。上の『同期して書き出し』で再同期できます。"
                        )
                        _rerun()
                else:
                    st.caption("原本が保存されていません")

                del_key = f"del_{sdir.name}"
                confirm = st.checkbox("この履歴を削除する", key=f"ck_{sdir.name}")
                if st.button("🗑 完全削除", key=del_key, disabled=not confirm):
                    try:
                        shutil.rmtree(sdir, ignore_errors=False)
                        st.success("削除しました。")
                        _rerun()
                        st.stop()
                    except Exception as e:
                        st.error(f"削除に失敗: {e}")
