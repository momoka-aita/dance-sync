# pages/3_動画音声履歴.py
import os
from pathlib import Path
import re
import streamlit as st
from auth_shared import page_scaffold, list_sessions, DATA_ROOT
import json
import numpy as np

try:
    import plotly.graph_objects as go
    from streamlit_plotly_events import plotly_events

    HAVE_PLOTLY = True
except Exception:
    HAVE_PLOTLY = False

import streamlit.components.v1 as components

with page_scaffold(
    title="📝 記録",
    page_title="記録 - Dance Sync",
    active_tab_label="記録",
    layout="centered",
):
    ss = st.session_state
    username = ss["auth_user"]

    # --------- サムネ生成（必要時のみ・キャッシュ） ---------
    @st.cache_data(show_spinner=False)
    def generate_thumb_once(video_path: str) -> str | None:
        """
        指定mp4のサムネjpgを動画と同じフォルダに生成（既にあれば再生成しない）。
        返り値: サムネの絶対パス or None
        """
        try:
            p = Path(video_path)
            if not p.exists():
                return None
            out = p.with_suffix(".thumb.jpg")
            if out.exists():  # 既存サムネを使う
                return str(out)

            from moviepy.editor import VideoFileClip

            clip = VideoFileClip(str(p))
            t = (
                min(0.3, (clip.duration or 1.0) - 0.01)
                if (clip.duration or 0) > 0.5
                else 0.0
            )
            frame = clip.get_frame(t)
            clip.close()

            from PIL import Image

            im = Image.fromarray(frame)
            im.thumbnail((480, 480))
            im.save(out, quality=85)
            try:
                out.chmod(0o600)
            except Exception:
                pass
            return str(out)
        except Exception:
            return None

    # --------- 履歴スキャン（結果はキャッシュ） ---------
    @st.cache_data(show_spinner=False)
    def scan_history(_username: str):
        data = []
        for sdir in list_sessions(_username):
            up_dir = sdir / "uploads"
            up_upper = next(iter(sorted(up_dir.glob("upper*"))), None)
            up_lower = next(iter(sorted(up_dir.glob("lower*"))), None)
            vids = sorted(sdir.glob("*.mp4"))
            wavs = sorted(sdir.glob("*.wav"))
            data.append(
                {
                    "sdir": str(sdir),
                    "vids": [str(v) for v in vids],
                    "wavs": [str(w) for w in wavs],
                    "up_upper": str(up_upper) if up_upper else None,
                    "up_lower": str(up_lower) if up_lower else None,
                }
            )
        return data

    def _load_session_meta(sdir: Path) -> dict | None:
        p = sdir / "session.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text("utf-8"))
        except Exception:
            return None

    def _media_url(path: Path) -> str:
        import base64

        p = Path(path)
        if not p.exists():
            return ""
        ext = p.suffix.lower()
        # ★ ここを明示マッピングに
        if ext == ".mp4":
            mime = "video/mp4"
        elif ext in (".wav",):
            mime = "audio/wav"
        elif ext in (".m4a", ".mp4a"):
            mime = "audio/mp4"  # ← 重要（Safari で必須）
        elif ext == ".mp3":
            mime = "audio/mpeg"
        else:
            mime = "application/octet-stream"

        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def _pair_supp_with_count_ranges(supp_segments, beats_src: np.ndarray):
        if beats_src is None or len(beats_src) == 0 or not supp_segments:
            return []
        segs = sorted(
            [(float(a), float(b)) for a, b in supp_segments], key=lambda x: x[0]
        )
        out = []
        last_j = None
        last_i_start, last_i_end = 0, -1
        for a, b in segs:
            j = int(np.searchsorted(beats_src, a, side="right") - 1)
            if j < 0:  # 最初のビートより前はスキップ
                continue
            if last_j is None:
                i_start, i_end = 0, j
            elif j > last_j:
                i_start, i_end = last_i_end + 1, j
            else:
                i_start, i_end = last_i_start, last_i_end
            if i_start > i_end or i_end < 0:
                i_start = max(0, min(j, len(beats_src) - 1))
                i_end = j
            out.append(((a, b), (i_start, i_end)))
            last_j, last_i_start, last_i_end = j, i_start, i_end
        return out

    def _make_timeline_fig_with_mapping(meta: dict):
        supp_u = [(float(a), float(b)) for a, b in meta.get("supp_u", [])]
        supp_l = [(float(a), float(b)) for a, b in meta.get("supp_l", [])]
        upper_beats = np.array(meta.get("upper_beats", []), dtype=float)
        lower_beats = np.array(meta.get("lower_beats", []), dtype=float)
        target_beats = np.array(meta.get("target_beats", []), dtype=float)
        target_end = float(
            meta.get("target_end", target_beats[-1] if len(target_beats) else 0.0)
        )
        head_pad = float(meta.get("head_pad", 0.0))
        tail_pad = float(meta.get("tail_pad", 0.0))
        _T = float(target_end + head_pad + tail_pad)

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

        # 原本補足（帯）
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

        # ▲マーカー（customdataに key を入れる）
        # ▲マーカー（customdata を "U-1" / "L-3" の文字列に）
        if supp_u:
            xs = [a for a, _ in supp_u]
            cds = [f"U-{i+1}" for i in range(len(supp_u))]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=[1 + 0.28] * len(xs),
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=12, color="#1f77b4"),
                    name="上半身▲",
                    customdata=cds,
                    hovertemplate="上半身 %{customdata}<extra></extra>",
                    selectedpoints=[],
                    showlegend=False,
                )
            )
        if supp_l:
            xs = [a for a, _ in supp_l]
            cds = [f"L-{i+1}" for i in range(len(supp_l))]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=[0 + 0.28] * len(xs),
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=12, color="#d62728"),
                    name="下半身▲",
                    customdata=cds,
                    hovertemplate="下半身 %{customdata}<extra></extra>",
                    selectedpoints=[],
                    showlegend=False,
                )
            )

        # 補足→直前カウント範囲（ターゲット時間軸）帯
        def _draw_map(segs, beats_src, lane_y, rgba):
            for (a, b), (i0, i1) in _pair_supp_with_count_ranges(
                segs, np.asarray(beats_src)
            ):
                tp = float(target_beats[i0])
                te = (
                    float(target_beats[i1 + 1])
                    if (i1 + 1) < len(target_beats)
                    else float(_T)
                )
                fig.add_shape(
                    type="rect",
                    x0=tp,
                    x1=te,
                    y0=lane_y - 0.40,
                    y1=lane_y + 0.40,
                    line=dict(color=f"rgba{rgba[:-1]},0.9)"),
                    fillcolor=f"rgba{rgba[:-1]},0.18)",
                )
                fig.add_annotation(
                    x=(tp + te) / 2,
                    y=lane_y + 0.48,
                    text=f"{'U' if lane_y==1 else 'L'}: {i0+1}-{i1+1}",
                    showarrow=False,
                    font=dict(size=11),
                )

        _draw_map(supp_u, upper_beats, 1, "(44,160,44)")
        _draw_map(supp_l, lower_beats, 0, "(255,127,14)")

        x_end = (
            max(
                target_end,
                *(b for _, b in supp_u),
                *(b for _, b in supp_l),
                (target_beats[-1] if len(target_beats) else 0.0),
            )
            + 0.2
        )
        fig.update_layout(
            height=320,
            showlegend=False,
            xaxis=dict(title="Time [s]", range=[0, x_end]),
            yaxis=dict(
                tickmode="array",
                tickvals=[0, 1],
                ticktext=["下半身", "上半身"],
                range=[-0.6, 1.6],
            ),
            margin=dict(l=40, r=10, t=20, b=40),
        )
        fig.update_layout(clickmode="event+select")
        return fig

    # ▼ 置換：履歴のホバープレビュー（unlockAllOnce 方式・メインと同等）
    def _history_show_hover_preview(sdir: Path, meta: dict):
        files = meta.get("files", {})
        base = sdir / files.get("base", "preview_base_silent.mp4")
        a_up = sdir / files.get("aud_upper", "audio_upper.m4a")
        a_lo = sdir / files.get("aud_lower", "audio_lower.m4a")
        a_bo = sdir / files.get("aud_both", "audio_both.m4a")
        if not (base.exists() and a_up.exists() and a_lo.exists() and a_bo.exists()):
            st.warning("プレビューに必要なファイルが不足しています。")
            return

        html = """
<div style="max-width: 900px;margin:auto;">
  <div id="modeBadge"
       style="position:relative;margin-bottom:8px;font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;">
    <span id="label"
          style="padding:6px 10px;border-radius:8px;background:#222;color:#fff;font-size:12px;">
      Hover area: BOTH
    </span>
  </div>

  <div id="playerWrap" style="position:relative;display:inline-block;">
    <video id="vid" src="__BASE__"
       style="width:100%;max-width:100%;height:auto;border-radius:8px;background:#000;display:block;"
       playsinline controls preload="metadata"></video>

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

  // --- メインと同じ：ユーザー操作一回で全トラックを解錠 ---
  let unlocked = false;
  function unlockAllOnce() {
    if (unlocked) return;
    unlocked = true;
    [up, lo, both].forEach(a => {
      try {
        a.muted = true;
        a.play().then(()=>{ try{ a.pause(); }catch(_){}}).catch(()=>{});
      } catch(_) {}
    });
  }
  for (const el of [document, wrap, vid]) {
    el.addEventListener('pointerdown', unlockAllOnce, { once: true, passive: true });
    el.addEventListener('touchstart',  unlockAllOnce, { once: true, passive: true });
    el.addEventListener('keydown',     unlockAllOnce, { once: true });
  }

  let mode = 'both';
  let syncing = false;

  function ensureSync(a){
    if (!a) return;
    const dt = Math.abs((a.currentTime||0) - (vid.currentTime||0));
    if (dt > 0.08) a.currentTime = vid.currentTime;
  }

  function applyMode(m){
    mode = m;
    const mu = (m==='upper'), ml = (m==='lower'), mb = (m==='both');
    up.muted   = !mu;
    lo.muted   = !ml;
    both.muted = !mb;
    [up, lo, both].forEach(a => {
      try { a.volume = 1.0; } catch(_){}
      ensureSync(a);
      if (!a.muted && !vid.paused) { a.play().catch(()=>{}); }
      else { a.pause(); }
    });
    label.textContent = 'Hover area: ' + (mu?'UPPER':ml?'LOWER':'BOTH');
  }

  function prime(a){ try{ a.load(); a.volume=1.0; }catch(_){}} 
  [up, lo, both].forEach(prime);
  up.muted=true; lo.muted=true; both.muted=false;

  vid.addEventListener('play', async () => {
    for (const a of [up, lo, both]) {
      try{
        ensureSync(a);
        if ((mode==='upper'&&a===up)||(mode==='lower'&&a===lo)||(mode==='both'&&a===both)){
          a.muted=false; await a.play();
        } else { a.pause(); }
      }catch(_){}
    }
  });
  vid.addEventListener('pause', ()=>{ [up,lo,both].forEach(a=>{ try{a.pause();}catch(_){}}); });
  vid.addEventListener('click', async ()=>{ if (vid.paused){ try{await vid.play();}catch(_){}} else { vid.pause(); }});
  vid.addEventListener('timeupdate', ()=>{ if (syncing) return; syncing=true; [up,lo,both].forEach(ensureSync); syncing=false; });
  vid.addEventListener('ended', ()=>{ [up,lo,both].forEach(a=>{ try{a.pause();}catch(_){}}); });

  function handlePointerMove(ev){
    const rect = vid.getBoundingClientRect();
    if (rect.height<=0) return;
    const y = (ev.clientY - rect.top) / rect.height;
    const m = (y < 0.40) ? 'upper' : (y > 0.60) ? 'lower' : 'both';
    if (m !== mode) applyMode(m);
  }
  vid.addEventListener('pointermove', handlePointerMove, {passive:true});
  vid.addEventListener('mousemove',   handlePointerMove, {passive:true});
  vid.addEventListener('mouseleave',  ()=>applyMode('both'));

  applyMode('both');
  [up,lo,both].forEach(ensureSync);
})();
</script>
"""
        html = (
            html.replace("__BASE__", _media_url(base))
            .replace("__AUP__", _media_url(a_up))
            .replace("__ALO__", _media_url(a_lo))
            .replace("__ABO__", _media_url(a_bo))
        )
        components.html(html, height=720, scrolling=False)

    # def _history_open_range_with_supp(sdir: Path, key: str) -> str | None:
    #     """
    #     key: "U-3" / "L-2" など → 合成済み(range_with_supp)があればそのパス、
    #     なければ無音の range.mp4 のパスを返す。見つからなければ None。
    #     """
    #     track = "U" if key.startswith("U-") else "L"
    #     idx = key.split("-", 1)[1]

    #     patterns = [
    #         f"orig_{track}-{idx}_*.range_with_supp.mp4",
    #         f"orig_{track}-{idx}_*.range.mp4",
    #         # ★ フォールバック：小数が入らない・ハイフンや桁ゆらぎなどに緩くマッチ
    #         f"orig_{track}-{idx}*.range_with_supp.mp4",
    #         f"orig_{track}-{idx}*.range.mp4",
    #     ]
    #     for pat in patterns:
    #         cand = sorted(sdir.glob(pat))
    #         if cand:
    #             return str(cand[0])
    #     return None

    # def _history_open_range_with_supp(sdir: Path, key: str) -> str | None:
    #     """
    #     key: "U-3" / "L-2" など → セグメントファイルを探索して返す
    #     """
    #     # key を元にどのファイルを再生するかを探す
    #     track = "U" if key.startswith("U-") else "L"
    #     idx = key.split("-", 1)[1]

    #     pat = re.compile(
    #         rf"^orig_{track}-{idx}_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)\.range(?:_with_supp)?\.mp4$"
    #     )
    #     candidates = []
    #     for f in sorted(Path(sdir).glob(f"orig_{track}-{idx}_*.mp4")):
    #         m = pat.match(f.name)
    #         if m:
    #             a = float(m.group(1))
    #             b = float(m.group(2))
    #             candidates.append((a, b, f))

    #     # range_with_supp を優先して探す
    #     for _, _, f in sorted(candidates):
    #         if "range_with_supp" in f.name:
    #             return str(f)

    #     # なければ range.mp4 を返す
    #     for _, _, f in sorted(candidates):
    #         if "range_with_supp" not in f.name and "range" in f.name:
    #             return str(f)

    #     return None

    def _history_open_range_with_supp(sdir: Path, key: str) -> str | None:
        """
        key: "U-3" / "L-2" など → indexベースで一致する range_with_supp → range.mp4 を探す
        """
        track = "U" if key.startswith("U-") else "L"
        idx = key.split("-", 1)[1]

        # indexのみでファイルを探す（時間範囲は無視）
        candidates = sorted(Path(sdir).glob(f"orig_{track}-{idx}_*.mp4"))

        # range_with_supp 優先
        for f in candidates:
            if "range_with_supp" in f.name:
                return str(f)

        # なければ range.mp4
        for f in candidates:
            if "range" in f.name and "with_supp" not in f.name:
                return str(f)

        return None

    def _scan_orig_segments_from_folder(sdir: Path):
        """
        session フォルダ直下の orig_U-1_5.987_7.346.mp4 等を集めて
        ミニ・タイムライン（▲）用に整形。
        返り値: (items_u, items_l)
        """
        pat = re.compile(r"^orig_(U|L)-(\d+)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?).mp4$")
        items_u, items_l = [], []
        for mp4 in sorted(Path(sdir).glob("orig_*-*.mp4")):
            m = pat.match(mp4.name)
            if not m:
                continue
            track, idx, a, b = (
                m.group(1),
                int(m.group(2)),
                float(m.group(3)),
                float(m.group(4)),
            )
            rec = {
                "key": f"{track}-{idx}",
                "a": a,
                "b": b,
                "mp4": str(mp4),
                "wav": str(mp4.with_suffix(".wav")),
            }
            (items_u if track == "U" else items_l).append(rec)
        return items_u, items_l

    # ===== UI本体 =====
    left, right = st.columns([3, 1])
    with left:
        st.caption(f"ユーザー: **{username}** / ルート: `{DATA_ROOT}`")
    with right:
        if st.button("キャッシュ更新（再スキャン）", use_container_width=True):
            scan_history.clear()
            generate_thumb_once.clear()
            st.rerun()

    items = scan_history(username)
    if not items:
        st.caption("まだ記録はありません。")
        st.stop()

    # 検索 & ページネーション
    q = st.text_input("🔎 セッション名/ファイル名でフィルタ", value="")
    per_page = st.selectbox("表示件数", [5, 10, 20, 50], index=1)
    page = st.number_input("ページ", min_value=1, value=1, step=1)

    def visible(it):
        if not q:
            return True
        ql = q.lower()
        if ql in Path(it["sdir"]).name.lower():
            return True
        for p in it["vids"]:
            if ql in Path(p).name.lower():
                return True
        for p in it["wavs"]:
            if ql in Path(p).name.lower():
                return True
        return False

    filtered = [it for it in items if visible(it)]
    total = len(filtered)
    start = (page - 1) * per_page
    end = min(start + per_page, total)
    st.caption(f"{total}件中 {start+1}〜{end} を表示")

    radio_seq = 0

    for it in filtered[start:end]:
        radio_seq += 1
        sdir = Path(it["sdir"])
        with st.expander(sdir.name, expanded=False):

            # === モード別再生エリア ===
            st.markdown("### 🎬 再生モード")
            p_upper = sdir / "preview_upper.mp4"
            p_lower = sdir / "preview_lower.mp4"
            p_both = sdir / "preview_both_mute.mp4"
            p_latest = sdir / "preview_synced.mp4"

            options = []
            if p_upper.exists():
                options.append("上半身（1）")
            if p_lower.exists():
                options.append("下半身（2）")
            if p_both.exists():
                options.append("上半身＋下半身（3）")
            if not options and p_latest.exists():
                options = ["最新プレビュー"]

            mode = st.radio(
                "表示する動画を選択",
                options=options,
                horizontal=True,
                key=f"mode-{sdir.name}-{radio_seq}",
            )

            to_show = None
            if mode == "上半身（1）":
                to_show = p_upper
            elif mode == "下半身（2）":
                to_show = p_lower
            elif mode == "上半身＋下半身（3）":
                to_show = p_both
            elif mode == "最新プレビュー":
                to_show = p_latest

            player_holder = st.empty()

            state_key_embed = f"hist_embed_{sdir.name}"  # "composite" or "orig"
            state_key_orig = f"hist_orig_{sdir.name}"  # 例: "U-1"
            ss.setdefault(state_key_embed, "composite")
            ss.setdefault(state_key_orig, None)

            items_u, items_l = _scan_orig_segments_from_folder(sdir)

            meta = _load_session_meta(sdir)

            if ss[state_key_embed] == "orig" and ss[state_key_orig]:
                key = ss[state_key_orig]
                src_list = items_u if key.startswith("U-") else items_l
                src = next((x for x in src_list if x["key"] == key), None)
                if src:
                    st.caption(
                        f"原本プレビュー：{('上半身' if key.startswith('U-') else '下半身')} "
                        f"{key}  {src['a']:.2f}s → {src['b']:.2f}s"
                    )
                    player_holder.video(src["mp4"])
                    if st.button(
                        "⬅ 合成プレビューに戻る",
                        key=f"back_{sdir.name}",
                        use_container_width=True,
                    ):
                        ss[state_key_embed] = "composite"
                        ss[state_key_orig] = None
                        st.rerun()
                else:
                    ss[state_key_embed] = "composite"

            if ss[state_key_embed] == "composite":
                # player_holder.video(str(to_show))  # ← 従来の無音/通常プレビューは使わない

                if meta:
                    st.markdown("### 🎧 プレビュー（ホバーで補足音声を切替・履歴）")
                    _history_show_hover_preview(sdir, meta)
                elif to_show and to_show.exists():
                    # session.json が無い古いセッション等のフォールバック
                    player_holder.video(str(to_show))
                else:
                    st.caption("このモードの動画は存在しません。")

            # --- 履歴：補足セグメント・タイムライン（▲で原本表示） ---
            if HAVE_PLOTLY and meta:
                st.markdown("### ⏱ 補足セグメント・タイムライン（履歴）")

                # ▼ ▲クリックで“常にここ”に再生させる専用ホルダ
                range_preview_holder = st.empty()
                safe_sid = sdir.name
                last_key = f"last_range_path__{safe_sid}"

                # 前回クリックしたプレビューがあれば先に表示
                if ss.get(last_key):
                    _p = Path(ss[last_key])
                    if _p.exists():
                        range_preview_holder.video(str(_p))
                    else:
                        ss[last_key] = None

                try:
                    fig = _make_timeline_fig_with_mapping(meta)
                    st.caption(
                        ":mag: debug: タイムラインに対するクリックイベントを監視中…"
                    )

                    ev = plotly_events(
                        fig,
                        click_event=True,
                        hover_event=False,
                        select_event=True,
                        override_height=320,
                        key=f"hist_tl_{sdir.name}_{radio_seq}",
                    )

                    if ev is not None:
                        st.write({"debug_event": ev})

                    if ev:
                        e = ev[0]
                        # 1) まず customdata があれば使う
                        key = (e.get("customdata") or "").strip()

                        # 2) 無ければ y と pointNumber から U/L と番号を推定
                        if not re.match(r"^[UL]-\d+$", key):
                            y = e.get("y", None)
                            pn = e.get("pointNumber", None)
                            # ▲は y=1.28 (上) / y=0.28 (下) 付近に置いているので 0.8 を閾値に
                            if y is not None and pn is not None:
                                track = "U" if y >= 0.8 else "L"
                                idx = int(pn) + 1  # 0始まり → 1始まり
                                key = f"{track}-{idx}"

                        # 3) key が決まったら動画ファイル探索→再生
                        if re.match(r"^[UL]-\d+$", key):
                            path = _history_open_range_with_supp(sdir, key)

                            # デバッグ（必要なら残す）
                            st.write({"debug_resolved_key": key, "debug_path": path})

                            if path and Path(path).exists():
                                st.info(
                                    f"原本プレビュー: {key}（直前カウント範囲＋補足音声）"
                                )
                                ss[last_key] = path
                                range_preview_holder.video(path)
                            else:
                                # どのパターンを探したかのヒント
                                st.warning("該当するプレビュー動画が見つかりません。")
                        else:
                            st.warning(
                                "クリック位置が▲マーカーとして解釈できませんでした。"
                            )
                except Exception as e:
                    st.warning(f"タイムラインの表示でエラー: {e}")

            elif not meta:
                st.caption(
                    "（このセッションには session.json が無いため、履歴タイムラインは表示できません）"
                )

            st.caption(f"セッションパス: `{sdir}`")

            # 原本復元→メインへ
            up_upper, up_lower = it["up_upper"], it["up_lower"]
            can_restore = bool(
                up_upper
                and up_lower
                and Path(up_upper).exists()
                and Path(up_lower).exists()
            )
            cols = st.columns([1, 1, 2])
            with cols[0]:
                if st.button(
                    "原本を現在入力に設定",
                    key=f"restore-{sdir.name}",
                    disabled=not can_restore,
                ):
                    ss.upper_path = up_upper
                    ss.lower_path = up_lower
                    ss.processed = False
                    ss.preview_path = None
                    st.success(
                        "原本を現在入力に設定しました。メインページで『同期して書き出し』を押してください。"
                    )
                    try:
                        st.switch_page("app18.py")  # ← メインのファイル名に合わせて
                    except Exception:
                        st.info("上部のページ切り替えからメインに戻ってください。")
            with cols[1]:
                st.download_button(
                    "セッションフォルダ名コピー",
                    data=sdir.name,
                    file_name="session_name.txt",
                    key=f"dl_{sdir.name}",
                )
