import os
import io
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# =========================
# 基本設定（JST とタイトル）
# =========================
try:
    from zoneinfo import ZoneInfo
    JST = ZoneInfo("Asia/Tokyo")
except Exception:
    JST = timezone(timedelta(hours=9))

st.set_page_config(page_title="関数テスト（問題→解説→採点）", layout="wide")
st.title("関数テスト（問題→解説→採点）")

# ==============
# ユーティリティ
# ==============
def find_files(root: str, pattern_exts: Tuple[str, ...]) -> List[Path]:
    from pathlib import Path
    p = Path(root)
    found = []
    for ext in pattern_exts:
        found.extend(sorted(p.glob(f"*{ext}")))
    return found

def load_answer_csv(csv_paths: List["Path"]) -> Optional[pd.DataFrame]:
    # 優先：ファイル名に「解答」または「answer」を含むCSV
    priority = [p for p in csv_paths if ("解答" in p.stem or "answer" in p.stem)]
    ordered = priority + [p for p in csv_paths if p not in priority]
    for enc in ("utf-8-sig", "utf-8", "cp932", "shift-jis"):
        for path in ordered:
            try:
                df = pd.read_csv(path, encoding=enc)
                df["__csv_path__"] = str(path)
                return df
            except Exception:
                continue
    return None

def as_str(x) -> str:
    if pd.isna(x):
        return ""
    # 数値は整数っぽければ小数部を落とす
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)

def seconds_to_hms(sec: int) -> str:
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h:
        return f"{h}時間{m}分{s}秒"
    return f"{m}分{s}秒"

# ======================
# ルートの PDF / CSV 収集
# ======================
root = "."
from pathlib import Path
pdfs = find_files(root, (".pdf",))
csvs = find_files(root, (".csv",))

# 問題/解説のペアを自動マッピング
# 例：問題1.pdf / 解答1.pdf, 解説1.pdf など
problems: Dict[int, Path] = {}
solutions: Dict[int, Path] = {}

for p in pdfs:
    name = p.stem
    if name.startswith("問題"):
        try:
            n = int(name.replace("問題", ""))
            problems[n] = p
        except Exception:
            pass
    elif name.startswith("解答") or name.startswith("解説"):
        # 解答1/解説1 → 1 として扱う
        try:
            n = int(name.replace("解答","").replace("解説",""))
            # 同じ番号で既に登録があっても、どちらかを保持（後勝ち）
            solutions[n] = p
        except Exception:
            pass

# CSV（解答仕様）読込
answer_df = load_answer_csv(csvs)

# 必須列チェック（存在しない列は自動追加）
required_cols = ["タイトル", "ID", "小問", "問題レベル", "答え", "解説動画", "解答時間", "累計時間"]
if answer_df is None:
    st.error("ルートにCSV（解答仕様）が見つかりませんでした。")
    st.stop()
for col in required_cols:
    if col not in answer_df.columns:
        answer_df[col] = pd.NA

# ID と 小問 は文字列で統一
answer_df["ID"] = answer_df["ID"].astype(str)
answer_df["小問"] = answer_df["小問"].astype(str)
answer_df["答え"] = answer_df["答え"].apply(as_str)

# 利用可能なID一覧（昇順）
available_ids = sorted({int(x) for x in answer_df["ID"].unique() if str(x).isdigit()})

# =================
# セッション状態管理
# =================
ss = st.session_state
if "phase" not in ss:  # "problem" / "solution" / "end"
    ss.phase = "problem"
if "current_id_idx" not in ss:
    ss.current_id_idx = 0  # available_ids のインデックス
if "start_time" not in ss:
    ss.start_time = time.time()  # 累計用
if "problem_start_time" not in ss:
    ss.problem_start_time = time.time()  # 問題画面の経過時間
if "answers" not in ss:
    # 収集形式：{(ID, 小問): {"入力": str, "正解": str, "判定": "正解/不正解", "経過秒": int}}
    ss.answers = {}
if "user_name" not in ss:
    ss.user_name = ""

def get_current_id() -> Optional[int]:
    if not available_ids:
        return None
    if ss.current_id_idx < 0 or ss.current_id_idx >= len(available_ids):
        return None
    return available_ids[ss.current_id_idx]

def rows_for_id(i: int) -> pd.DataFrame:
    return answer_df[answer_df["ID"] == str(i)].sort_values(by=["小問"], key=lambda s: s.astype(str))

def download_button_bytes(label: str, data: bytes, file_name: str, mime: str = "application/octet-stream"):
    st.download_button(label, data=data, file_name=file_name, mime=mime)

# =======================
# 画面：問題（Problem UI）
# =======================
def render_problem(i: int):
    st.subheader(f"問題 {i}")
    # 経過時間（この問題）
    elapsed = int(time.time() - ss.problem_start_time)
    st.caption(f"経過時間：{seconds_to_hms(elapsed)}　｜　累計時間：{seconds_to_hms(int(time.time() - ss.start_time))}")

    # 問題PDF 表示とDL
    colA, colB = st.columns([4,1])
    with colA:
        if i in problems:
            st.write(f"PDF: {problems[i].name}")
            # st.pdf は利用環境により未対応の可能性があります。未対応の場合は download のみでも動作します。
            try:
                st.pdf(problems[i].read_bytes())
            except Exception:
                st.info("この環境では PDF の埋め込み表示が未対応です。『問題DL』からご確認ください。")
        else:
            st.info("このIDに対応する問題PDFが見つかりませんでした。ファイル名は『問題{番号}.pdf』推奨です。")

    with colB:
        if i in problems:
            download_button_bytes("問題DL", problems[i].read_bytes(), problems[i].name, "application/pdf")

    # 操作ボタン
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("解答記入", use_container_width=True):
            ss.phase = "solution"  # 解答入力は解説画面でまとめて行う仕様に変更（採点ボタンも一緒に）
    with c2:
        if st.button("問題パス（解説へ）", use_container_width=True):
            ss.phase = "solution"
    st.stop()

# =======================
# 画面：解説（Solution UI）
# =======================
def render_solution(i: int):
    st.subheader(f"解説 {i}")

    # 上部：解説動画ボタン（CSV からURL検出）
    rows = rows_for_id(i)
    video_links = [as_str(u) for u in rows["解説動画"].tolist() if isinstance(u, str) and u.strip()]
    colTop = st.columns([1,1,1,4])
    with colTop[0]:
        if video_links:
            # 代表1本を表示ボタンに
            st.link_button("解説動画", video_links[0])
    with colTop[1]:
        if i in solutions:
            download_button_bytes("解説DL", solutions[i].read_bytes(), solutions[i].name, "application/pdf")
    with colTop[2]:
        if st.button("次の問題", use_container_width=True):
            ss.current_id_idx += 1
            if ss.current_id_idx >= len(available_ids):
                ss.phase = "end"
            else:
                ss.phase = "problem"
                ss.problem_start_time = time.time()
            st.rerun()

    # 解説PDF（あれば表示）
    if i in solutions:
        st.write(f"PDF: {solutions[i].name}")
        try:
            st.pdf(solutions[i].read_bytes())
        except Exception:
            st.info("この環境では PDF の埋め込み表示が未対応です。『解説DL』からご確認ください。")
    else:
        st.info("このIDに対応する解説PDFが見つかりませんでした。ファイル名は『解答{番号}.pdf』または『解説{番号}.pdf』推奨です。")

    st.divider()
    st.markdown("#### 解答記入欄")

    # 小問ごと回答欄（左：小問番号、右：入力＋判定）
    any_input = False
    for _, r in rows.iterrows():
        sub = as_str(r["小問"])
        key = (str(i), sub)  # (ID, 小問)
        colL, colM, colR = st.columns([1,2,2])
        with colL:
            st.write(f"小問 {sub}")
        with colM:
            # 10文字まで・数字を想定（ただし自由入力も許容）
            default_val = ss.answers.get(key, {}).get("入力", "")
            val = st.text_input("入力", value=default_val, max_chars=10, key=f"input_{i}_{sub}")
            if val != default_val:
                # 仮保存
                cur = ss.answers.get(key, {})
                cur["入力"] = val
                ss.answers[key] = cur
            any_input = any_input or bool(val.strip())
        with colR:
            result = ss.answers.get(key, {}).get("判定", "")
            if result:
                st.write(result)

    cA, cB = st.columns([1,3])
    with cA:
        if st.button("採点", type="primary", use_container_width=True):
            # 採点処理
            per_problem_elapsed = int(time.time() - ss.problem_start_time)
            total_elapsed = int(time.time() - ss.start_time)
            for _, r in rows.iterrows():
                sub = as_str(r["小問"])
                key = (str(i), sub)
                user_inp = ss.answers.get(key, {}).get("入力", "").strip()
                correct = as_str(r["答え"]).strip()
                judge = "正解！" if user_inp == correct else "不正解"
                ss.answers[key] = {
                    "入力": user_inp,
                    "正解": correct,
                    "判定": judge,
                    "経過秒": per_problem_elapsed,
                    "累計秒": total_elapsed,
                    "難易度": as_str(r["問題レベル"]),
                    "タイトル": as_str(r["タイトル"]),
                }
            st.rerun()
    with cB:
        st.caption("※『採点』を押すと各欄の右に 判定（正解/不正解） が表示されます。")

    st.divider()
    st.button("終了", use_container_width=True, on_click=lambda: setattr(ss, "phase", "end"))

# =======================
# 画面：終了（Export UI）
# =======================
def render_end():
    st.subheader("終了")
    st.write("結果のCSVをダウンロードできます。")
    ss.user_name = st.text_input("氏名を入力してください", value=ss.user_name, placeholder="例）千葉 太郎")

    # 集計テーブル作成
    rows: List[Dict] = []
    # キー順を安定化
    for (ID, sub) in sorted(ss.answers.keys(), key=lambda t: (int(t[0]), str(t[1]))):
        rec = ss.answers[(ID, sub)]
        rows.append({
            "タイトル": rec.get("タイトル", ""),
            "小問": sub,
            "難易度": rec.get("難易度", ""),
            "正誤": "正解" if rec.get("判定","") == "正解！" else "不正解",
            "経過時間": seconds_to_hms(int(rec.get("経過秒", 0)) ),
            "累計時間": seconds_to_hms(int(rec.get("累計秒", 0)) ),
            "入力": rec.get("入力", ""),
            "正解": rec.get("正解", ""),
            "ID": ID,
        })
    result_df = pd.DataFrame(rows, columns=["タイトル","小問","難易度","正誤","経過時間","累計時間","入力","正解","ID"])

    st.dataframe(result_df, use_container_width=True, hide_index=True)

    if ss.user_name:
        timestamp = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
        out_name = f"{ss.user_name}_結果_{timestamp}.csv"
        buf = io.StringIO()
        # ユーザー指定の列順：タイトル、小問、難易度、小問ごとの正解と不正解、経過時間、累計時間
        export_cols = ["タイトル","小問","難易度","正誤","経過時間","累計時間"]
        result_df[export_cols].to_csv(buf, index=False, encoding="utf-8-sig")
        st.download_button("結果CSVをダウンロード", buf.getvalue().encode("utf-8-sig"), file_name=out_name, mime="text/csv")
    else:
        st.info("氏名を入力するとダウンロードできます。")

    st.button("はじめから", on_click=lambda: [ss.clear()])

# ==============
# ルーター
# ==============
current_id = get_current_id()
if current_id is None:
    st.error("CSV内に有効な ID が見つかりません。CSV の『ID』列をご確認ください。")
    st.stop()

# ページトップに現在の進行表示
st.caption(f"進行状況： {ss.current_id_idx+1} / {len(available_ids)} 　｜　現在ID：{current_id}")

if ss.phase == "problem":
    render_problem(current_id)
elif ss.phase == "solution":
    render_solution(current_id)
else:
    render_end()
