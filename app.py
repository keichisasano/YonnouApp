import io
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types
from PIL import Image

# ---------------- ここだけ書き換えればOK ----------------
MODEL = "gemini-flash-lite-latest"  # 精度を上げたい時は "gemini-flash-latest"
MAX_SIDE = 1024                     # 画像の長辺(小さいほど安い)
PER_IP_PER_DAY = 10                  # 1人あたりの1日の回数上限
TOTAL_PER_DAY = 100                 # 全体の1日の回数上限(費用の安全装置)

PROMPT = """

あなたは、顔写真の見た目の印象を、イラストの'四脳タイプ'と照らし合わせる役です。 誰かを特定したり、その人の内面や性格を断定したりせず、あくまで「見た目の印象がどのタイプのイラストや特徴に近いか」だけを推測して出力してください。

【各タイプの特徴】

・支配者外向: 口元 笑う時は一見親しみやすく見えますが目が笑っていなくて口だけ笑っています / 表情の雰囲気 狐や蛇系統の顔です。爬虫類系統の顔

・支配者内向: 目つき 細くて鋭い眼光の時が多い/ 表情の雰囲気 ワニやポメラニアンの系統の顔です。爬虫類系統の顔

・他タイプは画像を参考にしてください

【判断のルール】


1. 画像を隅々まで見て16タイプを確認してください
2. 判断材料は「目つき」「口元」「表情」「顔全体の雰囲気」だけです。
3. 身振り、手の動き、姿勢、ポーズ、服装、髪色、髪型、アクセサリー、背景は判断材料にしないでください。写っていても完全に無視してください。
4. 上の「各タイプの特徴」と見比べて、どれだけ当てはまるかを0〜100の整数で採点してください。
5. reason は「〜という印象を受けます」「〜に近い雰囲気です」のように、印象として述べてください。「〜な性格だ」「〜な人だ」と断定しないでください。
6. reason には、判断材料に使った顔の部位(目つき・口元・表情)を必ず含めてこう言う特徴があると説明してください。
7. 【出力形式】 次のJSONだけを出力してください。説明文やコードブロックは付けないでください。 0~100の整数を出力してください{"scores": {"支配者外向": , "支配者内向": , "求道者進歩陽": , "求道者進歩陰": , "求道者調和陽": , "求道者調和陰": , "求道者憑依陽": , "求道者憑依陰": , "求道者演技陽": , "求道者演技陰": , "観察者内向": , "観察者外向": , "求道者ロマンチスト陰": , "求道者ロマンチスト陽": , "遊戯者内向": , "遊戯者外向": }, "best": "タイプ名", "reason": "根拠を2〜3文で"} 
8. reasonではイラストに近いと言うよりこのタイプに近いと言う言い方にしてください

"""
# ---------------------------------------------------------

st.set_page_config(page_title="四脳分類 顔診断", page_icon="🧠")

try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    API_KEY = os.environ.get("GEMINI_API_KEY")

REF_DIR = Path(__file__).parent / "refs"


@st.cache_resource
def get_client():
    return genai.Client(api_key=API_KEY)


@st.cache_resource
def get_counters():
    # 全ユーザーで共有されるカウンタ(アプリが再起動するとリセット)
    return {"ip": defaultdict(int), "total": defaultdict(int)}


def get_ip():
    try:
        h = st.context.headers
        return h.get("X-Forwarded-For", "unknown").split(",")[0].strip()
    except Exception:
        return "unknown"


def to_jpeg(img):
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def parse_result(text):
    """AIの返答からJSON部分を取り出す(```json などが付いていても対応)"""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def show_result(data):
    scores = data.get("scores", {})
    # スコアが高い順に並べる
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    max_len = max((len(k) for k, _ in items), default=1)

    st.subheader("診断結果")
    st.markdown(f"最も近いタイプ: **{data.get('best', items[0][0] if items else '')}**")

    # ヒストグラム風の横棒グラフ(HTMLで描画)
    rows = ""
    for i, (name, score) in enumerate(items):
        color = "#ff4b4b" if i == 0 else "#4b8bff"
        rows += f"""
        <div style="display:flex;align-items:center;margin:4px 0;">
          <div style="width:{max_len + 1}em;font-size:14px;">{name}</div>
          <div style="flex:1;background:#eee;border-radius:4px;height:20px;">
            <div style="width:{score}%;background:{color};height:100%;border-radius:4px;"></div>
          </div>
          <div style="width:3em;text-align:right;font-size:14px;">{score}</div>
        </div>
        """
    st.markdown(rows, unsafe_allow_html=True)

    st.markdown("#### 解説")
    st.write(data.get("reason", ""))


st.title("四脳分類 顔相診断")
st.caption("非公式の診断ツールです。精度はまだ高くないので上位5つくらいを参考にしてください。写真は保存されません。ご自身の写真、フリー素材のみ使ってください。\n四脳分類についてはこちらがおすすめです : \nhttps://note.com/novel_tucan8317\nhttps://www.youtube.com/watch?v=UVlgYfU_AFk")

if not API_KEY:
    st.error("APIキーが設定されていません。")
    st.stop()

file = st.file_uploader("顔写真(1枚)", type=["jpg", "jpeg", "png", "webp"])

if file is not None:
    photo = Image.open(file)
    st.image(photo, width=300)

    if st.button("診断する", type="primary"):
        today = time.strftime("%Y-%m-%d")
        ip = get_ip()
        c = get_counters()

        if c["total"][today] >= TOTAL_PER_DAY:
            st.warning("今日は利用が多いため終了しました。明日また試してください。")
            st.stop()
        if c["ip"][(today, ip)] >= PER_IP_PER_DAY:
            st.warning(f"1日{PER_IP_PER_DAY}回までです。明日また試してください。")
            st.stop()
        c["ip"][(today, ip)] += 1
        c["total"][today] += 1

        contents = []
        if REF_DIR.exists():
            for p in sorted(REF_DIR.glob("*")):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                    contents.append(f"[代表イラスト] {p.stem}")
                    contents.append(types.Part.from_bytes(data=to_jpeg(Image.open(p)), mime_type="image/jpeg"))
        contents.append("[診断したい顔写真]")
        contents.append(types.Part.from_bytes(data=to_jpeg(photo), mime_type="image/jpeg"))
        contents.append(PROMPT)

        with st.spinner("診断中..."):
            try:
                res = get_client().models.generate_content(model=MODEL, contents=contents)
                data = parse_result(res.text)
                if data is None:
                    st.error("結果の読み取りに失敗しました。もう一度試してください。")
                else:
                    show_result(data)
            except Exception:
                st.error("エラーが発生しました。少し待ってからもう一度試してください。")