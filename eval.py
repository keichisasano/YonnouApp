"""
四脳分類アプリの精度評価（top1 / top5）

フォルダ構成:
    eval.py
    refs/              ... アプリと同じ代表イラスト
    test/タイプ名/*.jpg ... テスト写真（フォルダ名 = 正解タイプ）

使い方:
    python eval.py            # 全部評価
    python eval.py --limit 1  # 各タイプ1枚だけ（動作確認用）
"""
import argparse
import io
import json
import re
import os
import sys
from collections import defaultdict
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

# ---------------- ここだけ書き換えればOK ----------------
MODEL = "gemini-flash-lite-latest"   # アプリと同じモデルにする
REF_MAX_SIDE = 1024                  # 代表イラストの長辺（アプリと同じ）
TEST_MAX_SIDE = 512                  # テスト写真の長辺（小さいほど安い）
USE_CACHE = True                     # キャッシュで代表イラストの送信を1回にする

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

TYPES = [
    "支配者外向", "支配者内向", "求道者進歩陽", "求道者進歩陰",
    "求道者調和陽", "求道者調和陰", "求道者憑依陽", "求道者憑依陰",
    "求道者演技陽", "求道者演技陰", "観察者内向", "観察者外向",
    "求道者ロマンチスト陰", "求道者ロマンチスト陽", "遊戯者内向", "遊戯者外向",
]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")
BASE = Path(__file__).parent
REF_DIR = BASE / "refs"
TEST_DIR = BASE / "test"
RESULT_FILE = BASE / "results.jsonl"
CSV_FILE = BASE / "wrong_list.csv"


def to_jpeg(img, side):
    img = img.convert("RGB")
    img.thumbnail((side, side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def parse_result(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def build_ref_contents():
    contents = []
    for p in sorted(REF_DIR.glob("*")):
        if p.suffix.lower() in IMG_EXTS:
            contents.append(f"[代表イラスト] {p.stem}")
            contents.append(types.Part.from_bytes(
                data=to_jpeg(Image.open(p), REF_MAX_SIDE), mime_type="image/jpeg"))
    return contents


def collect_tests(limit):
    items = []
    for t in TYPES:
        d = TEST_DIR / t
        if not d.exists():
            continue
        files = [p for p in sorted(d.glob("*")) if p.suffix.lower() in IMG_EXTS]
        if limit:
            files = files[:limit]
        items += [(t, p) for p in files]
    # フォルダ名の打ち間違いチェック
    if TEST_DIR.exists():
        for d in TEST_DIR.iterdir():
            if d.is_dir() and d.name not in TYPES:
                print(f"[警告] test/{d.name} はタイプ名と一致しないため無視します")
    return items


def rank_of(scores, true_type):
    """正解タイプの順位。同点は不利な順位にする（厳しめ）"""
    s = scores.get(true_type)
    if s is None:
        return None
    return 1 + sum(1 for k, v in scores.items() if k != true_type and v >= s)


def load_done():
    done = {}
    if RESULT_FILE.exists():
        for line in RESULT_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["path"]] = r
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="各タイプ最大N枚")
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print('先に export GEMINI_API_KEY="キー" を実行してください')
        sys.exit(1)
    if not REF_DIR.exists():
        print("refs フォルダが見つかりません。eval.py をアプリと同じ場所に置いてください")
        sys.exit(1)

    items = collect_tests(args.limit)
    if not items:
        print("test/タイプ名/ に写真を入れてください")
        sys.exit(1)
    print(f"テスト写真: {len(items)} 枚")

    client = genai.Client(api_key=key)
    ref_contents = build_ref_contents()

    # ----- キャッシュ作成（失敗したら通常送信） -----
    cache_name = None
    if USE_CACHE:
        try:
            cache = client.caches.create(
                model=MODEL,
                config=types.CreateCachedContentConfig(
                    contents=ref_contents,
                    system_instruction=PROMPT,
                    ttl="3600s",
                ),
            )
            cache_name = cache.name
            print("キャッシュ: 使用（代表イラストは1回だけ送信）")
        except Exception as e:
            print(f"キャッシュ: 使えなかったので通常送信にします（{str(e)[:80]}）")

    done = load_done()
    in_tok = out_tok = 0

    with open(RESULT_FILE, "a", encoding="utf-8") as f:
        for n, (true_type, path) in enumerate(items, 1):
            key_path = str(path.relative_to(BASE))
            if key_path in done:
                continue

            photo = types.Part.from_bytes(
                data=to_jpeg(Image.open(path), TEST_MAX_SIDE), mime_type="image/jpeg")
            try:
                if cache_name:
                    res = client.models.generate_content(
                        model=MODEL,
                        contents=["[診断したい顔写真]", photo],
                        config=types.GenerateContentConfig(
                            cached_content=cache_name, temperature=0),
                    )
                else:
                    res = client.models.generate_content(
                        model=MODEL,
                        contents=ref_contents + ["[診断したい顔写真]", photo, PROMPT],
                        config=types.GenerateContentConfig(temperature=0),
                    )
                data = parse_result(res.text)
                u = res.usage_metadata
                in_tok += u.prompt_token_count or 0
                out_tok += u.candidates_token_count or 0
            except Exception as e:
                print(f"[{n}/{len(items)}] エラー {key_path}: {str(e)[:80]}")
                continue

            if not data or "scores" not in data:
                print(f"[{n}/{len(items)}] 結果の読み取り失敗 {key_path}")
                continue

            r = {"path": key_path, "true": true_type, "scores": data["scores"]}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            done[key_path] = r
            rk = rank_of(data["scores"], true_type)
            print(f"[{n}/{len(items)}] {true_type} → 正解の順位 {rk}位  ({path.name})")

    # ----- 集計 -----
    rows = [done[str(p.relative_to(BASE))] for _, p in items
            if str(p.relative_to(BASE)) in done]
    if not rows:
        print("集計できる結果がありません")
        return

    per = defaultdict(lambda: [0, 0, 0])  # 枚数, top1, top5
    wrong = []
    for r in rows:
        rk = rank_of(r["scores"], r["true"])
        per[r["true"]][0] += 1
        if rk is not None and rk <= 1:
            per[r["true"]][1] += 1
        if rk is not None and rk <= 5:
            per[r["true"]][2] += 1
        if rk is None or rk > 5:
            top = sorted(r["scores"].items(), key=lambda kv: kv[1], reverse=True)[:3]
            wrong.append((r["path"], r["true"], rk, top))

    total = sum(v[0] for v in per.values())
    t1 = sum(v[1] for v in per.values())
    t5 = sum(v[2] for v in per.values())

    print("\n========== 結果 ==========")
    print(f"全体: {total}枚  top1 = {t1}/{total} = {t1/total:.1%}   top5 = {t5}/{total} = {t5/total:.1%}")
    print(f"ランダムの目安: top1 = {1/16:.1%}   top5 = {5/16:.1%}")
    print("\n--- タイプ別 ---")
    for t in TYPES:
        if t in per:
            n, a, b = per[t]
            print(f"{t:<14} {n:>3}枚  top1 {a/n:>6.1%}  top5 {b/n:>6.1%}")

    # 予測の偏り（AIが1位に出しがちなタイプ）
    cnt = defaultdict(int)
    for r in rows:
        cnt[max(r["scores"].items(), key=lambda kv: kv[1])[0]] += 1
    print("\n--- AIが1位に出したタイプの回数（偏りチェック） ---")
    for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
        print(f"{k:<14} {v}回")

    # top5に入らなかった写真をCSVに
    with open(CSV_FILE, "w", encoding="utf-8-sig") as f:
        f.write("写真,正解タイプ,正解の順位,AIの上位3つ\n")
        for path, true, rk, top in wrong:
            f.write(f"{path},{true},{rk},{' / '.join(f'{k}:{v}' for k, v in top)}\n")
    print(f"\ntop5に入らなかった写真: {len(wrong)}枚 → {CSV_FILE.name}")
    print(f"使用トークン 入力: {in_tok}  出力: {out_tok}")


if __name__ == "__main__":
    main()