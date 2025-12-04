import gradio as gr
from transformers import pipeline

# --- 1. 機械学習モデルの準備 ---
# 日本語の感情分析モデルを使用（huggingfaceからダウンロードされます）
print("モデルを読み込んでいます...")
sentiment_analyzer = pipeline("sentiment-analysis", model="koheiduck/bert-japanese-finetuned-sentiment")

# --- 2. オリジナル要素：感情と曲の紐づけロジック ---
# ここを自分の好きな曲やYouTubeリンクに書き換えるだけでOK！
def get_music_recommendation(label):
    recommendations = {
        "POSITIVE": {
            "song": "Sugar Rush Ride",
            "comment": "最高ですね！この爽快な曲でさらにテンションを上げましょう！",
            "link": "https://www.youtube.com/watch?v=P9tKTxbgdkk"
        },
        "NEGATIVE": {
            "song": "0X1=LOVESONG (I Know I Love You)",
            "comment": "辛い時は無理しないで。この曲のエモーショナルな歌声に浸りませんか。",
            "link": "https://www.youtube.com/watch?v=d5bbqKYu51w"
        },
        "NEUTRAL": {
            "song": "Chasing That Feeling",
            "comment": "落ち着いていますね。作業用やリラックスタイムにこの曲をどうぞ。",
            "link": "https://www.youtube.com/watch?v=IS8uaBlMgCI"
        }
    }
    # 辞書から取得（万が一エラーが出た場合はPOSITIVEを返す安全策）
    return recommendations.get(label, recommendations["POSITIVE"])

# --- アプリのメイン動作関数 ---
def app_main(text):
    # 空白チェック
    if not text:
        return "テキストを入力してください", "", ""
    
    # 感情分析を実行
    result = sentiment_analyzer(text)[0]
    label = result['label']  # 'POSITIVE', 'NEGATIVE', 'NEUTRAL' のいずれか
    score = result['score']  # 確信度
    
    # 曲を選定
    rec = get_music_recommendation(label)
    
    # 出力テキスト作成
    output_msg = f"判定結果: {label} (確信度: {score:.2f})\n\n{rec['comment']}"
    
    return output_msg, rec['song'], rec['link']

# --- 3. Gradioインターフェースの構築 ---
with gr.Blocks() as demo:
    gr.Markdown("# 🎵 感情に合わせてTXTの曲をおすすめするAI")
    gr.Markdown("今のあなたの気持ちや、今日あったことを入力してください。AIが感情を分析して、ぴったりの一曲を提案します。")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="今の気持ちを入力", placeholder="例：今日は課題がうまくいって最高！ / バイトでミスして落ち込んでる...")
            submit_btn = gr.Button("おすすめを聞く")
        
        with gr.Column():
            output_text = gr.Textbox(label="AIからのメッセージ")
            song_name = gr.Textbox(label="おすすめの曲")
            song_link = gr.Textbox(label="YouTubeリンク")

    # ボタンを押した時の動作
    submit_btn.click(
        fn=app_main,
        inputs=input_text,
        outputs=[output_text, song_name, song_link]
    )

# 起動
demo.launch()
