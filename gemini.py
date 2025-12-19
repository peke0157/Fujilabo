import gradio as gr
from transformers import pipeline

# --- 1. モデルの準備 (katsutakuモデルに変更) ---
# このモデルは8つの感情 (joy, sadness, anticipation, surprise, anger, fear, disgust, trust) を返します
print("モデルを読み込んでいます...")
sentiment_analyzer = pipeline("sentiment-analysis", model="katsutaku/wrime-sentiment-analyzer")

# --- 2. オリジナル要素：感情と曲の紐づけ ---
# 8種類の感情に合わせてキーを設定します
recommendations = {
    "joy": {
        "song": "Mrs. GREEN APPLE - ダンスホール",
        "msg": "最高にハッピーですね！この曲でさらに盛り上がりましょう！",
        "link": "https://www.youtube.com/..."
    },
    "sadness": {
        "song": "優里 - レオ",
        "msg": "辛い時は泣いてもいいんです。この曲が寄り添ってくれます。",
        "link": "https://www.youtube.com/..."
    },
    "anger": {
        "song": "Ado - うっせぇわ",
        "msg": "イライラする時はこの曲で発散しましょう！",
        "link": "https://www.youtube.com/..."
    },
    "surprise": {
        "song": "きゃりーぱみゅぱみゅ - PONPONPON",
        "msg": "驚きのニュース？そんな時はこの不思議な世界観へ。",
        "link": "https://www.youtube.com/..."
    },
    # ★残りの感情 (anticipation, fear, disgust, trust) も必要に応じて追加してください
    # 設定していない感情が来たらデフォルト（joyなど）に流す処理を下に書いています
}

def app_main(text):
    if not text:
        return "テキストを入力してください", "", ""
    
    # 感情分析
    result = sentiment_analyzer(text)[0]
    label = result['label']
    score = result['score']
    
    # 辞書から曲を取得（もし辞書にない感情が来たら joy を返す）
    rec = recommendations.get(label, recommendations["joy"])
    
    output_msg = f"判定結果: {label} (確信度: {score:.2f})\n\n{rec['msg']}"
    return output_msg, rec['song'], rec['link']

# --- 3. Gradioインターフェース ---
with gr.Blocks() as demo:
    gr.Markdown("# 🎵 8感情音楽レコメンドAI")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="今の気持ちを入力", placeholder="例：テストで満点取れた！ / 財布を落とした...")
            submit_btn = gr.Button("おすすめを聞く")
        
        with gr.Column():
            output_text = gr.Textbox(label="AIからのメッセージ")
            song_name = gr.Textbox(label="おすすめの曲")
            song_link = gr.Textbox(label="リンク")

    submit_btn.click(
        fn=app_main,
        inputs=input_text,
        outputs=[output_text, song_name, song_link]
    )

demo.launch()