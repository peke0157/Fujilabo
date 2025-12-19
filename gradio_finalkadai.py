import gradio as gr
from transformers import pipeline

print("モデルを読み込んでいます")
sentiment_analyzer= pipeline("sentiment-analysis", model="koheiduck/bert-japanese-finetuned-sentiment")

recommendations = {
    "POSITIVE":{
        "song": "Mrs.GREEN APPLE - ライラック",
        "msg": "ハッピーなこそミセスのこの曲を聞きましょう!!!!",
        "link": "https://youtu.be/QjrkrVmC-8M?si=kDFrsQ1auPI9Vat8"
    },
    "NEGATIVE":{
        "song": "Soala - すれ違い",
        "msg": "失恋して悲しいときはこれを聞くのがおすすめです",
        "link": "https://youtu.be/I5eu4XMWZR8?si=XLWzzVC_a0Corwml"
    },
    "NEUTRAL":{
        "song": "Mrs.GREEN APPLE - 天国",
        "msg": "感動したまたは感動したいときはこの曲しか勝たん!!!!",
        "link": "https://youtu.be/CO0Eoj9aPcs?si=TJUqBbN73wmJiadd"
    }    
}
    
def app_main(text):
    if not text:
        return "テキストを入力していください。","", ""
   
   # 感情分析
    result = sentiment_analyzer(text)[0]
    label = result['label']
    score = result['score']
    
    # 辞書から曲を取得
    rec = recommendations.get(label, recommendations["POSITIVE"])
    
    output_msg = f"判定結果: {label}(確信度: {score: .2f})\n\n{rec['msg']}"
    return output_msg, rec['song'], rec['link']


   




# Grインターフェースの作成
with gr.Blocks() as demo:
    gr.Markdown("おすすめの音楽を提案するよ")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="今の感情を入力")
            submit_btn = gr.Button('おすすめを聞く')
        
        with gr.Column():
            outputs_text = gr.Textbox(label='AIからのメッセージ')
            song_name = gr.Textbox(label='おすすめの曲')
            song_link = gr.Textbox(label='リンク')  
            
    submit_btn.click(
        fn=app_main,
        inputs=input_text,
        outputs=[outputs_text, song_name, song_link]
    ) 


# アプリケーションの気道
if __name__ == "__main__":
    demo.launch()