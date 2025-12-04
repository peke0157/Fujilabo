# モジュールの読み込み
import gradio as gr
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertJapaneseTokenizer
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
vocab_size = tokenizer.vocab_size
emb = nn.Embedding(vocab_size, 128)




# モデルの定義
class ClassifierRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(ClassifierRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(input_size=embed_size, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        output, hidden = self.rnn(x)
        batch_size = output.size(0)
        lengths = attention_mask.sum(dim=1)
        final_output = output[range(batch_size), lengths - 1, :]
        out = self.linear(final_output)
        return out
    
# モデルの作成，学習済パラメータの読み込み
model = ClassifierRNN(vocab_size=vocab_size, embed_size=128, hidden_size=64, output_size=3)
model.load_state_dict(torch.load('sentiment_wrime_rnn.pth'))
model.eval()


# クラス名
class_names = [
"negative", "nertural", "positive"
]

# 予測関数
def predict(text):
    model.eval()
    enc = tokenizer(text, add_special_tokens=False,
    return_tensors='pt')
    ids = enc['input_ids']
    mask = enc['attention_mask']
    # 推論
    with torch.no_grad():
        logits = model(ids, mask)
        
    # softmaxで確率に変換
    probs = F.softmax(logits, dim=-1)
    # 結果を辞書形式で返す
    results = {}
    for i, class_name in enumerate(class_names):
        results[class_name] = float(probs[0][i])
        
    return results

#Grインターフェースの作成
demo = gr.Interface(
fn=predict,
inputs=gr.Textbox(label="text"),
outputs=gr.Label(num_top_classes=3, label="予測結果"),
title="課題感情モデル",
)

# アプリケーションの起動
if __name__ == "__main__":
    demo.launch()
