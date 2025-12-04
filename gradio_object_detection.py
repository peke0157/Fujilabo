import gradio as gr
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection

model_name = "facebook/detr-resnet-50"

processor = DetrImageProcessor.from_pretrained(model_name, revision="no_timm")
model = DetrForObjectDetection.from_pretrained(model_name, revision="no_timm")
id2label = model.config.id2label

def detect_objects(image):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
    outputs, target_sizes=target_sizes, threshold=0.9)[0]

    values = []
    for label, box in zip(results["labels"], results["boxes"]):
        box = [int(round(i, 0)) for i in box.tolist()]
        name = id2label[label.item()]
        values.append([box, name])

    return image, values

demo = gr.Interface(
  fn=detect_objects,
  inputs=gr.Image(type="pil"),
  outputs=gr.AnnotatedImage(),
  title="DETRによる物体検出デモ",
)

if __name__ == "__main__":
    demo.launch()
