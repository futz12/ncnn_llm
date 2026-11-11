import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from PIL import Image

m = r"H:\work\deploy_model\ncnn_llm\models\Xiaomi-OCR-0"
proc = AutoProcessor.from_pretrained(m)
# model = Qwen3_5ForConditionalGeneration.from_pretrained(
#     m, torch_dtype=torch.bfloat16).eval()

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    m, torch_dtype=torch.bfloat16).to("cuda").eval()

img = Image.open(m + r"\example_pics\qianziwen.jpeg").convert("RGB")

sysp = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\n"
tail = ("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "Extract the text in the image.<|im_end|>\n\n"
        "<|im_start|>assistant\n<th" + "ink>\n\n</th" + "ink>\n\n")

inputs = proc(text=[sysp + tail], images=[img], return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}



out = model.generate(**inputs, max_new_tokens=160, do_sample=False)
print(proc.batch_decode(out, skip_special_tokens=True)[0])