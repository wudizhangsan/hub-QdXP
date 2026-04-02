import torch
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

# 1. 加载模型和处理器
model_name = "./model/chinese-clip-vit-base-patch16"
model = ChineseCLIPModel.from_pretrained(model_name)
processor = ChineseCLIPProcessor.from_pretrained(model_name)

# 2. 加载本地的一张小狗图片
image_path = "dog.png"  # 请确保你的文件夹里有这张图片
image = Image.open(image_path)


# 3. 准备候选文本标签 (这就是 Zero-Shot 的精髓！)
candidate_labels = [
    "一张小狗的照片",
    "一张小猫的照片",
    "一辆行驶的汽车",
    "一个美味的汉堡",
    "一只在天上飞的鸟"
]

# 4. 数据预处理 (把图片和文字全部转成大模型认识的 Tensor 张量)
inputs = processor(text=candidate_labels, images=image, return_tensors="pt", padding=True)

# 5. 推理计算 (不需要算梯度，所以用 torch.no_grad)
with torch.no_grad():
    outputs = model(**inputs)

# 6. 计算概率得分
# outputs.logits_per_image 包含了这张图片对应每一句话的原始匹配得分
logits_per_image = outputs.logits_per_image

# 使用 softmax 将得分转化为相加等于 100% 的概率分布
probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

# 7. 打印预测结果
print(f"当前检测图片: {image_path}")
print("各类别匹配概率:")
for label, prob in zip(candidate_labels, probs):
    print(f"  - {label}: {prob * 100:.2f}%")

# 找出最高概率的预测结果
best_idx = probs.argmax()
print(f"\n最终结论: CLIP 认为这大概率是 【{candidate_labels[best_idx]}】")
