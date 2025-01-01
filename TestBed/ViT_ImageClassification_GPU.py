import os
from PIL import Image
import torch
from transformers import AutoProcessor, ViTForImageClassification
import json

# 檢查是否有 CUDA 支援
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置: {device}")

# 定義模型路徑
model_paths = [
    "../models/ViT/ViT-Plant-Classifier-All",
    "../models/ViT/ViT-Plant-Classifier-All_B",
    "../models/ViT/ViT-Plant-Classifier-All_C"
]

# 定義標籤對應
label_All = {
    "00": "A-01",
    "01": "A-02",
    "02": "A-03",
    "03": "A-04",
    "04": "A-05",
    "05": "A-06",
    "06": "A-07",
    "07": "A-08",
    "08": "A-09",
    "09": "A-10",
    "10": "A-11",
    "11": "A-12",
    "12": "A-13",
    "13": "A-14",
    "14": "A-15",
    "15": "B-01",
    "16": "B-02",
    "17": "B-03",
    "18": "B-04",
    "19": "B-05",
    "20": "C-01",
    "21": "C-02",
    "22": "C-03"
}

label_All_B = {
    "00": "B-01",
    "01": "B-02",
    "02": "B-03",
    "03": "B-04",
    "04": "B-05",
}

label_All_C = {
    "00": "C-01",
    "01": "C-02",
    "02": "C-03"
}

# 加載第一個模型和處理器
current_model_index = 0
model = ViTForImageClassification.from_pretrained(model_paths[current_model_index])
processor = AutoProcessor.from_pretrained(model_paths[current_model_index], use_fast=True)

# 測試資料集路徑
# test_dir = "./data/750-testImage/"   #測試750張
# test_dir = "./data/miniTestImage/"  #迷你測試23種植物
test_dir = "./data/3W-repeat-testImage"  #3萬張重複40次

# 忽略大小不匹配加載模型
model = ViTForImageClassification.from_pretrained(
    model_paths[current_model_index],
    ignore_mismatched_sizes=True
)

# 驗證模型
print(model.classifier)


# 統計結果
correct = 0
incorrect = 0


# 確保模型輸出的類別數正確
num_classes = 23  # 確保模型輸出與實際類別數匹配


# 定義圖片預測函數
def predict_image(image_path):
    # 加載並轉換圖片為 RGB
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    # 禁用梯度計算進行預測
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

        # 獲取預測類別和信心值
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class_idx].item()

        # 確保概率分佈只有 num_classes 類別
        all_probabilities = probabilities.squeeze().tolist()[:num_classes]

    return predicted_class_idx, confidence, all_probabilities

# 遍歷測試資料夾中的圖片
for filename in os.listdir(test_dir):
    if filename.endswith(".png"):
        image_path = os.path.join(test_dir, filename)
        prediction_successful = False

        # 使用不同模型進行預測
        for model_index in range(len(model_paths)):
            if model_index != current_model_index:
                # 切換到新的模型
                current_model_index = model_index
                model = ViTForImageClassification.from_pretrained(
                    model_paths[current_model_index],
                    ignore_mismatched_sizes=True
                )
                processor = AutoProcessor.from_pretrained(model_paths[current_model_index], use_fast=True)

            # 根據模型選擇對應的標籤
            label_mapping = label_All
            if model_index == 1:
                label_mapping = label_All_B
            elif model_index == 2:
                label_mapping = label_All_C

            # 預測結果
            predicted_class_idx, confidence, all_probabilities = predict_image(image_path)

            # 提取正確的標籤前綴和預測標籤
            true_prefix = filename[:4]
            predicted_label = label_mapping.get(f"{predicted_class_idx:02}", "未知")

            # 驗證結果
            if predicted_label.startswith(true_prefix):
                print(f"正確: {filename} -> {predicted_label}, 信心值: {confidence:.4f}")
                correct += 1
                prediction_successful = True
                break
            else:
                print(f"--------ERROR-----------")
                print(f"模型 {model_index+1} 預測錯誤: {filename} -> 預測: {predicted_label}, 真實前綴: {true_prefix}, 信心值: {confidence:.4f}")
                # 輸出完整的概率分佈
                probabilities_str = ", ".join([f"{label_mapping.get(f'{idx:02}', '未知')}: {prob:.4f}" for idx, prob in enumerate(all_probabilities)])
                print(f"完整概率分佈: {probabilities_str}")

        # 如果所有模型均無法辨識
        if not prediction_successful:
            print(f"--------ERROR Result:-----------")
            print(f"無法辨識: {filename}")
            print(f"--------ERROR Result:-----------")
            incorrect += 1


# 統計和輸出總結
total = correct + incorrect
accuracy = correct / total * 100 if total > 0 else 0
print(f"-----------Results----------")
print(f"總正確數: {correct}")
print(f"總錯誤數: {incorrect}")
print(f"準確率: {accuracy:.2f}%")


