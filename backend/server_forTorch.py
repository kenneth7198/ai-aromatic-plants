from flask import Flask, request, jsonify, render_template
from torch import nn, optim
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import einops
import os

# 引入外部文件中的類別
from deformconv_snakeconv import DeformConv2d, DSConv_pro

# 初始化 Flask 應用
app = Flask(__name__)

# 定义混合卷积网络结构
class HybridConvNet(nn.Module):
    def __init__(self, num_classes):
        super(HybridConvNet, self).__init__()
        self.conv1 = DeformConv2d(3, 32, kernel_size=3, padding=1, modulation=True)
        self.conv2 = DeformConv2d(32, 64, kernel_size=3, padding=1, modulation=True)
        self.conv3 = DeformConv2d(64, 128, kernel_size=3, padding=1, modulation=True)
        self.conv4 = DSConv_pro(128, 128, kernel_size=3, extend_scope=1.0, morph=0)
        self.conv5 = DSConv_pro(128, 256, kernel_size=3, extend_scope=1.0, morph=1)
        self.conv6 = DSConv_pro(256, 256, kernel_size=3, extend_scope=1.0, morph=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 載入模型
num_classes = 23  # 根據實際數據集的類別數量設置
model_paths = {
    "best_model": "model/best_model.pth",
    "final_model": "model/final_trained_model.pth",
    "complete_model": "model/complete_model.pth"
}
models = {}
for key, path in model_paths.items():
    model = HybridConvNet(num_classes=num_classes)
    loaded_data = torch.load(path)
    if isinstance(loaded_data, dict):
        model.load_state_dict(loaded_data)
    else:
        # Assume it's a model instance, extract state_dict
        torch.save(loaded_data.state_dict(), path)
        model.load_state_dict(torch.load(path))
    model.eval()
    models[key] = model

# 定義影像預處理函數
def preprocess_image(image, target_size=(128, 128)):
    image = image.convert("RGB")  # 確保圖像為 RGB 模式
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform(image).unsqueeze(0)


# 設定首頁路由
@app.route('/')
def home():
    return render_template('index.html')

# 設定預測路由
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    model_type = request.form.get('model', 'best_model')
    if model_type not in models:
        return jsonify({'error': f'Invalid model type. Available models: {list(models.keys())}'})

    # 處理影像並進行預測
    image = Image.open(file)
    processed_image = preprocess_image(image)

    model = models[model_type]
    with torch.no_grad():
        output = model(processed_image)
    predicted_class = torch.argmax(output, dim=1).item()
    probabilities = torch.nn.functional.softmax(output, dim=1)[0].tolist()

    # 列印模型的預測結果
    print(f"Model: {model_type}, Prediction: {predicted_class}, Probabilities: {probabilities}")

    # 返回模型的預測結果和各類別百分比分佈
    return jsonify({
        'model': model_type,
        'classification_prediction': int(predicted_class),
        'classification_probabilities': [round(float(prob) * 100, 2) for prob in probabilities]
    })

# 運行應用
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
