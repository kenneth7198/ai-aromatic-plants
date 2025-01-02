# ai-aromatic-plants
 
# 後端-環境設置
### 1. 在Anaconda建構一個flask虛擬環境，並用python 3.9.20版
### 2. 安裝Flask
```
pip install flask
```
### 3.安裝PyTorch 
a. 無GPU支援
```
pip install torch torchvision torchaudio 
``` 
b.有CUDA支援 ( 根據電腦是否有獨立顯示卡，若有財安裝cuda版本torch )
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
### 4. 安裝 einops
```
pip install einops
```
### 5. 安裝 transformer
```
pip install transformers
```
### 6. 安裝必要套件
```
pip install rembg
pip install Pillow numpy
```
### 7. 安裝GPU支援的ONNX Runtime 或GPU支援cuDNN與Nvdia CUDA Toolkit
#### 使用GPU進行加速
```
pip install onnxruntime-gpu
```
#### 使用CPU進行加速
``` 
pip install onnxruntime

```
### 下載預訓練好的模型到backend/model/ViT-plant-classifier/
* https://drive.google.com/drive/folders/1vRxy3Af2KyyhdSRwSTa10_wzQ8ja7j0B?usp=sharing


# 啟動&運行
1. 切換目錄到\ai-aromatic-plants\backend\
2. 啟動flask
```
python server_forViT.py
```
3. 開啟瀏覽器，輸入http://127.0.0.1:5000，在local端確認有啟動即可

# 安裝node.js啟動前端
1. 安裝node.js v20.18版本  https://nodejs.org/dist/v20.18.1/node-v20.18.1-x64.msi 
2. 安裝nvm管理node.js的版本  https://github.com/coreybutler/nvm-windows/releases/download/1.1.12/nvm-setup.exe
3. 確認node -v 目前電腦內安裝的版本
```
node -v
```
4. 用nvm確認已有的node.js
```
nvm list
```
5. 切換使用node版本
```
nvm use 20.18.0
```
教學說明：https://ithelp.ithome.com.tw/articles/10275647
6. 安裝不同版本的node.js
```
nvm install v20.18.0
```
7. 執行前端互動介面
```
node index.js     //用來預測新版的模型版本
node index2.js    // 原本用來預測tensorflow版本
```


# 測試模型用環境建置-Anaconda
在Anaconda建構一個testBed_forViT虛擬環境，並用python 3.12.x版
## 使用python開發
### 安裝必要工具與套件
```
pip install -r requirements.txt
```
## 使用Jupyter開發
### 安裝Jupyter
```
pip install
```

# 安裝 Jupyter Notebook
```
pip install notebook
```

# 建構基本的Pytorch for Nvdia GPU
安裝 python 3.8 ~ 3.11

# 安裝cuda -> cu118
https://stevehublog.medium.com/%E4%B8%80%E6%AD%A5%E4%B8%80%E6%AD%A5%E6%95%99%E6%82%A8%E5%9C%A8windows%E4%B8%8A%E5%AE%89%E8%A3%9Dpytorch%E8%88%87nvidia-cuda-a369eb88ee58

## 安裝有cuda的 torch + cuda 11.8
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
## 安裝 CUDA Toolkit 11.8 for windows
https://developer.nvidia.com/cuda-11-8-0-download-archive
