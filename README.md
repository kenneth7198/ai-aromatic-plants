# ai-aromatic-plants
 
# 環境設置
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

# 啟動&運行
1. 切換目錄到\ai-aromatic-plants\backend\
2. 啟動flask
```
flask run
```
3. 開啟瀏覽器，輸入http://127.0.0.1:5000，在local端確認有啟動即可

# 安裝node.js啟動前端
1. 啟動前端，安裝node.js v20.18版本
 
