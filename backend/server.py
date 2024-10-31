from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import AdamW
import tensorflow as tf
import numpy as np
from tensorflow.keras.regularizers import l2
from PIL import Image, ImageEnhance
import io
import base64

# 初始化 Flask 應用
app = Flask(__name__)

# 定義 Autoencoder 類別
class Autoencoder(tf.keras.models.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # 編碼器
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(128, 128, 3)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=1, kernel_regularizer=l2(1e-6)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=1, kernel_regularizer=l2(1e-6)),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', strides=1, kernel_regularizer=l2(1e-6)),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', strides=1, kernel_regularizer=l2(1e-6)),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        ])

        # 解碼器
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(256, kernel_size=3, strides=2, activation='relu', padding='same', kernel_regularizer=l2(1e-6)),
            tf.keras.layers.Conv2DTranspose(128, kernel_size=3, strides=2, activation='relu', padding='same', kernel_regularizer=l2(1e-6)),
            tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same', kernel_regularizer=l2(1e-6)),
            tf.keras.layers.Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same')
        ])

        # 分類器
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(25, activation='softmax')
        ])
        
        # 初始化模型權重
        self.encoder.build((None, 128, 128, 3))
        self.decoder.build((None, 16, 16, 512))
        self.classifier.build((None, 16, 16, 512))

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        classification = self.classifier(encoded)
        return decoded, classification

    def get_config(self):
        config = super(Autoencoder, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls()

# 載入三個模型並設置優化器
model_All = load_model('model/All_model.h5', custom_objects={'Autoencoder': Autoencoder})
#model_A = load_model('model/A_model.h5', custom_objects={'Autoencoder': Autoencoder})
#model_B = load_model('model/B_model.h5', custom_objects={'Autoencoder': Autoencoder})
#model_C = load_model('model/C_model.h5', custom_objects={'Autoencoder': Autoencoder})

# 使用 Adam 優化器並設置 clipnorm 以防止梯度爆炸
optimizer = AdamW(learning_rate=0.00005, weight_decay=1e-6, clipnorm=1.0)

model_All.compile(optimizer=optimizer, loss='categorical_crossentropy')

#model_A.compile(optimizer=optimizer, loss='categorical_crossentropy')
#model_B.compile(optimizer=optimizer, loss='categorical_crossentropy')
#model_C.compile(optimizer=optimizer, loss='categorical_crossentropy')

# 定義影像預處理函數
def preprocess_image(image, target_size=(128, 128)):
    # 調整圖片大小
    image = image.resize(target_size)
    
    # 確保影像為 RGB 模式
    image = image.convert("RGB")
    
    # 增強亮度
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.5)
    
    # 轉換為 numpy 陣列
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # 正規化
    return img_array

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

    # 處理影像並進行預測
    image = Image.open(file)
    processed_image = preprocess_image(image)

    # 使用 All_model 進行預測
    _, classification_All = model_All.predict(processed_image)
    predicted_class_All = np.argmax(classification_All, axis=1)[0]
    probabilities_All = [round(float(prob) * 100, 2) for prob in classification_All[0]]  # 轉換為百分比並保留兩位小數

    # 列印模型的預測結果
    print(f"Model All Prediction: {predicted_class_All}, Probabilities: {probabilities_All}")

    # 返回模型的預測結果和各類別百分比分佈
    return jsonify({
        'classification_prediction_All': int(predicted_class_All),
        'classification_probabilities_All': probabilities_All
    })

# 運行應用
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)