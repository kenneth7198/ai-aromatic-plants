from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from tensorflow.keras.regularizers import l2
from PIL import Image

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

# 載入模型並使用 custom_objects 引入自訂層
model = load_model('model\A_model.h5', custom_objects={'Autoencoder': Autoencoder})

# 使用 Adam 優化器並設置 clipnorm 以防止梯度爆炸
optimizer = Adam(learning_rate=0.00005, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='categorical_crossentropy')

# 定義影像預處理函數
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

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
    image = Image.open(file).convert("RGB")
    processed_image = preprocess_image(image, target_size=(128, 128))

    # 獲取模型的兩個輸出
    decoded_image, classification = model.predict(processed_image)
    predicted_class = np.argmax(classification, axis=1)[0]

    # 返回預測結果
    return jsonify({
        'classification_prediction': int(predicted_class)
    })

# 運行應用
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
