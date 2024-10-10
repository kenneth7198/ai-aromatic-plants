const express = require('express');
const multer = require('multer');
const axios = require('axios');
const fs = require('fs');
const FormData = require('form-data');

const app = express();
const upload = multer({ dest: 'uploads/' });

// 主頁顯示表單
app.get('/', (req, res) => {
  res.send(`
    <h1>Upload an Image for Prediction</h1>
    <form id="uploadForm" enctype="multipart/form-data">
      <input type="file" name="image" required>
      <button type="submit">Upload and Predict</button>
    </form>
    <div id="result"></div>
    <div id="enhancedImage"></div>
    
    <script>
      const form = document.getElementById('uploadForm');
      form.addEventListener('submit', async (event) => {
        event.preventDefault();
        
        const formData = new FormData(form);
        
        try {
          const response = await fetch('/upload', {
            method: 'POST',
            body: formData,
          });
          
          if (!response.ok) {
            const errorText = await response.text();
            console.error("Server response error:", errorText);
            document.getElementById('result').innerHTML = 'Prediction failed';
            return;
          }
          
          const result = await response.json();
          document.getElementById('result').innerHTML = 'Prediction: ' + result.classification_prediction;
          
          // 確認 enhanced_image 存在再顯示
          if (result.enhanced_image) {
            const img = document.createElement('img');
            img.src = 'data:image/jpeg;base64,' + result.enhanced_image;
            img.alt = 'Enhanced Image';
            img.style.maxWidth = '100%';
            const enhancedImageDiv = document.getElementById('enhancedImage');
            enhancedImageDiv.innerHTML = '';
            enhancedImageDiv.appendChild(img);
          } else {
            console.error("Enhanced image data is missing.");
            document.getElementById('enhancedImage').innerHTML = 'No enhanced image received';
          }
          
        } catch (error) {
          console.error("Fetch error:", error);
          document.getElementById('result').innerHTML = 'Prediction failed: ' + error.message;
        }
      });
    </script>
  `);
});

// 接收圖片並轉發至 Flask 應用進行預測
app.post('/upload', upload.single('image'), async (req, res) => {
  const filePath = req.file.path;

  try {
    const formData = new FormData();
    formData.append('file', fs.createReadStream(filePath));

    // 發送圖片到 Flask 應用
    const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
      headers: {
        ...formData.getHeaders()
      }
    });

    // 返回 Flask 響應的結果
    res.json(response.data);

  } catch (error) {
    console.error('Error:', error.response ? error.response.data : error.message);
    res.status(500).json('Prediction failed: ' + (error.response ? error.response.data.error : error.message));
  } finally {
    fs.unlinkSync(filePath); // 刪除臨時文件
  }
});

app.listen(3000, () => {
  console.log('Node.js server is running on http://localhost:3000');
});
