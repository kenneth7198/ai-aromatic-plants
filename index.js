const express = require('express');
const multer = require('multer');
const axios = require('axios');
const fs = require('fs');
const FormData = require('form-data');

const app = express();

// 上傳到 public/uploads
const upload = multer({ dest: 'public/uploads/' });

// 設置靜態文件夾
app.use(express.static('public'));
app.use('/uploads', express.static('public/uploads'));

// 主頁顯示表單
app.get('/', (req, res) => {
  res.send(`

    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>AI香草辨識</title>
      <!-- 引入 Bootstrap CSS -->
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
      
    </head>
    <body class="bg-light">

    <div class="container py-5">
      <div class="row justify-content-center">
        <div class="col-md-10">
          <div class="card shadow">
            <div class="card-header bg-primary text-white">
              <h3 class="card-title text-center">AI香草辨識系統</h3>
            </div>
            <div class="card-body">
              <form id="uploadForm" enctype="multipart/form-data">
                <div class="mb-3">
                  <label for="model" class="form-label">選擇模型:</label>
                  <select id="model" name="model" class="form-select" required>
                    <option value="best_model">最佳模型 (Best)</option>
                    <option value="final_model">最終模型 (Final)</option>
                    <option value="complete_model">完整模型 (Complete)</option>
                  </select>
                </div>
                <div class="mb-3">
                  <label for="image" class="form-label">上傳圖片:</label>
                  <input type="file" name="image" id="image" class="form-control" required>
                </div>
                <button type="submit" class="btn btn-success btn-lg w-100">上傳 & 預測</button>
              </form>
            </div>
          </div>
        </div>
      </div>

      <div class="row justify-content-center mt-4">
        <div class="col-md-10">
          <div id="result" class="alert alert-secondary text-center" role="alert">
            預測結果將顯示在這裡。
          </div>
        </div>
      </div>

      <div class="row justify-content-center mt-2">
        <div class="col-md-6 text-center">
          <div id="uploadedImage"></div>
        </div>
      </div>
    </div>

    <!-- 引入 Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels"></script>
    <script src="script.js"></script>
    </body>
    </html>



  `);
});

// 接收圖片並轉發至 Flask 應用進行預測
app.post('/upload', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }

  const filePath = req.file.path;
  const selectedModel = req.body.model; // 接收用戶選擇的模型

  try {
    const formData = new FormData();
    formData.append('file', fs.createReadStream(filePath));
    formData.append('model', selectedModel); // 傳遞模型名稱到 Flask

    const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
      headers: {
        ...formData.getHeaders()
      }
    });

    res.json({
      prediction: response.data,
      imageUrl: `/uploads/${req.file.filename}`
    });
  } catch (error) {
    console.error('Error:', error.response ? error.response.data : error.message);
    res.status(500).json({ error: 'Prediction failed' });
  } finally {
    fs.unlink(filePath, (err) => {
      if (err) console.error("Failed to delete temp file:", err);
    });
  }
});



app.listen(3000, () => {
  console.log('Node.js server is running on http://localhost:3000');
});
