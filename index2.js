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
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels"></script>
    <h1>AI香草辨識 - 請上傳圖片進行預測</h1>
    <form id="uploadForm" enctype="multipart/form-data">
      <input type="file" name="image" required>
      <button type="submit">上傳&預測</button>
    </form>
    <div id="result"></div>
    <div id="uploadedImage"></div>
    
    <!-- 引入外部 JavaScript 文件 -->
    <script src="script.js"></script>


  `);
});

// 接收圖片並轉發至 Flask 應用進行預測
app.post('/upload', upload.single('image'), async (req, res) => {
  
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }

  // 打印文件信息以確認保存路徑
  console.log("File uploaded to:", req.file.path);
  
  const filePath = req.file.path;
  console.log("File saved at:", filePath); // 打印文件保存的路徑

  try {
    const formData = new FormData();
    formData.append('file', fs.createReadStream(filePath));

    // 發送圖片到 Flask 應用
    const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
      headers: {
        ...formData.getHeaders()
      }
    });

    // 返回 Flask 應用的響應結果和圖片URL到前端
    res.json({
      prediction: response.data,
      imageUrl: `/uploads/${req.file.filename}` // 直接使用 /uploads 路徑
    });
  } catch (error) {
    console.error('Error:', error.response ? error.response.data : error.message);
    if (!res.headersSent) {
      res.status(500).json({ error: 'Prediction failed: ' + (error.response ? error.response.data.error : error.message) });
    }
  } finally {
    fs.unlink(filePath, (err) => {
      if (err) {
        console.error("Failed to delete temp file:", err);
      }
    });
  }
});

app.listen(3000, () => {
  console.log('Node.js server is running on http://localhost:3000');
});
