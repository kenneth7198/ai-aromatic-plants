document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('uploadForm');
  const resultDiv = document.getElementById('result');
  const uploadedImageDiv = document.getElementById('uploadedImage');
  
  // 檢查元素是否存在
  if (!form || !resultDiv || !uploadedImageDiv) {
    console.error("Could not find one or more required elements: form, resultDiv, uploadedImageDiv");
    return;
  }
  
  // 使用 fetch 並行獲取三個描述 JSON
  const fetchDescriptions = () => {
    return Promise.all([
      fetch('descriptionsA.json').then(response => response.json()),
      fetch('descriptionsB.json').then(response => response.json()),
      fetch('descriptionsC.json').then(response => response.json())
    ]);
  };

  let descriptionsA = {};
  let descriptionsB = {};
  let descriptionsC = {};
  
  // 獲取描述數據
  fetchDescriptions()
    .then(([dataA, dataB, dataC]) => {
      descriptionsA = dataA;
      descriptionsB = dataB;
      descriptionsC = dataC;
    })
    .catch(error => {
      console.error('Failed to load descriptions:', error);
    });
  
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
        resultDiv.textContent = 'Prediction failed';
        return;
      }
      
      const result = await response.json();
      console.log('Result from Flask:', result);

      // 清空結果區域
      resultDiv.innerHTML = ''; // 清除之前的內容
      
      const createResultParagraph = (modelName, prediction, descriptions) => {
        const paragraph = document.createElement('p');
        const description = descriptions[prediction] || "Unknown prediction";
        paragraph.textContent = `${modelName}: ${prediction} - ${description}`;
        return paragraph;
      };

      // 顯示 Model A 的預測結果和說明
      resultDiv.appendChild(createResultParagraph("Prediction from Model A", result.prediction.classification_prediction_A, descriptionsA));

      // 顯示 Model B 的預測結果和說明
      resultDiv.appendChild(createResultParagraph("Prediction from Model B", result.prediction.classification_prediction_B, descriptionsB));

      // 顯示 Model C 的預測結果和說明
      resultDiv.appendChild(createResultParagraph("Prediction from Model C", result.prediction.classification_prediction_C, descriptionsC));

      // 顯示上傳的圖片
      uploadedImageDiv.innerHTML = ''; // 清空之前的圖片
      const img = document.createElement('img');
      img.src = result.imageUrl;
      img.alt = 'Uploaded Image';
      img.style.maxWidth = '100%';
      uploadedImageDiv.appendChild(img);
      
    } catch (error) {
      console.error("Fetch error:", error);
      resultDiv.textContent = 'Prediction failed: ' + error.message;
    }
  });
});
