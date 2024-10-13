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

      // 獲取嵌套在 prediction 屬性中的預測數據
      const predictions = result.prediction;

      // 顯示每個模型的預測結果和百分比
      const displayModelResults = (modelName, prediction, probabilities, descriptions) => {
        const container = document.createElement('div');
        container.classList.add('model-result');

        const header = document.createElement('p');
        const description = descriptions[prediction] || "未知";
        header.textContent = `${modelName}: 預測結果為 - 類別標籤： ${prediction} - ${description}`;
        container.appendChild(header);

        // 創建一個 Canvas 元素來繪製直條圖
        if (Array.isArray(probabilities) && probabilities.length > 0) {
          const canvas = document.createElement('canvas');
          container.appendChild(canvas);

          new Chart(canvas, {
            type: 'bar',
            data: {
              labels: probabilities.map((_, index) => `類別標籤： ${index}`), // 類別標籤
              datasets: [{
                label: 'Probability (%)',
                data: probabilities,
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
              }]
            },
            options: {
              scales: {
                y: {
                  beginAtZero: true,
                  max: 100
                }
              },
              plugins: {
                legend: {
                  display: false
                }
              }
            }
          });
        } else {
          const noData = document.createElement('p');
          noData.textContent = '沒有預測的結果';
          container.appendChild(noData);
        }

        resultDiv.appendChild(container);
      };

      // 顯示 Model A 的預測結果和百分比
      displayModelResults("預測為 Model A", predictions.classification_prediction_A, predictions.classification_probabilities_A, descriptionsA);

      // 顯示 Model B 的預測結果和百分比
      displayModelResults("預測為 Model B", predictions.classification_prediction_B, predictions.classification_probabilities_B, descriptionsB);

      // 顯示 Model C 的預測結果和百分比
      displayModelResults("預測為 Model C", predictions.classification_prediction_C, predictions.classification_probabilities_C, descriptionsC);

      // 顯示上傳的圖片
      // uploadedImageDiv.innerHTML = ''; // 清空之前的圖片
      // const img = document.createElement('img');
      // img.src = result.imageUrl;
      // img.alt = 'Uploaded Image';
      // img.style.maxWidth = '100%';
      // uploadedImageDiv.appendChild(img);
      
    } catch (error) {
      console.error("Fetch error:", error);
      resultDiv.textContent = 'Prediction failed: ' + error.message;
    }
  });
});
