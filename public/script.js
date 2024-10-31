document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('uploadForm');
  const resultDiv = document.getElementById('result');
  const uploadedImageDiv = document.getElementById('uploadedImage');
  
  // 檢查元素是否存在
  if (!form || !resultDiv || !uploadedImageDiv) {
    console.error("Could not find one or more required elements: form, resultDiv, uploadedImageDiv");
    return;
  }
  
  // 使用 fetch 獲取描述 JSON 和植物名稱 JSON
  const fetchDescriptions = () => fetch('descriptions.json').then(response => response.json());
  const fetchPlantNames = () => fetch('plants.json').then(response => response.json());

  let descriptions = {};
  let plantNames = {};

  // 獲取描述數據和植物名稱
  Promise.all([fetchDescriptions(), fetchPlantNames()])
    .then(([dataDescriptions, dataPlants]) => {
      descriptions = dataDescriptions;
      plantNames = dataPlants;
    })
    .catch(error => {
      console.error('Failed to load descriptions or plant names:', error);
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
      const prediction = result.prediction;

      // 顯示模型的預測結果和百分比
      const displayModelResults = (modelName, predictionClass, probabilities, descriptions) => {
        const container = document.createElement('div');
        container.classList.add('model-result');

        const header = document.createElement('p');
        const description = descriptions[predictionClass] || "Unknown prediction";
        header.textContent = `${modelName}: Prediction ${predictionClass} - ${description}`;
        container.appendChild(header);

        // 創建一個 Canvas 元素來繪製直條圖
        if (Array.isArray(probabilities) && probabilities.length > 0) {
          const canvas = document.createElement('canvas');
          container.appendChild(canvas);

          // 使用植物名稱替換類別標籤
          const labels = probabilities.map((_, index) => plantNames[index] || `Class ${index}`);

          new Chart(canvas, {
            type: 'bar',
            data: {
              labels: labels, // 使用植物名稱標籤
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
                  max: 120 // 將最大值設置為120%
                }
              },
              plugins: {
                legend: {
                  display: false
                },
                datalabels: {
                  anchor: 'end',
                  align: 'end',
                  formatter: (value) => `${value}%`,  // 格式化顯示為百分比
                  color: '#000',
                  font: {
                    weight: 'bold'
                  }
                }
              }
            },
            plugins: [ChartDataLabels]  // 啟用數據標籤插件
          });
        } else {
          const noData = document.createElement('p');
          noData.textContent = 'No probability data available';
          container.appendChild(noData);
        }

        resultDiv.appendChild(container);
      };

      // 顯示模型的預測結果和百分比
      displayModelResults(
        "Model Prediction",
        prediction.classification_prediction_All,
        prediction.classification_probabilities_All,
        descriptions
      );

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
