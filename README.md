# Breast Cancer Histology Analysis Using Deep Learning

## 📌 Project Overview
This project focuses on **Breast Cancer Histology Image Classification** using **Deep Learning**. It applies **Convolutional Neural Networks (CNNs)** to analyze and classify histology images into different categories, aiding in **early cancer diagnosis**. The dataset is sourced from the **ICIAR 2018 Grand Challenge on Breast Cancer Histology Images**.

## 🚀 Key Features
- **Medical Imaging Analysis:** Uses deep learning for classifying breast cancer histology images.
- **Deep Learning Architectures:** Implements and compares multiple CNN models.
- **Performance Evaluation:** Achieves high accuracy in classifying cancerous and non-cancerous tissues.
- **Optimized Model:** Uses **MobileNetV2**, **XceptionNet**, and other architectures for efficient classification.

## 📂 Dataset
- **Source:** ICIAR 2018 Breast Cancer Histology Images.
- **Categories:**
  - Normal (100 images)
  - Benign (100 images)
  - In situ carcinoma (100 images)
  - Invasive carcinoma (100 images)
- **Format:** `.tiff` images in RGB (2048x1536 pixels).

## 🏗️ Methodology
1. **Data Preprocessing**
   - Load and convert `.tiff` images to numpy arrays.
   - Resize images to **299x299** for uniform processing.
   - Normalize pixel values for better training stability.

2. **Model Training & Evaluation**
   - Implemented deep learning models:
     - **DenseNet**
     - **MobileNetV2** (Best Performing Model)
     - **InceptionNetV3**
     - **XceptionNet**
     - **Inception ResNetV2**
   - Performance measured using **Accuracy, Precision, and Recall**.

3. **Results Comparison**
   | Model               | Training Accuracy | Test Accuracy | Validation Accuracy |
   |--------------------|-----------------|--------------|------------------|
   | DenseNet           | 97.92%          | 87.5%        | 81.88%           |
   | **MobileNetV2**    | **99.17%**      | **90.62%**   | **90.62%**       |
   | InceptionNetV3     | 94.17%          | 70.63%       | 71.88%           |
   | XceptionNet        | 98.75%          | 92.5%        | 90.0%            |
   | Inception ResNetV2 | 95.63%          | 86.25%       | 81.25%           |

## 📊 Findings
- **MobileNetV2 achieved the highest accuracy (99.17% training, 90.62% test).**
- **XceptionNet performed well (98.75% training, 92.5% test).**
- **CNN-based architectures provide reliable classification, aiding in early breast cancer detection.**

## 🛠 Tech Stack
- **Programming Language:** Python 3.x
- **Deep Learning Frameworks:** TensorFlow, Keras
- **Data Processing:** OpenCV, NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn

## 🔧 Installation & Setup
1. **Clone the Repository**
   ```sh
   git clone https://github.com/rohii52/Breast-Cancer-Histology-Analysis.git
   cd Breast-Cancer-Histology-Analysis
   ```
2. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```
3. **Run the Training Script**
   ```sh
   python train_model.py
   ```
4. **Evaluate the Model**
   ```sh
   python evaluate_model.py
   ```

## 🔮 Future Enhancements
- Implementing **transfer learning** for better model generalization.
- Deploying the trained model as a **Flask/FastAPI Web Service**.
- Developing an interactive **dashboard using Streamlit** for real-time image classification.

## 📄 References
- **ICIAR 2018 Dataset**: [ICIAR Challenge](https://iciar2018-challenge.grand-challenge.org/)
- **Deep Learning for Medical Imaging**: Research Papers & Documentation

📧 Contact: rohithgofficial@gmail.com  
🔗 GitHub: [github.com/rohii52](https://github.com/rohii52)  
💼 LinkedIn: [linkedin.com/in/rohii52](https://linkedin.com/in/rohii52)  

🌟 If you find this project useful, give it a ⭐ on GitHub!
