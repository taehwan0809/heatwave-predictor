☀️ AI-Powered Heatwave Prediction and Response Guide
A Streamlit-based web application that predicts future heatwave trends using 10 years of temperature data and provides tailored response guides for various occupations.

🔍 Features
✅ Region Selection: Predict heatwaves for multiple cities (e.g., Seoul, Busan, Daegu, Paju).

🧠 LSTM-based Prediction: Uses deep learning (LSTM) to analyze 10 years of temperature data and forecast upcoming heatwaves.

👷 Job-Specific Response Guides: Provides customized safety tips and coping strategies for different professions (e.g., outdoor workers, office workers, seniors).

🌐 User-Friendly Interface: Built with Streamlit for real-time interaction and easy accessibility.

🚀 How It Works
Data Input: Loads regional temperature data from CSV files.

Model Prediction: LSTM model predicts future max temperatures.

Result Display: Shows graph and interpretation of predicted heatwaves.

Custom Guide: Displays personalized advice based on selected job type.

🛠 Tech Stack
Python

TensorFlow / Keras

Streamlit

Pandas, NumPy, Matplotlib

📌 Getting Started
bash
복사
편집
git clone https://github.com/your-username/heatwave-predictor.git
cd heatwave-predictor
pip install -r requirements.txt
streamlit run app.py
