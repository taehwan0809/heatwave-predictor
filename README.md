# 🔥 HeatWavePredictor: AI-Based Heatwave Forecast & Response Guide

## 🌍 Overview

**HeatWavePredictor** is a user-friendly AI-powered web application that forecasts future heatwaves using 10 years of temperature data. The app not only predicts extreme temperatures across different regions but also provides tailored response guides based on users’ jobs or situations. It aims to help people prepare and respond effectively to future heat risks.

## 🧠 Features

* 📈 **AI Forecast**: Uses LSTM (Long Short-Term Memory) neural networks to analyze temperature trends over the past decade and predict upcoming extreme heat.
* 🌐 **Regional Selection**: Users can choose between major Korean cities like Seoul, Busan, Daegu, and Paju.
* 🧑‍🏭 **Job-Based Guide**: Provides specific heatwave response tips based on the user's job or daily activity (e.g., construction workers, delivery workers, elderly, students).
* 🖥 **Real-Time Web App**: Built with [Streamlit](https://streamlit.io), enabling quick, interactive AI analysis and visualization directly in your browser.

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/heatwave-predictor.git
   cd heatwave-predictor
   ```

2. Install required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

## 📊 Technology Stack

* **Python**
* **Pandas, NumPy** – Data preprocessing
* **TensorFlow / Keras** – LSTM deep learning model
* **Streamlit** – Frontend dashboard
* **Matplotlib, Plotly** – Graphs and visualization

## 💡 Future Direction

* ⌚️ **Integration with Wearables**: Connect with smartwatches to give real-time heat alerts and health monitoring.
* 📲 **Mobile App**: Develop a lightweight version for smartphones.
* 🌡 **Live Weather API**: Integrate real-time weather data from national weather APIs.
* 🌍 **Global Expansion**: Adapt model to other countries’ climate data.

## 🙌 Project Motivation

With increasing global temperatures, heatwaves are becoming a serious public health threat. This project aims to use AI not only to predict but also **prepare people** for these dangers, especially those in vulnerable jobs or age groups.
