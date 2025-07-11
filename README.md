# ☀️ AI-Powered Heatwave Forecasting

A Streamlit-based web application that predicts heatwaves in South Korea for the next 7 days using LSTM (Long Short-Term Memory) deep learning models trained on past temperature data.

## 📌 Features

- Predicts the next 7 days of maximum daily temperature
- Classifies and visualizes potential heatwave days
- Provides daily heat safety recommendations
- User-friendly web UI with navigation tabs

## 📁 Dataset

This project uses two datasets:

- 2015~2025.csv: Historical maximum temperature data (2015–2025)
- 한 달.csv: Recent daily temperature data (most recent month)

> 📌 **Data Source**:  
> Both datasets were sourced from the [Korean Public Data Portal (공공데이터포털)](https://www.data.go.kr), specifically from temperature records provided by the Korea Meteorological Administration (KMA).

You may replace or supplement these datasets with real-time API data in future versions.

## 🔧 Tech Stack

- Python 3.11+
- Streamlit
- TensorFlow / Keras
- NumPy, Pandas
- Scikit-Learn
- Matplotlib

- ⚙️ How to Run
Clone this repository:

git clone https://github.com/yourusername/heatwave-predictor.git
cd heatwave-predictor

Install dependencies:

pip install -r requirements.txt

Create a data/ folder and add the required datasets:

heatwave-predictor/
├── data/
│   ├── 2015~2025.csv
│   └── 한 달.csv

Run the Streamlit app:

python -m streamlit run app.py
