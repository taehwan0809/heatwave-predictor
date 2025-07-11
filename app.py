import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
import os

# 스타일 설정 (폰트 변경)
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

st.title("AI 기반 폭염 예측 시스템")

# 파일 경로
DATA_DIR = "data"
file_1_path = os.path.join(DATA_DIR, "2015~2025.csv")
file_2_path = os.path.join(DATA_DIR, "한 달.csv")

if os.path.exists(file_1_path) and os.path.exists(file_2_path):
    df_all = pd.read_csv(file_1_path)
    df_recent = pd.read_csv(file_2_path)

    df_recent['날짜'] = pd.to_datetime(df_recent['날짜'])
    df_recent = df_recent[df_recent['날짜'] <= datetime.today()]
    df_recent = df_recent.sort_values(by='날짜')
    df_recent['일자'] = (df_recent['날짜'] - df_recent['날짜'].min()).dt.days

    # 폭염일 수
    hot_days = df_recent[df_recent['최고기온'] >= 33]

    # LSTM 학습
    temps = df_recent['최고기온'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    temps_scaled = scaler.fit_transform(temps)

    X, y = [], []
    sequence_length = 7
    for i in range(len(temps_scaled) - sequence_length):
        X.append(temps_scaled[i:i + sequence_length])
        y.append(temps_scaled[i + sequence_length])
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(Input(shape=(sequence_length, 1)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, verbose=0)

    # 7일 예측
    pred_seq = temps_scaled[-sequence_length:]
    future_predictions = []
    for _ in range(7):
        pred_input = pred_seq.reshape(1, sequence_length, 1)
        pred = model.predict(pred_input, verbose=0)
        future_predictions.append(pred[0][0])
        pred_seq = np.append(pred_seq[1:], pred, axis=0)
    future_temps = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

    # 대응 가이드
    guide = []
    for temp in future_temps:
        if temp >= 35:
            guide.append("실외 활동 자제 + 냉방 준비 강화, 야외근로자 건강 주의")
        elif temp >= 33:
            guide.append("폭염주의보 수준, 물 자주 마시기, 야외근로자 규칙적인 휴식 필수")
        elif temp >= 30:
            guide.append("더위 대비 필요")
        else:
            guide.append("무난한 날씨")

    # 평균 기온
    avg_temp = np.mean(future_temps)

    # 탭 구성
    tabs = st.tabs(["폭염일 수", "다음 7일 예측", "대응 가이드", "평균 기온"])

    with tabs[0]:
        st.subheader("최근 한 달간 폭염일 수")
        st.write(f"폭염일 수: {len(hot_days)}일")

    with tabs[1]:
        st.subheader("다음 7일간 최고기온 예측")
        df_pred = pd.DataFrame({
            "날짜": [(datetime.today() + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)],
            "예측 기온 (℃)": [f"{temp:.2f}" for temp in future_temps]
        })
        st.table(df_pred)

    with tabs[2]:
        st.subheader("폭염 대응 가이드")
        df_guide = pd.DataFrame({
            "날짜": [(datetime.today() + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 8)],
            "예측 기온 (℃)": [f"{temp:.2f}" for temp in future_temps],
            "대응 가이드": guide
        })
        st.table(df_guide)

    with tabs[3]:
        st.subheader("다음 주 평균 최고기온")
        st.write(f"{avg_temp:.2f} ℃")

else:
    st.error("'data/' 폴더에 '2015~2025.csv' 와 '한 달.csv' 파일이 존재해야 합니다.")
