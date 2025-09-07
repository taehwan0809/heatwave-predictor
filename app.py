import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler

# 페이지 설정
st.set_page_config(page_title="폭염 예측 대시보드", layout="wide")

# CSS 스타일
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}
.block-container {
    max-width: 900px;
    margin: auto;
    padding-top: 2rem;
}
.metric-box {
    border-radius: 8px;
    padding: 1rem;
    color: white;
    text-align: center;
}
.temp-box { background-color: #e63946; }
.gap-box  { background-color: #457b9d; }
</style>
""", unsafe_allow_html=True)

# 타이틀
st.markdown("<h1 style='text-align:center;'> <span style='color:#e63946;'>폭염 예측</span> 대시보드 </h1>", unsafe_allow_html=True)
st.markdown("### 한국 주요 도시의 여름철 기온 및 일교차 예측 결과")

# 데이터 경로
DATA_FILES = {
    "서울": {"temp": "data/서울/서울 2015~2025 기온.csv",
        "humid": "data/서울/서울 2015~2025 습도.csv",
        "rain": "data/서울/서울 2015~2025 강수량.csv",},

    "부산": {
        "temp": "data/부산/부산 2015~2025 기온.csv",
        "humid": "data/부산/부산 2015~2025 습도.csv",
        "rain": "data/부산/부산 2015~2025 강수량.csv",
    },
    "대구": {
        "temp": "data/대구/대구 2015~2025 기온.csv",
        "humid": "data/대구/대구 2015~2025 습도.csv",
        "rain": "data/대구/대구 2015~2025 강수량.csv",
    },
    "파주": {
        "temp": "data/파주/파주 2015~2025 기온.csv",
        "humid": "data/파주/파주 2015~2025 습도.csv",
        "rain": "data/파주/파주 2015~2025 강수량.csv",
    },
}

# 상태 관리
if 'confirmed' not in st.session_state:
    st.session_state.confirmed = False
region = st.selectbox("지역을 선택하세요", list(DATA_FILES.keys()))
user_role = st.selectbox("역할/직업 선택", ["전체 보기","노약자","야외근로자","학생","학교/기관","영유아 보호자","배달/택배업","소상공인","농업 종사자","건설 현장","반려동물 보호자","지체 장애인"])
if not st.session_state.confirmed:
    if st.button("확인"):
        st.session_state.confirmed = True
    st.stop()
if st.button("다시 선택"):
    st.session_state.confirmed = False
    st.rerun()

# 예측 함수
# 1) 전체 데이터 사용, 2) 스케일러 분리 학습, 3) 시퀀스 길이 늘리기
def predict_multivariate(df, target_col):
    feature_cols = ['최고기온(℃)', '습도', '강수량']

    for col in feature_cols:
        if col not in df.columns:
            raise KeyError(f"❌ 데이터에 '{col}' 컬럼이 없습니다. 현재 컬럼: {df.columns.tolist()}")

    target_idx = feature_cols.index(target_col)

    features = df[feature_cols].dropna().values
    scaler = MinMaxScaler().fit(features)
    scaled = scaler.transform(features)

    seq_len = 30
    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i+seq_len])
        y.append(scaled[i+seq_len][target_idx])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        Input(shape=(seq_len, X.shape[2])),
        LSTM(64, activation='relu'),
        Dense(1)
    ])
    model.compile('adam', 'mse')
    model.fit(X, y, epochs=20, verbose=0)

    seq_vals = scaled[-seq_len:]
    preds = []
    for _ in range(30):
        p = model.predict(seq_vals.reshape(1,seq_len,X.shape[2]), verbose=0)[0][0]
        preds.append(p)
        next_row = seq_vals[-1].copy()
        next_row[target_idx] = p
        seq_vals = np.append(seq_vals[1:], [next_row], axis=0)

    # 스케일 복원
    pred_array = np.zeros((len(preds), len(feature_cols)))
    pred_array[:, target_idx] = preds
    preds_rescaled = scaler.inverse_transform(pred_array)[:, target_idx]
    return preds_rescaled

def predict_heatwave_end(dates, temps):
    # 30℃ 이하 3일 연속이면 종료일로 판단
    for i in range(len(temps) - 2):
        if temps[i] <= 30 and temps[i+1] <= 30 and temps[i+2] <= 30:
            return dates[i].strftime("%Y-%m-%d")
    return "올해 여름 내내 30℃ 이상 유지"



def predict_yearly(df, col, year):
    # 1. 데이터 전처리
    series = df[['일시', col]].dropna().sort_values('일시')
    data_full = series[[col]].values.reshape(-1,1)
    scaler = MinMaxScaler().fit(data_full)
    scaled_full = scaler.transform(data_full)

    # 2. 학습 데이터 준비
    seq_len = 30
    X, y = [], []
    for i in range(len(scaled_full) - seq_len):
        X.append(scaled_full[i:i+seq_len])
        y.append(scaled_full[i+seq_len])
    X, y = np.array(X), np.array(y)

    # 3. LSTM 모델 학습
    model = Sequential([
        Input(shape=(seq_len,1)),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile('adam', 'mse')
    model.fit(X, y, epochs=50, verbose=0)

    # 4. 미래 2년치(730일) 예측
    seq_vals = scaled_full[-seq_len:]
    preds = []
    for _ in range(365*2):  
        p = model.predict(seq_vals.reshape(1,seq_len,1),verbose=0)[0][0]
        preds.append(p)
        seq_vals = np.append(seq_vals[1:], [[p]], axis=0)

    # 5. 예측값을 실제 값으로 변환
    preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    future_dates = pd.date_range(df['일시'].max()+pd.Timedelta(days=1), periods=len(preds))

    # 6. 원하는 연도(year)의 6~8월 평균
    future_df = pd.DataFrame({"일시": future_dates, col: preds_inv})
    summer_df = future_df[(future_df['일시'].dt.year == year) & (future_df['일시'].dt.month.isin([6,7,8]))]
    return summer_df[col].mean()


def load_and_clean_csv(path):
    # skiprows 제거 → 헤더 그대로 읽기
    df = pd.read_csv(path, encoding="cp949")

    df.columns = df.columns.str.strip().str.replace(r"[\t\n\r]", "", regex=True)

    # '일시' 컬럼 찾기
    candidates = [c for c in df.columns if "일시" in c or "날짜" in c or "관측" in c]
    if not candidates:
        raise ValueError(f"⚠️ '{path}' 파일에서 '일시' 컬럼을 찾을 수 없습니다. 현재 컬럼: {df.columns.tolist()}")
    date_col = candidates[0]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.rename(columns={date_col: "일시"})
    return df




    



# 메인
if region:
    # 데이터 로드 및 일시 설정
    
    df_temp = load_and_clean_csv(DATA_FILES[region]["temp"])
    df_humid = load_and_clean_csv(DATA_FILES[region]["humid"])
    df_rain  = load_and_clean_csv(DATA_FILES[region]["rain"])

    # 날짜 단위로 통일
    df_temp['일시'] = pd.to_datetime(df_temp['일시']).dt.date
    df_humid['일시'] = pd.to_datetime(df_humid['일시']).dt.date
    df_rain['일시']  = pd.to_datetime(df_rain['일시']).dt.date


    df = df_temp.merge(df_humid, on="일시", how="inner") \
            .merge(df_rain, on="일시", how="inner")
    
    if "강수량(mm)" in df.columns:
        df["강수량(mm)"] = df["강수량(mm)"].fillna(0)

    df = df.sort_values("일시").interpolate()

    numeric_cols = ["최고기온(℃)", "최저기온(℃)", "습도", "강수량"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    rename_map = {
        "강수량(mm)": "강수량",
        "최고기온(℃)": "최고기온(℃)",
        "최저기온(℃)": "최저기온(℃)",
        "평균기온(℃)": "평균기온(℃)",
        "습도": "습도"
    }
    df = df.rename(columns=rename_map)
    df.columns = df.columns.str.strip()


    today = pd.Timestamp.today().normalize()
    dates = [today + pd.Timedelta(days=i+1) for i in range(30)]

    # 예측
    temps = predict_multivariate(df, '최고기온(℃)')

    low_col = [c for c in df.columns if "최저" in c][0]
    df['일교차'] = df['최고기온(℃)'] - df[low_col]

    gaps  = df['일교차'].tail(30).values  # 최근 30일치만 사용

    

    # 팝업 경고 및 효과
    max_temp = max(temps)
    if max_temp >= 35:
        st.markdown(f"""
            <div style='background:#ff0000; padding:1rem; text-align:center; font-weight:bold; color:#000;'>🚨 주의 {region} 최고 {max_temp:.1f}℃ 예측!🚨 </div>
        """, unsafe_allow_html=True)
    elif max_temp >= 33:
        # 배경 대비 높은 가독성을 위해 텍스트 컬러를 검은색으로 변경
        st.markdown(
            f"<div style='background:#F1C40F; padding:1rem; text-align:center; font-weight:bold; color:#000;'>⚠️ {region} 최고 {max_temp:.1f}℃ 예측!</div>",
            unsafe_allow_html=True)
        
        

    # 예측 요약 테이블
    st.subheader("📋 예측 요약")
    df_tbl = pd.DataFrame({
        '일시':     [d.strftime('%m-%d') for d in dates],
        '최고기온(℃)': temps,
        '일교차':   gaps
    })
    st.dataframe(df_tbl.style.format({'최고기온(℃)':'{:.1f}℃','일교차':'{:.1f}℃'}), use_container_width=True)

    # KPI 카드
    avg_temp = np.mean(temps)
    avg_gap  = np.mean(gaps)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"<div class='metric-box temp-box'><h3>7일 평균<br>최고기온(℃)</h3><h1>{avg_temp:.1f}℃</h1></div>",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"<div class='metric-box gap-box'><h3>7일 평균<br>일교차</h3><h1>{avg_gap:.1f}℃</h1></div>",
            unsafe_allow_html=True
        )
    st.markdown("---")
        # 2026년, 2027년 예측 추가
    future_2026 = predict_yearly(df, '최고기온(℃)', 2026)
    future_2027 = predict_yearly(df, '최고기온(℃)', 2027)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown(
            f"<div class='metric-box temp-box'><h3>2026년 여름<br>예상 평균 최고기온(℃)</h3><h1>{future_2026:.1f}℃</h1></div>",
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            f"<div class='metric-box temp-box'><h3>2027년 여름<br>예상 평균 최고기온(℃)</h3><h1>{future_2027:.1f}℃</h1></div>",
            unsafe_allow_html=True
        )


        # 더위 종료일 예측
heatwave_end = predict_heatwave_end(dates, temps)

st.subheader("🌅 더위 종료일 예측")
st.markdown(f"**예상 종료일:** {heatwave_end}")





    # 상세 대응 가이드 (생략: 기존 all_guides & expander 로직 적용)
    # 상세 대응 가이드
st.markdown("---")
st.markdown("### 폭염 대응 가이드 (선택한 역할 기준)")

all_guides = {
                "노약자": {
                    "35": [
                        "🏠 실외 활동 전면 금지",
                        "❄️ 냉방기(에어컨, 선풍기 등) 24시간 가동",
                        "💧 미지근한 물로 자주 양치질",
                        "👩‍⚕️ 1일 2회 이상 체온·맥박 측정"
                    ],
                    "33": [
                        "🚫 장시간 외출 삼가기",
                        "🥤 30분마다 물 또는 이온음료 1잔 섭취",
                        "🧊 아이스팩 목·겨드랑이 부착",
                        "📞 가족·이웃에게 안부 확인 요청"
                    ],
                    "30": [
                        "☂️ 그늘 또는 실내 휴식 권장",
                        "🥣 수분 많은 과일(수박 등) 섭취",
                        "🎶 가벼운 음악 감상으로 스트레스 완화",
                        "🔄 2시간마다 시원한 곳으로 이동"
                    ],
                    "27": [
                        "🚶‍♀️ 무리 없는 가벼운 산책 가능",
                        "🎗️ 모자·양산으로 햇빛 차단",
                        "📋 수분 섭취 알람 설정",
                        "🛋️ 휴식 공간에 선풍기 배치"
                    ],
                    "0": [
                        "😊 쾌적한 날씨입니다. 평소처럼 활동하세요!",
                        "🗓️ 내일 예보 확인도 잊지 마세요."
                    ]
                },
                "야외근로자": {
                    "35": [
                        "⚠️ 작업 전면 금지 권고",
                        "🧊 얼음조끼 및 쿨링타월 착용",
                        "💧 물·이온음료 수시 섭취",
                        "⏱️ 30분 작업 후 15분 휴식"
                    ],
                    "33": [
                        "🏖️ 그늘막 설치 및 휴식 확보",
                        "🥤 30분마다 수분 보충",
                        "📏 주변 온도·습도 자주 확인",
                        "🎧 안전 교육 영상 시청"
                    ],
                    "30": [
                        "🛖 쉼터에서 휴식",
                        "🥣 과일·시원한 음료 섭취",
                        "📝 작업 강도 조절 계획 수립",
                        "🔄 1시간 작업 후 10분 휴식"
                    ],
                    "27": [
                        "👕 통풍 잘 되는 작업복 착용",
                        "🧢 모자·선글라스 필수",
                        "⛑️ 일일 작업 일정 재조정",
                        "📱 동료와 안전체크 주기적 실시"
                    ],
                    "0": [
                        "😊 쾌적한 날씨입니다. 작업에 큰 지장 없습니다.",
                        "📋 오늘 작업 계획을 재확인하세요."
                    ]
                },
                "학생": {
                    "35": [
                        "🏫 실외 체육 활동 전면 금지",
                        "🚌 통학버스 내부 냉방 확인",
                        "💧 체육 수업 전후 물 충분히 섭취",
                        "🩹 응급키트 및 얼음주머니 준비"
                    ],
                    "33": [
                        "🤸‍♂️ 실내 대체 체육 수업 진행",
                        "📚 쉬는 시간마다 휴식장소 제공",
                        "🥤 자기 이름이 적힌 물병 지참",
                        "📣 안내방송으로 주의 환기"
                    ],
                    "30": [
                        "🖥️ 실내 수업 전환 검토",
                        "🧴 자외선 차단제 및 모자 사용",
                        "🎈 수분 섭취 시간표 배포",
                        "📊 교실 내 온습도 측정 실시"
                    ],
                    "27": [
                        "🔆 야외활동 시 그늘 확보",
                        "🎒 모자·양산 지참 필수",
                        "🍉 간식으로 과일 제공",
                        "💬 교사·학생 간 수시 호흡 체크"
                    ],
                    "0": [
                        "😊 쾌적한 날씨입니다. 정상 수업 진행하세요.",
                        "🗓️ 오늘 일정 다시 확인하세요."
                    ]
                },
                "학교/기관": {
                    "35": [
                        "🏫 실외 수업 전면 금지 및 냉방 강화",
                        "📢 긴급 안내방송 실시",
                        "🧊 휴대용 얼음팩 비치",
                        "🚑 응급 대기팀 상시 대기"
                    ],
                    "33": [
                        "🔄 수업 시간 조정 및 단축",
                        "📋 냉방기 필터 점검",
                        "🥤 학생·교직원 수분 보충 권장",
                        "📈 교실 온습도 모니터링 강화"
                    ],
                    "30": [
                        "🛠️ 냉방기 유지보수 실시",
                        "📰 안내문 배포 (수분 섭취 안내)",
                        "🎤 방송을 통한 주기적 주의 환기",
                        "📊 일일 대응 매뉴얼 공유"
                    ],
                    "27": [
                        "🏘️ 그늘막 설치 및 휴식장소 마련",
                        "📚 실내 체험활동 권장",
                        "📋 폭염 매뉴얼 안내",
                        "👍 안전점검 점검표 작성"
                    ],
                    "0": [
                        "😊 평상시와 같이 운영하세요.",
                        "🗓️ 주간 예보 확인 바랍니다."
                    ]
                },
                "영유아 보호자": {
                    "35": [
                        "🚫 외출 자제 및 실내 활동만 허용",
                        "🍼 미지근한 물로 자주 수분 공급",
                        "🧺 차가운 수건으로 신체 닦아주기",
                        "🛏️ 실내 온도 26℃ 이하로 유지"
                    ],
                    "33": [
                        "👒 모자·양산 필수 착용",
                        "🥤 수시 물 또는 이온음료 제공",
                        "📏 30분마다 체온 체크",
                        "📷 외출 기록 이미지 남기기"
                    ],
                    "30": [
                        "🏠 실내 놀이로 대체",
                        "🍉 과일·야채 수분 보충",
                        "🍼 이유식 온도 조절",
                        "🔍 응급 증상 체크 리스트 확인"
                    ],
                    "27": [
                        "🚶‍♀️ 짧은 산책만 가능",
                        "🎒 휴대용 물통 지참",
                        "🎨 실내 놀이 교구 준비",
                        "👩‍👦 부모·보호자 밀착 관찰"
                    ],
                    "0": [
                        "😊 야외 활동 충분히 가능합니다.",
                        "🗓️ 평소 놀이 계획대로 진행하세요."
                    ]
                },
                "배달/택배업": {
                    "35": [
                        "🚫 배송 업무 잠정 중단",
                        "🧊 얼음조끼 및 손목 쿨링밴드 착용",
                        "💧 20분마다 물 섭취",
                        "📞 본사와 실시간 위치·체온 공유"
                    ],
                    "33": [
                        "🕶️ 그늘로 이동하며 배송",
                        "🥤 수시 수분 보충",
                        "🗺️ 배송 경로 재조정",
                        "📲 앱 알림으로 열지 상태 공지"
                    ],
                    "30": [
                        "🚲 전기 오토바이 이용 권장",
                        "🍧 휴게소 얼음음료 제공",
                        "📝 배송 스케줄 조정",
                        "⏱️ 1시간 배송 후 15분 휴식"
                    ],
                    "27": [
                        "👕 통풍 옷 착용",
                        "🧢 모자 필수",
                        "🔄 짧은 휴식 자주 갖기",
                        "📋 동료 간 업무 분담 강화"
                    ],
                    "0": [
                        "😊 정상 업무 가능",
                        "🗓️ 배송 계획 점검하세요."
                    ]
                },
                "소상공인": {
                    "35": [
                        "❄️ 매장 냉방 최대 가동",
                        "🥤 무료 시원 음료 제공",
                        "🏷️ 할인 이벤트 통한 고객 유치",
                        "🚑 직원 응급 교육 실시"
                    ],
                    "33": [
                        "🔄 영업시간 단축 검토",
                        "💦 실내 가습 및 환기",
                        "📣 고객 안내문 게시",
                        "🎁 선착순 얼음물 제공"
                    ],
                    "30": [
                        "🪧 선풍기 및 냉풍기 설치",
                        "🍧 시원 메뉴 개발 제공",
                        "🧑‍💻 온라인 판매 권장",
                        "📋 직원 순환 근무 실시"
                    ],
                    "27": [
                        "🥤 시원 음료판 마련",
                        "🆒 냉방 팁 안내 포스터 부착",
                        "🔆 햇빛 차단 커튼 설치",
                        "👥 직원 쿨링타임 확보"
                    ],
                    "0": [
                        "😊 정상 영업 가능",
                        "🗓️ 주간 매출 목표 재확인"
                    ]
                },
                "농업 종사자": {
                    "35": [
                        "🚜 작업 금지, 오전·야간 시간 활용",
                        "💧 물·이온음료 상시 지참",
                        "🧊 얼음조끼 및 목 타월 사용",
                        "🔄 환상작업 스케줄 도입"
                    ],
                    "33": [
                        "🏘️ 그늘막 설치 및 휴식"
                    ]
                }
            }

def get_level(temp):
                if temp >= 35:
                    return "35"
                elif temp >= 33:
                    return "33"
                elif temp >= 30:
                    return "30"
                elif temp >= 27:
                    return "27"
                else:
                    return "0"

for date, temp in zip(dates, temps):
                day = date.strftime("%m/%d (%a)")
                level = get_level(temp)

                if temp >= 35:
                    temp_level = "☀️ 극심한 폭염"
                    box_fn = st.error
                elif temp >= 33:
                    temp_level = "🥵 폭염주의보"
                    box_fn = st.warning
                elif temp >= 30:
                    temp_level = "🌤 고온 주의"
                    box_fn = st.info
                elif temp >= 27:
                    temp_level = "☁️ 더위 대비"
                    box_fn = st.info
                else:
                    temp_level = "😊 안전"
                    box_fn = st.success

                with st.expander(f"{day} - {temp:.1f}℃ {temp_level}"):
                    if level == "0":
                        st.success("무난한 날씨입니다.")
                        continue

                    if user_role == "전체 보기":
                        for role, guides in all_guides.items():
                            msgs = guides.get(level, [])
                            for msg in msgs:
                                box_fn(f"**{role}**: {msg}")
                    elif user_role in all_guides:
                        msgs = all_guides[user_role].get(level, [])
                        if msgs:
                            for msg in msgs:
                                box_fn(msg)
                        else:
                            st.info("선택한 역할에 대한 가이드가 없습니다.")
                    else:
                        st.info("직업/역할을 선택해 주세요.")