import pandas as pd

def load_weather_csv(path):
    # CSV 읽기
    df = pd.read_csv(path, encoding="cp949", skiprows=10)
    # 컬럼명 정리 (탭/공백 제거)
    df.columns = df.columns.str.strip()
    # 일시 컬럼 자동 탐색 후 변환
    date_col = [c for c in df.columns if "일시" in c][0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.rename(columns={date_col: "일시"})
    return df

# 서울 데이터 로드
df_temp  = load_weather_csv("data/서울/서울 2015~2025 기온.csv")
df_humid = load_weather_csv("data/서울/서울 2015~2025 습도.csv")
df_rain  = load_weather_csv("data/서울/서울 2015~2025 강수량.csv")

# 확인
print("기온 데이터:", df_temp.head(), "\n")
print("습도 데이터:", df_humid.head(), "\n")
print("강수량 데이터:", df_rain.head(), "\n")

# 최종 병합
df_all = df_temp.merge(df_humid, on="일시").merge(df_rain, on="일시")
print("병합 데이터:", df_all.head())
