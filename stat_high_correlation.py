import pandas as pd

# 결과 파일 불러오기
df = pd.read_excel("./correlation_results.xlsx")

# 상관계수 ≥ 0.9 (절댓값) & 서로 다른 변수 쌍만 남기기
high_corrs = df[
    (df['상관계수'].abs() >= 0.9) &
    (df['변수1'] != df['변수2'])
]

# 엑셀로 저장
high_corrs.to_excel("high_correlation_results.xlsx", index=False)

print("✅ 상관계수 |0.9| 이상인 결과가 'high_correlation_results.xlsx'에 저장되었습니다.")
