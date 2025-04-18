import os
import pandas as pd
import matplotlib.pyplot as plt

# 설정
CSV_DIR = "data/joint_visibility"  # csv 파일들이 있는 폴더
TARGET_PATTERN = "joint_visibility"  # 파일명 패턴
TOP_N = 10  # 그래프에 보여줄 상위 관절 수
OUTPUT_IMG = "joint_visibility_rank.png"

# csv 파일 불러오기
visibility_dfs = []
for file in os.listdir(CSV_DIR):
    if file.endswith(".csv") and TARGET_PATTERN in file:
        df = pd.read_csv(os.path.join(CSV_DIR, file))
        visibility_dfs.append(df)

# 데이터 병합 및 평균 계산
combined_df = pd.concat(visibility_dfs)
summary_df = combined_df.groupby(["Landmark_Index", "Landmark_Name"], as_index=False).agg({
    "Mean_Visibility": "mean"
}).sort_values(by="Mean_Visibility", ascending=False)

print("\n전체 영상 기준 인식이 잘된 관절 순위:\n")
print(summary_df.to_string(index=False))

# 그래프 그리기
plt.figure(figsize=(10, 6))
top_df = summary_df.head(TOP_N)
plt.bar(top_df["Landmark_Name"], top_df["Mean_Visibility"])
plt.title(f"Top {TOP_N} Most Visible Landmarks Across Videos")
plt.xlabel("Landmark")
plt.ylabel("Mean Visibility")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)

# 그래프 이미지 저장
plt.savefig(OUTPUT_IMG, dpi=300)  # 고해상도로 저장
print(f"\n 그래프 이미지 저장 완료: \"{OUTPUT_IMG}\"")

plt.show()
