import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_DIR = "data/joint_visibility"  # CSV 파일 경로
TARGET_PATTERN = "joint_visibility"  # 파일명 패턴
OUTPUT_IMG = "joint_landmark_results.png"  # 저장할 이미지 파일명

# 파일 로딩
visibility_dfs = []
for file in os.listdir(CSV_DIR):
    if file.endswith(".csv") and TARGET_PATTERN in file:
        df = pd.read_csv(os.path.join(CSV_DIR, file))
        visibility_dfs.append(df)

# 데이터 병합
combined_df = pd.concat(visibility_dfs)

# 33개 관절 전체 평균 가시성 계산
summary_df = combined_df.groupby(["Landmark_Index", "Landmark_Name"], as_index=False).agg({
    "Mean_Visibility": "mean"
}).sort_values(by="Mean_Visibility", ascending=False)

print("\n전체 영상 기준 인식이 잘된 관절 순위:\n")
print(summary_df.to_string(index=False))

# 그래프 출력
plt.figure(figsize=(14, max(10, len(summary_df) * 0.3)))
plt.bar(summary_df["Landmark_Name"], summary_df["Mean_Visibility"])
plt.title("Total Joint Mean Visibility")
plt.xlabel("Landmark")
plt.ylabel("Mean Visibility")
plt.xticks(rotation=45, fontsize=8)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)

# 그래프 이미지 저장
plt.savefig(OUTPUT_IMG, dpi=300)
print(f"\n그래프 이미지 저장 완료: \"{OUTPUT_IMG}\"")

plt.show()