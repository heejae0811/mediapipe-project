import pandas as pd
import glob
from scipy.stats import shapiro, pearsonr, spearmanr
import networkx as nx
import os

os.makedirs('./result', exist_ok=True)

def analyze_and_filter_with_grouping(df):
    metrics = [col for col in df.columns if col not in ['id', 'label']]
    results = []

    # 정규성에 따른 상관계수 계산
    for i in range(len(metrics)):
        for j in range(i+1, len(metrics)):
            var1, var2 = metrics[i], metrics[j]
            data = df[[var1, var2]].dropna()
            if len(data) < 3:
                continue

            x, y = data[var1], data[var2]
            p_norm_x = shapiro(x).pvalue
            p_norm_y = shapiro(y).pvalue

            if p_norm_x > 0.05 and p_norm_y > 0.05:
                corr, pval = pearsonr(x, y)
                method = 'Pearson'
            else:
                corr, pval = spearmanr(x, y)
                method = 'Spearman'

            results.append({
                '변수1': var1,
                '변수2': var2,
                '상관계수': round(corr, 4),
                '사용된방법': method,
                'var1_정규성_p': round(p_norm_x, 4),
                'var2_정규성_p': round(p_norm_y, 4)
            })

    res_df = pd.DataFrame(results)

    # 상관계수가 0.9 이상인 변수쌍 추출
    high_corr_df = res_df[res_df['상관계수'].abs() >= 0.9].copy()

    # 네트워크로 그룹화 후 제거 변수 결정
    G = nx.Graph()
    G.add_edges_from(zip(high_corr_df['변수1'], high_corr_df['변수2']))

    to_keep = []
    to_drop = []

    for group in nx.connected_components(G):
        group = list(group)
        to_keep.append(group[0])    # 그룹에서 하나만 남기고
        to_drop.extend(group[1:])   # 나머지는 제거

    # 그룹에 속하지 않는 변수 + 그룹에서 남긴 변수
    all_grouped = set(high_corr_df['변수1']).union(high_corr_df['변수2'])
    remaining_vars = [v for v in metrics if (v in to_keep) or (v not in all_grouped)]

    return res_df, high_corr_df, to_keep, to_drop, remaining_vars


# 데이터 불러오기
dfs = []
for file in glob.glob('./features_xlsx/*.xlsx'):
    df_tmp = pd.read_excel(file, sheet_name=0)
    dfs.append(df_tmp)

if not dfs:
    raise FileNotFoundError("❌ './features_xlsx/' 폴더에 .xlsx 파일이 없습니다.")

df = pd.concat(dfs, ignore_index=True)

# 분석
res_df, high_corr_df, to_keep, to_drop, remaining_vars = analyze_and_filter_with_grouping(df)

# 결과 출력
print("\n✅ 정규성에 따라 계산된 상관계수 결과 (앞부분):")
print(res_df.head())

print("\n✅ 상관계수 >= 0.9인 변수쌍:")
print(high_corr_df)

print("\n✅ 각 그룹에서 남긴 변수:")
print(to_keep)

print("\n✅ 제거한 변수:")
print(to_drop)

print("\n✅ 최종 남은 변수:")
print(remaining_vars)

# 결과 엑셀로 저장
with pd.ExcelWriter('./result/features_correlation.xlsx') as writer:
    res_df.to_excel(writer, sheet_name='모든_쌍_결과', index=False)
    high_corr_df.to_excel(writer, sheet_name='0.9이상_쌍', index=False)
    pd.DataFrame({'그룹별_남긴변수': to_keep}).to_excel(writer, sheet_name='그룹별_남긴변수', index=False)
    pd.DataFrame({'제거한변수': to_drop}).to_excel(writer, sheet_name='제거한변수', index=False)
    pd.DataFrame({'최종남은변수': remaining_vars}).to_excel(writer, sheet_name='최종남은변수', index=False)

print("\n 결과가 './result/features_correlation.xlsx' 에 저장되었습니다.")