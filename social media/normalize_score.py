import pandas as pd

# 讀入原始資料
df = pd.read_csv("ETH_sentiment_comparison.csv")


# 線性標準化：(-1 ~ +1) → (0 ~ 100)
df["finbert_score_100"] = (df["finbert_sentiment_score"] + 1) / 2 * 100
df["reddit_score_100"] = (df["reddit_sentiment_score"] + 1) / 2 * 100
df["cryptobert_score_100"] = (df["cryptobert_sentiment_score"] + 1) / 2 * 100

# 如果需要四捨五入
df[["finbert_score_100", "reddit_score_100", "cryptobert_score_100"]] = \
    df[["finbert_score_100", "reddit_score_100", "cryptobert_score_100"]].round(2)

# 儲存為新的檔案
df.to_csv("sentiment_with_scores_100.csv", index=False)

print("Done. 新增三欄位並已儲存為 sentiment_with_scores_100.csv")