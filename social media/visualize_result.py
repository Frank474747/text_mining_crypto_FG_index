import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

def plot_weekly_scores(df, date_col, score_cols, title, filename, 
                       ylabel="Score (0-100)", fill_zone=False, 
                       save_dir="visualized_results"):
    """
    繪製每週情緒分數比較圖
    
    df        : DataFrame
    date_col  : 日期欄位名稱（通常是 'date'）
    score_cols: 要畫的數值欄位 list
    title     : 圖片標題
    ylabel    : Y 軸標籤
    """
    plt.figure(figsize=(14,5))
    for col in score_cols:
        plt.plot(df[date_col], df[col], marker='o', label=col)

    plt.axhline(y=50, color='gray', linestyle='--', linewidth=1, label='Neutral')
    # 塗區域
    if fill_zone:
        for col in score_cols:
            scores = df[col]
            dates = df[date_col]
            plt.fill_between(dates, scores, 50,
                            where=(scores >= 50),
                            alpha=0.2, color='green', interpolate=True)
            plt.fill_between(dates, scores, 50,
                            where=(scores <= 50),
                            alpha=0.2, color='red', interpolate=True)
            
    plt.title(title)
    plt.xlabel("Week Start Date")
    plt.ylabel(ylabel)

    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.legend()
    plt.tight_layout()

    # 儲存檔案
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved plot to {save_path}")



# 讀取 weekly CSV
weekly = pd.read_csv("filled_weekly_scores.csv")
weekly["date"] = pd.to_datetime(weekly["date"])

# 平均值
plot_weekly_scores(
    weekly, 
    date_col="date",
    score_cols=["finbert_score_100", "reddit_score_100", "cryptobert_score_100"],
    title="Weekly Average Sentiment Scores",
    filename="model_comparison.png"
)

# finbert 
plot_weekly_scores(
    weekly, 
    date_col="date",
    score_cols=["finbert_score_100"],
    title="Weekly Average Sentiment Scores - FinBERT",
    filename="finbert_score.png",
    fill_zone=True
)

# reddit
plot_weekly_scores(
    weekly, 
    date_col="date",
    score_cols=["reddit_score_100"],
    title="Weekly Average Sentiment Scores - Reddit",
    filename="reddit_score.png",
    fill_zone=True
)   

# cryptobert
plot_weekly_scores( 
    weekly, 
    date_col="date",
    score_cols=["cryptobert_score_100"],
    title="Weekly Average Sentiment Scores - CryptoBERT",
    filename="cryptobert_score.png",
    fill_zone=True
)

# # 最大值
# plot_weekly_scores(
#     weekly, 
#     date_col="date",
#     score_cols=["finbert_max", "reddit_max", "cryptobert_max"],
#     title="Weekly Maximum Sentiment Scores"
# )

# # 最小值
# plot_weekly_scores(
#     weekly, 
#     date_col="date",
#     score_cols=["finbert_min", "reddit_min", "cryptobert_min"],
#     title="Weekly Minimum Sentiment Scores"
# )
