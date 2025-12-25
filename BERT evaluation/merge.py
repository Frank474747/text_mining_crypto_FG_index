import pandas as pd

# ----------------------------
# 1. 讀取資料
# ----------------------------
labels_df = pd.read_csv("labels.csv")
pred_df = pd.read_csv("predictions.csv")

# ----------------------------
# 2. 只保留有人工標記的文章（75 篇）
# ----------------------------
labels_df = labels_df[labels_df["labels"].notna()].copy()

# ----------------------------
# 3. 將 n/g/b 映射為情緒標籤
# ----------------------------
label_map = {
    "n": "negative",
    "g": "neutral",
    "b": "positive"
}

labels_df["gold_label"] = labels_df["labels"].map(label_map)

# ----------------------------
# 4. 從 predictions.csv 取出三個 BERT 的 label
# ----------------------------
pred_subset = pred_df[
    [
        "id",
        "finbert_label",
        "reddit_label",
        "cryptobert_label"
    ]
].copy()

# ----------------------------
# 5. 用 id 合併（inner join，確保只剩 75 篇）
# ----------------------------
eval_df = labels_df.merge(
    pred_subset,
    on="id",
    how="inner"
)

# ----------------------------
# 6. 只保留評估需要的欄位
# ----------------------------
eval_df = eval_df[
    [
        "id",
        "gold_label",
        "finbert_label",
        "reddit_label",
        "cryptobert_label"
    ]
]

print(f"Evaluation samples: {len(eval_df)}")
print(eval_df.head())

# ----------------------------
# 評價BERT成效
# ----------------------------
print("----------------------------")

from sklearn.metrics import classification_report

print("FinBERT")
print(
    classification_report(
        eval_df["gold_label"],
        eval_df["finbert_label"],
        digits=3
    )
)

print("Reddit BERT")
print(
    classification_report(
        eval_df["gold_label"],
        eval_df["reddit_label"],
        digits=3,
        zero_division=0 
    )
)

print("CryptoBERT")
print(
    classification_report(
        eval_df["gold_label"],
        eval_df["cryptobert_label"],
        digits=3
    )
)

from sklearn.metrics import f1_score

for model in ["finbert_label", "reddit_label", "cryptobert_label"]:
    macro_f1 = f1_score(
        eval_df["gold_label"],
        eval_df[model],
        average="macro"
    )
    print(f"{model}: Macro-F1 = {macro_f1:.3f}")
