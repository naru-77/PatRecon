import csv
import os
import random

# 3Dモデルのファイルパス
model_path = "../PatRecon/output_volume_D.bin"

# インデックスをシャッフル
indices = list(range(360))
random.shuffle(indices)

# インデックスをトレーニングセットと検証セットに分割（4:1の比率）
split_point = int(len(indices) * 0.8)
train_indices = indices[:split_point]
val_indices = indices[split_point:]


# トレーニングセットと検証セットのデータを作成
def create_dataset(indices, filename):
    data = {"model_path": model_path}
    for idx, view_idx in enumerate(indices):
        drr_path = os.path.join(
            "../PatRecon/DRRoutput/output_volume_D", f"projection_{view_idx}.png"
        )
        data[f"drr_path_{idx}"] = drr_path

    fieldnames = ["model_path"] + [f"drr_path_{i}" for i in range(len(indices))]
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(data)


# トレーニングセットと検証セットをCSVファイルとして保存
create_dataset(train_indices, "../PatRecon/train_dataset.csv")
create_dataset(val_indices, "../PatRecon/val_dataset.csv")
