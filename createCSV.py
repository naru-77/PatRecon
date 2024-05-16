import csv
import os

# CSVファイルのパス
csv_file = "../PatRecon/datasetD.csv"

# 3Dモデルのファイルパス
model_path = "../PatRecon/output_volume_D.bin"

# CSVファイルに書き込むデータの準備
data = {"model_path": model_path}
for view_idx in range(360):  # 各画像のパスを追加
    drr_path = os.path.join(
        "../PatRecon/DRRoutput/output_volume_D", f"projection_{view_idx}.png"
    )
    data[f"drr_path_{view_idx}"] = drr_path

# CSVファイルにデータを書き込む
fieldnames = ["model_path"] + [f"drr_path_{i}" for i in range(360)]
with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow(data)
