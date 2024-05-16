import pandas as pd
from PIL import Image
import numpy as np
from torchvision import transforms
from tqdm import tqdm

# CSVファイルの読み込み
df = pd.read_csv('../PatRecon/datasetD.csv')

# 画像データのロードと統計の計算
mean = np.zeros(3)
std = np.zeros(3)
n_samples = 0
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 画像サイズのリサイズ
    transforms.ToTensor()           # PIL画像をPyTorchテンソルに変換
])

for _, row in tqdm(df.iterrows(), total=len(df)):
    image_path = row['drr_path']
    image = Image.open(image_path).convert('RGB')  # RGB形式に変換
    image = transform(image)
    mean += image.mean((1, 2)).numpy()
    std += image.std((1, 2)).numpy()
    n_samples += 1

mean /= n_samples
std /= n_samples

print(f"Mean: {mean}")
print(f"Std: {std}")
