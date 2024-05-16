import os
import numpy as np
from PIL import Image
from skimage.transform import radon

def generate_2d_projections(volume, angles, volume_filename, output_dir):
    # 3Dボリュームファイル名から拡張子を除いた部分を使用
    model_name = os.path.splitext(volume_filename)[0]
    
    # モデル名に基づいた出力フォルダを作成
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    for angle in angles:
        projection = np.zeros((volume.shape[1], len(volume)), dtype=np.float32)
        for i, slice in enumerate(volume):
            slice_projection = radon(slice, theta=[angle], circle=True)
            projection[:, i] = slice_projection[:, 0]

        # 2D投影画像をスケーリングして保存
        proj_scaled = (projection - projection.min()) / (projection.max() - projection.min()) * 255.0
        proj_scaled = proj_scaled.astype(np.uint8)
        img = Image.fromarray(proj_scaled)
        img.save(os.path.join(model_output_dir, f"projection_{angle}.png"))

# 例: 3Dボリュームデータの読み込み
volume = np.fromfile('output_volume_D.bin', dtype=np.float32)
original_shape = (221, 512, 512)
volume = np.reshape(volume, original_shape)

# 0度から359度までの角度でDRRを生成し、指定されたフォルダに保存
angles = list(range(360))
volume_filename = "output_volume_D.bin"  # 3Dボリュームファイル名
output_dir = "../PatRecon/DRRoutput"  # DRR画像の保存先ディレクトリ
generate_2d_projections(volume, angles, volume_filename, output_dir)
