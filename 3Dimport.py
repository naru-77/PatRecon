import numpy as np
import matplotlib.pyplot as plt


def load_3d_volume(file_path, shape):
    # ファイルからデータを読み込む
    volume = np.fromfile(file_path, dtype=np.float32)
    # データを指定された形状にリシェイプする
    volume = volume.reshape(shape)
    return volume


def plot_slice(volume, slice_index, axis=0):
    if axis == 0:  # 軸横軸 (上下方向)
        slice = volume[slice_index, :, :]
    elif axis == 1:  # 冠状軸 (前後方向)
        slice = volume[:, slice_index, :]
    elif axis == 2:  # 矢状軸 (左右方向)
        slice = volume[:, :, slice_index]

    plt.imshow(slice, cmap="gray")
    plt.title(f"Slice {slice_index} along axis {axis}")
    plt.show()


file_path = "../PatRecon/exp/model/data/3D_CT.bin"  # ファイルパスを指定
volume_shape = (46, 128, 128)  # 実際のボリュームの形状を指定

# ボリュームデータを読み込む
volume = load_3d_volume(file_path, volume_shape)

# 視覚化するスライスのインデックスを指定
slice_index = 23  # 例として50番目のスライスを使用

# スライスを視覚化
plot_slice(volume, slice_index, axis=0)  # 軸横軸
plot_slice(volume, slice_index, axis=1)  # 冠状軸
plot_slice(volume, slice_index, axis=2)  # 矢状軸
