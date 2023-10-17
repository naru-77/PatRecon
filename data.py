import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class MedReconDataset(Dataset):
    """3D Reconstruction Dataset. 3D 再構成データセット"""

    def __init__(
        self, file_list, data_root, num_views, input_size, output_size, transform=None
    ):
        """
        初期化関数
        Args:
            file_list (string): アノテーションが記載されたcsvファイルへのパス。
            data_root (string): すべての画像が保存されているディレクトリ。
            transform (callable, optional): サンプルに適用されるオプションの変換。
        """
        self.df = pd.read_csv(file_list)  # csvファイルを読み込む
        self.data_root = data_root  # 画像ディレクトリのパス
        self.transform = transform  # 画像変換関数

        self.num_views = num_views  # ビューの数
        self.input_size = input_size  # 入力画像のサイズ
        self.output_size = output_size  # 出力画像のサイズ

    def __len__(self):
        return len(self.df)  # データセットの長さを返す

    def __getitem__(self, idx):
        # 入力画像の初期化 (高さ, 幅, チャンネル)
        projs = np.zeros(
            (self.input_size, self.input_size, self.num_views), dtype=np.uint8
        )

        # 2D投影をロード
        for view_idx in range(self.num_views):
            proj_path = self.df.iloc[idx]["view_%d" % (view_idx)]
            proj_path = os.path.join(self.data_root, proj_path[8:])

            # 2D画像のリサイズ
            proj = Image.open(proj_path).resize((self.input_size, self.input_size))
            projs[:, :, view_idx] = np.array(proj)

        # 変換関数が指定されている場合は、それを適用
        if self.transform:
            projs = self.transform(projs)

        # 3D画像をロード
        image_path = self.df.iloc[idx]["3d_model"]
        image_path = os.path.join(self.data_root, image_path[8:])
        image = np.fromfile(image_path, dtype=np.float32)
        image = np.reshape(image, (-1, self.output_size, self.output_size))

        # 3D画像のスケーリング正規化
        image = image - np.min(image)
        image = image / np.max(image)
        # 値が正しく正規化されていることを確認
        assert (np.max(image) - 1.0 < 1e-3) and (np.min(image) < 1e-3)

        image = torch.from_numpy(image)

        return (projs, image)  # 2D投影と3D画像をタプルとして返す
