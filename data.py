import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import random


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

        # 利用可能な画像パスのインデックスを取得
        available_indices = list(range(360))

        # ランダムに画像パスのインデックスを選択
        selected_indices = random.sample(available_indices, self.num_views)

        # DRR画像をロード
        # 選択された画像パスに基づいてDRR画像をロード
        for view_idx, image_idx in enumerate(selected_indices):
            drr_path = self.df.iloc[idx][f"drr_path_{image_idx}"]
            if self.data_root not in drr_path:
                proj_path = os.path.join(self.data_root, drr_path)
            else:
                proj_path = drr_path

            # 2D画像のリサイズ
            proj = Image.open(proj_path).resize((self.input_size, self.input_size))
            projs[:, :, view_idx] = np.array(proj)

        if self.transform:
            projs = self.transform(projs)

        # 3Dモデルのロード
        model_path = self.df.iloc[idx]["model_path"].replace("\\", "/")
        if "../PatRecon/" in model_path:
            image_path = model_path
        else:
            image_path = os.path.join(self.data_root, model_path)

        image = np.fromfile(image_path, dtype=np.float32)
        image = np.reshape(image, (-1, self.output_size, self.output_size))

        if image.shape[0] > 128:
            image = image[:128, :, :]

        # 3D画像の正規化
        image = image - np.min(image)
        image = image / np.max(image)
        assert (np.max(image) - 1.0 < 1e-3) and (np.min(image) < 1e-3)

        image = torch.from_numpy(image)

        return (projs, image)

    # def __getitem__(self, idx):
    #     # 入力画像の初期化 (高さ, 幅, チャンネル)
    #     projs = np.zeros(
    #         (self.input_size, self.input_size, self.num_views), dtype=np.uint8
    #     )

    #     # DRR画像をロード
    #     for view_idx in range(self.num_views):
    #         drr_path = self.df.iloc[idx]["drr_path_%d" % view_idx]  # 例: drr_path_0, drr_path_1, ...
    #         proj_path = os.path.join(self.data_root, drr_path)
    #          # 2D画像のリサイズ
    #         proj = Image.open(proj_path).resize((self.input_size, self.input_size))
    #         projs[:, :, view_idx] = np.array(proj)

    #     # 変換関数が指定されている場合は、それを適用
    #     if self.transform:
    #         projs = self.transform(projs)

    #     # 3D画像をロード#
    #     # image_path = self.df.iloc[idx]["3d_model"]
    #     # image_path = os.path.join(self.data_root, image_path[8:])
    #     # image = np.fromfile(image_path, dtype=np.float32)
    #     # image = np.reshape(image, (-1, self.output_size, self.output_size))
    #     model_path = self.df.iloc[idx]["model_path"]  # 3Dモデルのファイルパスを取得
    #     image_path = os.path.join(self.data_root, model_path)
    #     image = np.fromfile(image_path, dtype=np.float32)
    #     image = np.reshape(image, (-1, self.output_size, self.output_size))

    #     # 3D画像のスケーリング正規化
    #     image = image - np.min(image)
    #     image = image / np.max(image)
    #     # 値が正しく正規化されていることを確認
    #     assert (np.max(image) - 1.0 < 1e-3) and (np.min(image) < 1e-3)

    #     image = torch.from_numpy(image)

    #     return (projs, image)  # 2D投影と3D画像をタプルとして返す


# 以下にデータセットの初期化を行うコード
if __name__ == "__main__":
    file_list = "../PatRecon/datasetD.csv"  # CSVファイルへのパス
    data_root = "../PatRecon/DRRoutput"  # 画像が保存されているディレクトリへのパス
    num_views = 360  # CSVファイルには0から360までの角度があるため
    input_size = 128  # 例: 入力画像のサイズを128x128とする
    output_size = 128  # 例: 出力画像のサイズを256x256とする

    # データセットのインスタンスを作成
    dataset = MedReconDataset(file_list, data_root, num_views, input_size, output_size)

    # ここでデータセットをテストするためのコードを追加することもできます
    # 例: データセットの最初の要素を取得して内容を表示
    projs, image = dataset[0]
    print("Projections shape:", projs.shape)
    print("Image shape:", image.shape)
