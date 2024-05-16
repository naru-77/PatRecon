import os
from data_loader import get_train_val_data_loaders
from net import ReconNet
from trainer import Trainer_ReconNet
import pandas as pd
from sklearn.model_selection import train_test_split
from data_loader import get_data_loader


class Args:
    def __init__(self):
        self.exp = "experiment_name"
        self.arch = "ReconNet"
        self.print_freq = 10
        self.output_path = "../PatRecon/Result"
        self.resume = None
        self.loss = "l2"
        self.optim = "adam"
        self.lr = 0.00002
        self.weight_decay = 0.0001
        self.num_views = 360
        self.input_size = 128
        self.output_size = 128
        self.batch_size = 1
        self.num_workers = 0
        self.data_root = "../PatRecon/DRRoutput"
        self.output_channel = 1
        self.init_gain = 0.02  # 重み初期化に使用するgain
        self.init_type = "normal"  # 重み初期化のタイプ
        self.pin_memory = True


def main():
    args = Args()  # Args オブジェクトの作成
    # トレーナークラスのインスタンス化
    trainer = Trainer_ReconNet(args)

    # トレーニングと検証用のデータローダーを取得
    train_loader, val_loader = get_train_val_data_loaders(
        train_file="../PatRecon/datasetD.csv",
        val_file="../PatRecon/datasetD.csv",
        args=args,
    )

    # トレーニングループ
    num_epochs = 100
    for epoch in range(num_epochs):
        # トレーニング
        train_loss = trainer.train_epoch(train_loader, epoch)

        # 検証
        val_loss = trainer.validate(val_loader)

        # モデルの保存
        trainer.save(val_loss, epoch)


if __name__ == "__main__":
    main()
