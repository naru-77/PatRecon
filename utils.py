import numpy as np
import os.path as osp


def save_record(output_path, epoch, train_loss, val_loss):
    """
    トレーニングと検証の損失の記録をファイルに保存します。

    Parameters:
    - output_path: 保存するパス
    - epoch: 現在のエポック数
    - train_loss: トレーニングの損失
    - val_loss: 検証の損失
    """
    filename = osp.join(output_path, "loss_record.npz")
    np.savez(filename, epoch=epoch, train_loss=train_loss, val_loss=val_loss)


class AverageMeter(object):
    """計算と現在の値と平均値の保存を行います"""

    def __init__(self):
        self.reset()

    def reset(self):
        """変数を初期化します"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        新しい値を使用して統計を更新します。

        Parameters:
        - val: 新しい値
        - n: 値の出現回数（デフォルトは1）
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(lr, lr_ratio, lr_decay, optimizer, epoch):
    """
    学習率を調整します。初期の学習率は、特定のエポック数ごとにlr_ratioで減少します。

    Parameters:
    - lr: 現在の学習率
    - lr_ratio: 学習率を減少させる比率
    - lr_decay: 学習率を減少させるエポック間隔
    - optimizer: 対象のオプティマイザ
    - epoch: 現在のエポック数

    Returns:
    - lr: 調整後の学習率
    """
    lr = lr * ((1.0 / lr_ratio) ** (epoch // lr_decay))

    # オプティマイザ内のすべてのパラメーターグループの学習率を設定
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr
