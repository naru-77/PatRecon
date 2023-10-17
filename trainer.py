import shutil
import os.path as osp
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from net import ReconNet
from utils import AverageMeter

class Trainer_ReconNet(nn.Module):
    def __init__(self, args):
        super(Trainer_ReconNet, self).__init__()

        # 引数からパラメータを設定
        self.exp_name = args.exp  # 実験の名前
        self.arch = args.arch  # モデルのアーキテクチャ
        self.print_freq = args.print_freq  # トレーニング情報をどの頻度で表示するか
        self.output_path = args.output_path  # モデルの重みを保存するパス
        self.resume = args.resume  # チェックポイントからトレーニングを再開するかどうか
        self.best_loss = 1e5  # 最良の損失を非常に高い値として初期化


        # モデルのインスタンスを作成
        print("=> モデルを作成中...")
        if self.arch == 'ReconNet':  
            # ReconNetモデルを初期化
            self.model = ReconNet(in_channels=args.num_views, out_channels=args.output_channel, gain=args.init_gain, init_type=args.init_type)
            self.model = nn.DataParallel(self.model).cuda()
        else:
            # モデルのアーキテクチャが認識されない場合
            assert False, print('実装されていないモデル: {}'.format(self.arch))

        # 引数に基づいて損失関数を定義
        if args.loss == 'l1':
            # L1損失（絶対差）
            self.criterion = nn.L1Loss(size_average=True, reduce=True).cuda() 
        elif args.loss == 'l2':
            # L2損失（平均二乗誤差）
            self.criterion = nn.MSELoss(size_average=True, reduce=True).cuda()
        else:
            # 損失タイプが認識されない場合
            assert False, print('実装されていない損失: {}'.format(args.loss))

        # 引数に基づいてオプティマイザを定義
        if args.optim == 'adam':
            # Adamオプティマイザを使用
            self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                            lr=args.lr,
                                            betas=(0.5, 0.999),
                                            weight_decay=args.weight_decay)
        else:
            # オプティマイザのタイプが認識されない場合
            assert False, print('実装されていないオプティマイザ: {}'.format(args.optim))



    def train_epoch(self, train_loader, epoch):
        
        #モデルを1エポック訓練します。
        
        train_loss = AverageMeter()

       # モデルを訓練モードに変更
        self.model.train()

        # トレーニングバッチをイテレーション
        for i, (input, target) in enumerate(train_loader):
            # 入力とターゲットをGPUと互換性のある変数に変換
            input_var, target_var = Variable(input).cuda(), Variable(target).cuda()

            # モデルの出力を計算
            output = self.model(input_var)

            # 損失を計算
            loss = self.criterion(output, target_var)
            train_loss.update(loss.data.item(), input.size(0))

            # 逆伝播とオプティマイザのステップを実行
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 定期的にトレーニング情報を表示
            if i % self.print_freq == 0:
                print('エポック: [{0}] \t'
                      'イテレーション: [{1}/{2}]\t'
                      'トレーニング損失: {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                       epoch, i, len(train_loader), 
                       loss=train_loss))

         # 現在のエポックを完了
        print('エポック完了: [{0}]\t'
              '平均トレーニング損失: {loss.avg:.5f}\t'.format(
               epoch, loss=train_loss))

        return train_loss.avg


       def validate(self, val_loader):
        """
        モデルを検証モードで実行し、検証セットの平均損失を計算します。
        """
        val_loss = AverageMeter()  # 損失の記録のためのユーティリティ
        batch_time = AverageMeter()  # バッチ処理時間の記録のためのユーティリティ

        # モデルを評価モードに変更
        self.model.eval()

        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            # 入力とターゲットをGPUと互換性のある変数に変換
            input_var, target_var = Variable(input).cuda(), Variable(target).cuda()

            # モデルの出力を計算
            output = self.model(input_var)

            # 損失を計算
            loss = self.criterion(output, target_var)
            val_loss.update(loss.data.item(), input.size(0))

            # 経過時間を計測
            batch_time.update(time.time() - end)
            end = time.time()

            # 定期的に検証情報を表示
            print('検証: [{0}/{1}]\t'
                  '時間 {batch_time.val: .3f} ({batch_time.avg:.3f})\t'
                  '損失 {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                   i, len(val_loader), 
                   batch_time=batch_time, 
                   loss=val_loss))

        return val_loss.avg

    def save(self, curr_val_loss, epoch):
        """
        現在のモデルとその最良の状態を保存します。
        """
        # 現在の検証損失が最良のものか確認し、必要に応じて更新
        is_best = curr_val_loss < self.best_loss
        self.best_loss = min(curr_val_loss, self.best_loss)

        # チェックポイントの状態を定義
        state = {
            'epoch': epoch + 1,
            'arch': self.arch,
            'state_dict': self.model.state_dict(),
            'best_loss': self.best_loss,
            'optimizer': self.optimizer.state_dict(),
        }

        # チェックポイントのファイル名を定義
        filename = osp.join(self.output_path, 'curr_model.pth.tar')
        best_filename = osp.join(self.output_path, 'best_model.pth.tar')

        print('! チェックポイントを保存中: {}'.format(filename))
        torch.save(state, filename)

        if is_best:
            print('!! 最良のチェックポイントを保存中: {}'.format(best_filename))
            shutil.copyfile(filename, best_filename)

    def load(self):
        """
        保存されたモデルのチェックポイントを読み込みます。
        """
        if self.resume == 'best':
            ckpt_file = osp.join(self.output_path, 'best_model.pth.tar')
        elif self.resume == 'final':
            ckpt_file = osp.join(self.output_path, 'curr_model.pth.tar')
        else:
            assert False, print("利用可能なチェックポイントが見つかりません '{}'".format(ckpt_file))

        if osp.isfile(ckpt_file):
            print("=> チェックポイントを読み込み中 '{}'".format(ckpt_file))
            checkpoint = torch.load(ckpt_file)
            start_epoch = checkpoint['epoch']

            # モデルとオプティマイザの状態を読み込み
            self.best_loss = checkpoint['best_loss']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> チェックポイントを読み込みました '{}' (エポック {})".format(ckpt_file, checkpoint['epoch']))
        else:
            print("=> チェックポイントが見つかりませんでした '{}'".format(ckpt_file))

        return start_epoch


