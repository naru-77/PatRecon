import argparse
import os
import sys
import shutil
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from net import ReconNet
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from skimage.measure import compare_mse, compare_nrmse, compare_psnr, compare_ssim
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
plt.switch_backend('agg')


# コマンドライン引数を解析するためのArgumentParserオブジェクトを作成
parser = argparse.ArgumentParser(description='PyTorch 3D Reconstruction Training')
# 各種のコマンドライン引数を追加
# これにより、ユーザーはコマンドラインからスクリプトの動作をカスタマイズできます
# 例: 実験のインデックス、乱数のシード、入力のビュー数など
parser.add_argument('--exp', type=int, default=1,
                    help='experiments index')
parser.add_argument('--seed', type=int, default=1, 
                    metavar='N', help='manual seed for GPUs to generate random numbers')
parser.add_argument('--num-views', type=int, default=1,
                    help='number of views/projections in inputs')
parser.add_argument('--input-size', type=int, default=128,
                    help='dimension of input view size')
parser.add_argument('--output-size', type=int, default=128,
                    help='dimension of ouput 3D model size')
parser.add_argument('--output-channel', type=int, default=0,
                    help='dimension of ouput 3D model size')
parser.add_argument('--start-slice', type=int, default=0,
                    help='the idx of start slice in 3D model')
parser.add_argument('--test', type=int, default=1,
                    help='number of total testing samples')
parser.add_argument('--vis_plane', type=int, default=0,
                    help='visualization plane of 3D images: [0,1,2]')

# メイン関数の定義
# この関数は、スクリプトが実行されたときに最初に呼び出される関数です
def main():
    global args
    global exp_path
    # コマンドライン引数を解析して、'args'オブジェクトに格納
    args = parser.parse_args()
    # exp_path = './exp{}'.format(args.exp)
    # 実験データを保存するためのパスを設定
    exp_path = './exp'

    # set random seed for GPUs for reproducible
    # 再現性を確保するための乱数のシードを設定
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # 現在の実験のインデックスを表示
    print('Testing Experiement {} ...................'.format(args.exp))
    # 実験のインデックスに基づいて、出力チャンネルの数を設定
    if args.exp == 1:
        args.output_channel = 46
    else:
        assert False, print('Not legal experiment index!')

    # define model
    # モデルの定義
    # ReconNetは、入力として2D投影を受け取り、3Dモデルを出力するネットワークです
    model = ReconNet(in_channels=args.num_views, out_channels=args.output_channel)
    # モデルをGPU上で実行するための設定
    model = torch.nn.DataParallel(model).cuda()

    # define loss function
    # 損失関数の定義
    # この場合、平均二乗誤差(MSE)を使用しています
    criterion = nn.MSELoss(size_average=True, reduce=True).cuda()


    # enable CUDNN benchmark
    # CUDNNのベンチマークモードを有効にする
    # これにより、CUDNNは最適なアルゴリズムを動的に選択して、計算を高速化します
    cudnn.benchmark = True

    # customized dataset
    # カスタムデータセットの定義
    # このデータセットは、2D投影画像と3Dモデルのペアを返します
    class MedReconDataset(Dataset):
        """ 3D Reconstruction Dataset."""
        def __init__(self, csv_file=None, data_dir=None, transform=None):
            self.transform = transform

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            # 2D投影画像を格納するための配列を初期化
            images = np.zeros((args.input_size, args.input_size, args.num_views), dtype=np.uint8)      ### input image size (H, W, C)
            ### load image
            # 各投影画像を読み込み、配列に格納
            for view_idx in range(args.num_views):
                image_path = os.path.join(exp_path, 'data/2D_projection_{}.jpg'.format(view_idx+1))
                ### resize 2D images
                img = Image.open(image_path).resize((args.input_size, args.input_size))
                images[:, :, view_idx] = np.array(img)
            if self.transform:
                images = self.transform(images)

            ### load target
            # 3Dモデルを読み込み
            volume_path = os.path.join(exp_path, 'data/3D_CT.bin')
            volume = np.fromfile(volume_path, dtype=np.float32)
            volume = np.reshape(volume, (-1, args.output_size, args.output_size))

            ### scaling normalize
            # 3Dモデルの値を[0,1]の範囲に正規化
            volume = volume - np.min(volume)
            volume = volume / np.max(volume)
            volume = torch.from_numpy(volume)

            return (images, volume)

    # データの前処理を定義
    # ここでは、画像をテンソルに変換し、平均と標準偏差で正規化しています
    normalize = transforms.Normalize(mean=[0.516], std=[0.264])
    test_dataset = MedReconDataset(
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        normalize,
                        ]))
    # データローダーを作成
    # これにより、データセットからバッチ単位でデータを効率的に読み込むことができます
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1, 
        shuffle=False,
        num_workers=4, 
        pin_memory=True)

    # load model 
    # 事前にトレーニングされたモデルの重みを読み込む
    ckpt_file = os.path.join(exp_path, 'model/model.pth.tar')
    if os.path.isfile(ckpt_file):
        print("=> loading checkpoint '{}' ".format(ckpt_file))
        checkpoint = torch.load(ckpt_file)
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' ".format(ckpt_file))
    else:
        print("=> no checkpoint found at '{}'".format(ckpt_file))

    # test evaluation 
    # テストデータでのモデルの評価を実行
    loss, pred_data = test(test_loader, model, criterion, mode='Test')

     # 予測された3Dモデルと実際の3Dモデルを比較し、結果を保存
    for idx in range(args.test):
        save_path = os.path.join(exp_path, 'result/sample_{}'.format(idx+1))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        print('Evaluate testing sample {} *********************************'.format(idx+1))
        print("Save prediction to: {}".format(save_path))
        pred = getPred(pred_data, idx)
        groundtruth = getGroundtruth(idx, normalize=True)
        getErrorMetrics(im_pred=pred, im_gt=groundtruth)
        imageSave(pred, groundtruth, args.vis_plane, save_path)

    return 


# 以下の関数は、テストデータでのモデルの評価、3Dモデルの正規化、予測と実際の3Dモデルの取得、
# 予測と実際の3Dモデルの誤差メトリクスの計算、結果の保存など、さまざまな補助的なタスクを実行するためのもの。


def test(val_loader, model, criterion, mode):
    model.eval()
    losses = AverageMeter()
    pred = np.zeros((args.test, args.output_channel, args.output_size, args.output_size), dtype=np.float32)
    for i, (input, target) in enumerate(val_loader):
        input_var, target_var = Variable(input), Variable(target)
        input_var, target_var = input_var.cuda(), target_var.cuda()

        output = model(input_var)
        loss = criterion(output, target_var)
        losses.update(loss.data.item(), input.size(0))
        pred[i, :, :, :] = output.data.float()

        print('{0}: [{1}/{2}]\t'
          'Val Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
           mode, i, len(val_loader), loss=losses))
    print('Average {} Loss: {y:.5f}\t'.format(mode, y=losses.avg))

    save_path = os.path.join(exp_path, 'result')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = os.path.join(save_path, 'test_prediction.npz')
    print("=> saving test prediction results: '{}'".format(file_name))
    np.savez(file_name, pred=pred)
    
    return losses.avg, pred

def dataNormalize(data):
    # scaling to [0,1]
    data = data - np.min(data)
    data = data / np.max(data)
    assert((np.max(data) - 1.0 < 1e-3) and (np.min(data) < 1e-3))
    return data

def getPred(data, idx, test_idx=None):
    pred = data[idx, ...]
    return pred

def getGroundtruth(idx, normalize=True):
    model_path = os.path.join(exp_path, 'data/3D_CT.bin')
    ct3d = np.fromfile(model_path, dtype=np.float32)
    ctslices = np.reshape(ct3d, (-1, args.output_size, args.output_size))
    if normalize:
        ctslices = dataNormalize(ctslices)
    return ctslices

def getErrorMetrics(im_pred, im_gt, mask=None):
    im_pred = np.array(im_pred).astype(np.float)
    im_gt = np.array(im_gt).astype(np.float)
    # sanity check
    assert(im_pred.flatten().shape==im_gt.flatten().shape)
    # RMSE
    rmse_pred = compare_nrmse(im_true=im_gt, im_test=im_pred)
    # PSNR
    psnr_pred = compare_psnr(im_true=im_gt, im_test=im_pred)
    # SSIM
    ssim_pred = compare_ssim(X=im_gt, Y=im_pred)
    # MSE
    mse_pred = mean_squared_error(y_true=im_gt.flatten(), y_pred=im_pred.flatten())
    # MAE
    mae_pred = mean_absolute_error(y_true=im_gt.flatten(), y_pred=im_pred.flatten())
    print("Compare prediction with groundtruth CT:")
    print('mae: {mae_pred:.4f} | mse: {mse_pred:.4f} | rmse: {rmse_pred:.4f} | psnr: {psnr_pred:.4f} | ssim: {ssim_pred:.4f}'
          .format(mae_pred=mae_pred, mse_pred=mse_pred, rmse_pred=rmse_pred, psnr_pred=psnr_pred, ssim_pred=ssim_pred))
    return mae_pred, mse_pred, rmse_pred, psnr_pred, ssim_pred

def imageSave(pred, groundtruth, plane, save_path):
    seq = range(pred.shape[plane])
    for slice_idx in seq:
        if plane == 0:
            pd = pred[slice_idx, :, :]
            gt = groundtruth[slice_idx, :, :]
        elif plane == 1:
            pd = pred[:, slice_idx, :]
            gt = groundtruth[:, slice_idx, :]
        elif plane == 2:
            pd = pred[:, :, slice_idx]
            gt = groundtruth[:, :, slice_idx]
        else:
            assert False
        f = plt.figure()
        f.add_subplot(1,3,1)
        plt.imshow(pd, interpolation='none', cmap='gray')
        plt.title('Prediction')
        plt.axis('off')
        f.add_subplot(1,3,2)
        plt.imshow(gt, interpolation='none', cmap='gray')
        plt.title('Groundtruth')
        plt.axis('off')
        f.add_subplot(1,3,3)
        plt.imshow(gt-pd, interpolation='none', cmap='gray')
        plt.title('Difference image')
        plt.axis('off')
        f.savefig(os.path.join(save_path, 'Plane_{}_ImageSlice_{}.png'.format(plane, slice_idx+1)))
        plt.close()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
