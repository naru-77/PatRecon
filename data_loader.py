import os.path as osp
import numpy as np
from data import MedReconDataset
from torch.utils.data import DataLoader
from torchvision import transforms


def get_train_val_data_loaders(train_file, val_file, args):
	# 各種設定パラメータをargsから取得
	data_root = args.data_root
	data_list_dir = args.data_list_dir
	num_views = args.num_views
	input_size = args.input_size
	output_size = args.output_size
	batch_size = args.batch_size
	num_workers = args.num_workers
# 学習データの平均値と標準偏差を読み込み
	train_stats = np.load(osp.join(args.data_list_dir, '{}_train_stats.npz'.format(args.exp)))
	# 正規化のための変換を設定
	normalize = transforms.Normalize(mean=list(train_stats['mean']),
	# 変換の一連の流れをComposeで定義								std=list(train_stats['std']))
	transform = transforms.Compose([transforms.ToTensor(),
									normalize,])

# 学習用と検証用のデータローダを取得
	train_loader = get_data_loader(file_list=train_file,
									data_root=data_root,
									num_views=num_views,
									input_size=input_size,
									output_size=output_size,
									transform=transform,
									batch_size=batch_size,
									train=True,
									num_workers=num_workers)

	val_loader = get_data_loader(file_list=val_file,
									data_root=data_root,
									num_views=num_views,
									input_size=input_size,
									output_size=output_size,
									transform=transform,
									batch_size=1,
									train=False,
									num_workers=num_workers)

	return train_loader, val_loader




def get_data_loader(file_list, data_root, num_views, 
					input_size, output_size, transform,
					batch_size, train, num_workers):
# 指定されたパラメータでデータセットを作成
	dataset = MedReconDataset(file_list=file_list,
				        data_root=data_root,
				        num_views=num_views,
				        input_size=input_size,
				        output_size=output_size,
				        transform=transform)
# データセットからデータローダを作成
	loader = DataLoader(dataset=dataset, 
				        batch_size=batch_size, 
				        shuffle=train,
				        num_workers=num_workers, 
				        pin_memory=True)

	return loader


