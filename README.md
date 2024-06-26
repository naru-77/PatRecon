# Patient-specific reconstruction of volumetric computed tomography images from few-view projections via deep learning

## Contents

- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Instructions for Use](#instructions-for-use)
- [License](./LICENSE)
- [Citation](#citation)

# 1. Overview

This project ([paper link](https://www.nature.com/articles/s41551-019-0466-4)) provides a deep-learning framework for generating volumetric tomographic X-ray images with ultra-sparse 2D projections as input. Using the code requires users to have basic knowledge about python, PyTorch, and deep neural networks.

このプロジェクトは、超疎な 2D 投影を入力として、体積断層 X 線画像を生成するためのディープラーニングフレームワークを提供しています。コードの使用には、Python、PyTorch、およびディープニューラルネットワークに関する基本的な知識が必要です。

# 2. Repo Contents

- [test.py](./test.py): main code to run evaluation.
- [net.py](./net.py): network definition of proposed framework.
- [trainer.py](./trainer.py): model training functions.
- [data.py](./data.py): dataset definition for model training.
- [data_loader.py](./data_loader.py): data loader definition for model training.
- [utils.py](./utils.py): util functions definition.
- [exp/model/model.pth.tar](./exp/model/model.pth.tar): trained model for running experiment. [Download link](https://drive.google.com/file/d/1wiwH3vHAA4zFSIbi8JT8XwdzZOP4MAT8/view?usp=sharing)
- [exp/data/2D_projection.jpg](./exp/data/2D_projection.jpg): 2D projection of the data sample, which is the input of model. [Download link](https://drive.google.com/file/d/1G63gUOHgyukGWqstcpLWPcvcvfIPa75l/view?usp=sharing)
- [exp/data/3D_CT.jpg](./exp/data/3D_CT.bin): 3D CT volume of the data sample, which will be used as groundtruth to compare with the output (prediction) results. [Download link](https://drive.google.com/file/d/1aNtf0gbo9C5kt6st8-Qqly24K_59QvwB/view?usp=sharing)
- [exp/result](./exp/result): output folder to save the model prediction as .png files.
- Please put the trained model and data sample under `exp` folder as above to run the code.
- Please note the trained model is for the specific patient sample. We suggest retraining the model to apply to the customized data according to training strategy in the paper.

test.py: 評価を実行するための主要なコード。
net.py: 提案されたフレームワークのネットワーク定義。
trainer.py: モデルトレーニング関数。
data.py: モデルトレーニングのためのデータセット定義。
data_loader.py: モデルトレーニングのためのデータローダー定義。
utils.py: ユーティリティ関数の定義。
その他、トレーニング済みのモデルやデータサンプル、結果を保存するためのフォルダなどが含まれています。

# 3. System Requirements システム要件

## Hardware Requirements ハードウェア要件

Loading and running deep network requires a standard computer with enough memory to support the model defined by a user. For optimal performance, a GPU card can largely accelerate computation. In our experiment, we use a NVIDIA Tesla V100 GPU card with about 32 GB of memory.

<!-- we recommend a computer with the following specs:
RAM: 16+ GB
CPU: 4+ cores, 3.3+ GHz/core -->

The runtimes below are generated using a computer with a NVIDIA Tesla V100 GPU.

ディープネットワークのロードと実行には、ユーザーが定義したモデルをサポートする十分なメモリを持つ標準的なコンピュータが必要です。最適なパフォーマンスのためには、GPU カードが計算を大幅に加速することができます。私たちの実験では、約 32GB のメモリを持つ NVIDIA Tesla V100 GPU カードを使用しています。

## OS Requirements OS 要件

This package is supported for _Linux_ operating systems. The package has been tested on the following systems:

Linux: Ubuntu 16.04

このパッケージは Linux オペレーティングシステムでサポートされています。このパッケージは次のシステムでテストされています：

Linux: Ubuntu 16.04

# 4. Installation Guide インストールガイド

Before running this package, users should have `Python`, `PyTorch`, and several python packages (`numpy`, `sklearn`, `skimage`, `PIL`, and `matplotlib`) installed.

## Installing Python version 3.5.5 on Ubuntu 16.04

The Python can be installed in Linux by running following command from terminal:

```
sudo apt-get update
sudo apt-get install python3.5
```

which should install in about 30 seconds.

Python は、ターミナルから次のコマンドを実行することで Linux にインストールできます

```
sudo apt-get update
sudo apt-get install python3.5
```

## Package Versions

This code functions with following dependency packages. The versions of software are, specifically:

```
pytorch: 0.4.1
numpy: 1.15.0
sklearn: 0.19.1
skimage: 0.14.0
PIL: 5.1.0
matplotlib: 2.2.2
```

## Package Installment パッケージのインストール

Users should install all the required packages shown above prior to running the algorithm. Most packages can be installed by running following command in terminal on Linux. To install of PyTorch, please refer to their official [website](https://pytorch.org).

```
pip install package-name
```

which will install in about 30 mins on a recommended machine.

ユーザーは、アルゴリズムを実行する前に、上記に示すすべての必要なパッケージをインストールする必要があります。ほとんどのパッケージは、Linux のターミナルで次のコマンドを実行することでインストールできます。PyTorch のインストールについては、公式ウェブサイトを参照してください。

推奨されるマシンで約 30 分でインストールされるはずです。

# 5. Instructions for Use 使用方法

To running the trained model to evaluate reconstruction performance on data samples, please type in following comman in terminal. Parameter `exp` is used to run different experiments. Running a model inference on one data sample should take approximately 2 mins using a computer with a NVIDIA Tesla V100 GPU.

トレーニング済みモデルを使用して、データサンプル上での再構築パフォーマンスを評価するには、ターミナルで次のコマンドを入力してください

## Running Experiment

Please use the parameter `vis_plane` to get output image slices on different planes of 3D pancreas CT. The prediction results are saved under path `exp/test_results/sample_1`. The results contains `.png` files, which are named after `Plane_[0/1/2]_ImageSlice_[].png`. Each file shows the prediction, ground truth, and difference image for one slice along the chosen plane. Specifically, the following commands could be run in terminal to get model results and visualize based on three different planes [0: axial, 1: sagittal, 2: coronal] of 3D pancreas CT.

```
python3 test.py --vis_plane 0
python3 test.py --vis_plane 1
python3 test.py --vis_plane 2
```

# 6. License

A provisional patent application for the reported work has been filed. The codes are copyrighted by Stanford University and are for research only. Correspondence should be addressed to the corresponding author in the paper. Licensing of the reported technique and codes is managed by the Office of Technology Licensing (OTL) of Stanford University (Ref. Docket #S18-464).

# 6. ライセンス (License)

報告された作業のための暫定的な特許申請が行われています。コードはスタンフォード大学によって著作権が保護されており、研究のみのためです。

# 7. Citation

If you find the code are useful, please consider citing the paper.

```
@article{shen2019patient,
  title={Patient-specific reconstruction of volumetric computed tomography images from a single projection view via deep learning},
  author={Shen, Liyue and Zhao, Wei and Xing, Lei},
  journal={Nature biomedical engineering},
  volume={3},
  number={11},
  pages={880--888},
  year={2019},
  publisher={Nature Publishing Group}
}
```
