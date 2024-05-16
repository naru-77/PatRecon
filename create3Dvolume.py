import os
import pydicom
import numpy as np

def load_dicom_series(directory):
    # DICOMファイルのリストを取得
    file_list = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.dcm')]
    # DICOMファイルを読み込み、スライス位置に基づいて並べ替え
    dicom_series = [pydicom.dcmread(f) for f in file_list]
    dicom_series.sort(key=lambda x: float(x.SliceLocation))
    # 3Dボリュームの作成
    volume = np.stack([ds.pixel_array for ds in dicom_series], axis=0)
    volume = volume.astype(np.float32)  # データ型の変換
    return volume

# 例：Dフォルダ内のDICOMシリーズを読み込む
directory = '100CT_cases/K101u/D/'
volume = load_dicom_series(directory)

# 出力ファイル名にフォルダ名を含める
folder_name = os.path.basename(os.path.normpath(directory))
output_file_name = f'output_volume_{folder_name}.bin'

# ボリュームデータを.binファイルとして保存
volume.tofile(output_file_name)
