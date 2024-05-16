import pydicom
import numpy as np
import pandas as pd
import os

def load_dicom_and_apply_window(filename, window_center, window_width):
    dicom_data = pydicom.dcmread(filename)
    pixel_array = dicom_data.pixel_array

    lower_bound = window_center - window_width // 2
    upper_bound = window_center + window_width // 2
    windowed_image = np.clip(pixel_array, lower_bound, upper_bound)
    windowed_image = (windowed_image - lower_bound) / (upper_bound - lower_bound) * 255

    return windowed_image.astype(np.uint8)

def process_dicom_directory(directory, window_center, window_width, csv_filename):
    all_images = []

    # DICOMファイルの読み込みと処理
    for i in range(221): # 000000.dcmから000220.dcmまで
        filename = os.path.join(directory, f"{i:06d}.dcm")
        windowed_image = load_dicom_and_apply_window(filename, window_center, window_width)
        all_images.append(windowed_image.flatten())

    # 全ての画像をDataFrameに変換してCSVファイルとして保存
    df = pd.DataFrame(all_images)
    df.to_csv(csv_filename, index=False)

# ディレクトリとウィンドウ処理のパラメータ
directory = "100CT_cases/K101u/D/"
window_center = 50
window_width = 200
csv_filename = "c:/Users/81805/Working/PatRecon/ct_images.csv"


# 処理の実行
process_dicom_directory(directory, window_center, window_width, csv_filename)
