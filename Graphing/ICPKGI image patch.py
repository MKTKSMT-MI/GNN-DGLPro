import os
import glob

# フォルダのパスを指定
folder_path = '../data/car/**/*'  # フォルダのパスを適切に設定してください

# フォルダ内のファイル名を取得
file_names = glob.glob(folder_path)


# 取得したファイル名を表示
for file_name in file_names:
    print(file_name)