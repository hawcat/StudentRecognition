# 归一处理

import os
import glob
from PIL import Image


def produceImage(file_in, width, height, file_out):
    image = Image.open(file_in)
    resized_image = image.resize((width, height), Image.ANTIALIAS)
    resized_image.save(file_out)


if __name__ == '__main__':
    path = r"D:\Action-Recognition-plus\Action-Recognition-plus\data\class_new\stand_03-13-13-22-37-869"
    out_save_path = r"D:\Action-Recognition-plus\Action-Recognition-plus\data\class_new\stand_03-13-13-22-37-869"
    file_list = glob.glob(os.path.join(path, '*.jpg'))  # 获取所有jpg文件
    width = 1280  # 调整的分辨率大小
    height = 720

    for file_in in file_list:
        file_out = os.path.join(path, out_save_path, os.path.basename(file_in))
        produceImage(file_in, width, height, file_out)  # 遍历所有图片，修改分辨率并保存
    print("done")