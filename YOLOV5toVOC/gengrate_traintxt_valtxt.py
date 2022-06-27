# 第一步:yolo2voc.py生成的xml标签
# 第二步: 生成VOC格式的train.txt and val.txt
import os
import argparse


def generate_txt(img_path, txt_path):
    img_paths = img_path
    txt_path = txt_path

    g = os.listdir(img_paths)
    name_withot_jpg = []
    for item in g:
        if os.path.isfile(os.path.join(img_paths, item)):
            name_withot_jpg.append(item)

    name_withot_jpg.sort()  # 为了将文件名排序
    print(name_withot_jpg)
    # 需要保存的文件名
    f = open(txt_path, mode="w", encoding="utf-8")
    for name in name_withot_jpg:
        print(name.split(".")[0])
        f.write(name.split(".")[0] + "\n")
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="/home/zranguai/Python-Code/YOLOV5toVOC/images")
    parser.add_argument('--txt_path', type=str, default="val_test1.txt")

    opt = parser.parse_args()
    generate_txt(opt.image_path, opt.txt_path)
