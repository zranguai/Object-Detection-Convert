"""
进行分离voc的文件夹，用于后续的voc2coco进行处理
"""

import argparse
import os
import shutil


def read_txt_and_move_file(txt_file, input_dir, out_put_dir):
    # current_dir = os.getcwd()
    current_dir = "/home/zranguai/Python-Code/Object-Detection-Convert"
    file_dir = []
    txt_file = os.path.join(current_dir, txt_file)
    f =  open(txt_file, mode="r", encoding='utf-8')
    while True:
        line = f.readline()
        if line:
            file_dir.append(line)
        else:
            break
    f.close()

    for file in file_dir:
        file_ = os.path.join(input_dir, file.strip()+".jpg")  # 这里写切换文件的后缀名

        src = os.path.join(current_dir, file_)
        dst = os.path.join(current_dir, out_put_dir)
        # print(src)
        # print(dst)
        shutil.move(src, dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="move_file")
    parser.add_argument('--txt_file', type=str, help='', default="VOCdevkit/VOC2007/ImageSets/Main/val.txt")
    parser.add_argument('--input_dir', type=str, help='directory to xml files', default="./Annotations")
    parser.add_argument('--output_dir', type=str, help='path to output json files', default="./data.json")

    args = parser.parse_args()
    # read txt and move file
    read_txt_and_move_file(args.txt_file, args.input_dir, args.output_dir)

