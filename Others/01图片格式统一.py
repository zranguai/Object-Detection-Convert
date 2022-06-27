# -*- coding: utf-8 -*-
# author: zranguai
# Email: zranguai@gmail.com

import os
import cv2


def listfiles(rootDir):
    list_dirs = os.walk(rootDir)
    for root, dirs, files in list_dirs:  # 遍历文件夹下的图片
        for d in dirs:
            print((os.path.join(root, d)))
        for f in files:
            fileid = f.split('.')[0]  # 获得图片的名字，不含后缀
            filepath = os.path.join(root, f)
            print(filepath)
            try:
                src = cv2.imread(filepath, 1)  # 读取原始图片，数据会加载到内存中
                print("src=", filepath, src.shape)
                os.remove(filepath) # 移除原来的图片
                cv2.imwrite(os.path.join(root, fileid + ".jpg"), src)  # 保存经过格式转换的图片
            except:
                os.remove(filepath)
                continue

path = "/home/zranguai/Python-Code/Tianchi/people-face_learn/Emotion_Recognition_File/img_type_test/"  # 输入图片路径即可，可以在这个文件夹下放置各种后缀名的图片，代码会将所有图片统一成 jpg 格式
listfiles(path)