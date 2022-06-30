import numpy as np
import os
from pathlib import Path
import cv2
import shutil
import shutil
from PIL import Image
from torchvision import transforms
import xml.dom.minidom
import re

class copycode(object):
    def __init__(self,
        img_path = '../coco/images/12_31zhong/',
        txt_path = '../coco/labels/12_31zhong/',
        img_train_save = '../coco/images/new_train_14_1230/',
        txt_train_save = '../coco/labels/new_train_14_1230/',

        img_val_save = '../coco/images/new_val_14_1230/',
        txt_val_save = '../coco/labels/new_val_14_1230/',
        copy_time = 10
        ):

        self.img_path = img_path
        self.txt_path = txt_path
        self.img_train_save = img_train_save
        self.txt_train_save =  txt_train_save
        self.img_val_save =  img_val_save
        self.txt_val_save = txt_val_save
        self.copy_time =copy_time
   
        if Path(self.img_train_save).exists() == False:
            os.mkdir(self.img_train_save)
        if Path(self.img_val_save).exists() == False:
            os.mkdir(self.img_val_save)

        if Path(self.txt_train_save).exists() == False:
            os.mkdir(self.txt_train_save)
        if Path(self.txt_val_save).exists() == False:
            os.mkdir(self.txt_val_save)

        for img in os.listdir(self.img_path):
            print(img)
            if  '中' in img :
                pname = img[0:8]+img[9:-4]
                txt = img[0:-4] + ".txt"     
     	        
                pic_img = pname + ".jpg"
                pic_label = pname + ".txt"
            else :
                pname = img[0:-4]
                txt = img[0:-4] + '.txt'
                pic_img = pname + ".jpg"
                pic_label = pname + ".txt"
            print(pname,txt,pic_img,pic_label)
            print(self.img_path + img, self.img_val_save + pic_img)
            
            if os.path.exists(self.txt_path + txt) and os.path.exists(self.txt_path + txt):
                shutil.copy(self.img_path + img, self.img_val_save + pic_img)
                shutil.copy(self.txt_path + txt,  self.txt_val_save + pic_label)
            
                for i in range(self.copy_time):
                    txt =  img[0:-4] + ".txt"
                    pic_img = pname + '_' + str(i) + ".jpg"
                    pic_label = pname + '_' + str(i) + ".txt"
                    shutil.copy(self.img_path + img, self.img_train_save + pic_img)
                    shutil.copy(self.txt_path + txt,  self.txt_train_save + pic_label)
            else:
                print(self.img_path + img,self.txt_path + txt)
                print('tiaoguo')
                continue

        print('成功')

'''
if __name__ == '__main__':
    img_path = './01_17zhong/'
    txt_path = './01_17zhong/'
    img_train_save = './new_train_14_1230/'
    txt_train_save = './new_train_14_1230/'

    img_val_save = './new_val_14_1230/'
    txt_val_save = './new_val_14_1230/'
    copy_time = 10
    copycode(img_path,txt_path,img_train_save,txt_train_save,img_val_save,txt_val_save,copy_time) 
    print('chenggong')
'''
