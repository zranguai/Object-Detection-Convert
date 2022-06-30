import os
import os
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import glob
from copy_code_test import copycode 
import cv2
#for loop查找图片，并且去label中查找，然后加入到新的文件夹
def get_img(img_,save_):
    listdir = os.listdir(img_)

    #assert len(listdir)%2 ==0,"The number of JPG and XML does not match"

    for i in os.listdir(img_):
        if '.xml' in i:
            shutil.move(img_ + i, save_ + i)

def get_txt(save_,txt_):
    for i in os.listdir(save_):
        if '.xml' in i:
            shutil.move(save_ + i, txt_ + i)

def make_txt_list(img_,save_):
    with open(save_, 'w') as f:
        for i in os.listdir(img_):
            f.write(img_ + i + "\n")


if __name__ == '__main__':

    """6-30
    
    #data and copy_time parameter
    data1 = '3'#'test_228'
    data2 = 'color'#'test_02_28'    #15 is class,0106 is data
    copy_time = 20


    # xml2txt  parameter
    img_dir = '/data1/coco/images/'+data1+'/'#你需要把xml转成txt文件所在的位置
    save_dir = '/data1/coco/labels/'+ data1+'/'#储存xml文件的位置（txt也会生成在这个路径下）
    xml_dir = '/data1/coco/labels/'+data1+'_/'#最后将txt和xml文件分离的路径（会将xml文件分离到这里）
    txt_list_save_dir = '/data1/coco/'+data1+'.txt'#储存save_dir中xml路径的文件
    
    

    # copy code parameter
    
    #img_path = '../coco/images/new_train_14_'

    img_train_save='../coco/images/train_'+ data2 + '/'
    txt_train_save= '../coco/labels/train_'+ data2 +'/'

    img_val_save = '../coco/images/val_'+ data2 +'/'
    txt_val_save = '../coco/labels/val_'+ data2 +'/' #train_14_
    #print(img_dir,save_dir,xml_dir,txt_list_save_dir,img_train_save,txt_train_save,img_val_save,txt_val_save)
    
    """

    # -------------6-30--------------------
    data1 = '1'
    data2 = 'color'

    copy_time = 2  # 用于复制数据集(应为epoch转换的时候比较耗时间)

    # xml2txt parameter
    img_dir = '/home/zranguai/Python-Code/Object-Detection-Convert_temp/voc2yolov5/images/'+data1+'/'  # 包括原始的jpg和xml
    save_dir = '/home/zranguai/Python-Code/Object-Detection-Convert_temp/voc2yolov5/labels/'+data1+'/'  # 存放转换好的yolov5txt文件
    xml_dir = '/home/zranguai/Python-Code/Object-Detection-Convert_temp/voc2yolov5/labels/'+data1+'_/'  # 将img_dir中的xml转换到这里
    txt_list_save_dir = '/home/zranguai/Python-Code/Object-Detection-Convert_temp/voc2yolov5/'+data1+'.txt'  # xml文件的路径

    # copy code parameter
    img_train_save = '/home/zranguai/Python-Code/Object-Detection-Convert_temp/voc2yolov5/images/train_' + data2 + '/'
    txt_train_save = '/home/zranguai/Python-Code/Object-Detection-Convert_temp/voc2yolov5/labels/train_' + data2 + '/'

    img_val_save = '/home/zranguai/Python-Code/Object-Detection-Convert_temp/voc2yolov5/images/val_' + data2 + '/'
    txt_val_save = '/home/zranguai/Python-Code/Object-Detection-Convert_temp/voc2yolov5/labels/val_' + data2 + '/'  # train_14_
    # -------------------------------------
    
    #判断文件夹在不在，不在就make一个
    if Path(save_dir).exists() == False:
        os.mkdir(save_dir)

        # 判断文件夹在不在，不在就make一个
    if Path(xml_dir).exists() == False:
        os.mkdir(xml_dir)

    get_img(img_dir,save_dir)
    #print(txt_list_save_dir)
    make_txt_list(save_dir,txt_list_save_dir)

    sets = [data1]#与 txt_list_save_dir一致


    #classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
    #           '11', '12', '13', '14','15','16','17','18','19','20','21','22','23']  # 自己训练的类别


    classes = ['0','1', '2', '3', '4', '5','6','7','8','9','10', '11', '12','13','14','15','16','17','18','19','20','21','22','23','24', '25', '26', '27','28', '29', '30','31' ,'32','33','34','35','36','37','38']

    def convert(size, box):
        #size = [1080,1440]
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)


    def convert_annotation(image_id):
        print('image_id',image_id)
        if os.path.exists(image_id[0:-4] + '.xml') :
            in_file = open('%s.xml' % (image_id[0:-4]))
            #print(image_id[0:-4])
            out_file = open('%s.txt' % (image_id[0:-4]), 'w')

            tree = ET.parse(in_file)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in classes or int(difficult) == 1:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                     float(xmlbox.find('ymax').text))
                bb = convert((w, h), b)
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        else:
            print(image_id," this file was Skiped")
            pass

    wd = getcwd()
    for image_set in sets:
        #print('image_set',image_set)
       
        if not os.path.exists('./labels/'):
            os.makedirs('./labels/')

        # image_ids = open('/data1/coco/%s.txt' % (image_set)).read().strip().split()  # 6-30

        img_file_path = '/home/zranguai/Python-Code/Object-Detection-Convert_temp/voc2yolov5'  # 项目根目录
        image_ids = open(img_file_path + '/%s.txt' % (image_set)).read().strip().split()  # 6-30

        list_file = open('./%s.txt' % (image_set), 'w')
        for image_id in image_ids:
            #print(image_ids)
            list_file.write('./%s\n' % (image_id))
            #print(list_file)
            convert_annotation(image_id)
        list_file.close()

    get_txt(save_dir,xml_dir)
    


    copy_img = copycode(img_dir,save_dir,img_train_save,txt_train_save,img_val_save,txt_val_save,copy_time)  # 复制数据集用，可以注释





