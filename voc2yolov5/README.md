## VOC2YOLOV5

+ data1: 用于xml转换txt
+ data2: 用于复制图片
+ copy_time 用于复制数据集(应为epoch转换的时候比较耗时间)


+ img_dir = '/home/zranguai/Python-Code/Object-Detection-Convert_temp/voc2yolov5/images/'+data1+'/'  # 包括原始的jpg和xml

+ save_dir = '/home/zranguai/Python-Code/Object-Detection-Convert_temp/voc2yolov5/labels/'+data1+'/'  # 存放转换好的yolov5txt文件

+ xml_dir = '/home/zranguai/Python-Code/Object-Detection-Convert_temp/voc2yolov5/labels/'+data1+'_/'  # 将img_dir中的xml转换到这里

+ txt_list_save_dir = '/home/zranguai/Python-Code/Object-Detection-Convert_temp/voc2yolov5/'+data1+'.txt'  # xml文件的路径
