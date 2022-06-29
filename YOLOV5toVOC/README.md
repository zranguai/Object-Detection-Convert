## U版YOLOV5格式数据 -> VOC数据格式
+ 转换格式目录结构
```txt
YOLOV5toVOC
├── images
│   ├── (51).jpg
│   └── (67).jpg
├── labels
│   ├── (51).txt
│   └── (67).txt
├── save_xml_path
├── classes.names
├── gengrate_traintxt_valtxt.py
├── README.md
└── yolo2voc.py
```
+ 说明
```txt
images目录下放置图片
labels目录下放置yolov5格式标签
save_xml_path目录下将要放置转换的xml文件
classes.names文件下写类别名字
gengrate_traintxt_valtxt.py产生VOC格式
最后按照VOC的格式放置即可
```
+ note
  + yolo2voc.py下的line14注意更改images格式是.jpg或.png等
+ step1: 转换xml格式
```text
python yolo2voc.py --classes_file classes.names --source_txt_path labels/ --source_img_path images/ --save_xml_path save_xml_path/
```
+ step2: 转换成VOC格式
```text
python gengrate_traintxt_valtxt.py --image_path /path/to/you/img --txt_path /path/to/you/save_txt
```
+ ref

[转换数据集格式1](https://blog.csdn.net/qq_38109843/article/details/90783347)

[参考1-转换合集](https://blog.csdn.net/qq_38109843/article/details/90783347?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_utm_term-0&spm=1001.2101.3001.4242)

[参考2-yolo2coco](https://github.com/RapidAI/YOLO2COCO)
