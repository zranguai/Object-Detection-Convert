# COCO2VOC
> converting MScoco json file format to PASCAL VOC xml format in object
> detectin
1. coco128是coco格式的文件
2. xml_files是转换后的xml文件

+ 使用:
```bash
python coco2voc.py --json_path path/to/data.json  --output path/to/save/xml_files
```

+ ref:
  + [coco2voc](https://github.com/bot66/coco2voc)
  + [使用labelImg进行可视化](https://github.com/tzutalin/labelImg)