Usage
=============

A python script for converting PASCAL VOC xml file format to MSCOCO json file format in **Object Detection**.

```
python voc2coco.py --xml_dir path/to/xml_files  --output path/to/save/data.json 

```



type `python voc2coco.py -h` for more details.

If you want to customize COCO categories key-value pairs, you can create a text file like this:

```
1,class1
2,class2
3,class3
...
n,classn
```

then add `--labelmap` argument with text file path behind  `python voc2coco.py` like this :

`python voc2coco.py --labelmap path/to/labelmap.txt`



