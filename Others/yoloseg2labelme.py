"""
Create on June 29, 2023
@author: zrg
"""
import argparse
import cv2
import json
import os

# 定义类别 id映射
label_id_map = {
    0: "ZangWu",
    1: "HuangBian",
    2: "WuZhuangSeCha",
    3: "YinYangPian",
    4: "DaBingPian",
    5: "Scratch",
    6: "SeBan",
    7: "YouWu"
}


class YOLO2Labelme(object):

    def __init__(self, txt_dir, img_dir, json_dir):
        self._txt_dir = txt_dir
        self._img_dir = img_dir
        self._json_dir = json_dir
        self._label_id_map = label_id_map

    def _get_label_id_map(self, label):
        label = int(label)
        return self._label_id_map[label]

    def _parse_yolo_label(self, txt_path, imageHeight, imageWidth):
        """
        Args:
            txt_path: yolo 文件夹的txt路径

        Returns:

        """
        with open(txt_path, "r") as file:
            txt_labels = file.readlines()

            shapes = list()
            for txt_label in txt_labels:

                shape = dict()
                txt_lists = [float(i) for i in txt_label.split(" ")]
                shape["label"] = self._get_label_id_map(txt_lists.pop(0))

                points = list()
                # for point in [txt_lists[i: i + 2] for i in range(0, len(txt_lists), 2)]:
                for index, point in enumerate([txt_lists[i: i + 2] for i in range(0, len(txt_lists), 2)]):
                    # point_x = point[0] * imageHeight
                    # point_y = point[1] * imageWidth

                    # 注意这里预测的x对应图片的宽, y对应图片的高
                    point_x = point[0] * imageWidth
                    point_y = point[1] * imageHeight

                    # 不要点这么多
                    # if index % 2 == 0: continue

                    points.append([point_x, point_y])
                shape["points"] = points
                shape["group_id"] = None
                shape["shape_type"] = "polygon"
                shape["flags"] = dict()
                shapes.append(shape)
            return shapes

    def _save_labelme_label(self, img_mat, img_path, shapes, imageHeight, imageWidth):
        img_basename = os.path.basename(img_path)
        assert img_basename.replace(".jpg", ".json")  # 针对图片不是jpg情况
        json_basename = img_basename.replace(".jpg", ".json")
        json_full_path = os.path.join(self._json_dir, json_basename)

        new_j = {}
        new_j["version"] = "5.0.1"
        new_j["flags"] = dict()
        new_j["shapes"] = shapes
        new_j["imagePath"] = img_basename
        new_j["imageData"] = None
        new_j["imageHeight"] = imageHeight
        new_j["imageWidth"] = imageWidth

        with open(json_full_path, mode="w", encoding="utf-8") as out:
            out.write(json.dumps(new_j, ensure_ascii=False))

    def convert(self):
        txt_names = [file_name for file_name in os.listdir(self._txt_dir) \
                      if os.path.isfile(os.path.join(self._txt_dir, file_name)) and \
                      file_name.endswith('.txt')]
        folders = [file_name for file_name in os.listdir(self._txt_dir) \
                   if os.path.isdir(os.path.join(self._txt_dir, file_name))]

        for txt_name in txt_names:
            txt_path = os.path.join(self._txt_dir, txt_name)
            img_path = os.path.join(self._img_dir, txt_name.replace(".txt", ".jpg"))
            # 读取图片获取图片宽高
            img_mat = cv2.imread(img_path)
            assert len(img_mat.shape) == 3
            imageHeight, imageWidth = img_mat.shape[0: 2]
            # 解析txt文档
            shapes = self._parse_yolo_label(txt_path, imageHeight, imageWidth)

            self._save_labelme_label(img_mat, img_path, shapes, imageHeight, imageWidth)

            print(f"{txt_name} IS CONVERT DONE")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt_dir', type=str, default=r'C:\CV_code\YOLO-Segmentation\ultralytics-main\tools\yolo_data\labels',
                        help='yolov8-seg txt files dir')
    parser.add_argument('--img_dir', type=str,
                        default=r'C:\CV_code\YOLO-Segmentation\ultralytics-main\tools\yolo_data\images',
                        help='yolov8-seg image files dir')
    parser.add_argument('--json_dir', type=str,
                        default=r'C:\CV_code\YOLO-Segmentation\ultralytics-main\tools\yolo_data\labelme_jsons',
                        help='convert labelme json files dir')
    args = parser.parse_args()
    convertor = YOLO2Labelme(args.txt_dir, args.img_dir, args.json_dir)
    convertor.convert()
