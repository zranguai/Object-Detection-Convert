"""
step1 将多边形包裹的目标按照最小的box坐标裁剪出来，记录裁剪后的多边形坐标
step2 挑选合适的背景图，并在背景图合适的位置裁剪一块和step1大小相同的背景小区域，使用掩膜方式，将其与目标图融合
step3: 将step2中融合后的图片放回到背景图中，注意与step2中扣取的坐标一致

原则:  粘贴的目标不会与任何现有目标重叠。这增加了小目标位置的多样性，同时确保这些目标出现在正确的上下文中
论文: https://arxiv.org/pdf/1902.07296.pdf
"""
import argparse
import base64
import copy
import cv2
import glob
import json
import os
import random
from tqdm import tqdm
import numpy as np


class SmallObjectAugmentation(object):
    def __init__(self, thresh=64*64, prob=0.5, copy_times=3, epochs=30, all_objects=False, one_object=False,
                 exclude_border=400, category=list()):
        """
        sample = {'img':img, 'annot':annots}
        img = [height, width, 3]
        annot = [xmin, ymin, xmax, ymax, label]
        thresh：the detection threshold of the small object. If annot_h * annot_w < thresh, the object is small
        prob: the prob to do small object augmentation
        epochs: the epochs to do
        exclude_border: 靠近边界的像素为黑色，不要贴目标
        category: 需要贴图的类别
        """
        self.thresh = thresh
        self.prob = prob
        self.copy_times = copy_times
        self.epochs = epochs
        self.all_objects = all_objects
        self.one_object = one_object
        self.exclude_boder = exclude_border
        self.category = category
        if self.all_objects or self.one_object:
            self.copy_times = 1

    def _is_small_object(self, h, w):
        if h * w <= self.thresh:
            return True
        else:
            return False

    def _is_choose_object(self, choose_category):
        if choose_category in self.category:
            return True
        else:
            return False

    def _compute_overlap(self, annot_a, annot_b):
        if annot_a is None: return False
        left_max = max(annot_a[0], annot_b[0])
        top_max = max(annot_a[1], annot_b[1])
        right_min = min(annot_a[2], annot_b[2])
        bottom_min = min(annot_a[3], annot_b[3])
        inter = max(0, (right_min - left_max)) * max(0, (bottom_min - top_max))
        if inter != 0:
            return True
        else:
            return False

    def _donot_overlap(self, new_annot, annots):
        for annot in annots:
            if self._compute_overlap(new_annot, annot): return False
        return True

    def _create_copy_annot(self, h, w, annot, annots):
        annot_4 = annot[:4].astype(int)
        annot_h, annot_w = annot_4[3] - annot_4[1], annot_4[2] - annot_4[0]
        for epoch in range(self.epochs):
            random_x, random_y = np.random.randint(int(annot_w / 2), int(w - annot_w / 2)), \
                                 np.random.randint(int(annot_h / 2), int(h - annot_h / 2))
            xmin, ymin = random_x - annot_w / 2, random_y - annot_h / 2
            xmax, ymax = xmin + annot_w, ymin + annot_h
            # if xmin < 0 or xmax > w or ymin < 0 or ymax > h:
            #     continue
            # 需要限制贴图的区域在较为中心的位置，贴到黑边训练效果不好
            if xmin < self.exclude_boder or xmax > (w - self.exclude_boder) or \
                    ymin < self.exclude_boder or ymax > (h - self.exclude_boder):
                continue
            # new_annot = np.array([xmin, ymin, xmax, ymax, annot[4]]).astype(int)
            new_annot = np.array([xmin, ymin, xmax, ymax]).astype(int)
            # todo: 改annotation
            annot_ = copy.deepcopy(annot)
            seg_annot = annot_[4:][0]

            seg_points = seg_annot["points"]
            for point in seg_points:
                point[0] = point[0] - annot_4[0] + xmin
                point[1] = point[1] - annot_4[1] + ymin

            if self._donot_overlap(new_annot, annots) is False:
                continue

            new_annot_ = [new_annot[0], new_annot[1], new_annot[2], new_annot[3], annot_[4:][0]]

            return np.array(new_annot_)
        return None

    def _add_patch_in_img(self, annot, copy_annot, image):
        copy_annot = copy_annot[:4].astype(int)
        image[annot[1]:annot[3], annot[0]:annot[2], :] = image[copy_annot[1]:copy_annot[3], copy_annot[0]:copy_annot[2], :]

        return image

    def __call__(self, sample):
        if self.all_objects and self.one_object: return sample
        if np.random.rand() > self.prob: return sample

        img, annots = sample['img'], sample['annot']
        h, w = img.shape[0], img.shape[1]

        small_object_list = list()
        for idx in range(annots.shape[0]):
            annot = annots[idx]
            annot_h, annot_w = annot[3] - annot[1], annot[2] - annot[0]
            if self._is_small_object(annot_h, annot_w):
                small_object_list.append(idx)

        l = len(small_object_list)
        # No Small Object
        if l == 0: return sample

        # Refine the copy_object by the given policy
        # Policy 2:
        copy_object_num = np.random.randint(0, l)
        # Policy 3:
        if self.all_objects:
            copy_object_num = l
        # Policy 1:
        if self.one_object:
            copy_object_num = 1

        random_list = random.sample(range(l), copy_object_num)
        annot_idx_of_small_object = [small_object_list[idx] for idx in random_list]
        select_annots = annots[annot_idx_of_small_object, :]
        annots = annots.tolist()
        for idx in range(copy_object_num):
            annot = select_annots[idx]
            annot_h, annot_w = annot[3] - annot[1], annot[2] - annot[0]

            choose_category = annot[4]['label']
            if self._is_choose_object(choose_category) is False: continue  # 判断该类别是否需要贴图

            if self._is_small_object(annot_h, annot_w) is False: continue  # 判断该大小是否要贴图

            for i in range(self.copy_times):
                # new_annot = self._create_copy_annot(h, w, annot, annots, )
                new_annot = self._create_copy_annot(h, w, annot, annots, )
                if new_annot is not None:
                    img = self._add_patch_in_img(new_annot[:4], annot, img)
                    annots.append(new_annot.tolist())

        return {'img': img, 'annot': np.array(annots)}


class ReadJsonImg(object):
    def __init__(self, json_path="", img_path=""):
        self.json_path = json_path
        self.img_path = img_path
        self.sample = dict()

    def _min_inside_rect(self, points):
        """
        分割点的最小内接矩
        """
        point_minx = 100000000000
        point_miny = 100000000000
        point_maxx = -1
        point_maxy = -1
        for point in points:
            # 最小最大x
            if point[0] < point_minx:
                point_minx = point[0]
            if point[0] > point_maxx:
                point_maxx = point[0]
            # 最小最大y
            if point[1] < point_miny:
                point_miny = point[1]
            if point[1] > point_maxy:
                point_maxy = point[1]
        return [point_minx, point_miny, point_maxx, point_maxy]

    def _get_annot(self):
        f_json = open(self.json_path, mode='r', encoding='utf-8')
        j_json = json.loads(f_json.read())
        shapes = j_json["shapes"]

        annots = []
        for shape in shapes:
            x1, y1, x2, y2 = self._min_inside_rect(shape["points"])
            annot = [x1, y1, x2, y2, shape]  # 这里的1为类别的信息
            annots.append(annot)
        annots = np.array(annots)
        self.sample['annot'] = annots

    def _get_img(self):
        img = cv2.imread(self.img_path)
        self.sample['img'] = img

    def __call__(self):
        self._get_img()
        self._get_annot()
        return self.sample


class WriteJsonImg(object):
    def __init__(self, json_path="", img_path="", Sample=dict, img_basename=""):
        self.json_path = json_path
        self.img_path = img_path
        self.Sample = Sample
        self.basename = img_basename

    def _write_img(self):
        cv2.imwrite(self.img_path, self.Sample["img"])

    def _get_shapes(self):
        annot = self.Sample["annot"].tolist()
        shapes = list()
        for ann in annot:
            shapes.append(ann[-1])
        return shapes

    def _get_image_data(self):
        retrval, buffer = cv2.imencode('.jpg', self.Sample['img'])
        img_data = base64.b64encode(buffer)
        return img_data

    def _write_json(self):
        wf_json = open(self.json_path, mode='r', encoding='utf-8')
        wj_json = json.loads(wf_json.read())

        new_j = {}
        new_j["version"] = wj_json["version"]
        new_j["flags"] = wj_json["flags"]
        new_j["shapes"] = self._get_shapes()
        # new_j["imagePath"] = wj_json["imagePath"]
        new_j["imagePath"] = self.basename
        new_j["imageData"] = None
        new_j["imageHeight"] = wj_json["imageHeight"]
        new_j["imageWidth"] = wj_json["imageWidth"]

        out_json = open(self.json_path, mode='w', encoding='utf-8')
        out_json.write(json.dumps(new_j, ensure_ascii=False))

    def __call__(self):
        self._write_img()
        self._write_json()
        # return f"WRITE {self.img_path} DONE"


def parse_args():
    parser = argparse.ArgumentParser(description="augmentation data")
    parser.add_argument('--path', default="trans/", help='the labelme file paths')
    parser.add_argument('--choose_category', default=["Scratch", "WuZhuangSeCha"], help='贴图的类别')
    parser.add_argument('--exclude_border', default=400, help='边缘不要贴图的距离')
    parser.add_argument('--SOA_THRESH', default=200 * 200, help='贴图筛选的面积')
    parser.add_argument('--SOA_PROB', default=1, help='')
    parser.add_argument('--SOA_COPY_TIMES', default=3, help='')
    parser.add_argument('--SOA_EPOCHS', default=30, help='')
    parser.add_argument('--SOA_ONE_OBJECT', default=False, help='')
    parser.add_argument('--SOA_ALL_OBJECTS', default=True, help='')
    args = parser.parse_args()

    return args


def main_process():
    """
    输入与输出文件夹形式均为labelme格式(对源文件夹进行覆盖操作)，即:

    输入:
    trans
    -------1-jxzh-s_s.jpg
    -------1-jxzh-s_s.json
    -------1-jxzh-2_s.jpg
    -------1-jxzh-2_s.json

    输出:
    trans
    -------1-jxzh-s_s.jpg
    -------1-jxzh-s_s.json
    -------1-jxzh-2_s.jpg
    -------1-jxzh-2_s.json
    """
    json_paths = glob.glob(r"trans/*.json")  # trans为输入文件夹的名字
    for json_path in json_paths:
        img_path = json_path.replace(".json", ".jpg")

        # 6-16添加新功能 按照类别进行筛选
        choose_category = ["Scratch", "WuZhuangSeCha"]
        read_json_img = ReadJsonImg(json_path=json_path, img_path=img_path)
        Sample = read_json_img()

        """   SMALL OBJECT AUGMENTATION   """
        # Defaultly perform Policy 2, if you want to use
        # Policy 1, make SOA_ONE_OBJECT = Ture, or if you
        # want to use Policy 3, make SOA_ALL_OBJECTS = True
        # SOA_THRESH = 64 * 64
        SOA_THRESH = 200 * 200
        SOA_PROB = 1
        SOA_COPY_TIMES = 3
        SOA_EPOCHS = 30
        SOA_ONE_OBJECT = False
        SOA_ALL_OBJECTS = True  # False -> True Policy 3
        augmenter = SmallObjectAugmentation(SOA_THRESH, SOA_PROB, SOA_COPY_TIMES, SOA_EPOCHS, SOA_ALL_OBJECTS,
                                            SOA_ONE_OBJECT, category=choose_category)
        Sample = augmenter(Sample)
        # print(Sample)

        # 重写文件和json
        img_basename = os.path.basename(img_path)
        write_json_img = WriteJsonImg(json_path=json_path, img_path=img_path, Sample=Sample, img_basename=img_basename)
        flag = write_json_img()
        print(flag)


def main():
    args = parse_args()

    path = args.path
    json_paths = glob.glob(f"{path}*.json")
    pbar = tqdm(json_paths, leave=True)
    for json_path in pbar:
        pbar.set_description(json_path)
        img_path = json_path.replace(".json", ".jpg")

        read_json_img = ReadJsonImg(json_path=json_path, img_path=img_path)
        Sample = read_json_img()

        augmenter = SmallObjectAugmentation(args.SOA_THRESH, args.SOA_PROB, args.SOA_COPY_TIMES, args.SOA_EPOCHS,
                                            args.SOA_ALL_OBJECTS, args.SOA_ONE_OBJECT, args.exclude_border, args.choose_category)
        Sample = augmenter(Sample)

        img_basename = os.path.basename(img_path)
        write_json_img = WriteJsonImg(json_path, img_path, Sample, img_basename)
        write_json_img()


if __name__ == '__main__':
    # main_process()  # 参数写在代码里
    main()
