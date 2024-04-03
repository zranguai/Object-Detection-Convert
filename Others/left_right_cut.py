import copy
import cv2
import json
import numpy as np
import os
import glob
from tqdm import tqdm


def polygon_iou(poly0, poly1):
    """
    多边形iou
    """
    # 计算两个多边形角点围城的外接矩形的宽高。
    pts = np.vstack([poly0, poly1])
    # x, y, w, h = cv2.boundingRect(pts)
    maxx, maxy = np.max(pts, 0)
    # w = maxx + 20  # +20,增加背景区域,无实际意义。
    # h = maxy + 20
    w = maxx
    h = maxy

    canv0 = np.zeros((h, w), np.uint8)
    canv1 = np.zeros_like(canv0)
    canvRes = np.zeros_like(canv0)

    cv2.fillConvexPoly(canv0, poly0, 255)
    cv2.fillConvexPoly(canv1, poly1, 255)
    canvRes = cv2.bitwise_and(canv0, canv1)
    poly0Area = cv2.countNonZero(canv0)
    poly1Area = cv2.countNonZero(canv1)
    intersection = cv2.countNonZero(canvRes)
    union = poly0Area + poly1Area - intersection
    # print("inter=%.2f, union=%.2f, iou=%.2f" % (intersection, union, intersection / union))

    if intersection == 0:  # 交集为0
        define_iou = 0
        return define_iou, canvRes
    elif intersection == min(poly0Area, poly1Area):  # 交集是等于小的面积，说明包含在大的里面
        define_iou = 1
        return define_iou, canvRes
    else:
        define_iou = 0.5  # 定义一个小于1的数
        return define_iou, canvRes


def cut_one(merge_dict):
    """
    左右进行切割
    """
    base_img_name = os.path.basename(merge_dict["src_image_name"])[:-4]
    origin_img_name = base_img_name + str("_origin_") + os.path.basename(merge_dict["src_image_name"])[-4:]
    left_img_name = base_img_name + str("_left_") + os.path.basename(merge_dict["src_image_name"])[-4:]
    right_img_name = base_img_name + str("_right_") + os.path.basename(merge_dict["src_image_name"])[-4:]
    base_dst_path = merge_dict["dst_file_dir"]
    origin_img_path = os.path.join(base_dst_path, origin_img_name)
    origin_json_path = origin_img_path.replace(".jpg", ".json")
    left_img_path = os.path.join(base_dst_path, left_img_name)
    left_json_path = left_img_path.replace(".jpg", ".json")
    right_img_path = os.path.join(base_dst_path, right_img_name)
    right_json_path = right_img_path.replace(".jpg", ".json")

    # 第一份，拷贝原始的到dst_data中
    origin_j = copy.deepcopy(merge_dict["new_j"])
    origin_j["imagePath"] = origin_img_name
    try:
        cv2.imwrite(origin_img_path, merge_dict["src_img_mat"])
        with open(origin_json_path, mode="w", encoding="utf-8") as out:
            out.write(json.dumps(origin_j, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}, 写入{origin_img_path}失败")

    # 第二份，左边的图片和json进行切割
    try:
        left_j = {}
        left_j['version'] = merge_dict["new_j"]['version']
        left_j['flags'] = merge_dict["new_j"]['flags']

        left_j['imagePath'] = left_img_name

        left_j['imageHeight'] = merge_dict["new_j"]['imageHeight']
        left_j['imageWidth'] = int(merge_dict["new_j"]['imageWidth'] / 2)

        left_box = [[0, 0], [left_j['imageWidth'], 0], [left_j['imageWidth'], left_j['imageHeight']], [0, left_j['imageHeight']]]
        left_img_mat = merge_dict["src_img_mat"][0:left_j['imageHeight'], 0:left_j['imageWidth']]

        # 写入左边图片
        cv2.imwrite(left_img_path, left_img_mat)

        # 找属于左边的点
        left_shapes = list()
        new_left_shapes = copy.deepcopy(merge_dict["new_j"]["shapes"])
        for nl_shape in new_left_shapes:
            nl_points = np.int32(nl_shape["points"])
            l_box = np.int32(left_box)
            l_iou, l_canvRes = polygon_iou(l_box, nl_points)
            if 0 == l_iou:
                # 缺陷在区域外
                continue
            elif 1 == l_iou:
                # 缺陷在区域内
                left_shapes.append(nl_shape)
            else:
                # 缺陷在交线上
                # 处理之前膨胀下, 放置之前传过来的有空洞
                l_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                l_dilate_canvRes = cv2.dilate(l_canvRes, l_dilate_kernel)
                l_points_contours, l_hierarchy = cv2.findContours(l_dilate_canvRes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                l_inter_points = l_points_contours[0].reshape(-1, 2).tolist()

                inter_nl_shape = dict()
                inter_nl_shape["label"] = nl_shape["label"]
                inter_nl_shape["points"] = l_inter_points
                inter_nl_shape["group_id"] = nl_shape["group_id"]
                inter_nl_shape["shape_type"] = nl_shape["shape_type"]
                inter_nl_shape["flags"] = nl_shape["flags"]
                left_shapes.append(inter_nl_shape)

        left_j['shapes'] = left_shapes
        left_j["imageData"] = None

        # 写入左边json
        with open(left_json_path, mode="w", encoding="utf-8") as left_out:
            left_out.write(json.dumps(left_j, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}, 写入{left_img_path}失败")

    # 第三份，右边的图片和json进行切割
    try:
        right_j = {}
        right_j['version'] = merge_dict["new_j"]['version']
        right_j['flags'] = merge_dict["new_j"]['flags']

        right_j['imagePath'] = right_img_name

        right_j['imageHeight'] = merge_dict["new_j"]['imageHeight']
        right_j['imageWidth'] = int(merge_dict["new_j"]['imageWidth'] / 2)

        right_box = [[right_j['imageWidth'], 0], [merge_dict["new_j"]['imageWidth'], 0],
                     [merge_dict["new_j"]['imageWidth'], right_j['imageHeight']],
                     [right_j['imageWidth'], right_j['imageHeight']]]
        right_img_mat = merge_dict["src_img_mat"][0:right_j['imageHeight'], right_j['imageWidth']:merge_dict["new_j"]['imageWidth']]

        # 写入右边图片
        cv2.imwrite(right_img_path, right_img_mat)

        # 找属于右边的点
        right_shapes = list()
        new_right_shapes = copy.deepcopy(merge_dict["new_j"]["shapes"])
        for nr_shape in new_right_shapes:
            nr_points = np.int32(nr_shape["points"])
            r_box = np.int32(right_box)
            r_iou, r_canvRes = polygon_iou(r_box, nr_points)
            if 0 == r_iou:
                # 缺陷在区域外
                continue
            elif 1 == r_iou:
                # 缺陷在区域内
                # 需要减去宽度
                sub_nr_points = [[x[0] - right_j['imageWidth'], x[1]] for x in nr_shape["points"]]
                nr_shape["points"] = sub_nr_points
                right_shapes.append(nr_shape)
            else:
                # 缺陷在交线上
                # 处理之前膨胀下, 防止之前传过来的有空洞
                r_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                r_dilate_canvRes = cv2.dilate(r_canvRes, r_dilate_kernel)
                r_points_contours, r_hierarchy = cv2.findContours(r_dilate_canvRes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                r_inter_points = r_points_contours[0].reshape(-1, 2).tolist()

                # 需要减去宽度
                sub_r_inter_points = [[x[0] - right_j['imageWidth'], x[1]] for x in r_inter_points]
                
                inter_nl_shape = dict()
                inter_nl_shape["label"] = nr_shape["label"]
                inter_nl_shape["points"] = sub_r_inter_points
                inter_nl_shape["group_id"] = nr_shape["group_id"]
                inter_nl_shape["shape_type"] = nr_shape["shape_type"]
                inter_nl_shape["flags"] = nr_shape["flags"]
                right_shapes.append(inter_nl_shape)

        right_j['shapes'] = right_shapes
        right_j["imageData"] = None

        # 写入左边json
        with open(right_json_path, mode="w", encoding="utf-8") as right_out:
            right_out.write(json.dumps(right_j, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}, 写入{right_img_path}失败")


def left_right_cut_main(src_file_dirs, dst_file_dirs):
    """
    读取图片和json进行处理
    """
    src_images = glob.glob(src_file_dirs + "/*.jpg")
    TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format
    pbar = tqdm(src_images, total=len(src_images), bar_format=TQDM_BAR_FORMAT)
    for src_image in pbar:
        src_json = src_image.replace(".jpg", ".json")

        # 图片读取
        try:
            src_img_mat = cv2.imread(src_image)
        except Exception as e:
            print(f"\033[1;31m Error:{e}, 该{src_image}图片读取不到，进行异常处理\033[0m")
            continue
        # json读取
        try:
            with open(src_json, mode="r", encoding="utf-8") as f_src:
                src_j = json.loads(f_src.read())

                new_j = {}
                new_j['version'] = src_j['version']
                new_j['flags'] = src_j['flags']
                new_j['shapes'] = src_j['shapes']
                new_j['imagePath'] = src_j['imagePath']
                new_j['imageData'] = None
                new_j['imageHeight'] = src_j['imageHeight']
                new_j['imageWidth'] = src_j['imageWidth']

                merge_dict = {
                    "src_image_name": src_image,
                    "dst_file_dir": dst_file_dirs,
                    "src_img_mat": src_img_mat,
                    "new_j": new_j
                }

                cut_one(merge_dict)

        except Exception as e:
            print(f"\033[1;34m Error:{e}, 该{src_json}异常，进行异常处理\033[0m")
            continue


if __name__ == '__main__':
    src_file_dirs = "./src_data"
    dst_file_dirs = "./dst_data"
    left_right_cut_main(src_file_dirs, dst_file_dirs)
