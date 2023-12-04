# 根据输入的图片大小， 切割大小， shave等进行切割操作
import cv2
import json
import numpy as np
import os
import base64
import glob


def polygon2mask(imgsz, polygons, downsample_ratio=1, color=1):
    """
    Args:
        imgsz (tuple): The image size.
        polygons (np.ndarray): [N, M], N is the number of polygons, M is the number of points(Be divided by 2).
        color (int): color
        downsample_ratio (int): downsample ratio
    """
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons)
    polygons = polygons.astype(np.int32)
    shape = polygons.shape
    polygons = polygons.reshape(shape[0], -1, 2)
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)
    # NOTE: fillPoly firstly then resize is trying the keep the same way
    # of loss calculation when mask-ratio=1.
    mask = cv2.resize(mask, (nw, nh))
    return mask


def polygons2masks_overlap(imgsz, segments, downsample_ratio=1):
    """
    返回 imgsz overlap mask
    """
    masks = np.zeros((imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),
                     dtype=np.int32 if len(segments) > 255 else np.uint8)
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(imgsz, [segments[si].reshape(-1)], downsample_ratio=downsample_ratio, color=1)
        ms.append(mask)
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        # mask = ms[i] * (i + 1)
        mask = ms[i]
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index


def format_segments(segment_labels, img_w, img_h, mask_ratio=1):
    """
    将重采样后的分割点按照类别转成masks 方便后续分割
    """
    format_segs = list()
    for segment_label in segment_labels.keys():
        segment_label_key = segment_label
        segment_label_value = segment_labels[segment_label_key]
        segment_label_value = np.stack(segment_label_value, axis=0)
        masks, sorted_idx = polygons2masks_overlap((img_h, img_w), segment_label_value, downsample_ratio=mask_ratio)

        label_mask = {"label": segment_label_key,
                      "masks": masks,
                      "mask_sorted_idx": sorted_idx}
        format_segs.append(label_mask)
    return format_segs


def resample_segments(chop_batch, n=1000):
    """
    对分割点进行重采样(将少数的点采样到1000个， 方便后续分割)
    """
    # 将同类别的segment cat到一起
    segment_label = dict()
    for cls_shape in chop_batch["new_j"]["shapes"]:  # 统计多少个类别
        cls_label = cls_shape["label"]
        if cls_label not in segment_label.keys():
            segment_label[cls_label] = list()

    for shape in chop_batch["new_j"]["shapes"]:
        "处理points点"
        segment = np.array(shape["points"])  # 转成numpy类型处理
        # 重采样处理
        s = np.concatenate((segment, segment[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segment = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)],
                                     dtype=np.float32).reshape(2, -1).T  # segment xy
        shape["segment"] = segment
        shape_label = shape["label"]
        if shape_label in segment_label.keys():
            segment_label[shape_label].append(segment)
    return segment_label


def _chop_batch(ori_img, chop_size=(640, 640), shave=10):
    """
                        根据原图大小，和切割大小得到切割份数，切割坐标，以及恢复到原图坐标点
                          -------->(x, ori_img_w)
                          |
                          |
                          |(y, ori_img_h)
                        """
    import math
    ori_img_h, ori_img_w = ori_img.shape[:2]
    x_chop_nums = math.ceil(ori_img_w / chop_size[0])
    y_chop_nums = math.ceil(ori_img_h / chop_size[1])
    # 按照先行 后列进行切割
    chop_img_batch = list()
    for y_index in range(y_chop_nums):
        for x_index in range(x_chop_nums):
            # 最左边进行切割
            if x_index == x_chop_nums - 1 and y_index != y_chop_nums - 1:
                # 切割坐标
                chop_xmax = ori_img_w
                chop_xmin = ori_img_w - chop_size[0] - shave
                chop_ymax = y_index * chop_size[1] + chop_size[1] + shave
                chop_ymin = y_index * chop_size[1]
                restore_coord = (chop_xmin, chop_ymin)  # 恢复坐标
                chop_img = ori_img[chop_ymin:chop_ymax, chop_xmin:chop_xmax]
                chop_img_dict = {"chop_img": chop_img, "restore_coord": restore_coord}
                chop_img_batch.append(chop_img_dict)
            # 最下边进行切割
            if y_index == y_chop_nums - 1 and x_index != x_chop_nums - 1:
                # 切割坐标
                chop_ymax = ori_img_h
                chop_ymin = ori_img_h - chop_size[1] - shave
                chop_xmax = x_index * chop_size[0] + chop_size[0] + shave
                chop_xmin = x_index * chop_size[0]
                # 恢复坐标
                restore_coord = (chop_xmin, chop_ymin)
                chop_img = ori_img[chop_ymin:chop_ymax, chop_xmin:chop_xmax]
                chop_img_dict = {"chop_img": chop_img, "restore_coord": restore_coord}
                chop_img_batch.append(chop_img_dict)
            # 最左下角区域进行切割
            if x_index == x_chop_nums - 1 and y_index == y_chop_nums - 1:
                # 切割坐标
                chop_xmax = ori_img_w
                chop_xmin = ori_img_w - chop_size[0] - shave
                chop_ymax = ori_img_h
                chop_ymin = ori_img_h - chop_size[1] - shave
                # 恢复坐标
                restore_coord = (chop_xmin, chop_ymin)
                chop_img = ori_img[chop_ymin:chop_ymax, chop_xmin:chop_xmax]
                chop_img_dict = {"chop_img": chop_img, "restore_coord": restore_coord}
                chop_img_batch.append(chop_img_dict)
            elif x_index != x_chop_nums - 1 and y_index != y_chop_nums - 1:
                # 非特殊区域
                chop_xmin = x_index * chop_size[0]
                chop_xmax = x_index * chop_size[0] + chop_size[0] + shave
                chop_ymin = y_index * chop_size[1]
                chop_ymax = y_index * chop_size[1] + chop_size[1] + shave
                # 恢复坐标
                restore_coord = (chop_xmin, chop_ymin)  # 第一个维度是y或者高度
                chop_img = ori_img[chop_ymin:chop_ymax, chop_xmin:chop_xmax]
                chop_img_dict = {"chop_img": chop_img, "restore_coord": restore_coord}
                chop_img_batch.append(chop_img_dict)
    return chop_img_batch


def chop_imgs_labels(chop_batch, format_segs):
    """
    切割图片和标签，并把标签组成需要格式
    """
    # todo: chop_size, shave后续写成输入参数
    chop_img_batch = _chop_batch(chop_batch["src_img_mat"], chop_size=(640, 640), shave=0)  # 切割得到每一小份的img
    chop_label_choose = [list() for _ in range(len(chop_img_batch))]
    for format_seg in format_segs:
        chop_label_batch = _chop_batch(format_seg["masks"], chop_size=(640, 640), shave=0)  # 切割得到每一小份的masks

        # 找到小块的缺陷区域，并集合到一起
        for chop_label_index, chop_label in enumerate(chop_label_batch):
            if chop_label["chop_img"].sum() > 0:  # 选择有缺陷的区域
                contours, hierarchy = cv2.findContours(chop_label["chop_img"], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = [np.reshape(contours[i], (len(contours[i]), -1)) for i in range(len(contours))]
                contours_info = {"label": format_seg["label"], "contours": contours,
                                 "hierarchy": hierarchy, "coord": chop_label["restore_coord"]}
                chop_label_choose[chop_label_index].append(contours_info)

    # 正式开始切割图片，标签
    for chop_choose_index, chop_choose in enumerate(chop_label_choose):
        if len(chop_choose) == 0:  # 该块没有标注 跳过
            continue
        # 有标注 进行处理图片和标注
        base_img_name = os.path.basename(chop_batch["src_image_name"])[:-4]
        chop_img_name = base_img_name + str(chop_choose_index) + os.path.basename(chop_batch["src_image_name"])[-4:]
        chop_img_path = os.path.join(chop_batch["dst_file_dir"], chop_img_name)
        chop_img_mat = chop_img_batch[chop_choose_index]["chop_img"]
        # 写入图片
        try:
            cv2.imwrite(chop_img_path, chop_img_mat)
        except Exception as e:
            print(f"Error: {e}, 写入{chop_img_path}失败")
        # 写入json
        chop_j = {}
        shapes = list()
        for choosed in chop_choose:
            shape_dict = dict()
            choosed_label = choosed["label"]
            for choosed_contours in choosed["contours"]:
                shape_dict["label"] = choosed_label
                shape_dict["points"] = choosed_contours.tolist()
                shape_dict["group_id"] = None
                shape_dict["shape_type"] = "polygon"
                shape_dict["flags"] = dict()
            shapes.append(shape_dict)
        chop_j["version"] = chop_batch["new_j"]["version"]
        chop_j["flags"] = chop_batch["new_j"]["flags"]
        chop_j["shapes"] = shapes
        chop_j["imagePath"] = chop_img_name
        f = open(chop_img_path, "rb")
        base64_encode = base64.b64encode(f.read()).decode('utf-8')
        chop_j["imageData"] = base64_encode
        chop_j["imageHeight"] = chop_img_mat.shape[:2][0]
        chop_j["imageWidth"] = chop_img_mat.shape[:2][1]

        chop_json_path = chop_img_path.replace(".jpg", ".json")
        with open(chop_json_path, mode="w", encoding="utf-8") as out:
            out.write(json.dumps(chop_j, ensure_ascii=False))  # 目标json

        print(f"{chop_img_path} is done")


def chop_batch_img_json(chop_batch):
    """
    将图片和json进行切开
    """
    # 第一步 对shapes里面的点进行重采样, 得到同一个类别的分割点放在一起
    segment_labels = resample_segments(chop_batch)

    # 第二步 对重采样后的1000个分割点按照类别转成masks
    img_h, img_w = chop_batch["new_j"]["imageHeight"], chop_batch["new_j"]["imageWidth"]  # 图片宽高
    format_segs = format_segments(segment_labels, img_w, img_h)

    # 第三步 进行切割图片和标签
    chop_imgs_labels(chop_batch, format_segs)


def chop_cut_main(src_file_dir="", dst_file_dir="", chop_size=(640, 640), shave=10):
    """
    读取src目录下的图片和json
    """
    src_images = glob.glob(src_file_dir + "/*.jpg")
    for src_image in src_images:
        src_json = src_image.replace(".jpg", ".json")

        try:
            src_img_mat = cv2.imread(src_image)  # 读取图片并做异常处理
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
                new_j['imageData'] = src_j['imageData']
                new_j['imageHeight'] = src_j['imageHeight']
                new_j['imageWidth'] = src_j['imageWidth']

                # 将读取好的图片和json进行处理
                chop_batch = {"src_image_name": src_image,
                              "dst_file_dir": dst_file_dir,
                              "chop_size": chop_size,
                              "shave": shave,
                              "src_img_mat": src_img_mat,
                              "new_j": new_j,
                              }
                chop_batch_img_json(chop_batch)

        except Exception as e:
            print(f"\033[1;34m Error:{e}, 该{src_json}异常，进行异常处理\033[0m")
            continue


if __name__ == '__main__':
    src_file_dir = r"../Images"
    dst_file_dir = r"../dst_data"
    chop_cut_main(src_file_dir=src_file_dir, dst_file_dir=dst_file_dir)
