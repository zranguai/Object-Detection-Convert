"""
+ 天池比赛: mmdetection中将目标检测的结果转换成json格式方便提交
1. 输入为待检测目录dir,config(配置文件), checkpoint(训练好的权重文件)
2. 向其中有image信息但是没有annotation信息的里面写入annotation信息
"""
import os
import json
from argparse import ArgumentParser

from mmdet.apis import init_detector, inference_detector, show_result_pyplot


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('dir', help='Image file dir')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    # parser.add_argument(
    #     '--score-thr', type=float, default=0.3, help='bbox score threshold')

    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()
    return args


def single_result(args, img_path, img_id, annotations):
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, img_path)
    # show the results
    # show_result_pyplot(model, img_path, result, score_thr=args.score_thr)  # 是否可视化结果

    # add to json
    for cls_id, cls in enumerate(result):
        if len(cls) == 0:
            continue
        else:
            for num_cls in cls:

                if num_cls[-1] < args.score_thr:
                    continue
                else:
                    # 将坐标的(x1, y1, x2, y2) 转换为 (x1, y1, w, h)
                    x1 = num_cls[0]
                    y1 = num_cls[1]
                    x2 = num_cls[2]
                    y2 = num_cls[3]
                    w = x2 - x1
                    h = y2 - y1
                    num_cls[:-1] = [x1, y1, w, h]
                    # 写成字典形式 --> 因为cls_id是从0开始所以需要+1
                    dic=dict(image_id=img_id, category_id=cls_id + 1, bbox=[int(i) for i in num_cls[:-1]], score=float(num_cls[-1]))
                    annotations.append(dic)


def main(args):
    annotations = []

    # load json
    with open("/home/zranguai/Cv-Code/detection/mmdetection-2.17.0/data/coco/demo-anno/demo-tianchi.json", "r") as load_f:
        load_dict = json.load(load_f)
        images = load_dict["images"]


    for img in images:
        img_path = os.path.join(args.dir, img["file_name"])
        img_id = img["id"]
        single_result(args, img_path, img_id, annotations)
    #
    # # 将数据写入json文件中
    with open("/home/zranguai/Cv-Code/detection/mmdetection-2.17.0/data/coco/demo-anno/demo-tianchi.json", "w") as f:
        annotations = dict(annotations=annotations)
        load_dict.update(annotations)
        json.dump(load_dict, f, ensure_ascii=False)
        print("加载json文件完成")


if __name__ == '__main__':
    args = parse_args()
    main(args)
