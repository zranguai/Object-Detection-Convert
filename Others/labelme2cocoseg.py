import json
import glob
import PIL.Image
import PIL.ImageDraw
import os
import base64
import io
import numpy as np
import PIL.ExifTags
import PIL.Image
import PIL.ImageOps
import shutil


def img_b64_to_arr(img_b64):
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    img_arr = np.array(PIL.Image.open(f))
    return img_arr


def img_arr_to_b64(img_arr):
    img_pil = PIL.Image.fromarray(img_arr)
    f = io.BytesIO()
    img_pil.save(f, format='PNG')
    img_bin = f.getvalue()
    if hasattr(base64, 'encodebytes'):
        img_b64 = base64.encodebytes(img_bin)
    else:
        img_b64 = base64.encodestring(img_bin)
    return img_b64


def img_data_to_png_data(img_data):
    with io.BytesIO() as f:
        f.write(img_data)
        img = PIL.Image.open(f)

        with io.BytesIO() as f:
            img.save(f, 'PNG')
            f.seek(0)
            return f.read()


def apply_exif_orientation(image):
    try:
        exif = image._getexif()
    except AttributeError:
        exif = None

    if exif is None:
        return image

    exif = {
        PIL.ExifTags.TAGS[k]: v
        for k, v in exif.items()
        if k in PIL.ExifTags.TAGS
    }

    orientation = exif.get('Orientation', None)

    if orientation == 1:
        # do nothing
        return image
    elif orientation == 2:
        # left-to-right mirror
        return PIL.ImageOps.mirror(image)
    elif orientation == 3:
        # rotate 180
        return image.transpose(PIL.Image.ROTATE_180)
    elif orientation == 4:
        # top-to-bottom mirror
        return PIL.ImageOps.flip(image)
    elif orientation == 5:
        # top-to-left mirror
        return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_270))
    elif orientation == 6:
        # rotate 270
        return image.transpose(PIL.Image.ROTATE_270)
    elif orientation == 7:
        # top-to-right mirror
        return PIL.ImageOps.mirror(image.transpose(PIL.Image.ROTATE_90))
    elif orientation == 8:
        # rotate 90
        return image.transpose(PIL.Image.ROTATE_90)
    else:
        return image


class labelme2coco(object):
    def __init__(self, labelme_json=[], save_json_path='./new.json'):

        """
        Args: labelme_json: paths of labelme json files
        : save_json_path: saved path
        """

        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        # self.data_coco = {}
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0

        self.save_json()

    def data_transfer(self):
        for num, json_file in enumerate(self.labelme_json):
            with open(json_file, 'r') as fp:
                print("json_name=", json_file)
                data = json.load(fp)
                (prefix, res) = os.path.split(json_file)
                (file_name, extension) = os.path.splitext(res)
                self.images.append(self.image(data, num, file_name))
                for shapes in data['shapes']:
                    # try:  # todo: 3-7号debug, 解决在转换动态切图要try打开，sahi时关闭
                    label = shapes['label']
                    if label not in self.label:
                        self.categories.append(self.categorie(label))
                        self.label.append(label)
                    points = shapes['points']
                    self.annotations.append(self.annotation(points, label, num))
                    self.annID += 1
                    # except:
                    #     print(f"{shapes}发生异常")
                    #     continue

    def image(self, data, num, file_name):
        image = {}
        img = img_b64_to_arr(data['imageData'])

        height, width = img.shape[:2]
        img = None
        image['height'] = height
        image['width'] = width
        image['id'] = int(num + 1)
        image['file_name'] = file_name + '.jpg'

        self.height = height
        self.width = width

        return image

    def categorie(self, label):
        categorie = {}
        categorie['supercategory'] = label
        categorie['id'] = int(len(self.label) + 1)
        categorie['name'] = label

        return categorie

    def annotation(self, points, label, num):
        annotation = {}
        annotation['iscrowd'] = 0
        annotation['image_id'] = int(num + 1)

        annotation['bbox'] = list(map(float, self.getbbox(points)))

        point = []
        for p in points:
            point.append(p[0])
            point.append(p[1])

        annotation['segmentation'] = [point]  # at least 6 points

        annotation['category_id'] = self.getcatid(label)
        annotation['id'] = int(self.annID)
        # add area info
        annotation['area'] = self.height * self.width  # the area is not used for detection
        return annotation

    def getcatid(self, label):
        for categorie in self.categories:

            if label == categorie['name']:
                return categorie['id']

        return -1

    def getbbox(self, points):
        # img = np.zeros([self.height,self.width],np.uint8)
        # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA)
        # cv2.fillPoly(img, [np.asarray(points)], 1)
        polygons = points
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]

        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        return [left_top_c, left_top_r, right_bottom_c - left_top_c,
                right_bottom_r - left_top_r]  # [x1,y1,w,h] for coco_cut1 box format

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        categoryName = open("categoryName.txt", 'w')

        for i in self.categories:
            categoryName.write(i['name'] + '\n')

        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()

        json.dump(self.data_coco, open(self.save_json_path, 'w', encoding='utf-8'), indent=4, separators=(',', ': '),
                  cls=MyEncoder)


# type check when save json files
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def copyImage(imgdir):
    # imgdir = './image/'
    # imgdir = './dst_data/'
    # imgdir = '..//data//luhou/'
    # imgdir = './src_data/'
    for file in os.listdir(imgdir):
        if file.endswith('.jpg'):
            shutil.copyfile(imgdir + file, './coco-luhou/train/' + file)


def copyCategory():
    shutil.copyfile('categoryName.txt', "E:\\detectron2-master_cuda101_vs2019_win10\\train_detector2\\categoryName.txt")


if __name__ == '__main__':
    # labelme_json = glob.glob('./image/*.json')
    # if not os.path.exists("./coco_cut1/annotations/"):
    #     os.makedirs('./coco_cut1/annotations/')
    # if not os.path.exists('./coco_cut1/train/'):
    #     os.makedirs('./coco_cut1/train/')
    # copyImage()
    # data-convert(labelme_json, './coco_cut1/annotations/train.json')
    # copyCategory()

    # labelme_json = glob.glob('./dst_data/*.json')
    labelme_json = glob.glob('./luhou-datasets/*.json')  # json路径
    if not os.path.exists("coco-luhou/annotations/"):
        os.makedirs('coco-luhou/annotations/')
    if not os.path.exists('coco_luhou/train/'):
        os.makedirs('coco-luhou/train/')

    imgdir = './luhou-datasets/'  # 图片路径
    copyImage(imgdir)
    labelme2coco(labelme_json, 'coco-luhou/annotations/train_coco.json')
    # copyCategory()
