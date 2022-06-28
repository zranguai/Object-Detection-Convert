import sys
import os
import json
import base64
import cv2
import numpy as np


if len(sys.argv) != 3:
    print('Usage:\n\tpython ' + sys.argv[0] + ' [src json directory] [dst json directory]')
    sys.exit(1)

src_dir = sys.argv[1]
dst_dir = sys.argv[2]

if not os.path.exists(src_dir):
    print('ERROR: src directory does not exist')
    sys.exit(1)

if os.path.exists(dst_dir):
    print('ERROR: dst directory already exists, please use another directory')
    sys.exit(1)

for name in os.listdir(src_dir):
    if not name.endswith('.json'):
        continue

    src_path = os.path.join(src_dir, name)
    dst_path = os.path.join(dst_dir, name)

    with open(src_path, mode='r', encoding='utf-8') as f:
        j = json.loads(f.read())

        # img_bin = base64.b64decode(j['imageData'])
        # img = cv2.imdecode(np.frombuffer(img_bin, dtype=np.uint8), cv2.IMREAD_COLOR)
        # cv2.imwrite('test.jpg', img)

        new_j = {}
        new_j['version'] = j['version']
        new_j['flags'] = j['flags']
        new_j['shapes'] = []
        new_j['imagePath'] = j['imagePath']
        new_j['imageData'] = j['imageData']
        new_j['imageHeight'] = j['imageHeight']
        new_j['imageWidth'] = j['imageWidth']

        shapes = j['shapes']
        for item in shapes:
            shape_type = item['shape_type']
            label = item['label']
            points = item['points']

            if label in ('Tennis', 'Audience', 'Coach'): # 自己添加其他类别
                continue

            new_j['shapes'].append(item) # 或者新建一个dict()，把需要的元素塞进去，再append到new_j['shapes']里面。如果需要，可以添加复杂的逻辑，比如把点和矩形关联起来，再 append
        
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        with open(dst_path, mode='w', encoding='utf-8') as out:
            out.write(json.dumps(new_j, ensure_ascii=False))

print('done')
