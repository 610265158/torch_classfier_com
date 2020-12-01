import sys
sys.path.append('.')
import numpy as np
from pycocotools.coco import COCO
import os

from lib.helper.logger import logger


## read coco data
class CocoMeta_bbox:
    """ Be used in PoseInfo. """


    ##this is the class_ map
    '''klass:(cat_id,'name),
         {0: (1, 'person'), 1: (2, 'bicycle'), 2: (3, 'car'), 3: (4, 'motorcycle'), 4: (5, 'airplane'), 5: (6, 'bus'),
         6: (7, 'train'), 7: (8, 'truck'), 8: (9, 'boat'), 9: (10, 'traffic light'), 10: (11, 'fire hydrant'),
         11: (13, 'stop sign'), 12: (14, 'parking meter'), 13: (15, 'bench'), 14: (16, 'bird'), 15: (17, 'cat'),
         16: (18, 'dog'), 17: (19, 'horse'), 18: (20, 'sheep'), 19: (21, 'cow'), 20: (22, 'elephant'),
         21: (23, 'bear'), 22: (24, 'zebra'), 23: (25, 'giraffe'), 24: (27, 'backpack'), 25: (28, 'umbrella'),
         26: (31, 'handbag'), 27: (32, 'tie'), 28: (33, 'suitcase'), 29: (34, 'frisbee'), 30: (35, 'skis'),
         31: (36, 'snowboard'), 32: (37, 'sports ball'), 33: (38, 'kite'), 34: (39, 'baseball bat'), 35: (40, 'baseball glove'),
         36: (41, 'skateboard'), 37: (42, 'surfboard'), 38: (43, 'tennis racket'), 39: (44, 'bottle'), 40: (46, 'wine glass'),
         41: (47, 'cup'), 42: (48, 'fork'), 43: (49, 'knife'), 44: (50, 'spoon'), 45: (51, 'bowl'),
         46: (52, 'banana'), 47: (53, 'apple'), 48: (54, 'sandwich'), 49: (55, 'orange'),   50: (56, 'broccoli'),
         51: (57, 'carrot'), 52: (58, 'hot dog'), 53: (59, 'pizza'), 54: (60, 'donut'), 55: (61, 'cake'),
         56: (62, 'chair'), 57: (63, 'couch'), 58: (64, 'potted plant'), 59: (65, 'bed'), 60: (67, 'dining table'),
         61: (70, 'toilet'), 62: (72, 'tv'), 63: (73, 'laptop'), 64: (74, 'mouse'), 65: (75, 'remote'),
         66: (76, 'keyboard'), 67: (77, 'cell phone'), 68: (78, 'microwave'), 69: (79, 'oven'), 70: (80, 'toaster'),
         71: (81, 'sink'), 72: (82, 'refrigerator'), 73: (84, 'book'), 74: (85, 'clock'), 75: (86, 'vase'),
         76: (87, 'scissors'), 77: (88, 'teddy bear'), 78: (89, 'hair drier'), 79: (90, 'toothbrush')}

        '''
    def __init__(self, idx, img_url, bbox):
        self.idx = idx
        self.img_url = img_url
        self.img = None
        self.bbox = bbox

        #############reshape the keypoints for coco type,
        ################make the parts in legs unvisible

        #########################


class BoxInfo:
    """ Use COCO for pose estimation, returns images with people only. """

    def __init__(self, image_base_dir, anno_path):
        self.metas = []
        # self.data_dir = data_dir
        # self.data_type = data_type
        self.image_base_dir = image_base_dir
        self.anno_path = anno_path
        self.coco = COCO(self.anno_path)
        self.get_image_annos()
        # self.image_list = os.listdir(self.image_base_dir)

    def get_image_annos(self):
        """Read JSON file, and get and check the image list.
        Skip missing images.
        """
        images_ids = self.coco.getImgIds()
        cats = self.coco.loadCats(self.coco.getCatIds())

        cat_klass_map={}

        for _cat in cats:
            cat_klass_map[_cat['id']]=_cat['name']

        nms = [cat['name'] for cat in cats]
        print('COCO categories: \n{}\n'.format(' '.join(nms)))

        print(cat_klass_map)

        len_imgs = len(images_ids)
        for idx in range(len_imgs):

            images_info = self.coco.loadImgs([images_ids[idx]])
            image_path = os.path.join(self.image_base_dir, images_info[0]['file_name'])
            # filter that some images might not in the list
            # if not os.path.exists(image_path):
            #     print("[skip] json annotation found, but cannot found image: {}".format(image_path))
            #     continue

            annos_ids = self.coco.getAnnIds(imgIds=[images_ids[idx]])
            annos_info = self.coco.loadAnns(annos_ids)



            bboxs=[]
            for ann in annos_info:

                if ann["iscrowd"]:
                    continue
                bbox = ann['bbox']
                cat = ann['category_id']
                klass = nms.index(cat_klass_map[cat])

                if bbox[2]<1 or bbox[3]<1:
                    continue

                bboxs.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], klass])

            if len(bboxs) > 0:
                tmp_meta = CocoMeta_bbox(images_ids[idx], image_path, bboxs)
                self.metas.append(tmp_meta)

            # sort from the biggest person to the smallest one

        logger.info("Overall get {} valid images from {} and {}".format(
            len(self.metas), self.image_base_dir, self.anno_path))

    def load_images(self):
        pass

    def get_image_list(self):
        img_list = []
        for meta in self.metas:
            img_list.append(meta.img_url)
        return img_list

    def get_bbox(self):
        box_list = []
        for meta in self.metas:
            box_list.append(meta.bbox)
        return box_list

#############bbox example
coco_ann_path = '/Users/liangzi/lz/sina/pubdata/mscoco/annotations/instances_val2017.json'
coco_img_path = '/Users/liangzi/lz/sina/pubdata/mscoco/val2017'
coco_box = BoxInfo(coco_img_path, coco_ann_path)





def expand_box(image,box):


    h,w,_=image.shape
    width=(box[2]-box[0])
    height=box[3]-box[1]

    center_x=(box[2]+box[0])//2
    center_y = (box[3] + box[1]) // 2

    x1=center_x-width*1.8/2
    y1 = center_y - height * 1.8 / 2
    x2 = center_x + width * 1.8 / 2
    y2 = center_y + height * 1.8 / 2

    x1= x1 if  x1 > 0 else 0
    y1 = y1 if y1 > 0 else 0
    x2 = x2 if x2 <w  else w-1
    y2 = y2 if y2 <h  else h-1

    return [x1,y1,x2,y2,box[4]]

def area(box):
    return (box[2]-box[0])*(box[3]-box[1])
import cv2


import time

if not os.access('traffic_tools' ,os.F_OK):

    os.mkdir('traffic_tools')


for meta in coco_box.metas:
    fname, bboxs = meta.img_url, meta.bbox

    image = cv2.imread(fname)
    image_show = image.copy()

    for bbox in bboxs:


        newbox=expand_box(image,bbox)

        newbox=[int(x) for x in newbox]
        if newbox[4] <= 8 and newbox[4] >= 1:

            croped_image=image[newbox[1]:newbox[3],newbox[0]:newbox[2],:]

            if area(newbox)>7200:
                if newbox[4]==1:
                    cur_dir='./traffic_tools/bicycle'
                    if not os.access(cur_dir,os.F_OK):
                        os.mkdir(cur_dir)

                    save_path=os.path.join(cur_dir,str(time.time())+'.jpg')

                    cv2.imwrite(save_path,croped_image)

                if newbox[4] == 2:
                    cur_dir = './traffic_tools/car'
                    if not os.access(cur_dir, os.F_OK):
                        os.mkdir(cur_dir)

                    save_path = os.path.join(cur_dir, str(time.time()) + '.jpg')

                    cv2.imwrite(save_path, croped_image)
                if newbox[4] == 3:
                    cur_dir = './traffic_tools/motorcycle'
                    if not os.access(cur_dir, os.F_OK):
                        os.mkdir(cur_dir)

                    save_path = os.path.join(cur_dir, str(time.time()) + '.jpg')

                    cv2.imwrite(save_path, croped_image)

                if newbox[4] == 4:
                    cur_dir = './traffic_tools/airplane'
                    if not os.access(cur_dir, os.F_OK):
                        os.mkdir(cur_dir)

                    save_path = os.path.join(cur_dir, str(time.time()) + '.jpg')

                    cv2.imwrite(save_path, croped_image)

                if newbox[4] == 5:
                    cur_dir = './traffic_tools/bus'
                    if not os.access(cur_dir, os.F_OK):
                        os.mkdir(cur_dir)

                    save_path = os.path.join(cur_dir, str(time.time()) + '.jpg')

                    cv2.imwrite(save_path, croped_image)

                if newbox[4] == 6:
                    cur_dir = './traffic_tools/train'
                    if not os.access(cur_dir, os.F_OK):
                        os.mkdir(cur_dir)

                    save_path = os.path.join(cur_dir, str(time.time()) + '.jpg')

                    cv2.imwrite(save_path, croped_image)


                if newbox[4] == 7:
                    cur_dir = './traffic_tools/truck'
                    if not os.access(cur_dir, os.F_OK):
                        os.mkdir(cur_dir)

                    save_path = os.path.join(cur_dir, str(time.time()) + '.jpg')

                    cv2.imwrite(save_path, croped_image)

                if newbox[4] == 8:
                    cur_dir = './traffic_tools/boat'
                    if not os.access(cur_dir, os.F_OK):
                        os.mkdir(cur_dir)

                    save_path = os.path.join(cur_dir, str(time.time()) + '.jpg')

                    cv2.imwrite(save_path, croped_image)



                # cv2.imshow('tmp', croped_image)
                # cv2.waitKey(0)
    #         cv2.rectangle(image_show, (int(bbox[0]), int(bbox[1])),
    #                       (int(bbox[2]), int(bbox[3])), (255, 0, 0), 7)
    #
    # cv2.imshow('tmp', image_show)
    # cv2.waitKey(0)


