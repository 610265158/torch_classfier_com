# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.09.09
""" python demo usage about MNN API """
import sys
sys.path.append('.')

import numpy as np
import MNN
import cv2
import os


from train_config import config as cfg

def preprocess( image, target_height, target_width, label=None):
    ###sometimes use in objs detects
    h, w, c = image.shape

    bimage = np.zeros(shape=[target_height, target_width, c], dtype=image.dtype)

    scale_y = target_height / h
    scale_x = target_width / w

    scale = min(scale_x, scale_y)

    image = cv2.resize(image, None, fx=scale, fy=scale)

    h_, w_, _ = image.shape

    dx = (target_width - w_) // 2
    dy = (target_height - h_) // 2
    bimage[dy:h_ + dy, dx:w_ + dx, :] = image

    return bimage, scale, scale, dx, dy



def inference(mnn_model_path,img_dir,thres=0.3):
    """ inference mobilenet_v1 using a specific picture """
    interpreter = MNN.Interpreter(mnn_model_path)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)

    img_list=os.listdir(img_dir)
    for pic in img_list:
        image = cv2.imread(os.path.join(img_dir,pic))
        #cv2 read as bgr format
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        image = cv2.resize(image,(224,224))
        #change to rgb format


        image_show=image.copy()

        image = image.astype(np.float32)

        tmp_input = MNN.Tensor((1, 224,224,3 ), MNN.Halide_Type_Float,\
                        image, MNN.Tensor_DimensionType_Tensorflow)
        #construct tensor from np.ndarray
        input_tensor.copyFrom(tmp_input)

        ### caution!!!!!!!!!!!!!!!! the model is nhwc

        interpreter.resizeSession(session)
        interpreter.runSession(session)

        output_tensor = interpreter.getSessionOutputAll(session)
        output=output_tensor['output'].getData()
        print(output)
        cv2.imshow('mnn result',image_show)
        cv2.waitKey(0)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mnn_model', type=str, default='./centernet.mnn', help='the mnn model ', required=False)
    parser.add_argument('--imgDir', type=str, default='../pubdata/mscoco/val2017', help='the image dir to detect')
    parser.add_argument('--thres', type=float, default=0.3, help='the thres for detect')
    args = parser.parse_args()

    data_dir = args.imgDir
    model_path=args.mnn_model
    thres=args.thres
    inference(model_path,data_dir,thres)
