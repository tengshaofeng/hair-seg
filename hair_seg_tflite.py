import glob
import os
import sys
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
# from src.models.modnet_tbq import MODNet
import time
import skimage.measure as measure
import tensorflow as tf
import tqdm
print(tf.__version__)
import io
# from mtcnn.mtcnn import MTCNN
# mtcnn_detector = MTCNN()
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from models.mtcnn_pytorch.src.detector import FaceDetectModel
detect_model = FaceDetectModel()
atts = {'background':0, 'skin': 1, 'left_brow': 2, 'right_brow': 2, 'left_eye': 3, 'right_eye': 3, 'left_ear': 4, 'right_ear': 4, 'ear_ring': 5,
        'nose': 6, 'mouth': 7, 'up_lip': 8, 'lower_lip': 8, 'hair': 9}


def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    # part_colors_19 = [[255, 0, 0], [255, 85, 0], [255, 0, 85],
    #                [255, 0, 85], [0, 255, 0],
    #                [0, 255, 0], [85, 255, 0], [170, 255, 0],
    #                [170, 255, 0], [0, 255, 170],
    #                [0, 0, 255], [0, 0, 0], [170, 0, 255],
    #                [0, 85, 255], [0, 170, 255],
    #                [255, 255, 0], [255, 255, 85], [255, 255, 170],
    #                [255, 0, 255], [255, 85, 255], [255, 170, 255],
    #                [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    part_colors = [[255, 0, 0], [255, 85, 0], [255, 0, 85],
                      [0, 255, 0], [0, 0, 255],
                      [0, 0, 255], [0, 0, 0], [170, 0, 255],
                      [0, 85, 255], [0, 170, 255],
                      [255, 255, 0], [255, 255, 85], [255, 255, 170],
                      [255, 0, 255], [255, 85, 255], [255, 170, 255],
                      [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    # atts = [background, 'skin', 'left_brow', 'right_brow', 'left_eye', 'right_eye', 'eye_glass', 'left_ear', 'right_ear', 'ear_ring',
    #         'nose', 'mouth', 'up_lip', 'lower_lip', 'neck', 'neck_lace', 'cloth', 'hair', 'hat']
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)

    # # tbq resize start
    # vis_parsing_anno = cv2.resize(vis_parsing_anno.astype(np.uint8), (1024, 1024), cv2.INTER_NEAREST)
    # print('nearest')
    # vis_im = cv2.resize(vis_im.astype(np.uint8), (1024, 1024), cv2.INTER_NEAREST)
    # # tbq resize end
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im_out = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        # cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im_out, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        cv2.imwrite(save_path[:-4] + '_org.png', vis_im[:,:,::-1])

class FaceParsingModel(object):
    # Data transforms
    db_features = []
    db_names = []
    mean = [0.5, 0.5, 0.5]
    stdv = [0.5, 0.5, 0.5]
    im_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def __init__(self, **kwargs):
        model_path = kwargs.get('model_path')
        self.save_path = kwargs.get('save_path')
        # Load the TFLite model and allocate tensors
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Test the model on random input data
        self.input_shape = self.input_details[0]['shape']
        self.output_details[0]['shape'] = np.array([1, 1, 512, 512]).astype(np.int32)
        self.output_details[0]['shape_signature'] = np.array([1, 1, 512, 512]).astype(np.int32)


    def predict(self, image):
        """
        get a feature representation given an face image tensor
        :param image: pil image
        :return: a scalar
        """
        tensor_images = self.preprocess(image)
        # tensor_images = np.load('tbq.npy')  # tbq debug
        ### get face features
        self.interpreter.set_tensor(self.input_details[0]['index'], tensor_images)  # tensor_images[0].unsqueeze(0)
        # print('tensor_images[0].unsqueeze(0).shape:', tensor_images[0].unsqueeze(0).shape)
        # print('self.input_details[0]shape:', self.input_details[0]['shape'])
        # self.interpreter.resize_tensor_input(self.input_details[0]['index'], [1, 3, 512, 512])
        self.interpreter.invoke()
        # get_tensor() returns a copy of the tensor data
        # use tensor() in order to get a pointer to the tensor
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data


    def preprocess(self, image):
        """
        process the data given an image buff
        :param image: type of pil image
        :return: torch tensor
        """
        # unify image channels to 3
        im = np.asarray(image)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        im_rw, im_rh = 512, 512
        im = cv2.resize(im, (im_rw, im_rh), cv2.INTER_CUBIC)

        # convert image to PyTorch tensor
        im = Image.fromarray(im)
        im = self.im_transform(im)

        # add mini-batch dim
        im_tensor = im[None, :, :, :]  # im = im[None, :, :, :]
        return im_tensor

def face_crop(im_pil):
    wid, hei = im_pil.size
    dets = mtcnn_detector.detect_faces(np.array(im_pil))  # cv2_frame:  rgb
    if len(dets) < 1:
        assert "No face detected!"
    det = dets[0]
    x, y, width, height = det['box']

    # bounding_boxes, landmarks = detect_model.detect_faces(im_pil, min_face_size=30)
    # index = 0
    # x0, y0, x1, y1 = [int(t) for t in bounding_boxes[index][:4]]  # x0, y0 ,x1, y1
    x0, y0, x1, y1 = int(x), int(y), int(x+width), int(y+height)
    w, h = x1 - x0, y1 - y0
    c_x, c_y = (x0 + x1) // 2, (y0 + y1) // 2
    x0_new = max(0, c_x - w)
    y0_new = max(0, c_y - h)
    x1_new = min(wid, c_x + w)
    y1_new = min(hei, c_y + h)
    w_new = x1_new - x0_new
    h_new = y1_new - y0_new
    dif  = abs(w_new - h_new)
    if w_new > h_new:
        x0_new += dif // 2
        x1_new -= dif // 2
    else:
        y0_new += dif // 2
        y1_new -= dif // 2

    im_pil = im_pil.crop([x0_new, y0_new, x1_new, y1_new])
    im_pil = im_pil.resize((512, 512), Image.ANTIALIAS)
    return im_pil

def get_outsize(im_pil, short=500):
    w, h = im_pil.size
    if min(h, w) < short:
        return w, h
    if w > h:
        th = short
        tw = int(w / h * th)
    else:
        tw = short
        th = int(h / w * tw)
    return tw, th


def exapand_box(im_pil, box):
    w, h = im_pil.size
    x0, y0, x1, y1 = box
    box_w = x1 - x0
    box_h = y1 - y0
    base = max(box_h, box_w)
    x0 = x0 - base * 1.5  # x0 - box_w * 1.5
    x1 = x1 + base * 1.5  # x1 + box_w * 1.5
    y0 = y0 - base  # y0 - box_h
    y1 = y1 + base * 2.2  # y1 + box_h * 2
    x0 = max(0, x0)
    x1 = min(w, x1)
    y0 = max(0, y0)
    y1 = min(h, y1)
    return int(x0), int(y0), int(x1), int(y1)


def face_crop_for_hair(im_pil):
    # 缩小后检测节省时间
    w, h = im_pil.size
    tw, th = get_outsize(im_pil, short=630)
    tmp_img = im_pil.resize((tw, th))  # 原图等比放缩后的图
    bounding_boxes, landmarks = detect_model.detect_faces(tmp_img)
    if len(bounding_boxes) < 1 or bounding_boxes[0][-1] < 0.9:  # 未检测到人脸忽略
        assert "No face detected!"
    box = bounding_boxes[0]
    rect = [int(t * (w / tw)) for t in box[:4]]  # x0,y0,x1,y1
    x0, y0, x1, y1 = exapand_box(im_pil, rect)
    img_tmp = im_pil.crop((x0, y0, x1, y1))
    # tw_box, th_box = get_outsize(img_tmp, short=512)
    # img_tmp = img_tmp.resize((tw_box, th_box), Image.ANTIALIAS)
    # img_tmp = img_tmp.crop([0, 0, 512, 512])
    img_tmp = img_tmp.resize((512, 512), Image.ANTIALIAS)
    return img_tmp, [x0, y0, x1, y1]


def get_hair_mask(im_pil):
    image, crop_info = face_crop_for_hair(im_pil)  # 512x512 头发分割2分类
    out = model.predict(image)
    parsing = out * 255  # .squeeze(0).argmax(0)
    res_mask = Image.new('L', im_pil.size, 0)
    crop_w = crop_info[2] - crop_info[0]
    crop_h = crop_info[3] - crop_info[1]
    res_mask.paste(Image.fromarray(parsing).resize((crop_w, crop_h), Image.ANTIALIAS), crop_info)
    # cv2.imwrite('tbq1.png', out * 255)
    return res_mask  # 对应原图im_pil的0~255 的mask,

ref_size = 512
ckpt_path = 'hair_seg.tflite'  # ../pretrained_models/
model = FaceParsingModel(model_path=ckpt_path)
print('load from:', ckpt_path)

if __name__ == '__main__':

    # random_sample()  # tbq debug
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, help='path of input images')
    parser.add_argument('--output-path', type=str, help='path of output images')
    parser.add_argument('--ckpt-path', type=str, help='path of pre-trained MODNet')
    args = parser.parse_args()

    # args.tflite_path = 'hair_seg.tflite' # 'face_parsing_cls10.tflite'
    # print(args.tflite_path)
    args.input_path = '/home/ateam/code/MODNet/demo/image_matting/colab/input' # './input'
    args.output_path = '/home/ateam/code/MODNet/demo/image_matting/colab/output' # './output'

    # define hyper-parameters
    # ref_size = 512

    # model = FaceParsingModel(model_path=args.tflite_path)

    # inference images
    im_names = os.listdir(args.input_path)
    im_names = [t for t in im_names if '.json' not in t]
    # im_names = glob.glob(args.input_path + '/video (1)*')[::4]
    im_names.sort()
    tic = time.time()
    forward_time = 0
    total_time = 0
    im_names = im_names#[:6]
    for im_name in tqdm.tqdm(im_names):  # im_names[600:650]:
        # im_name = '06993.png'# '06990.png' #'cf.jpg'
        im_name = os.path.basename(im_name)
        im_name = 'fangcheng1.jpg'
        print('Process image: {0}'.format(im_name))
        fname = os.path.join(args.input_path, im_name)
        # fname = '/home/ateam/code/stylegan2-ada-pytorch/test_imgs/border-case3/男正瑕疵脸有阴影.jpg'
        data = open(fname, 'rb').read()
        image = Image.open(io.BytesIO(data)).convert('RGB')

        # # image = face_crop(image)  # 人脸框 512x512  人脸解析10类或者19类
        # image, crop_info = face_crop_for_hair(image)  # 512x512 头发分割2分类
        # # tbq add
        # # im_w, im_h = image.size
        # tic = time.time()
        # out = model.predict(image)
        # total_time += time.time() - tic
        #
        # parsing = out  # .squeeze(0).argmax(0)
        # cv2.imwrite('tbq1.png', parsing*255)
        # image.save('tbq.png')

        parsing = get_hair_mask(image)
        # tbq add
        # vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=os.path.join(args.output_path, im_name))
    print('avg time:', (total_time)/len(im_names))
    print('avg forward time:', forward_time / len(im_names))

# python inference.py --input-path input  --output-path output --ckpt-path ../../../pretrained/modnet_photographic_portrait_matting.ckpt