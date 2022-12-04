'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-12-03 16:25:22
Email: haimingzhang@link.cuhk.edu.cn
Description: The demo script.
'''

import torch
import os
import cv2
import numpy as np
from easydict import EasyDict
import yaml

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (5,5)

import model.model_mobilenetv2_seg_small as modellib
from util.image_utils import pred_single


## 1) Load the config file
config_path = './config/model_mobilenetv2_with_two_auxiliary_losses.yaml'
config_path = "./config/model_mobilenetv2_with_prior_channel.yaml"

with open(config_path, 'rb') as f:
    cont = f.read()
cf = yaml.load(cont, yaml.FullLoader)

print('finish load config file ...')

exp_args = EasyDict()
exp_args.istrain = False
exp_args.task = cf['task']
exp_args.datasetlist = cf['datasetlist']  # ['EG1800', ATR', 'MscocoBackground', 'supervisely_face_easy']

exp_args.model_root = cf['model_root']
exp_args.data_root = cf['data_root']
exp_args.file_root = cf['file_root']

# the height of input images, default=224
exp_args.input_height = cf['input_height']
# the width of input images, default=224
exp_args.input_width = cf['input_width']

# if exp_args.video=True, add prior channel for input images, default=False
exp_args.video = cf['video']
# the probability to set empty prior channel, default=0.5
exp_args.prior_prob = cf['prior_prob']

# whether to add boundary auxiliary loss, default=False
exp_args.addEdge = cf['addEdge']
# whether to add consistency constraint loss, default=False
exp_args.stability = cf['stability']

# input normalization parameters
exp_args.padding_color = cf['padding_color']
exp_args.img_scale = cf['img_scale']
# BGR order, image mean, default=[103.94, 116.78, 123.68]
exp_args.img_mean = cf['img_mean']
# BGR order, image val, default=[0.017, 0.017, 0.017]
exp_args.img_val = cf['img_val']

# if exp_args.useUpsample==True, use nn.Upsample in decoder, else use nn.ConvTranspose2d
exp_args.useUpsample = cf['useUpsample']
# if exp_args.useDeconvGroup==True, set groups=input_channel in nn.ConvTranspose2d
exp_args.useDeconvGroup = cf['useDeconvGroup']

# print ('===========> loading model <===========')
netmodel_video = modellib.MobileNetV2(n_class=2,
                                      useUpsample=exp_args.useUpsample,
                                      useDeconvGroup=exp_args.useDeconvGroup,
                                      addEdge=exp_args.addEdge,
                                      channelRatio=1.0,
                                      minChannel=16,
                                      weightInit=True,
                                      video=exp_args.video).cuda()

bestModelFile = os.path.join(exp_args.model_root, 'model_best.pth.tar')
if os.path.isfile(bestModelFile):
    checkpoint_video = torch.load(bestModelFile, encoding='latin1')
    netmodel_video.load_state_dict(checkpoint_video['state_dict'])
    print("minLoss: ", checkpoint_video['minLoss'], checkpoint_video['epoch'])
    print("=> loaded checkpoint '{}' (epoch {})".format(bestModelFile, checkpoint_video['epoch']))
else:
    print("=> no checkpoint found at '{}'".format(bestModelFile))


## Background blur
image_name = "00308"
image_name = "02252"
image_name = "zhanghm"
save_dir = f"./samples/{image_name}"
os.makedirs(save_dir, exist_ok=True)

# img_ori = cv2.imread(f"./Data/EG1800/Images/{image_name}.png")
img_ori = cv2.imread(f"./samples/{image_name}.jpg")
# mask_ori = cv2.imread("/home/dongx12/Data/EG1800/Labels/00457.png")

prior = None
height, width, _ = img_ori.shape

background = img_ori.copy()
background = cv2.blur(background, (17, 17))

alphargb, pred = pred_single(netmodel_video, exp_args, img_ori, prior)
# plt.imshow(alphargb)
# plt.show()

alphargb = cv2.cvtColor(alphargb, cv2.COLOR_GRAY2BGR)
alphargb_img = np.asarray(alphargb * 255.0, dtype=np.uint8)
cv2.imwrite(f"./{save_dir}/alphargb.png", alphargb_img)
cv2.imwrite(f"./{save_dir}/img_ori.png", img_ori)

## save the bluring result
result = np.uint8(img_ori * alphargb + background * (1 - alphargb))
cv2.imwrite(f"./{save_dir}/blur_result.png", result)

## save the background replacement result
background = cv2.imread("./samples/landscape.png")
background = cv2.resize(background, (width, height))
result = np.uint8(img_ori * alphargb + background * (1 - alphargb))
cv2.imwrite(f"./{save_dir}/background_replacement_result.png", result)


myImg = np.ones((height, width * 2 + 20, 3)) * 255
myImg[:, :width, :] = img_ori
myImg[:, width + 20:, :] = result
cv2.imwrite(f"./{save_dir}/result_landscape_prior.png", myImg)

# plt.imshow(myImg[:, :, ::-1] / 255)
# plt.yticks([])
# plt.xticks([])
# plt.show()
# plt.savefig("./samples/result_landscape.png")
