# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import importlib
import os
import argparse
import copy
import datetime
import random
import sys
import json
import pathlib

import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models
import torch.multiprocessing as mp
from torchvision import transforms

# My libs
from core.utils import Stack, ToTorchFormatTensor


w, h = 288, 288
ref_length = 10
neighbor_stride = 1
default_fps = 24

_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor(),
    transforms.Lambda(lambda x: x.cuda())])


# sample reference frames from the whole video 
def get_ref_index(neighbor_ids, length):
    ref_index = []
    for i in range(0, length, ref_length):
        if not i in neighbor_ids:
            ref_index.append(i)
    return ref_index


def test(imgs, ckptpath, ckptnumber, whichmodel):
    # set up models 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = importlib.import_module('model.sttn')
    
    # model
    model = net.InpaintGenerator().to(device)
    model_path = os.path.join(ckptpath,"gen_"+whichmodel+"_"+ ckptnumber.zfill(5)+".pth")
    data = torch.load(model_path, map_location=device)
    model.load_state_dict(data["netG_"+whichmodel])
    # print("loading model "+whichmodel+" from: {}".format(ckptpath))
    model.eval()

    # prepare datset, encode all frames into deep space 
    frames = imgs
    w, h=frames[0].size

    video_length = len(frames)

    masks = [Image.fromarray(np.array(m.convert('L'))*0+255) for m in frames]
    feats = _to_tensors(frames).unsqueeze(0)*2-1
    masks = _to_tensors(masks).unsqueeze(0)
    frames = [np.array(f).astype(np.uint8) for f in frames]
    comp_frames = [None]*video_length
    
    with torch.no_grad():
        feats = model.encoder(feats.view(video_length,3, h, w))
        _, c, feat_h, feat_w = feats.size()
        feats = feats.view(1, video_length, c, feat_h, feat_w)

    # completing holes by spatial-temporal transformers
    for f in range(0, video_length, neighbor_stride):
        neighbor_ids = [i for i in range(max(0, f-neighbor_stride), min(video_length, f+neighbor_stride+1))]
        ref_ids = get_ref_index(neighbor_ids, video_length)
        with torch.no_grad():
            pred_feat = model.infer(
                feats[0, neighbor_ids+ref_ids, :, :, :], masks[0, neighbor_ids+ref_ids, :, :, :])
            pred_img = torch.tanh(model.decoder(
                pred_feat[:len(neighbor_ids), :, :, :])).detach()
            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy()*255
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(
                    np.uint8)
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(
                        np.float32)*0.5 + img.astype(np.float32)*0.5
    comp=[]
    for f in range(video_length):
        comp.append(np.array(comp_frames[f]).astype(
            np.uint8))
    return comp
