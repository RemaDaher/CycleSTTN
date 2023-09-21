import os
import cv2
import json
import random
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from core.utils import ZipReader
from core.utils import Stack, ToTorchFormatTensor


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args: dict, split='train', debug=False):
        self.shifted=args['shifted']
        self.masking=args['masking']
        self.Dil=args['Dil']
        self.args = args
        self.split = split
        self.sample_length = args['sample_length']
        self.size = self.w, self.h = (args['w'], args['h'])

        with open(os.path.join(args['data_root'], args['name'], split+'.json'), 'r') as f:
            self.video_dict = json.load(f)
        self.video_names = list(self.video_dict.keys())
        if debug or split != 'train':
            self.video_names = self.video_names[:100]

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(), ])

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('Loading error in video {}'.format(self.video_names[index]))
            item = self.load_item(0)
        return item
    

    def load_item(self, index):
        video_name = self.video_names[index]
        if 'frame_limit' in self.args:
            all_frames = [f"{str(i).zfill(5)}.jpg" for i in range(min(self.video_dict[video_name],self.args['frame_limit']))]
        else:
            all_frames = [f"{str(i).zfill(5)}.jpg" for i in range(self.video_dict[video_name])]            
        if self.masking=='empty' or self.masking=='mixed' or self.masking=='simple mixed':
            all_masks = Image.fromarray((np.ones((self.h, self.w))*255).astype(np.uint8))
            all_masks = [all_masks.convert('L')]*len(all_frames)

        ref_index = get_ref_index(len(all_frames), self.sample_length)
        # read video frames
        frames = []
        framesB = []
        masks = []
        masks_T=[]
        empty_masks = []
        for idx in ref_index:
            zfilelist = ZipReader.filelist('{}/{}/JPEGImages/{}.zip'.format(
                self.args['data_root'], self.args['name'], video_name)) #used since all_frames counts from 0 whereas zfilelist checks the correct naming of files
            img = ZipReader.imread('{}/{}/JPEGImages/{}.zip'.format(
                self.args['data_root'], self.args['name'], video_name), zfilelist[idx]).convert('RGB')
            img = img.resize(self.size)
            frames.append(img)
            if self.masking=='empty':
                empty_mask=all_masks[idx]
                all_mask=empty_mask.copy()
                all_mask_T=empty_mask.copy()
                imgB = ZipReader.imread('{}/{}/JPEGImagesNS/{}.zip'.format(
                    self.args['data_root'], self.args['name'], video_name), zfilelist[idx]).convert('RGB')
                imgB = imgB.resize(self.size)
                framesB.append(imgB)
            elif self.masking=='loaded':
                m = ZipReader.imread('{}/{}/Annotations/{}.zip'.format(
                    self.args['data_root'], self.args['name'], video_name), zfilelist[idx]).convert('RGB')
                m = m.resize(self.size)
                m = np.array(m.convert('L'))
                m = np.array(m > 199).astype(np.uint8) #Rema:from 0 to 199 changes to binary better
                m = cv2.dilate(m, cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (self.Dil,self.Dil)), iterations=1) #Rema:Dilate only 1 iteration change 3,3 to 55(tried it in quantifyResults.ipyb
                m_T=np.copy(m)
                if self.shifted:
                    M = np.float32([[1,0,50],[0,1,0]])
                    m_T = cv2.warpAffine(m,M,self.size)
                    m_T[m!=0]=0
                all_mask_T=Image.fromarray(m_T*255)
                all_mask=Image.fromarray(m*255)
                empty_mask=all_mask.copy()
                framesB=frames.copy()
            elif self.masking=='mixed' or self.masking=="load_add":
                if self.masking=='mixed': empty_mask=all_masks[idx]
                imgB = ZipReader.imread('{}/{}/JPEGImagesNS/{}.zip'.format(
                    self.args['data_root'], self.args['name'], video_name), zfilelist[idx]).convert('RGB')
                imgB = imgB.resize(self.size)
                framesB.append(imgB)
                m = ZipReader.imread('{}/{}/Annotations/{}.zip'.format(
                    self.args['data_root'], self.args['name'], video_name), zfilelist[idx]).convert('RGB')
                m = m.resize(self.size)
                m = np.array(m.convert('L'))
                m = np.array(m > 199).astype(np.uint8) #Rema:from 0 to 199 changes to binary better
                m = cv2.dilate(m, cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (self.Dil,self.Dil)), iterations=1) #Rema:Dilate only 1 iteration change 3,3 to 55(tried it in quantifyResults.ipyb
                m_T=np.copy(m)
                if self.shifted:
                    M = np.float32([[1,0,50],[0,1,0]])
                    m_T = cv2.warpAffine(m,M,self.size)
                    m_T[m!=0]=0
                all_mask_T=Image.fromarray(m_T*255)
                all_mask=Image.fromarray(m*255)
                if self.masking=='load_add': empty_mask=all_mask.copy()

            elif self.masking=='simple mixed':
                empty_mask=all_masks[idx]
                m = ZipReader.imread('{}/{}/Annotations/{}.zip'.format(
                    self.args['data_root'], self.args['name'], video_name), zfilelist[idx]).convert('RGB')
                m = m.resize(self.size)
                m = np.array(m.convert('L'))
                m = np.array(m > 199).astype(np.uint8) #Rema:from 0 to 199 changes to binary better
                m = cv2.dilate(m, cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (self.Dil,self.Dil)), iterations=1) #Rema:Dilate only 1 iteration change 3,3 to 55(tried it in quantifyResults.ipyb
                m_T=np.copy(m)
                if self.shifted:
                    M = np.float32([[1,0,50],[0,1,0]])
                    m_T = cv2.warpAffine(m,M,self.size)
                    m_T[m!=0]=0
                all_mask_T=Image.fromarray(m_T*255)
                all_mask=Image.fromarray(m*255)
                framesB=frames.copy()

            masks_T.append(all_mask_T)
            masks.append(all_mask)
            empty_masks.append(empty_mask)
            
        # if self.split == 'train':
        #     frames = GroupRandomHorizontalFlip()(frames)

        # To tensors
        framesB_tensors = self._to_tensors(framesB)*2.0 - 1.0
        frame_tensors = self._to_tensors(frames)*2.0 - 1.0
        mask_tensors = self._to_tensors(masks)
        mask_T_tensors = self._to_tensors(masks_T)
        empty_masks_tensors = self._to_tensors(empty_masks)
        return frame_tensors, framesB_tensors, mask_tensors, mask_T_tensors, empty_masks_tensors


def get_ref_index(length, sample_length):
    if random.uniform(0, 1) > 0.5:
        ref_index = random.sample(range(length), sample_length)
        ref_index.sort()
    else:
        pivot = random.randint(0, length-sample_length)
        ref_index = [pivot+i for i in range(sample_length)]
    return ref_index
