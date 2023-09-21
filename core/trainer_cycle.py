import os
import glob
from tqdm import tqdm
import importlib
import numpy as np
from PIL import Image
import itertools

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter

from core.dataset_cycle import Dataset
from core.loss import AdversarialLoss
import sys
epsilon=sys.float_info.epsilon


class Trainer():
    def __init__(self, config, debug=False):
        self.config = config
        self.epoch = 0
        self.iteration = 0
        if debug:
            self.config['trainer']['save_freq'] = 5
            self.config['trainer']['valid_freq'] = 5
            self.config['trainer']['iterations'] = 5

        # setup data set and data loader
        self.train_dataset = Dataset(config['data_loader'], split='train',  debug=debug)
        self.train_sampler = None
        self.train_args = config['trainer']
        if config['distributed']:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=config['world_size'], 
                rank=config['global_rank'])
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_args['batch_size'] // config['world_size'],
            shuffle=(self.train_sampler is None), 
            num_workers=self.train_args['num_workers'],
            sampler=self.train_sampler)

        # set loss functions 
        self.adversarial_loss = AdversarialLoss(type=self.config['losses']['GAN_LOSS'])
        self.adversarial_loss = self.adversarial_loss.to(self.config['device'])
        self.l1_loss = nn.L1Loss()

        # setup models including generator and discriminator
        net = importlib.import_module('model.'+config['model'])
        self.netG_A = net.InpaintGenerator()
        self.netG_A = self.netG_A.to(self.config['device'])
        self.netG_B = net.InpaintGenerator()
        self.netG_B = self.netG_B.to(self.config['device'])

        self.netD_A = net.Discriminator(
            in_channels=3, use_sigmoid=config['losses']['GAN_LOSS'] != 'hinge')
        self.netD_A = self.netD_A.to(self.config['device'])        
        self.netD_B = net.Discriminator(
            in_channels=3, use_sigmoid=config['losses']['GAN_LOSS'] != 'hinge')
        self.netD_B = self.netD_B.to(self.config['device'])

        self.optimG = torch.optim.Adam(
            itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), 
            lr=config['trainer']['lr'],
            betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))
        self.optimD = torch.optim.Adam(
            itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), 
            lr=config['trainer']['lr'],
            betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))
        self.load_initialmodel() #added by Rema for loading initializing model
        self.load_initialmodel_sep() #added by Rema for loading initializing model
        self.load()


        if config['distributed']:
            self.netG_A = DDP(
                self.netG_A, 
                device_ids=[self.config['local_rank']], 
                output_device=self.config['local_rank'],
                broadcast_buffers=True, 
                find_unused_parameters=False)
            self.netG_B = DDP(
                self.netG_B, 
                device_ids=[self.config['local_rank']], 
                output_device=self.config['local_rank'],
                broadcast_buffers=True, 
                find_unused_parameters=False)
            self.netD_A = DDP(
                self.netD_A, 
                device_ids=[self.config['local_rank']], 
                output_device=self.config['local_rank'],
                broadcast_buffers=True, 
                find_unused_parameters=False)
            self.netD_B = DDP(
                self.netD_B, 
                device_ids=[self.config['local_rank']], 
                output_device=self.config['local_rank'],
                broadcast_buffers=True, 
                find_unused_parameters=False)

        # set summary writer
        self.dis_A_writer = None
        self.gen_A_writer = None
        self.dis_B_writer = None
        self.gen_B_writer = None
        self.summary = {}
        if self.config['global_rank'] == 0 or (not config['distributed']):
            self.dis_A_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'dis_A'))
            self.gen_A_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'gen_A'))
            self.dis_B_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'dis_B'))
            self.gen_B_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'gen_B'))

    # get current learning rate
    def get_lr(self):
        return self.optimG.param_groups[0]['lr']

     # learning rate scheduler, step
    def adjust_learning_rate(self):
        decay = 0.1**(min(self.iteration,
                          self.config['trainer']['niter_steady']) // self.config['trainer']['niter'])
        new_lr = self.config['trainer']['lr'] * decay
        if new_lr != self.get_lr():
            for param_group in self.optimG.param_groups:
                param_group['lr'] = new_lr
            for param_group in self.optimD.param_groups:
                param_group['lr'] = new_lr

    # add summary
    def add_summary(self, writer, name, val):
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        if writer is not None and self.iteration % 100 == 0:
            writer.add_scalar(name, self.summary[name]/100, self.iteration)
            self.summary[name] = 0

    # load netG and netD
    def load(self):
        model_path = self.config['save_dir']
        if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
            latest_epoch = open(os.path.join(
                model_path, 'latest.ckpt'), 'r').read().splitlines()[-1]
        else:
            ckpts = [os.path.basename(i).split('.pth')[0] for i in glob.glob(
                os.path.join(model_path, '*.pth'))]
            ckpts.sort()
            latest_epoch = ckpts[-1] if len(ckpts) > 0 else None
        if latest_epoch is not None:
            gen_A_path = os.path.join(
                model_path, 'gen_A_{}.pth'.format(str(latest_epoch).zfill(5)))
            dis_A_path = os.path.join(
                model_path, 'dis_A_{}.pth'.format(str(latest_epoch).zfill(5)))
            gen_B_path = os.path.join(
                model_path, 'gen_B_{}.pth'.format(str(latest_epoch).zfill(5)))
            dis_B_path = os.path.join(
                model_path, 'dis_B_{}.pth'.format(str(latest_epoch).zfill(5)))
            if self.config['global_rank'] == 0:
                print('Loading model from {}...'.format(gen_A_path))
            data = torch.load(gen_A_path, map_location=self.config['device'])
            self.netG_A.load_state_dict(data['netG_A'])
            data = torch.load(dis_A_path, map_location=self.config['device'])
            self.netD_A.load_state_dict(data['netD_A'])
            data = torch.load(gen_B_path, map_location=self.config['device'])
            self.netG_B.load_state_dict(data['netG_B'])
            data = torch.load(dis_B_path, map_location=self.config['device'])
            self.netD_B.load_state_dict(data['netD_B'])

            opt_path = os.path.join(
                model_path, 'opt_{}.pth'.format(str(latest_epoch).zfill(5)))
            data = torch.load(opt_path, map_location=self.config['device'])
            self.optimG.load_state_dict(data['optimG'])
            self.optimD.load_state_dict(data['optimD'])
            self.epoch = data['epoch']
            self.iteration = data['iteration']
        else:
            if self.config['global_rank'] == 0:
                print(
                    'Warnning: There is no trained model found. An initialized model will be used.')

                
    def load_initialmodel_sep(self):#added by Rema for loading initializing model         
        if os.path.isdir(self.config['initialmodelA']):
            gen_A_path = os.path.join(
                self.config['initialmodelA'], 'gen_{}.pth'.format((self.config['chosen_epochA']).zfill(5)))
            dis_A_path = os.path.join(
                self.config['initialmodelA'], 'dis_{}.pth'.format((self.config['chosen_epochA']).zfill(5)))
            if self.config['global_rank'] == 0:
                print('Loading model from {}...'.format(gen_A_path))
            data = torch.load(gen_A_path, map_location=self.config['device'])
            self.netG_A.load_state_dict(data['netG'])
            data = torch.load(dis_A_path, map_location=self.config['device'])
            self.netD_A.load_state_dict(data['netD'])

        if os.path.isdir(self.config['initialmodelB']):
            gen_B_path = os.path.join(
                self.config['initialmodelB'], 'gen_{}.pth'.format((self.config['chosen_epochB']).zfill(5)))
            dis_B_path = os.path.join(
                self.config['initialmodelB'], 'dis_{}.pth'.format((self.config['chosen_epochB']).zfill(5)))
            if self.config['global_rank'] == 0:
                print('Loading model from {}...'.format(gen_B_path))
            data = torch.load(gen_B_path, map_location=self.config['device'])
            self.netG_B.load_state_dict(data['netG'])
            data = torch.load(dis_B_path, map_location=self.config['device'])
            self.netD_B.load_state_dict(data['netD'])

        # can't do this because opt is different now
        # opt_path = os.path.join(
        #     self.config['initialmodelA'], 'opt_{}.pth'.format((self.config['chosen_epochA']).zfill(5)))
        # data = torch.load(opt_path, map_location=self.config['device'])
        # self.optimG.load_state_dict(data['optimG'])
        # self.optimD.load_state_dict(data['optimD'])
        # self.epoch = data['epoch']
        # self.iteration = data['iteration']

    def load_initialmodel(self):#added by Rema for loading initializing model    
        if os.path.isfile(os.path.join(self.config['initialmodel'], 'latest.ckpt')):
            latest_epoch = open(os.path.join(
                self.config['initialmodel'], 'latest.ckpt'), 'r').read().splitlines()[-1]
            if latest_epoch is not None:        
                gen_A_path = os.path.join(
                    self.config['initialmodel'], 'gen_A_{}.pth'.format((self.config['chosen_epoch']).zfill(5)))
                dis_A_path = os.path.join(
                    self.config['initialmodel'], 'dis_A_{}.pth'.format((self.config['chosen_epoch']).zfill(5)))
                gen_B_path = os.path.join(
                    self.config['initialmodel'], 'gen_B_{}.pth'.format((self.config['chosen_epoch']).zfill(5)))
                dis_B_path = os.path.join(
                    self.config['initialmodel'], 'dis_B_{}.pth'.format((self.config['chosen_epoch']).zfill(5)))
                if self.config['global_rank'] == 0:
                    print('Loading model from {}...'.format(gen_A_path))
                data = torch.load(gen_A_path, map_location=self.config['device'])
                self.netG_A.load_state_dict(data['netG_A'])
                data = torch.load(dis_A_path, map_location=self.config['device'])
                self.netD_A.load_state_dict(data['netD_A'])
                data = torch.load(gen_B_path, map_location=self.config['device'])
                self.netG_B.load_state_dict(data['netG_B'])
                data = torch.load(dis_B_path, map_location=self.config['device'])
                self.netD_B.load_state_dict(data['netD_B'])

                opt_path = os.path.join(
                    self.config['initialmodel'], 'opt_{}.pth'.format((self.config['chosen_epoch']).zfill(5)))
                data = torch.load(opt_path, map_location=self.config['device'])
                self.optimG.load_state_dict(data['optimG'])
                self.optimD.load_state_dict(data['optimD'])
                self.epoch = data['epoch']
                self.iteration = data['iteration']
            else:
                if self.config['global_rank'] == 0:
                    print(
                        'Warnning: There is no trained model found. An initialized model will be used.')
         
                
    # save parameters every eval_epoch
    def save(self, it):
        if self.config['global_rank'] == 0:
            gen_A_path = os.path.join(
                self.config['save_dir'], 'gen_A_{}.pth'.format(str(it).zfill(5)))
            dis_A_path = os.path.join(
                self.config['save_dir'], 'dis_A_{}.pth'.format(str(it).zfill(5)))
            gen_B_path = os.path.join(
                self.config['save_dir'], 'gen_B_{}.pth'.format(str(it).zfill(5)))
            dis_B_path = os.path.join(
                self.config['save_dir'], 'dis_B_{}.pth'.format(str(it).zfill(5)))
            opt_path = os.path.join(
                self.config['save_dir'], 'opt_{}.pth'.format(str(it).zfill(5)))
            print('\nsaving model to {} and {} ...'.format(gen_A_path, gen_B_path))
            if isinstance(self.netG_A, torch.nn.DataParallel) or isinstance(self.netG_A, DDP):
                netG_A = self.netG_A.module
                netD_A = self.netD_A.module
                netG_B = self.netG_B.module
                netD_B = self.netD_B.module
            else:
                netG_A = self.netG_A
                netD_A = self.netD_A
                netG_B = self.netG_B
                netD_B = self.netD_B
            torch.save({'netG_A': netG_A.state_dict()}, gen_A_path)
            torch.save({'netD_A': netD_A.state_dict()}, dis_A_path)
            torch.save({'netG_B': netG_B.state_dict()}, gen_B_path)
            torch.save({'netD_B': netD_B.state_dict()}, dis_B_path)
            torch.save({'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimG': self.optimG.state_dict(),
                        'optimD': self.optimD.state_dict()}, opt_path)
            os.system('echo {} > {}'.format(str(it).zfill(5),
                                            os.path.join(self.config['save_dir'], 'latest.ckpt')))

    # train entry
    def train(self):
        pbar = range(int(self.train_args['iterations']))
        if self.config['global_rank'] == 0:
            pbar = tqdm(pbar, initial=self.iteration, dynamic_ncols=True, smoothing=0.01)
        
        while True:
            self.epoch += 1
            if self.config['distributed']:
                self.train_sampler.set_epoch(self.epoch)

            if self.config['data_loader']['masking'] =='empty': self._train_epoch_InpTargFrames_nomask(pbar)
            elif self.config['data_loader']['masking'] =='loaded': self._train_epoch_InpFrame_masks(pbar)
            elif (self.config['data_loader']['masking'] =='mixed' and self.config["data_loader"]["shifted"]): self._train_epoch_InpTargFrames_mixedmasksShifted(pbar)
            elif self.config['data_loader']['masking'] =='mixed': self._train_epoch_InpTargFrames_mixedmasks(pbar)
            elif self.config['data_loader']['masking'] =='simple mixed': self._train_epoch_InpFrame_mixedmasks(pbar)
            elif self.config['data_loader']['masking'] =='load_add': self._train_epoch_InpTargFrames_masks(pbar)
            if self.iteration > self.train_args['iterations']:
                break
        print('\nEnd training....')

    # process input and calculate loss every training epoch
    # only load input frames and generate target frames 
    # Input and target have masks
    def _train_epoch_InpFrame_masks(self, pbar):
        device = self.config['device']

        for frames, _, masks, masks_T, _ in self.train_loader:
            self.adjust_learning_rate()
            self.iteration += 1
            frames, masks, masks_T = frames.to(device), masks.to(device), masks_T.to(device)
            b, t, c, h, w = frames.size()
            masked_frame = (frames * (1 - masks).float())
            masked_frame_T=(frames * (1 - masks_T).float())
            
            # Real, Predicted, Identity, Rectified
            pred_A_img_MT = self.netG_A(masked_frame_T, masks_T)
            pred_A_img_M = self.netG_A(masked_frame, masks)
            idt_A = self.netG_A(frames,masks_T)
            rec_A = self.netG_B(pred_A_img_M.view(b, t, c, h, w), masks)

            pred_B_img_M = self.netG_B(masked_frame, masks)
            pred_B_img_MT = self.netG_B(masked_frame_T, masks_T)
            idt_B = self.netG_B(frames,masks)
            rec_B = self.netG_A(pred_B_img_MT.view(b, t, c, h, w), masks_T)

            frames = frames.view(b*t, c, h, w)
            masks = masks.view(b*t, 1, h, w)
            masks_T = masks_T.view(b*t, 1, h, w)
            comp_A_img_MT = frames*(1.-masks_T) + masks_T*pred_A_img_MT
            comp_B_img_M = frames*(1.-masks) + masks*pred_B_img_M

            # comp_B_img_MT = frames*(1.-masks_T) + masks_T*pred_B_img_MT
            comp_A_img_M = frames*(1.-masks) + masks*pred_A_img_M

            gen_loss = 0
            dis_A_loss = 0
            dis_B_loss = 0

            # discriminator adversarial loss A
            real_A_vid_feat = self.netD_A(frames)
            fake_A_vid_feat = self.netD_A(comp_A_img_MT.detach())
            dis_A_real_loss = self.adversarial_loss(real_A_vid_feat, True, True)
            dis_A_fake_loss = self.adversarial_loss(fake_A_vid_feat, False, True)
            dis_A_loss += (dis_A_real_loss + dis_A_fake_loss) / 2
            self.add_summary(
                self.dis_A_writer, 'loss/dis_A_vid_fake', dis_A_fake_loss.item())
            self.add_summary(
                self.dis_A_writer, 'loss/dis_A_vid_real', dis_A_real_loss.item())


            # discriminator adversarial loss B
            real_B_vid_feat = self.netD_B(frames)
            fake_B_vid_feat = self.netD_B(comp_B_img_M.detach())
            dis_B_real_loss = self.adversarial_loss(real_B_vid_feat, True, True)
            dis_B_fake_loss = self.adversarial_loss(fake_B_vid_feat, False, True)
            dis_B_loss += (dis_B_real_loss + dis_B_fake_loss) / 2
            self.add_summary(
                self.dis_B_writer, 'loss/dis_B_vid_fake', dis_B_fake_loss.item())
            self.add_summary(
                self.dis_B_writer, 'loss/dis_B_vid_real', dis_B_real_loss.item())

            # backward D
            self.optimD.zero_grad()
            dis_A_loss.backward()
            dis_B_loss.backward()
            self.optimD.step()

            # generator adversarial loss A
            gen_A_vid_feat = self.netD_A(comp_A_img_M) #M instead of MT -> so now not like STTN original. to make like original add T
            gan_A_loss = self.adversarial_loss(gen_A_vid_feat, True, False)
            gan_A_loss = gan_A_loss * self.config['losses']['adversarial_weight']
            gen_loss += gan_A_loss
            self.add_summary(
                self.gen_A_writer, 'loss/gan_A_loss', gan_A_loss.item())

            # generator adversarial loss B
            gen_B_vid_feat = self.netD_B(comp_B_img_M)
            gan_B_loss = self.adversarial_loss(gen_B_vid_feat, True, False)
            gan_B_loss = gan_B_loss * self.config['losses']['adversarial_weight']
            gen_loss += gan_B_loss
            self.add_summary(
                self.gen_B_writer, 'loss/gan_B_loss', gan_B_loss.item())

            # generator l1 loss
            hole_A_loss = self.l1_loss(pred_A_img_MT*masks_T, frames*masks_T)
            hole_A_loss = hole_A_loss / max(torch.mean(masks_T), epsilon) * self.config['losses']['hole_weight']
            gen_loss += hole_A_loss 
            self.add_summary(
                self.gen_A_writer, 'loss/hole_A_loss', hole_A_loss.item())

            valid_A_loss = self.l1_loss(pred_A_img_MT*(1-masks_T), frames*(1-masks_T))
            valid_A_loss = valid_A_loss / max(torch.mean(1-masks_T), epsilon) * self.config['losses']['valid_weight']
            gen_loss += valid_A_loss 
            self.add_summary(
                self.gen_A_writer, 'loss/valid_A_loss', valid_A_loss.item())


            # generator l1 loss
            hole_B_loss = self.l1_loss(pred_B_img_M*masks, frames*masks)
            hole_B_loss = hole_B_loss / max(torch.mean(masks), epsilon) * self.config['losses']['hole_weight']
            gen_loss += hole_B_loss 
            self.add_summary(
                self.gen_B_writer, 'loss/hole_B_loss', hole_B_loss.item())

            valid_B_loss = self.l1_loss(pred_B_img_M*(1-masks), frames*(1-masks))
            valid_B_loss = valid_B_loss / max(torch.mean(1-masks), epsilon) * self.config['losses']['valid_weight']
            gen_loss += valid_B_loss 
            self.add_summary(
                self.gen_B_writer, 'loss/valid_B_loss', valid_B_loss.item())
            
            # identity and cycle weights
            lambda_idt = self.config['losses']['idt_weight']
            lambda_A = self.config['losses']['cycle_A_weight']
            lambda_B = self.config['losses']['cycle_B_weight']

            # Identity loss
            if lambda_idt > 0:
                # G_A should be identity if real_B is fed: ||G_A(B) - B||
                idt_A_loss = self.l1_loss(idt_A, frames) * lambda_A * lambda_idt
                # G_B should be identity if real_A is fed: ||G_B(A) - A||
                idt_B_loss = self.l1_loss(idt_B, frames) * lambda_B * lambda_idt
                gen_loss += idt_A_loss + idt_B_loss
                self.add_summary(
                    self.gen_A_writer, 'loss/idt_A_loss', idt_A_loss.item())
                self.add_summary(
                    self.gen_B_writer, 'loss/idt_B_loss', idt_B_loss.item())
                printedidt = f"idt_A: {idt_A_loss.item():.3f}; idt_B: {idt_B_loss.item():.3f};"
            else:
                idt_A_loss = 0
                idt_B_loss = 0
                printedidt = ""

            if lambda_A > 0 or lambda_B > 0:
                # Forward cycle loss || G_A(G_B(B)) - B||
                cycle_A_loss = self.l1_loss(rec_A, frames) * lambda_A
                # Backward cycle hole and valid loss || G_B(G_A(A)) - A||
                cycle_B_loss = self.l1_loss(rec_B, frames) * lambda_B
                gen_loss += cycle_A_loss + cycle_B_loss
                self.add_summary(
                    self.gen_A_writer, 'loss/cycle_A_loss', cycle_A_loss.item())
                self.add_summary(
                    self.gen_B_writer, 'loss/cycle_B_loss', cycle_B_loss.item())
                printedcycle=f"cA: {cycle_A_loss.item():.3f}; cB: {cycle_B_loss.item():.3f}"
            else:
                cycle_A_loss = 0
                cycle_B_loss = 0      
                printedcycle=""

            # if lambda_A > 0 or lambda_B > 0:
            #     # Forward cycle hole and valid loss || G_A(G_B(B)) - B||
            #     cycle_hole_A_loss = self.l1_loss(rec_A*masks, frames*masks)
            #     cycle_hole_A_loss = cycle_hole_A_loss / max(torch.mean(masks), epsilon) * lambda_A

            #     cycle_valid_A_loss = self.l1_loss(rec_A*(1-masks), frames*(1-masks))
            #     cycle_valid_A_loss = cycle_valid_A_loss / max(torch.mean(1-masks), epsilon) * lambda_A

            #     # Backward cycle hole and valid loss || G_B(G_A(A)) - A||
            #     cycle_hole_B_loss = self.l1_loss(rec_B*masks_T, frames*masks_T)
            #     cycle_hole_B_loss = cycle_hole_B_loss / max(torch.mean(masks_T), epsilon) * lambda_B

            #     cycle_valid_B_loss = self.l1_loss(rec_B*(1-masks_T), frames*(1-masks_T))
            #     cycle_valid_B_loss = cycle_valid_B_loss / max(torch.mean(1-masks_T), epsilon) * lambda_B

            #     gen_loss += cycle_hole_A_loss + cycle_hole_B_loss + cycle_valid_A_loss + cycle_valid_B_loss
            #     self.add_summary(
            #         self.gen_A_writer, 'loss/cycle_hole_A_loss', cycle_hole_A_loss.item())
            #     self.add_summary(
            #         self.gen_A_writer, 'loss/cycle_valid_A_loss', cycle_valid_A_loss.item())
            #     self.add_summary(
            #         self.gen_B_writer, 'loss/cycle_hole_B_loss', cycle_hole_B_loss.item())
            #     self.add_summary(
            #         self.gen_B_writer, 'loss/cycle_valid_B_loss', cycle_valid_B_loss.item())
            #     printedcycle=f"chA: {cycle_hole_A_loss.item():.3f}; cvA {cycle_valid_A_loss.item():.3f}; chB: {cycle_hole_B_loss.item():.3f}; cvB: {cycle_valid_B_loss.item():.3f}"
            # else:
            #     cycle_hole_A_loss = 0
            #     cycle_valid_A_loss = 0
            #     cycle_hole_B_loss = 0
            #     cycle_valid_B_loss = 0    
            #     printedcycle=""

            self.optimG.zero_grad()
            gen_loss.backward()
            self.optimG.step()

            # console logs
            if self.config['global_rank'] == 0:
                pbar.update(1)
                pbar.set_description((
                    f"dA: {dis_A_loss.item():.3f}; gA: {gan_A_loss.item():.3f};"
                    f"holeA: {hole_A_loss.item():.3f}; validA: {valid_A_loss.item():.3f};"
                    # f"cycle_holeA: {cycle_hole_A_loss.item():.3f}; cycle_validA: {cycle_valid_A_loss.item():.3f};"
                    f"dB: {dis_B_loss.item():.3f}; gB: {gan_B_loss.item():.3f};"
                    # f"cycle_holeB: {cycle_hole_B_loss.item():.3f}; cycle_validB: {cycle_valid_B_loss.item():.3f};"
                    f"{printedidt}, {printedcycle}"
                    )
                )
            # saving models
            if self.iteration % self.train_args['save_freq'] == 0:
                self.save(int(self.iteration//self.train_args['save_freq']))
            if self.iteration > self.train_args['iterations']:
                break

    # this method means that we load target as well as input frames. whereas in train_epoch we only load input frames and generate target frames
    # input is images with spec and target is images that are inpainted and now have no specs
    # empty masks are used here
    def _train_epoch_InpTargFrames_nomask(self, pbar):
        device = self.config['device']

        for frames, framesNS, _, _, empty_masks in self.train_loader:
            self.adjust_learning_rate()
            self.iteration += 1
            frames, framesNS, empty_masks = frames.to(device), framesNS.to(device), empty_masks.to(device)
            b, t, c, h, w = frames.size()
            
            # Real, Predicted, Identity, Rectified
            pred_A_img = self.netG_A(frames, empty_masks)
            idt_A = self.netG_A(framesNS,empty_masks)
            rec_A = self.netG_B(pred_A_img.view(b, t, c, h, w), empty_masks)

            pred_B_img = self.netG_B(framesNS, empty_masks)
            idt_B = self.netG_B(frames,empty_masks)
            rec_B = self.netG_A(pred_B_img.view(b, t, c, h, w), empty_masks)

            frames = frames.view(b*t, c, h, w)
            framesNS = framesNS.view(b*t, c, h, w)
            empty_masks = empty_masks.view(b*t, 1, h, w)

            gen_loss = 0
            dis_A_loss = 0
            dis_B_loss = 0

            # discriminator adversarial loss A
            real_A_vid_feat = self.netD_A(framesNS)
            fake_A_vid_feat = self.netD_A(pred_A_img.detach())
            dis_A_real_loss = self.adversarial_loss(real_A_vid_feat, True, True)
            dis_A_fake_loss = self.adversarial_loss(fake_A_vid_feat, False, True)
            dis_A_loss += (dis_A_real_loss + dis_A_fake_loss) / 2
            self.add_summary(
                self.dis_A_writer, 'loss/dis_A_vid_fake', dis_A_fake_loss.item())
            self.add_summary(
                self.dis_A_writer, 'loss/dis_A_vid_real', dis_A_real_loss.item())


            # discriminator adversarial loss B
            real_B_vid_feat = self.netD_B(frames)
            fake_B_vid_feat = self.netD_B(pred_B_img.detach())
            dis_B_real_loss = self.adversarial_loss(real_B_vid_feat, True, True)
            dis_B_fake_loss = self.adversarial_loss(fake_B_vid_feat, False, True)
            dis_B_loss += (dis_B_real_loss + dis_B_fake_loss) / 2
            self.add_summary(
                self.dis_B_writer, 'loss/dis_B_vid_fake', dis_B_fake_loss.item())
            self.add_summary(
                self.dis_B_writer, 'loss/dis_B_vid_real', dis_B_real_loss.item())

            # backward D
            self.optimD.zero_grad()
            dis_A_loss.backward()
            dis_B_loss.backward()
            self.optimD.step()

            # generator adversarial loss A
            gen_A_vid_feat = self.netD_A(pred_A_img)
            gan_A_loss = self.adversarial_loss(gen_A_vid_feat, True, False)
            gan_A_loss = gan_A_loss * self.config['losses']['adversarial_weight']
            gen_loss += gan_A_loss
            self.add_summary(
                self.gen_A_writer, 'loss/gan_A_loss', gan_A_loss.item())

            # generator adversarial loss B
            gen_B_vid_feat = self.netD_B(pred_B_img)
            gan_B_loss = self.adversarial_loss(gen_B_vid_feat, True, False)
            gan_B_loss = gan_B_loss * self.config['losses']['adversarial_weight']
            gen_loss += gan_B_loss
            self.add_summary(
                self.gen_B_writer, 'loss/gan_B_loss', gan_B_loss.item())

            # identity and cycle weights
            lambda_idt_A = self.config['losses']['idt_A_weight']
            lambda_idt_B = self.config['losses']['idt_B_weight']
            lambda_A = self.config['losses']['cycle_A_weight']
            lambda_B = self.config['losses']['cycle_B_weight']

            # generator l1 loss A Masked
            hole_A_loss = self.l1_loss(pred_A_img, framesNS) * self.config['losses']['hole_A_weight']
            gen_loss += hole_A_loss 
            self.add_summary(
                self.gen_A_writer, 'loss/hole_A_loss', hole_A_loss.item())

            # generator l1 loss B Masked
            hole_B_loss = self.l1_loss(pred_B_img, frames) * self.config['losses']['hole_B_weight']
            gen_loss += hole_B_loss 
            self.add_summary(
                self.gen_B_writer, 'loss/hole_B_loss', hole_B_loss.item())


            # generator l1 loss
            if lambda_idt_A > 0:
                idt_A_loss = self.l1_loss(idt_A, framesNS) * lambda_idt_A
                gen_loss += idt_A_loss 
                self.add_summary(
                    self.gen_A_writer, 'loss/idt_A_loss', idt_A_loss.item())
                printedidtA=f"idtA: {idt_A_loss.item():.3f}"
            else:
                idt_A_loss = 0
                printedidtA=""

            if lambda_idt_B > 0:
                idt_B_loss = self.l1_loss(idt_B, frames) * lambda_idt_B
                gen_loss += idt_B_loss 
                self.add_summary(
                    self.gen_B_writer, 'loss/idt_B_loss', idt_B_loss.item())
                printedidtB=f"idtB: {idt_B_loss.item():.3f}"
            else:
                idt_B_loss = 0
                printedidtB=""

            if lambda_A > 0 or lambda_B > 0:
                # Forward cycle loss || G_B(G_A(A)) - A||
                cycle_A_loss = self.l1_loss(rec_A, frames) * lambda_A
                # Backward cycle hole and valid loss || G_A(G_B(B)) - B||
                cycle_B_loss = self.l1_loss(rec_B, framesNS) * lambda_B
                gen_loss += cycle_A_loss + cycle_B_loss
                self.add_summary(
                    self.gen_A_writer, 'loss/cycle_A_loss', cycle_A_loss.item())
                self.add_summary(
                    self.gen_B_writer, 'loss/cycle_B_loss', cycle_B_loss.item())
                printedcycle=f"cA: {cycle_A_loss.item():.3f}; cB: {cycle_B_loss.item():.3f}"
            else:
                cycle_A_loss = 0
                cycle_B_loss = 0      
                printedcycle=""

            self.optimG.zero_grad()
            gen_loss.backward()
            self.optimG.step()

            # console logs
            if self.config['global_rank'] == 0:
                pbar.update(1)
                pbar.set_description((
                    f"dA: {dis_A_loss.item():.3f}; gA: {gan_A_loss.item():.3f};"
                    f"{printedidtA}; {printedidtB};"
                    f"dB: {dis_B_loss.item():.3f}; gB: {gan_B_loss.item():.3f};"
                    f"{printedcycle}"
                    )
                )
            # saving models
            if self.iteration % self.train_args['save_freq'] == 0:
                self.save(int(self.iteration//self.train_args['save_freq']))
            if self.iteration > self.train_args['iterations']:
                break

    # this method means that we load target as well as input frames. but in one direction we have a mask and in the other we don't
    # input is images with spec and target is images that are inpainted and now have no specs
    def _train_epoch_InpTargFrames_mixedmasks(self, pbar):
        device = self.config['device']

        for frames, framesNS, M, _, empty_masks in self.train_loader:
            self.adjust_learning_rate()
            self.iteration += 1
            frames, framesNS, empty_masks, M = frames.to(device), framesNS.to(device), empty_masks.to(device), M.to(device)
            b, t, c, h, w = frames.size()
            masked_frame=(frames * (1 - M).float())
            # masked_frame_T=(frames * (1 - M_T).float())

            # pred_A_img_MT = self.netG_A(masked_frame_T, M_T)
            # pred_B_img_M = self.netG_B(masked_frame, M)
            
            # Real, Predicted, Identity, Rectified
            pred_A_img = self.netG_A(masked_frame, M)
            idt_A = self.netG_A(framesNS,M)
            rec_A = self.netG_B(pred_A_img.view(b, t, c, h, w), empty_masks)

            pred_B_img = self.netG_B(framesNS, empty_masks)
            idt_B = self.netG_B(frames,empty_masks)
            rec_B = self.netG_A(pred_B_img.view(b, t, c, h, w), M)

            frames = frames.view(b*t, c, h, w)
            framesNS = framesNS.view(b*t, c, h, w)
            empty_masks = empty_masks.view(b*t, 1, h, w)
            M = M.view(b*t, 1, h, w)
            # M_T = M_T.view(b*t, 1, h, w)

            comp_A_img = frames*(1.-M) + M*pred_A_img
            # comp_B_img_M = framesNS*(1.-M) + M*pred_B_img_M

            gen_loss = 0
            dis_A_loss = 0
            dis_B_loss = 0

            # discriminator adversarial loss A Masked
            real_A_vid_feat_M = self.netD_A(framesNS)
            fake_A_vid_feat_M = self.netD_A(comp_A_img.detach())
            dis_A_real_M_loss = self.adversarial_loss(real_A_vid_feat_M, True, True)
            dis_A_fake_M_loss = self.adversarial_loss(fake_A_vid_feat_M, False, True)
            dis_A_loss += (dis_A_real_M_loss + dis_A_fake_M_loss) / 2
            self.add_summary(
                self.dis_A_writer, 'loss/dis_A_vid_fake_M', dis_A_fake_M_loss.item())
            self.add_summary(
                self.dis_A_writer, 'loss/dis_A_vid_real_M', dis_A_real_M_loss.item())

            # discriminator adversarial loss B
            real_B_vid_feat = self.netD_B(frames)
            fake_B_vid_feat = self.netD_B(pred_B_img.detach())
            dis_B_real_loss = self.adversarial_loss(real_B_vid_feat, True, True)
            dis_B_fake_loss = self.adversarial_loss(fake_B_vid_feat, False, True)
            dis_B_loss += (dis_B_real_loss + dis_B_fake_loss) / 2
            self.add_summary(
                self.dis_B_writer, 'loss/dis_B_vid_fake', dis_B_fake_loss.item())
            self.add_summary(
                self.dis_B_writer, 'loss/dis_B_vid_real', dis_B_real_loss.item())

            # backward D
            self.optimD.zero_grad()
            dis_A_loss.backward()
            dis_B_loss.backward()
            self.optimD.step()

            # generator adversarial loss A Masked
            gen_A_vid_feat_M = self.netD_A(comp_A_img)
            gan_A_loss_M = self.adversarial_loss(gen_A_vid_feat_M, True, False)
            gan_A_loss_M = gan_A_loss_M * self.config['losses']['adversarial_weight']
            gen_loss += gan_A_loss_M
            self.add_summary(
                self.gen_A_writer, 'loss/gan_A_M_loss', gan_A_loss_M.item())

            # generator adversarial loss B
            gen_B_vid_feat = self.netD_B(pred_B_img)
            gan_B_loss = self.adversarial_loss(gen_B_vid_feat, True, False)
            gan_B_loss = gan_B_loss * self.config['losses']['adversarial_weight']
            gen_loss += gan_B_loss
            self.add_summary(
                self.gen_B_writer, 'loss/gan_B_loss', gan_B_loss.item())

            # generator l1 loss A Masked
            hole_A_loss = self.l1_loss(pred_A_img*M, framesNS*M)
            hole_A_loss = hole_A_loss / max(torch.mean(M), epsilon) * self.config['losses']['hole_A_weight']
            gen_loss += hole_A_loss 
            self.add_summary(
                self.gen_A_writer, 'loss/hole_A_loss', hole_A_loss.item())

            valid_A_loss = self.l1_loss(pred_A_img*(1-M), framesNS*(1-M))
            valid_A_loss = valid_A_loss / max(torch.mean(1-M), epsilon) * self.config['losses']['valid_A_weight']
            gen_loss += valid_A_loss 
            self.add_summary(
                self.gen_A_writer, 'loss/valid_A_loss', valid_A_loss.item())

            # generator l1 loss B Masked
            hole_B_loss = self.l1_loss(pred_B_img, frames) * self.config['losses']['hole_B_weight']
            # hole_B_loss = hole_B_loss / max(torch.mean(M), epsilon) 
            gen_loss += hole_B_loss 
            self.add_summary(
                self.gen_B_writer, 'loss/hole_B_loss', hole_B_loss.item())

            # valid_B_loss = self.l1_loss(pred_B_img_M*(1-M), framesNS*(1-M))
            # valid_B_loss = valid_B_loss / max(torch.mean(1-M), epsilon) * self.config['losses']['valid_B_weight']
            # gen_loss += valid_B_loss 
            # self.add_summary(
            #     self.gen_B_writer, 'loss/valid_B_loss', valid_B_loss.item())

            # identity and cycle weights
            lambda_idt_A = self.config['losses']['idt_A_weight']
            lambda_idt_B = self.config['losses']['idt_B_weight']
            lambda_A = self.config['losses']['cycle_A_weight']
            lambda_B = self.config['losses']['cycle_B_weight']

            # generator l1 loss
            if lambda_idt_A > 0:
                idt_A_loss = self.l1_loss(idt_A, framesNS) * lambda_idt_A
                gen_loss += idt_A_loss 
                self.add_summary(
                    self.gen_A_writer, 'loss/idt_A_loss', idt_A_loss.item())
                printedidtA=f"idtA: {idt_A_loss.item():.3f}"
            else:
                idt_A_loss = 0
                printedidtA=""

            if lambda_idt_B > 0:
                idt_B_loss = self.l1_loss(idt_B, frames) * lambda_idt_B
                gen_loss += idt_B_loss 
                self.add_summary(
                    self.gen_B_writer, 'loss/idt_B_loss', idt_B_loss.item())
                printedidtB=f"idtB: {idt_B_loss.item():.3f}"
            else:
                idt_B_loss = 0
                printedidtB=""

            if lambda_A > 0 or lambda_B > 0:
                # Forward cycle loss || G_B(G_A(A)) - A||
                cycle_A_loss = self.l1_loss(rec_A, frames) * lambda_A
                # Backward cycle hole and valid loss || G_A(G_B(B)) - B||
                cycle_B_loss = self.l1_loss(rec_B, framesNS) * lambda_B
                gen_loss += cycle_A_loss + cycle_B_loss
                self.add_summary(
                    self.gen_A_writer, 'loss/cycle_A_loss', cycle_A_loss.item())
                self.add_summary(
                    self.gen_B_writer, 'loss/cycle_B_loss', cycle_B_loss.item())
                printedcycle=f"cA: {cycle_A_loss.item():.3f}; cB: {cycle_B_loss.item():.3f}"
            else:
                cycle_A_loss = 0
                cycle_B_loss = 0      
                printedcycle=""

            self.optimG.zero_grad()
            gen_loss.backward()
            self.optimG.step()

            # console logs
            if self.config['global_rank'] == 0:
                pbar.update(1)
                pbar.set_description((
                    f"dA: {dis_A_loss.item():.3f}; gA: {gan_A_loss_M.item():.3f};"
                    f"{printedidtA}; {printedidtB};"
                    f"dB: {dis_B_loss.item():.3f}; gB: {gan_B_loss.item():.3f};"
                    f"{printedcycle}"
                    )
                )
            # saving models
            if self.iteration % self.train_args['save_freq'] == 0:
                self.save(int(self.iteration//self.train_args['save_freq']))
            if self.iteration > self.train_args['iterations']:
                break

    # like train_epoch (no target loaded) but with mixed masking
    def _train_epoch_InpFrame_mixedmasks(self, pbar):
        device = self.config['device']

        for frames, _, M, M_T, empty_masks in self.train_loader:
            self.adjust_learning_rate()
            self.iteration += 1
            frames, M, M_T, empty_masks = frames.to(device), M.to(device), M_T.to(device), empty_masks.to(device)
            b, t, c, h, w = frames.size()
            masked_frame=(frames * (1 - M).float())
            masked_frame_T=(frames * (1 - M_T).float())

            pred_A_img_MT = self.netG_A(masked_frame_T, M_T)
            
            # Real, Predicted, Identity, Rectified
            pred_A_img_M = self.netG_A(masked_frame, M)
            rec_A = self.netG_B(pred_A_img_M.view(b, t, c, h, w), empty_masks)

            frames = frames.view(b*t, c, h, w)
            empty_masks = empty_masks.view(b*t, 1, h, w)
            M = M.view(b*t, 1, h, w)
            M_T = M_T.view(b*t, 1, h, w)

            comp_A_img_MT = frames*(1.-M_T) + M_T*pred_A_img_MT 

            gen_loss = 0
            dis_A_loss = 0
            dis_B_loss = 0

            # discriminator adversarial loss A Masked
            real_A_vid_feat_M = self.netD_A(frames)
            fake_A_vid_feat_M = self.netD_A(comp_A_img_MT.detach())
            dis_A_real_M_loss = self.adversarial_loss(real_A_vid_feat_M, True, True)
            dis_A_fake_M_loss = self.adversarial_loss(fake_A_vid_feat_M, False, True)
            dis_A_loss += (dis_A_real_M_loss + dis_A_fake_M_loss) / 2
            self.add_summary(
                self.dis_A_writer, 'loss/dis_A_vid_fake_M', dis_A_fake_M_loss.item())
            self.add_summary(
                self.dis_A_writer, 'loss/dis_A_vid_real_M', dis_A_real_M_loss.item())


            # discriminator adversarial loss B
            real_B_vid_feat = self.netD_B(frames)
            fake_B_vid_feat = self.netD_B(rec_A.detach())
            dis_B_real_loss = self.adversarial_loss(real_B_vid_feat, True, True)
            dis_B_fake_loss = self.adversarial_loss(fake_B_vid_feat, False, True)
            dis_B_loss += (dis_B_real_loss + dis_B_fake_loss) / 2
            self.add_summary(
                self.dis_B_writer, 'loss/dis_B_vid_fake', dis_B_fake_loss.item())
            self.add_summary(
                self.dis_B_writer, 'loss/dis_B_vid_real', dis_B_real_loss.item())

            # backward D
            self.optimD.zero_grad()
            dis_A_loss.backward()
            dis_B_loss.backward()
            self.optimD.step()

            # generator adversarial loss A Masked
            gen_A_vid_feat_M = self.netD_A(comp_A_img_MT)
            gan_A_loss = self.adversarial_loss(gen_A_vid_feat_M, True, False)
            gan_A_loss = gan_A_loss * self.config['losses']['adversarial_weight']
            gen_loss += gan_A_loss
            self.add_summary(
                self.gen_A_writer, 'loss/gan_A_loss', gan_A_loss.item())


            # generator adversarial loss B
            gen_B_vid_feat = self.netD_B(rec_A)
            gan_B_loss = self.adversarial_loss(gen_B_vid_feat, True, False)
            gan_B_loss = gan_B_loss * self.config['losses']['adversarial_weight']
            gen_loss += gan_B_loss
            self.add_summary(
                self.gen_B_writer, 'loss/gan_B_loss', gan_B_loss.item())

            # generator l1 loss A Masked
            hole_A_loss = self.l1_loss(pred_A_img_MT*M_T, frames*M_T)
            hole_A_loss = hole_A_loss / max(torch.mean(M_T), epsilon) * self.config['losses']['hole_A_weight']
            gen_loss += hole_A_loss 
            self.add_summary(
                self.gen_A_writer, 'loss/hole_A_loss', hole_A_loss.item())

            valid_A_loss = self.l1_loss(pred_A_img_MT*(1-M_T), frames*(1-M_T))
            valid_A_loss = valid_A_loss / max(torch.mean(1-M_T), epsilon) * self.config['losses']['valid_A_weight']
            gen_loss += valid_A_loss 
            self.add_summary(
                self.gen_A_writer, 'loss/valid_A_loss', valid_A_loss.item())

            # identity and cycle weights
            # lambda_idt_A = self.config['losses']['idt_A_weight']
            # lambda_idt_B = self.config['losses']['idt_B_weight']
            lambda_A = self.config['losses']['cycle_A_weight']
            lambda_B = self.config['losses']['cycle_B_weight']


            if lambda_A > 0 or lambda_B > 0:
                # Forward cycle loss || G_B(G_A(A)) - A||
                cycle_A_loss = self.l1_loss(rec_A, frames) * lambda_A
                # Backward cycle hole and valid loss || G_A(G_B(B)) - B||
                gen_loss += cycle_A_loss
                self.add_summary(
                    self.gen_A_writer, 'loss/cycle_A_loss', cycle_A_loss.item())
                printedcycle=f"cA: {cycle_A_loss.item():.3f}"
            else:
                cycle_A_loss = 0
                printedcycle=""

            self.optimG.zero_grad()
            gen_loss.backward()
            self.optimG.step()

            # console logs
            if self.config['global_rank'] == 0:
                pbar.update(1)
                pbar.set_description((
                    f"dA: {dis_A_loss.item():.3f}; gA: {gan_A_loss.item():.3f};"
                    # f"{printedidtA}; {printedidtB};"
                    f"dB: {dis_B_loss.item():.3f}; gB: {gan_B_loss.item():.3f};"
                    f"{printedcycle}"
                    )
                )
            # saving models
            if self.iteration % self.train_args['save_freq'] == 0:
                self.save(int(self.iteration//self.train_args['save_freq']))
            if self.iteration > self.train_args['iterations']:
                break

    # this method means that we load target as well as input frames. but in one direction we have a shifted mask and in the other we don't
    # input is images with spec and target is images that are inpainted and now have no specs
    def _train_epoch_InpTargFrames_mixedmasksShifted(self, pbar):
        device = self.config['device']

        for frames, framesNS, M, M_T, empty_masks in self.train_loader:
            self.adjust_learning_rate()
            self.iteration += 1
            frames, framesNS, empty_masks, M, M_T = frames.to(device), framesNS.to(device), empty_masks.to(device), M.to(device), M_T.to(device)
            b, t, c, h, w = frames.size()
            masked_frame=(frames * (1 - M).float())
            masked_frame_T=(frames * (1 - M_T).float())
            # cv2.imwrite("masked_frame.png",frames.float()[0,0].cpu().numpy().transpose(1,2,0)*255)
            # cv2.imwrite("M.png",empty_masks[0,0].cpu().numpy().transpose(1,2,0)*255)
            # exit(1)

            pred_A_img_MT = self.netG_A(masked_frame_T, M_T)
            # pred_B_img_M = self.netG_B(masked_frame, M)
            
            # Real, Predicted, Identity, Rectified
            pred_A_img = self.netG_A(masked_frame, M)
            idt_A = self.netG_A(framesNS,M)
            rec_A = self.netG_B(pred_A_img.view(b, t, c, h, w), empty_masks)

            pred_B_img = self.netG_B(framesNS, empty_masks)
            idt_B = self.netG_B(frames,empty_masks)
            rec_B = self.netG_A(pred_B_img.view(b, t, c, h, w), M_T)

            frames = frames.view(b*t, c, h, w)
            framesNS = framesNS.view(b*t, c, h, w)
            empty_masks = empty_masks.view(b*t, 1, h, w)
            M = M.view(b*t, 1, h, w)
            M_T = M_T.view(b*t, 1, h, w)

            comp_A_img_MT = frames*(1.-M_T) + M_T*pred_A_img_MT
            # comp_B_img_M = framesNS*(1.-M) + M*pred_B_img_M

            gen_loss = 0
            dis_A_loss = 0
            dis_B_loss = 0

            # discriminator adversarial loss A Masked
            real_A_vid_feat_M = self.netD_A(frames)
            fake_A_vid_feat_M = self.netD_A(comp_A_img_MT.detach())
            dis_A_real_M_loss = self.adversarial_loss(real_A_vid_feat_M, True, True)
            dis_A_fake_M_loss = self.adversarial_loss(fake_A_vid_feat_M, False, True)
            dis_A_loss += (dis_A_real_M_loss + dis_A_fake_M_loss) / 2
            self.add_summary(
                self.dis_A_writer, 'loss/dis_A_vid_fake_M', dis_A_fake_M_loss.item())
            self.add_summary(
                self.dis_A_writer, 'loss/dis_A_vid_real_M', dis_A_real_M_loss.item())

            # discriminator adversarial loss B
            real_B_vid_feat = self.netD_B(frames)
            fake_B_vid_feat = self.netD_B(pred_B_img.detach())
            dis_B_real_loss = self.adversarial_loss(real_B_vid_feat, True, True)
            dis_B_fake_loss = self.adversarial_loss(fake_B_vid_feat, False, True)
            dis_B_loss += (dis_B_real_loss + dis_B_fake_loss) / 2
            self.add_summary(
                self.dis_B_writer, 'loss/dis_B_vid_fake', dis_B_fake_loss.item())
            self.add_summary(
                self.dis_B_writer, 'loss/dis_B_vid_real', dis_B_real_loss.item())

            # backward D
            self.optimD.zero_grad()
            dis_A_loss.backward()
            dis_B_loss.backward()
            self.optimD.step()

            # generator adversarial loss A Masked
            gen_A_vid_feat_M = self.netD_A(comp_A_img_MT)
            gan_A_loss_M = self.adversarial_loss(gen_A_vid_feat_M, True, False)
            gan_A_loss_M = gan_A_loss_M * self.config['losses']['adversarial_weight']
            gen_loss += gan_A_loss_M
            self.add_summary(
                self.gen_A_writer, 'loss/gan_A_M_loss', gan_A_loss_M.item())

            # generator adversarial loss B
            gen_B_vid_feat = self.netD_B(pred_B_img)
            gan_B_loss = self.adversarial_loss(gen_B_vid_feat, True, False)
            gan_B_loss = gan_B_loss * self.config['losses']['adversarial_weight']
            gen_loss += gan_B_loss
            self.add_summary(
                self.gen_B_writer, 'loss/gan_B_loss', gan_B_loss.item())

            # generator l1 loss A Masked
            hole_A_loss = self.l1_loss(pred_A_img_MT*M_T, frames*M_T)
            hole_A_loss = hole_A_loss / max(torch.mean(M_T), epsilon) * self.config['losses']['hole_A_weight']
            gen_loss += hole_A_loss 
            self.add_summary(
                self.gen_A_writer, 'loss/hole_A_loss', hole_A_loss.item())

            valid_A_loss = self.l1_loss(pred_A_img_MT*(1-M_T), frames*(1-M_T))
            valid_A_loss = valid_A_loss / max(torch.mean(1-M_T), epsilon) * self.config['losses']['valid_A_weight']
            gen_loss += valid_A_loss 
            self.add_summary(
                self.gen_A_writer, 'loss/valid_A_loss', valid_A_loss.item())

            # generator l1 loss B Masked
            hole_B_loss = self.l1_loss(pred_B_img, frames) * self.config['losses']['hole_B_weight']
            # hole_B_loss = hole_B_loss / max(torch.mean(M), epsilon) 
            gen_loss += hole_B_loss 
            self.add_summary(
                self.gen_B_writer, 'loss/hole_B_loss', hole_B_loss.item())

            # valid_B_loss = self.l1_loss(pred_B_img_M*(1-M), framesNS*(1-M))
            # valid_B_loss = valid_B_loss / max(torch.mean(1-M), epsilon) * self.config['losses']['valid_B_weight']
            # gen_loss += valid_B_loss 
            # self.add_summary(
            #     self.gen_B_writer, 'loss/valid_B_loss', valid_B_loss.item())

            # identity and cycle weights
            lambda_idt_A = self.config['losses']['idt_A_weight']
            lambda_idt_B = self.config['losses']['idt_B_weight']
            lambda_A = self.config['losses']['cycle_A_weight']
            lambda_B = self.config['losses']['cycle_B_weight']

            # generator l1 loss
            if lambda_idt_A > 0:
                idt_A_loss = self.l1_loss(idt_A, framesNS) * lambda_idt_A
                gen_loss += idt_A_loss 
                self.add_summary(
                    self.gen_A_writer, 'loss/idt_A_loss', idt_A_loss.item())
                printedidtA=f"idtA: {idt_A_loss.item():.3f}"
            else:
                idt_A_loss = 0
                printedidtA=""

            if lambda_idt_B > 0:
                idt_B_loss = self.l1_loss(idt_B, frames) * lambda_idt_B
                gen_loss += idt_B_loss 
                self.add_summary(
                    self.gen_B_writer, 'loss/idt_B_loss', idt_B_loss.item())
                printedidtB=f"idtB: {idt_B_loss.item():.3f}"
            else:
                idt_B_loss = 0
                printedidtB=""

            if lambda_A > 0 or lambda_B > 0:
                # Forward cycle loss || G_B(G_A(A)) - A||
                cycle_A_loss = self.l1_loss(rec_A, frames) * lambda_A
                # Backward cycle hole and valid loss || G_A(G_B(B)) - B||
                cycle_B_loss = self.l1_loss(rec_B, frames) * lambda_B
                gen_loss += cycle_A_loss + cycle_B_loss
                self.add_summary(
                    self.gen_A_writer, 'loss/cycle_A_loss', cycle_A_loss.item())
                self.add_summary(
                    self.gen_B_writer, 'loss/cycle_B_loss', cycle_B_loss.item())
                printedcycle=f"cA: {cycle_A_loss.item():.3f}; cB: {cycle_B_loss.item():.3f}"
            else:
                cycle_A_loss = 0
                cycle_B_loss = 0      
                printedcycle=""

            self.optimG.zero_grad()
            gen_loss.backward()
            self.optimG.step()

            # console logs
            if self.config['global_rank'] == 0:
                pbar.update(1)
                pbar.set_description((
                    f"dA: {dis_A_loss.item():.3f}; gA: {gan_A_loss_M.item():.3f};"
                    f"holeA: {hole_A_loss.item():.3f}; validA: {valid_A_loss.item():.3f};"
                    f"dB: {dis_B_loss.item():.3f}; gB: {gan_B_loss.item():.3f};"
                    f"holeB: {hole_B_loss.item():.3f};"
                    f"{printedidtA}; {printedidtB};"
                    f"{printedcycle}"
                    )
                )
            # saving models
            if self.iteration % self.train_args['save_freq'] == 0:
                self.save(int(self.iteration//self.train_args['save_freq']))
            if self.iteration > self.train_args['iterations']:
                break

    # this method means that we load target as well as input frames. but in one direction we have a mask and in the other we have !MT
    # input is images with spec and target is images that are inpainted and now have no specs
    def _train_epoch_InpTargFrames_masks(self, pbar):
        device = self.config['device']

        for frames, framesNS, M, M_T, _ in self.train_loader:
            self.adjust_learning_rate()
            self.iteration += 1
            frames, framesNS, M_T, M = frames.to(device), framesNS.to(device), M_T.to(device), M.to(device)
            b, t, c, h, w = frames.size()
            masked_frame=(frames * (1 - M).float())
            # masked_frame_T=(frames * (1 - M_T).float())

            # pred_A_img_MT = self.netG_A(masked_frame_T, M_T)
            # pred_B_img_M = self.netG_B(masked_frame, M)
            
            # Real, Predicted, Identity, Rectified
            pred_A_img = self.netG_A(masked_frame, M)
            idt_A = self.netG_A(framesNS,M)
            rec_A = self.netG_B(pred_A_img.view(b, t, c, h, w), (1-M_T))

            pred_B_img = self.netG_B(framesNS, (1-M_T))
            idt_B = self.netG_B(frames,(1-M_T))
            rec_B = self.netG_A(pred_B_img.view(b, t, c, h, w), M)

            frames = frames.view(b*t, c, h, w)
            framesNS = framesNS.view(b*t, c, h, w)
            M_T = M_T.view(b*t, 1, h, w)
            M = M.view(b*t, 1, h, w)

            comp_A_img = frames*(1.-M) + M*pred_A_img
            comp_B_img = frames*(M_T) + (1-M_T)*pred_B_img

            gen_loss = 0
            dis_A_loss = 0
            dis_B_loss = 0

            # discriminator adversarial loss A Masked
            real_A_vid_feat_M = self.netD_A(framesNS)
            fake_A_vid_feat_M = self.netD_A(comp_A_img.detach())
            dis_A_real_M_loss = self.adversarial_loss(real_A_vid_feat_M, True, True)
            dis_A_fake_M_loss = self.adversarial_loss(fake_A_vid_feat_M, False, True)
            dis_A_loss += (dis_A_real_M_loss + dis_A_fake_M_loss) / 2
            self.add_summary(
                self.dis_A_writer, 'loss/dis_A_vid_fake_M', dis_A_fake_M_loss.item())
            self.add_summary(
                self.dis_A_writer, 'loss/dis_A_vid_real_M', dis_A_real_M_loss.item())

            # discriminator adversarial loss B
            real_B_vid_feat = self.netD_B(frames)
            fake_B_vid_feat = self.netD_B(comp_B_img.detach())
            dis_B_real_loss = self.adversarial_loss(real_B_vid_feat, True, True)
            dis_B_fake_loss = self.adversarial_loss(fake_B_vid_feat, False, True)
            dis_B_loss += (dis_B_real_loss + dis_B_fake_loss) / 2
            self.add_summary(
                self.dis_B_writer, 'loss/dis_B_vid_fake', dis_B_fake_loss.item())
            self.add_summary(
                self.dis_B_writer, 'loss/dis_B_vid_real', dis_B_real_loss.item())

            # backward D
            self.optimD.zero_grad()
            dis_A_loss.backward()
            dis_B_loss.backward()
            self.optimD.step()

            # generator adversarial loss A Masked
            gen_A_vid_feat_M = self.netD_A(comp_A_img)
            gan_A_loss_M = self.adversarial_loss(gen_A_vid_feat_M, True, False)
            gan_A_loss_M = gan_A_loss_M * self.config['losses']['adversarial_weight']
            gen_loss += gan_A_loss_M
            self.add_summary(
                self.gen_A_writer, 'loss/gan_A_M_loss', gan_A_loss_M.item())

            # generator adversarial loss B
            gen_B_vid_feat = self.netD_B(comp_B_img)
            gan_B_loss = self.adversarial_loss(gen_B_vid_feat, True, False)
            gan_B_loss = gan_B_loss * self.config['losses']['adversarial_weight']
            gen_loss += gan_B_loss
            self.add_summary(
                self.gen_B_writer, 'loss/gan_B_loss', gan_B_loss.item())

            # generator l1 loss A Masked
            hole_A_loss = self.l1_loss(pred_A_img*M, framesNS*M)
            hole_A_loss = hole_A_loss / max(torch.mean(M), epsilon) * self.config['losses']['hole_A_weight']
            gen_loss += hole_A_loss 
            self.add_summary(
                self.gen_A_writer, 'loss/hole_A_loss', hole_A_loss.item())

            valid_A_loss = self.l1_loss(pred_A_img*(1-M), framesNS*(1-M))
            valid_A_loss = valid_A_loss / max(torch.mean(1-M), epsilon) * self.config['losses']['valid_A_weight']
            gen_loss += valid_A_loss 
            self.add_summary(
                self.gen_A_writer, 'loss/valid_A_loss', valid_A_loss.item())

            # generator l1 loss B Masked
            hole_B_loss = self.l1_loss(pred_B_img*M_T, frames*M_T)
            hole_B_loss = hole_B_loss / max(torch.mean(M_T), epsilon) * self.config['losses']['hole_A_weight']
            gen_loss += hole_B_loss 
            self.add_summary(
                self.gen_B_writer, 'loss/hole_B_loss', hole_B_loss.item())

            valid_B_loss = self.l1_loss(pred_B_img*(1-M_T), frames*(1-M_T))
            valid_B_loss = valid_B_loss / max(torch.mean(1-M_T), epsilon) * self.config['losses']['valid_A_weight']
            gen_loss += valid_B_loss 
            self.add_summary(
                self.gen_B_writer, 'loss/valid_B_loss', valid_B_loss.item())

            # identity and cycle weights
            lambda_idt_A = self.config['losses']['idt_A_weight']
            lambda_idt_B = self.config['losses']['idt_B_weight']
            lambda_A = self.config['losses']['cycle_A_weight']
            lambda_B = self.config['losses']['cycle_B_weight']

            # generator l1 loss
            if lambda_idt_A > 0:
                idt_A_loss = self.l1_loss(idt_A, framesNS) * lambda_idt_A
                gen_loss += idt_A_loss 
                self.add_summary(
                    self.gen_A_writer, 'loss/idt_A_loss', idt_A_loss.item())
                printedidtA=f"idtA: {idt_A_loss.item():.3f}"
            else:
                idt_A_loss = 0
                printedidtA=""

            if lambda_idt_B > 0:
                idt_B_loss = self.l1_loss(idt_B, frames) * lambda_idt_B
                gen_loss += idt_B_loss 
                self.add_summary(
                    self.gen_B_writer, 'loss/idt_B_loss', idt_B_loss.item())
                printedidtB=f"idtB: {idt_B_loss.item():.3f}"
            else:
                idt_B_loss = 0
                printedidtB=""

            if lambda_A > 0 or lambda_B > 0:
                # Forward cycle loss || G_B(G_A(A)) - A||
                cycle_A_loss = self.l1_loss(rec_A, frames) * lambda_A
                # Backward cycle hole and valid loss || G_A(G_B(B)) - B||
                cycle_B_loss = self.l1_loss(rec_B, framesNS) * lambda_B
                gen_loss += cycle_A_loss + cycle_B_loss
                self.add_summary(
                    self.gen_A_writer, 'loss/cycle_A_loss', cycle_A_loss.item())
                self.add_summary(
                    self.gen_B_writer, 'loss/cycle_B_loss', cycle_B_loss.item())
                printedcycle=f"cA: {cycle_A_loss.item():.3f}; cB: {cycle_B_loss.item():.3f}"
            else:
                cycle_A_loss = 0
                cycle_B_loss = 0      
                printedcycle=""

            self.optimG.zero_grad()
            gen_loss.backward()
            self.optimG.step()

            # console logs
            if self.config['global_rank'] == 0:
                pbar.update(1)
                pbar.set_description((
                    f"dA: {dis_A_loss.item():.3f}; gA: {gan_A_loss_M.item():.3f};"
                    f"{printedidtA}; {printedidtB};"
                    f"dB: {dis_B_loss.item():.3f}; gB: {gan_B_loss.item():.3f};"
                    f"{printedcycle}"
                    )
                )
            # saving models
            if self.iteration % self.train_args['save_freq'] == 0:
                self.save(int(self.iteration//self.train_args['save_freq']))
            if self.iteration > self.train_args['iterations']:
                break
