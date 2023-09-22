import copy
import os
import time

import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common_sift import (get_camera_from_tensor, get_samples_sift,
                        get_tensor_from_camera, proj_3D_2D, replace_zero_depth, ray_to_3D)

from src.common import (get_samples)


from src.loss_functions.loss import huber_loss

from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer

import torch.nn as nn

debug = False
class Tracker(object):
    def __init__(self, cfg, args, slam
                 ):
        self.cfg = cfg
        self.args = args

        self.scale = cfg['scale']
        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.sync_method = cfg['sync_method']

        self.idx = slam.idx
        self.nice = slam.nice
        self.bound = slam.bound
        self.mesher = slam.mesher
        self.output = slam.output
        self.verbose = slam.verbose
        self.shared_c = slam.shared_c
        self.renderer = slam.renderer
        self.gt_c2w_list = slam.gt_c2w_list
        self.low_gpu_mem = slam.low_gpu_mem
        self.mapping_idx = slam.mapping_idx
        self.mapping_cnt = slam.mapping_cnt
        self.shared_decoders = slam.shared_decoders
        self.estimate_c2w_list = slam.estimate_c2w_list

        self.cam_lr = cfg['tracking']['lr']
        self.device = cfg['tracking']['device']
        self.num_cam_iters = cfg['tracking']['iters']
        self.gt_camera = cfg['tracking']['gt_camera']
        self.tracking_pixels = cfg['tracking']['pixels']
        self.seperate_LR = cfg['tracking']['seperate_LR']
        self.w_color_loss = cfg['tracking']['w_color_loss']
        self.ignore_edge_W = cfg['tracking']['ignore_edge_W']
        self.ignore_edge_H = cfg['tracking']['ignore_edge_H']
        self.handle_dynamic = cfg['tracking']['handle_dynamic']
        self.use_color_in_tracking = cfg['tracking']['use_color_in_tracking']
        self.const_speed_assumption = cfg['tracking']['const_speed_assumption']

        self.every_frame = cfg['mapping']['every_frame']
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']

        self.prev_mapping_idx = -1
        self.frame_reader = get_dataset(
            cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(
            self.frame_reader, batch_size=1, shuffle=False, num_workers=1)
        self.visualizer = Visualizer(freq=cfg['tracking']['vis_freq'], inside_freq=cfg['tracking']['vis_inside_freq'],
                                     vis_dir=os.path.join(self.output, 'vis' if 'Demo' in self.output else 'tracking_vis'),
                                     renderer=self.renderer, verbose=self.verbose, device=self.device)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy


    def cam_pose_optimization_sift(self, camera_tensor, gt_color, gt_depth, batch_size, 
                                   optimizer, prev_camera_tensor, gt_depth_prev, gt_color_prev, idx):
        """
        Do one iteration of camera iteration. Sample pixels, render depth/color, calculate loss and backpropagation.

        Args:
            camera_tensor (tensor): camera tensor.  - is the optimized thing weight update backprop
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            batch_size (int): batch size, number of sampling rays. - additional 100 sift selected added in get_samples_sift()
            optimizer (torch.optim): camera optimizer.

        Returns:
            loss (float): The value of loss.
        """
        # random_numbers = torch.cuda.FloatTensor(7).uniform_(0.01, 0.4)
        # camera_tensor = prev_camera_tensor + random_numbers
        # print("camera tensor inside: ", camera_tensor)
        # print("camera tensor previous inside: ", prev_camera_tensor)




        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        optimizer.zero_grad()
        print("camera_tensor inside optimize 2d2d: \n", camera_tensor)
        
        
        print("prev_camera_tensor inside optimize 2d2d: \n", prev_camera_tensor)

        # print("optimizer inside optimize: \n", optimizer.grad)

        c2w = get_camera_from_tensor(camera_tensor)
        prev_c2w = get_camera_from_tensor(prev_camera_tensor)
        Wedge = self.ignore_edge_W
        Hedge = self.ignore_edge_H
        
        if debug == True:
            print("wedge and hedge are: ", Wedge, Hedge)
            batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(
            Hedge, H-Hedge, Wedge, W-Wedge, batch_size, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)

        # get sift rays for both images
        # then calculate loss
        # will probably need initial poses - 
            '''
            hedge and wedge are 20 - the input to get_samples_sift() are 20 and imagesize-20
            this means on every edge 20 pixels are removed and the removed image is put inside
            1 is previous
            2 is current
            '''

        uv_prev, uv_cur, sbatch_rays_o, sbatch_rays_d, sbatch_gt_depth, sbatch_gt_color, sbatch_rays_o2, sbatch_rays_d2, sbatch_gt_depth2, sbatch_gt_color2 = get_samples_sift(
            Hedge, H-Hedge, Wedge, W-Wedge, batch_size, H, W, fx, fy, cx, cy, gt_depth_prev, gt_color_prev, prev_c2w, gt_depth, gt_color, c2w, idx, self.device)

        uv_prev = uv_prev.float()
        uv_cur = uv_cur.float()
        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(
            Hedge, H-Hedge, Wedge, W-Wedge, batch_size, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)

        sift_feature_size = 100



        # get 3D points for the sift feature rays
        point_3D_current = ray_to_3D(sbatch_rays_o2, sbatch_rays_d2, sbatch_gt_depth2, gt_depth, batch_size, sift_feature_size)
        point_3D_prev = ray_to_3D(sbatch_rays_o, sbatch_rays_d2, sbatch_gt_depth2, gt_depth_prev, batch_size, sift_feature_size)

        # previous image 3D points in current frame 2D / current image 3D points in previous frame 2D
        prev_in_cur = proj_3D_2D(point_3D_prev, W, fx, fy, cx, cy, c2w, self.device)  # is float
        cur_in_prev = proj_3D_2D(point_3D_current, W, fx, fy, cx, cy, prev_c2w, self.device)  # is float

        cur_in_prev = cur_in_prev
        prev_in_cur = prev_in_cur


        # print("PREV IN CURRRRRRRRRRRR. ", prev_in_cur)
        if debug:
            # outputs the projected 3D point of current frame backprojected into 2D
            # should be the same as sift feature uv
            # check difference for different data sets
            uv_in_cur_test = proj_3D_2D(point_3D_current, W, fx, fy, cx, cy, c2w, self.device)   # is float
            # uv_in_cur_test = np.round(uv_in_cur_test).astype(int)
            print("uv_prev and uv_cur\n", uv_prev[:5],"\n", uv_cur[:5])
            print("uv_in_cur_test:\n ", uv_in_cur_test[:5])


        # these go into loss and are backpropagated    
        print("uv_prev: \n", uv_prev,"\n")
        print("cur_in_prev:\n ", cur_in_prev)

        

        loss_out_test = torch.nn.functional.huber_loss(cur_in_prev, uv_prev, reduction='mean', delta=1.0)

        # loss_out_test = torch.tensor(loss_out_test, requires_grad=True)
        # print("testing loss of huber_loss output:\n", loss_out_test)
        
        print("printing loss_out_test should be fn: ", loss_out_test)


        # print("loss_out_test after all: \n", loss_out_test)
        # print("gradient before: ", camera_tensor.grad)

        if camera_tensor.grad is not None:
            print("gradient before: ", camera_tensor.grad)

        # L1_loss = nn.L1Loss()
        # print("gradient cam_tensor before: \n", camera_tensor.grad)

        loss_out_test.backward()
        optimizer.step()
        # print("gradient cam_tensor after: \n", camera_tensor.grad)

        print("loss_out_test: ", loss_out_test)
        if camera_tensor.grad is not None:
            print("gradient before: ", camera_tensor.grad)

        # print("camera_tensor inside optimize 2d2d AFTER OPTIM: \n", camera_tensor)

        optimizer.zero_grad()
        return loss_out_test.item()



    def update_para_from_mapping(self):
        """
        Update the parameters of scene representation from the mapping thread.

        """
        if self.mapping_idx[0] != self.prev_mapping_idx:
            if self.verbose:
                print('Tracking: update the parameters from mapping')
            self.decoders = copy.deepcopy(self.shared_decoders).to(self.device)
            for key, val in self.shared_c.items():
                val = val.clone().to(self.device)
                self.c[key] = val
            self.prev_mapping_idx = self.mapping_idx[0].clone()

    def run(self):


        # true for using gt c2w otherwise run normal
        toggle_gt_c2w = False

        device = self.device
        self.c = {}
        if self.verbose:
            pbar = self.frame_loader
        else:
            pbar = tqdm(self.frame_loader)

        gt_color_prev = None
        gt_depth_prev = None
        prev_idx = None
        prev_camera_tensor = None
        # batch_rays_d_prev = None 
        # batch_rays_o_prev = None
        
        # main loop for tracker - runs for all frames
        for idx, gt_color, gt_depth, gt_c2w in pbar:
            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0]}")

            idx = idx[0]
            gt_depth = gt_depth[0]
            gt_color = gt_color[0]
            gt_c2w = gt_c2w[0]

            if debug == True and idx>1:
                print("this is the current idx: ", idx, "this is the previous idx: ", prev_idx, "\n")

            if self.sync_method == 'strict':
                # strictly mapping and then tracking
                # initiate mapping every self.every_frame frames
                if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1):
                    while self.mapping_idx[0] != idx-1:
                        time.sleep(0.1)
                    pre_c2w = self.estimate_c2w_list[idx-1].to(device)
            elif self.sync_method == 'loose':
                # mapping idx can be later than tracking idx is within the bound of
                # [-self.every_frame-self.every_frame//2, -self.every_frame+self.every_frame//2]
                while self.mapping_idx[0] < idx-self.every_frame-self.every_frame//2:
                    time.sleep(0.1)
            elif self.sync_method == 'free':
                # pure parallel, if mesh/vis happens may cause inbalance
                pass

            self.update_para_from_mapping()

            if self.verbose:
                print(Fore.MAGENTA)
                print("Tracking Frame ",  idx.item())
                print(Style.RESET_ALL)
            # print("this is index: ", idx)
            # for the first frame initializes as gt camera pose
            if idx == 0 or self.gt_camera:
                c2w = gt_c2w
                if not self.no_vis_on_first_frame:
                    self.visualizer.vis(
                        idx, 0, gt_depth, gt_color, c2w, self.c, self.decoders)
                prev_camera_tensor = get_tensor_from_camera(c2w)
                prev_camera_tensor = Variable(prev_camera_tensor.to(device), requires_grad=True)
                
            else:
                gt_camera_tensor = get_tensor_from_camera(gt_c2w)
                if self.const_speed_assumption and idx-2 >= 0:
                    pre_c2w = pre_c2w.float()
                    # print("pre_c2w at beginning: \n", pre_c2w)

                    delta = pre_c2w@self.estimate_c2w_list[idx-2].to(
                        device).float().inverse()
                    estimated_new_cam_c2w = delta@pre_c2w
                else:
                    estimated_new_cam_c2w = pre_c2w 

                #camera_tensor = get_tensor_from_camera(
                #    estimated_new_cam_c2w.detach())
                

                camera_tensor = get_tensor_from_camera(
                estimated_new_cam_c2w.detach())
                print("estimated_new_cam_c2w: \n", estimated_new_cam_c2w)


                camera_tensor = Variable(
                    camera_tensor.to(device), requires_grad=True)
                cam_para_list = [camera_tensor]
                optimizer_camera = torch.optim.Adam(
                    cam_para_list, lr=self.cam_lr)

                initial_loss_camera_tensor = torch.abs(
                    gt_camera_tensor.to(device)-camera_tensor).mean().item()
                candidate_cam_tensor = None
                current_min_loss = 10000000000.

                """
                Similar to SPARF to get loss for tracking
                gets sparf rays and computes 3D point coordinates

                """
                for cam_iter in range(self.num_cam_iters):      # run optimization   self.num_cam_iters
                    print("cam_iter: ", cam_iter)

                    # print("current cam_tensor first: ", camera_tensor)

                    self.visualizer.vis(
                        idx, cam_iter, gt_depth, gt_color, camera_tensor, self.c, self.decoders)
                    loss = self.cam_pose_optimization_sift(
                        camera_tensor, gt_color, gt_depth, self.tracking_pixels, optimizer_camera, 
                        prev_camera_tensor, gt_depth_prev, gt_color_prev, idx)
                    
                    # print("batch_Rays_d_prev = ", batch_rays_d_prev)
                    # print("size of batch_rays_d_prev = ", batch_rays_d_prev.size())
                    # print("size of batch_rays_o_prev = ", batch_rays_o_prev.size())

                    if cam_iter == 0:
                        initial_loss = loss
                        print("initial_loss cor cam iter 0 = ", initial_loss)
                    loss_camera_tensor = torch.abs(gt_camera_tensor.to(device)-camera_tensor).mean().item()
                    if self.verbose:
                        if cam_iter == self.num_cam_iters-1:
                            print(
                                f'Re-rendering loss: {initial_loss:.2f}->{loss:.2f} ' +
                                f'camera tensor error: {initial_loss_camera_tensor:.4f}->{loss_camera_tensor:.4f}')

                    print("loss: ", loss)   
                    # print("current cam_tensor second: ", camera_tensor)

                    print("this is current min loss: ", current_min_loss)
                    if loss < current_min_loss:
                        current_min_loss = loss
                        print("current min loss: ",current_min_loss)
                    candidate_cam_tensor = camera_tensor.clone().detach()
                # if debug == True:


                # renders and outputs the image
                # every out_num image is rendered and saved in rendered_images
                if (debug == True):
                    out_num = 10
                    if (idx%out_num == 0):
                        self.visualizer.vis_rendered(
                            idx, cam_iter, gt_depth, gt_color, camera_tensor, self.c, self.decoders)




                bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
                    [1, 4])).type(torch.float32).to(self.device)
                c2w = get_camera_from_tensor(
                    candidate_cam_tensor.clone().detach())
                c2w = torch.cat([c2w, bottom], dim=0)
                # print("this is c2w in normal: \n", c2w)
            self.estimate_c2w_list[idx] = c2w.clone().cpu()
            self.gt_c2w_list[idx] = gt_c2w.clone().cpu()                #gt of c2w list is read like this
            pre_c2w = c2w.clone()
            # print("pre_c2w at end: \n", pre_c2w)

            self.idx[0] = idx
            if self.low_gpu_mem:
                torch.cuda.empty_cache()

            # save current gt_color image into gt_color_prev, same with depth, current camera position in tensor
            gt_color_prev = gt_color.clone().detach()
            gt_depth_prev = gt_depth.clone().detach()
            prev_idx = idx.clone().detach()
            if idx > 0:
                prev_camera_tensor = candidate_cam_tensor.clone().detach()
            # if batch_rays_o_prev is not None:
            #     batch_rays_o_prev = batch_rays_o.clone()
            #     batch_rays_d_prev = batch_rays_d.clone()
                