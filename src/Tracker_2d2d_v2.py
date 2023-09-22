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
                        get_tensor_from_camera, proj_3D_2D)
from src.loss_functions.loss import huber_loss

from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer

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
            camera_tensor (tensor): camera tensor.
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            batch_size (int): batch size, number of sampling rays. - additional 100 sift selected added in get_samples_sift()
            optimizer (torch.optim): camera optimizer.

        Returns:
            loss (float): The value of loss.
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        optimizer.zero_grad()
        c2w = get_camera_from_tensor(camera_tensor)
        prev_c2w = get_camera_from_tensor(prev_camera_tensor)
        Wedge = self.ignore_edge_W
        Hedge = self.ignore_edge_H
        
        if debug == True:
            print("wedge and hedge are: ", Wedge, Hedge)
        # batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(
        #     Hedge, H-Hedge, Wedge, W-Wedge, batch_size, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)

        # get sift rays for both images
        # then calculate loss
        # will probably need initial poses - 
        '''
        hedge and wedge are 20 - the input to get_samples_sift() are 20 and imagesize-20
        this means on every edge 20 pixels are removed and the removed image is put inside
        '''
        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, batch_rays_o2, batch_rays_d2, batch_gt_depth2, batch_gt_color2 = get_samples_sift(
            Hedge, H-Hedge, Wedge, W-Wedge, batch_size, H, W, fx, fy, cx, cy, gt_depth_prev, gt_color_prev, prev_c2w, gt_depth, gt_color, c2w, idx, self.device)
        # print("in cam_pose_optimization gt_color size: ", gt_color.size())
        # print("in cam_pose_optimization H and W size: ", H, W)

        max_sift = batch_size + 100
        # sizes of rays_o and rays_d is [100,3]
        s_rays_o = batch_rays_o[batch_size:max_sift]
        s_rays_d = batch_rays_d[batch_size:max_sift]
        s_rays_o2 = batch_rays_o2[batch_size:max_sift]
        s_rays_d2 = batch_rays_d2[batch_size:max_sift]

        # maybe change to range 
        # s_depth sizes are [100]
        s_depth = batch_gt_depth[batch_size:max_sift]
        s_depth2 = batch_gt_depth2[batch_size:max_sift]
        


        # need to add this to get rid of depth = 0 - projection error
        # for the Huber loss does not matter since it's only in 2D
        s_depth     += 0.1
        s_depth2    += 0.1
        # print("sdepth2 ", s_depth2[:10])


        # 3D coordinates in 3D
        point_3D_prev    = s_rays_o + s_rays_d * s_depth.unsqueeze(1) # output size is [100,3]
        point_3D_current = s_rays_o2 + s_rays_d2 * s_depth2.unsqueeze(1)

        # 
        uv_in_cur = proj_3D_2D(point_3D_prev, fx, fy, cx, cy, prev_c2w)  # is float
        uv_in_cur = torch.from_numpy(uv_in_cur).to(self.device)

        if debug:
            # outputs the projected 3D point of current frame backprojected into 2D
            # should be the same as sift feature uv
            # check difference for different data sets
            uv_in_cur_test = proj_3D_2D(point_3D_current, fx, fy, cx, cy, c2w)  # is float
            uv_in_cur_test[:, 0] = W-uv_in_cur_test[:, 0]
            uv_in_cur_test[:, 0] = uv_in_cur_test[:, 0] - 2
            uv_in_cur_test = np.round(uv_in_cur_test).astype(int)
            print("uv_in_cur_test:\n ", uv_in_cur_test[:10])

        # prev rays in cur frame
        prev_in_cur = proj_3D_2D(point_3D_prev, fx, fy, cx, cy, c2w)  # is float
        prev_in_cur[:, 0] = W-prev_in_cur[:, 0]
        prev_in_cur[:, 0] = prev_in_cur[:, 0] - 3
        #print("prev_in_cur: \n", prev_in_cur[:10])
        prev_in_cur = torch.from_numpy(prev_in_cur).to(self.device)

        # cur rays in prev frame
        cur_in_prev = proj_3D_2D(point_3D_current, fx, fy, cx, cy, prev_c2w)  # is float
        cur_in_prev[:, 0] = W-cur_in_prev[:, 0]
        cur_in_prev[:, 0] = cur_in_prev[:, 0] - 3
        #print("cur_in_prev: \n", cur_in_prev[:10])
        cur_in_prev = torch.from_numpy(cur_in_prev).to(self.device)

        loss_out_test = huber_loss(uv_in_cur, cur_in_prev, delta=1)   
        loss_out_test = loss_out_test.to(torch.float64).requires_grad_()
        print("testing loss of huber_loss output:\n", loss_out_test)

        if self.nice:
            # should pre-filter those out of bounding box depth value
            with torch.no_grad():
                det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                t = (self.bound.unsqueeze(0).to(device)-det_rays_o)/det_rays_d
                t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                inside_mask = t >= batch_gt_depth
            batch_rays_d = batch_rays_d[inside_mask]
            batch_rays_o = batch_rays_o[inside_mask]
            batch_gt_depth = batch_gt_depth[inside_mask]
            batch_gt_color = batch_gt_color[inside_mask]

        ret = self.renderer.render_batch_ray(
            self.c, self.decoders, batch_rays_d, batch_rays_o,  self.device, stage='color',  gt_depth=batch_gt_depth)
        depth, uncertainty, color = ret

        uncertainty = uncertainty.detach()
        if self.handle_dynamic:
            tmp = torch.abs(batch_gt_depth-depth)/torch.sqrt(uncertainty+1e-10)
            mask = (tmp < 10*tmp.median()) & (batch_gt_depth > 0)
        else:
            mask = batch_gt_depth > 0

        loss = (torch.abs(batch_gt_depth-depth) /
                torch.sqrt(uncertainty+1e-10))[mask].sum()

        if self.use_color_in_tracking:
            color_loss = torch.abs(
                batch_gt_color - color)[mask].sum()
            loss_out_test += self.w_color_loss*color_loss
        print("this is the loss: ", loss)
        loss_out_test.backward()
        optimizer.step()
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

        gt_color_prev = None
        gt_depth_prev = None
        prev_idx = None
        prev_camera_tensor = None


        device = self.device
        self.c = {}
        if self.verbose:
            pbar = self.frame_loader
        else:
            pbar = tqdm(self.frame_loader)

        for idx, gt_color, gt_depth, gt_c2w in pbar:
            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0]}")

            idx = idx[0]
            gt_depth = gt_depth[0]
            gt_color = gt_color[0]
            gt_c2w = gt_c2w[0]

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
                    delta = pre_c2w@self.estimate_c2w_list[idx-2].to(
                        device).float().inverse()
                    estimated_new_cam_c2w = delta@pre_c2w
                else:
                    estimated_new_cam_c2w = pre_c2w

                camera_tensor = get_tensor_from_camera(
                    estimated_new_cam_c2w.detach())
                # print("estimated_new_cam_c2w: \n", estimated_new_cam_c2w)

                if self.seperate_LR:
                    camera_tensor = camera_tensor.to(device).detach()
                    T = camera_tensor[-3:]
                    quad = camera_tensor[:4]
                    cam_para_list_quad = [quad]
                    quad = Variable(quad, requires_grad=True)
                    T = Variable(T, requires_grad=True)
                    camera_tensor = torch.cat([quad, T], 0)
                    cam_para_list_T = [T]
                    cam_para_list_quad = [quad]
                    optimizer_camera = torch.optim.Adam([{'params': cam_para_list_T, 'lr': self.cam_lr},
                                                         {'params': cam_para_list_quad, 'lr': self.cam_lr*0.2}])
                else:
                    camera_tensor = Variable(
                        camera_tensor.to(device), requires_grad=True)
                    cam_para_list = [camera_tensor]
                    optimizer_camera = torch.optim.Adam(
                        cam_para_list, lr=self.cam_lr)

                initial_loss_camera_tensor = torch.abs(
                    gt_camera_tensor.to(device)-camera_tensor).mean().item()
                candidate_cam_tensor = None
                current_min_loss = 10000000000.
                for cam_iter in range(1):      #self.num_cam_iters
                    if self.seperate_LR:
                        camera_tensor = torch.cat([quad, T], 0).to(self.device)

                    self.visualizer.vis(
                        idx, cam_iter, gt_depth, gt_color, camera_tensor, self.c, self.decoders)

                    loss = self.cam_pose_optimization_sift(
                        camera_tensor, gt_color, gt_depth, self.tracking_pixels, optimizer_camera, 
                        prev_camera_tensor, gt_depth_prev, gt_color_prev, idx)
                    if cam_iter == 0:
                        initial_loss = loss

                    loss_camera_tensor = torch.abs(
                        gt_camera_tensor.to(device)-camera_tensor).mean().item()
                    if self.verbose:
                        # if cam_iter == self.num_cam_iters-1:
                        print(
                            f'Re-rendering loss: {initial_loss:.2f}->{loss:.2f} ' +
                            f'camera tensor error: {initial_loss_camera_tensor:.4f}->{loss_camera_tensor:.4f}')
                    if loss < current_min_loss:
                        current_min_loss = loss
                    candidate_cam_tensor = camera_tensor.clone().detach()

                out_num = 10
                if (idx%out_num == 0):
                    self.visualizer.vis_rendered(
                        idx, cam_iter, gt_depth, gt_color, camera_tensor, self.c, self.decoders)


# mean value sghould be around 1
# check scale
# differnet data set different depth scale






                bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
                    [1, 4])).type(torch.float32).to(self.device)
                c2w = get_camera_from_tensor(
                    candidate_cam_tensor.clone().detach())
                c2w = torch.cat([c2w, bottom], dim=0)
            self.estimate_c2w_list[idx] = c2w.clone().cpu()
            self.gt_c2w_list[idx] = gt_c2w.clone().cpu()
            pre_c2w = c2w.clone()
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