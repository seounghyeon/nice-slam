import copy
import os
import time

import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common_f import (get_camera_from_tensor, get_samples,
                        get_tensor_from_camera, proj_3D_2D, ray_to_3D, get_rays_from_uv)


from src.sift import SIFTMatcher

from src.loss_functions.loss import (huber_loss, huber_loss_sum)

from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer

import torch.nn as nn


from src.conv_onet.models.decoder import MLP_no_xyz
from src.utils.Mesher import Mesher


from src.image_processing import img_pre

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

    def optimize_cam_in_batch(self, camera_tensor, gt_color, gt_depth, batch_size, optimizer,
                               prev_camera_tensor, gt_depth_prev, gt_color_prev, idx, uv_prev, uv_cur, index_prev, index_cur, i_prev, j_prev, i_cur, j_cur, gt_depth_prev_batch_sift, gt_depth_batch_sift, W1, H0, colors_cur, gt_color_clone):
        """
        Do one iteration of camera iteration. Sample pixels, render depth/color, calculate loss and backpropagation.

        Args:
            camera_tensor (tensor): camera tensor.
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            batch_size (int): batch size, number of sampling rays.
            optimizer (torch.optim): camera optimizer.

        Returns:
            loss (float): The value of loss.
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        optimizer.zero_grad()
        c2w = get_camera_from_tensor(camera_tensor)
        Wedge = self.ignore_edge_W
        Hedge = self.ignore_edge_H
        prev_c2w = get_camera_from_tensor(prev_camera_tensor)

        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(
            Hedge, H-Hedge, Wedge, W-Wedge, batch_size, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)
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



        # sakdjalksdj

        # 3D loss
        rays_o_cur, rays_d_cur = get_rays_from_uv(i_cur, j_cur, c2w, H, W, fx, fy, cx, cy, device)
        rays_o_prev, rays_d_prev = get_rays_from_uv(i_prev, j_prev, prev_c2w, H, W, fx, fy, cx, cy, device)
        point_3D_current = ray_to_3D(rays_o_cur, rays_d_cur, gt_depth_batch_sift, gt_depth)
        point_3D_prev = ray_to_3D(rays_o_prev, rays_d_prev, gt_depth_prev_batch_sift, gt_depth_prev)

        p_test_prev = point_3D_prev 
        # print("p_test_prev point: \n", p_test_prev[:3])

        p_test = point_3D_current        

        occupancy_and_color_cur = self.mesher.eval_points(p_test, self.decoders, self.c, 'fine', device)
        occupancy_and_color_prev = self.mesher.eval_points(p_test_prev, self.decoders, self.c, 'fine', device)
        occupancy_middle_cur = self.mesher.eval_points(p_test, self.decoders, self.c, 'middle', device)
        occupancy_middle_prev = self.mesher.eval_points(p_test_prev, self.decoders, self.c, 'middle', device)

        color_cur = self.mesher.eval_points(p_test, self.decoders, self.c, 'color', device)
        color_prev= self.mesher.eval_points(p_test_prev, self.decoders, self.c, 'color', device)


        # fine_loss = torch.nn.L1Loss()(occupancy_and_color_cur, occupancy_and_color_prev)
        # print("fine loss: ", fine_loss)
        # middle_loss = torch.nn.L1Loss()(occupancy_middle_cur, occupancy_middle_prev)
        # print("middle loss: ",middle_loss)
        # color_loss3D = torch.nn.L1Loss()(color_cur, color_prev)
        # print("color loss: ", color_loss)
        # distance3D_loss = torch.nn.L1Loss()(point_3D_current, point_3D_prev)
        # print("3D distance loss: ", distance3D_loss)




        fine_loss = torch.abs(occupancy_and_color_cur - occupancy_and_color_prev).sum()
        middle_loss = torch.abs(occupancy_middle_cur - occupancy_middle_prev).sum()
        color_loss3D = torch.abs(color_cur - color_prev).sum()
        distance3D_loss = torch.abs(point_3D_current - point_3D_prev).sum()






        # 2D loss
        # the calculated uv_prev_in_cur should be the target (reference) and the variable(prediction) is the current output color and occupancy
        # uv_prev_in_cur = proj_3D_2D(point_3D_prev, W, fx, fy, cx, cy, c2w, self.device)   # is float
        # # index_1 = (v_reshaped_1 * W1) + u_reshaped_1
        # uv_prev_in_cur[:, 0] = W-uv_prev_in_cur[:, 0]
        # prev_in_cur = torch.from_numpy(prev_in_cur).to(self.device)

        # outputs the projected 3D point of current frame backprojected into 2D
        # should be the same as sift feature uv
        # # check difference for different data sets









        # uv_in_cur_test = proj_3D_2D(point_3D_current, W1, H0, fx, fy, cx, cy, c2w, self.device)  # is float
        # # print("uv_in_cur_test:\n ", uv_in_cur_test[:10])
        # uv_in_cur_test = uv_in_cur_test.to(torch.float32)

        # print("uv_inCUrtest ", uv_in_cur_test[:10])
        
        # print("Type of uv_in_cur_test:", type(uv_in_cur_test))
        # # print("size of image after img_pre: ", image_pred.size())

        # image_pred = img_pre(gt_color_clone)

        # colors_test = torch.tensor([image_pred[int(v), int(u)] for u, v in uv_in_cur_test], dtype=torch.float32, device=self.device)
        # print("color test of current 3D point in current 3D frame \n", colors_test[:10])
























        uv_prev_in_cur = proj_3D_2D(point_3D_prev, W1, H0, fx, fy, cx, cy, c2w, self.device)  # is float
        loss_2d =  huber_loss_sum(uv_cur, uv_prev_in_cur, delta=1)
        # loss_2d_mean =  huber_loss(uv_cur, uv_prev_in_cur, delta=1)

        print("loss 2d huber sum: ", loss_2d)
        # print("loss 2d huber mean: ", loss_2d_mean)
        # print("uv_cur size asdad ", uv_cur.shape)
        


        """   
            test the 3D point from test of current 3D point in current frame (get uv) for color and check if it matches with the one given from sift 
            3D point color can be taken from above - have to check if the scaling or format is the same
            then do loss
        """


        print("shape of color fine: ", color_cur.shape)
        color_cur_copy = color_cur.clone()
        color_cur_2D = color_cur_copy[:, :3]

        color_cur_2D = torch.clamp(color_cur_2D, 0, 1)

        # If color values are in [0,1], scale to [0,255]
        if color_cur_2D.max() <= 1.0:
            color_cur_2D = (color_cur_2D * 255).to(torch.uint8)
        # color_cur_2D = torch.round((color_cur_2D + 1.0) * 127.5).to(torch.int)
        # print("color_cur 2d",color_cur_2D[:10])


        color_loss_2d = huber_loss_sum(colors_cur, color_cur_2D)
        print("2d color loss 0.1: ", color_loss_2d*0.1)
        # total_grid_loss = fine_loss + middle_loss + color_loss3D + 20*distance3D_loss + loss_2d
        total_grid_loss = loss_2d +color_loss_2d *0.1





        supervis_loss = (torch.abs(batch_gt_depth-depth) /
                torch.sqrt(uncertainty+1e-10))[mask].sum()
        



        # print("fine loss: ", fine_loss)
        # print("middle loss: ",middle_loss)
        # print("color loss: ", color_loss3D)
        # print("3D distance loss: ", distance3D_loss)
        # print("loss depth: ", supervis_loss)


        loss = total_grid_loss + supervis_loss 


        if self.use_color_in_tracking:
            color_loss = torch.abs(
                batch_gt_color - color)[mask].sum()
            loss += self.w_color_loss*color_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

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

        for idx, gt_color, gt_depth, gt_c2w in pbar:


            H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
            Wedge = self.ignore_edge_W
            Hedge = self.ignore_edge_H


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




                # get sift_matcher
                sift_matcher1 = SIFTMatcher()  # Instantiate the class
                H0 = Hedge 
                H1 = H-Hedge
                W0 = Wedge
                W1 = W-Wedge
                gt_color_prev_clone = gt_color_prev.clone().detach()
                gt_color_clone = gt_color.clone().detach()
                gt_color_prev_clone = gt_color_prev_clone[H0:H1, W0:W1]
                gt_color_clone = gt_color_clone[H0:H1, W0:W1]


                gt_depth_prev_clone = gt_depth_prev.clone().detach()
                gt_depth_prev_clone = gt_depth_prev_clone[H0:H1, W0:W1]

                gt_depth_clone = gt_depth.clone().detach()
                gt_depth_clone = gt_depth_clone[H0:H1, W0:W1]


                # Hedge, H-Hedge, Wedge, W-Wedge, 
                    # print("in select uv before match() h0,h1,w0,w1: ", H0, H1, W0, W1)
                i, j = torch.meshgrid(torch.linspace(
                    W0, W1-1, W1-W0).to(device='cpu'), torch.linspace(H0, H1-1, H1-H0).to(device='cpu'))
                
                
                i = i.t()  # transpose
                j = j.t()
                i = i.reshape(-1)
                j = j.reshape(-1)
                gt_depth_prev_clone = gt_depth_prev_clone.reshape(-1)
                gt_depth_clone = gt_depth_clone.reshape(-1)

                # color_cur = color_cur[H0:H1, W0:W1]    
                # color_prev = color_prev[H0:H1, W0:W1]
                uv_prev, uv_cur, index_prev, index_cur, colors_cur = sift_matcher1.match(i, j, idx, gt_color_prev_clone, gt_color_clone)
                # print("uv_cur outside: ", uv_cur[:10])

                i_prev = i[index_prev]  
                j_prev = j[index_prev]  

                i_cur = i[index_cur]  
                j_cur = j[index_cur]  



                gt_depth_prev_batch_sift = gt_depth_prev[H0:H1, W0:W1]
                gt_depth_batch_sift = gt_depth[H0:H1, W0:W1]
                gt_depth_prev_batch_sift = gt_depth_prev_batch_sift.reshape(-1)
                gt_depth_batch_sift = gt_depth_batch_sift.reshape(-1)

                gt_depth_prev_batch_sift = gt_depth_prev_batch_sift[index_prev]  # (n)
                gt_depth_batch_sift = gt_depth_batch_sift[index_cur]  # (n)



                # print("OUTSIDE CAM ITER  UV CUR: ",uv_cur[:10])


                # GETS UV AND IDX NOW NEED TO GET THE DEPTHS OF UVs inside get_samples - not random




                for cam_iter in range(self.num_cam_iters):
                    if self.seperate_LR:
                        camera_tensor = torch.cat([quad, T], 0).to(self.device)

                    self.visualizer.vis(
                        idx, cam_iter, gt_depth, gt_color, camera_tensor, self.c, self.decoders)


                    loss = self.optimize_cam_in_batch(
                        camera_tensor, gt_color, gt_depth, self.tracking_pixels, optimizer_camera,
                        prev_camera_tensor, gt_depth_prev, gt_color_prev, idx, uv_prev, uv_cur, index_prev, index_cur, i_prev, j_prev, i_cur, j_cur, gt_depth_prev_batch_sift, gt_depth_batch_sift,W1, H0, colors_cur, gt_color_clone)

                    if cam_iter == 0:
                        initial_loss = loss

                    loss_camera_tensor = torch.abs(
                        gt_camera_tensor.to(device)-camera_tensor).mean().item()
                    if self.verbose:
                        if cam_iter == self.num_cam_iters-1:
                            print(
                                f'Re-rendering loss: {initial_loss:.2f}->{loss:.2f} ' +
                                f'camera tensor error: {initial_loss_camera_tensor:.4f}->{loss_camera_tensor:.4f}')
                    if loss < current_min_loss:
                        current_min_loss = loss
                        candidate_cam_tensor = camera_tensor.clone().detach()
                        print("current min loss: ", current_min_loss)

                # renders and outputs the image
                # every out_num image is rendered and saved in rendered_images

                out_num = 100
                if (idx%out_num == 0):
                    self.visualizer.vis_rendered(
                        idx, cam_iter, gt_depth, gt_color, camera_tensor, self.c, self.decoders)





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
