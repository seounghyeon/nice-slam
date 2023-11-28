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
                        get_tensor_from_camera, proj_3D_2D, ray_to_3D)

from src.common import (get_samples)


from src.loss_functions.loss import huber_loss

from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer

import torch.nn as nn


from src.conv_onet.models.decoder import MLP_no_xyz
from src.utils.Mesher import Mesher

debug = False
debug2 = True
debug_multi = False
class Tracker(object):
    def __init__(self, cfg, args, slam,
                 dim=3, c_dim=32,
                 coarse_grid_len=2.0,  middle_grid_len=0.16, fine_grid_len=0.16,
                 color_grid_len=0.16, hidden_size=32, coarse=False, pos_embedding_method='fourier'
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

        self.mesher = Mesher(cfg, args, slam)  # Create an instance of Mesher

        self.color_list = []



    def cam_pose_optimization_sift(self, camera_tensor, gt_color, gt_depth, batch_size, 
                                   optimizer, prev_camera_tensor, gt_depth_prev, gt_color_prev, idx, color_list):
        """
        Do one iteration of camera iteration. Sample pixels, render depth/color, calculate loss and backpropagation.
        Uses GT images and depth
        Args:
            camera_tensor (tensor): camera tensor.  - is the optimized thing weight update backprop
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
        # print("camera_tensor inside optimize 2d2d: \n", camera_tensor)
        print("COLOR_LIST SIZE:",  len(self.color_list))
        # print("prev_camera_tensor inside optimize 2d2d: \n", prev_camera_tensor)

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
            this means on every edge ccc pixels are removed and the removed image is put inside
            1 is previous
            2 is current
            '''
        uv_prev, uv_cur, sbatch_rays_o, sbatch_rays_d, sbatch_gt_depth, sbatch_gt_color, sbatch_rays_o2, sbatch_rays_d2, sbatch_gt_depth2, sbatch_gt_color2 = get_samples_sift(
            Hedge, H-Hedge, Wedge, W-Wedge, batch_size, H, W, fx, fy, cx, cy, gt_depth_prev, gt_color_prev, prev_c2w, gt_depth, gt_color, c2w, idx, self.device, color_list)

        # print("from 2 size: ", sbatch_rays_d2.size(), sbatch_rays_o2.size())
        # print("from 1 size: ", sbatch_rays_d.size(), sbatch_rays_o.size())
        # print("uv_cur size: ", uv_cur.size())
        sift_feature_size = uv_cur.shape[0]



        # ret_cur = self.renderer.render_batch_ray(
        #     self.c, self.decoders, sbatch_rays_d2, sbatch_rays_o2,  self.device, stage='color',  gt_depth=sbatch_gt_depth2)
        # depth_cur, uncertainty_cur, color_cur = ret_cur


        # ret_prev = self.renderer.render_batch_ray(
        #     self.c, self.decoders, sbatch_rays_d, sbatch_rays_o,  self.device, stage='color',  gt_depth=sbatch_gt_depth)
        # depth_prev, uncertainty_prev, color_prev = ret_prev
        # fix THIS MAYBE WRONG
        # get 3D points for the sift feature rays
        point_3D_current = ray_to_3D(sbatch_rays_o2, sbatch_rays_d2, sbatch_gt_depth2, gt_depth, batch_size, sift_feature_size)
        point_3D_prev = ray_to_3D(sbatch_rays_o, sbatch_rays_d, sbatch_gt_depth, gt_depth_prev, batch_size, sift_feature_size)
        # previous image 3D points in current frame 2D / current image 3D points in previous frame 2D
        prev_in_cur = proj_3D_2D(point_3D_prev, W, fx, fy, cx, cy, c2w, self.device)  # is float
        cur_in_prev = proj_3D_2D(point_3D_current, W, fx, fy, cx, cy, prev_c2w, self.device)  # is float
        # intermediate value dont need CHANGE THIS
        # cur_in_prev = cur_in_prev.requires_grad_(True)
        # prev_in_cur = prev_in_cur.requires_grad_(True)


        # print("PREV IN CURRRRRRRRRRRR. ", prev_in_cur)
        if debug:
            print("POINT 3D: ", point_3D_current[:10])
            # outputs the projected 3D point of current frame backprojected into 2D
            # should be the same as sift feature uv
            # check difference for different data sets
            uv_in_cur_test = proj_3D_2D(point_3D_current, W, fx, fy, cx, cy, c2w, self.device)   # is float
            # uv_in_cur_test = np.round(uv_in_cur_test).astype(int)
            print("uv_prev and uv_cur\n", uv_prev[:5],"\n", uv_cur[:5])
            print("uv_in_cur_test:\n ", uv_in_cur_test[:5])


        # these go into loss and are backpropagated    
        # print("uv_prev: \n", uv_prev,"\n")
        # print("cur_in_prev:\n ", cur_in_prev)

        




        p_test_prev = point_3D_prev 
        # print("p_test_prev point: \n", p_test_prev[:3])

        p_test = point_3D_current
        # print("p_test point: \n", p_test[:3])


        # print("print the first 3D current: ", p)
        # # c_grid = None
        # stage = 'middle'
        # if stage == 'middle':
        #             middle_occ = self.fine_decoder(p, self.c)
        #             middle_occ = middle_occ.squeeze(0)
        #             raw = torch.zeros(middle_occ.shape[0], 4).to(device).float()
        #             raw[..., -1] = middle_occ
        #             print("raw_output: \n", raw)
        #             # print("RAW IN middle:\n", raw.size())
        #             return raw
        

        # # Call the forward method of the mlp_model to sample voxels
        # sampled_voxels = self.mlp_model(point_3D_current[:10], self.c)
        # # print("sample_voxel: \n", sampled_voxels)

        # # Print the sampled voxels or process them further as needed
        # print("Sampled Voxels: ", sampled_voxels)


        occupancy_and_color_cur = self.mesher.eval_points(p_test, self.decoders, self.c, 'fine', device)
        # print("occupancy and color out:\n ",occupancy_and_color_cur[:3])
        occupancy_and_color_prev = self.mesher.eval_points(p_test_prev, self.decoders, self.c, 'fine', device)
        # print("occupancy and color out_prev:\n ",occupancy_and_color_prev[:3])


        occupancy_middle_cur = self.mesher.eval_points(p_test, self.decoders, self.c, 'middle', device)
        # print("occupancy and color out middle:\n ",occupancy_middle_cur[:3])
        occupancy_middle_prev = self.mesher.eval_points(p_test_prev, self.decoders, self.c, 'middle', device)
        

        color_cur = self.mesher.eval_points(p_test, self.decoders, self.c, 'color', device)
        color_prev= self.mesher.eval_points(p_test_prev, self.decoders, self.c, 'color', device)


        fine_loss = torch.nn.L1Loss()(occupancy_and_color_cur, occupancy_and_color_prev)
        middle_loss = torch.nn.L1Loss()(occupancy_middle_cur, occupancy_middle_prev)
        color_loss = torch.nn.L1Loss()(color_cur, color_prev)

        distance3D_loss = torch.nn.L1Loss()(point_3D_current, point_3D_prev)

        total_grid_loss = fine_loss + middle_loss + color_loss + distance3D_loss


        # prediction, target
        # loss_out_test = torch.nn.functional.huber_loss(uv_prev, cur_in_prev, reduction='mean', delta=1.0)
        # loss_out_test = torch.nn.functional.huber_loss(uv_cur, prev_in_cur, reduction='mean', delta=1.0)
        loss_out_test = total_grid_loss
        # loss_out_test = torch.tensor(loss_out_test, requires_grad=True)
        # print("testing loss of huber_loss output:\n", loss_out_test)
        
        # print("printing loss_out_test should be fn: ", loss_out_test)


        # print("loss_out_test after all: \n", loss_out_test)
        # print("gradient before: ", camera_tensor.grad)

        # L1_loss = nn.L1Loss()
        # print("gradient cam_tensor before: \n", camera_tensor.grad)

        loss_out_test.backward()
        optimizer.step()
        # print("gradient cam_tensor after: \n", camera_tensor.grad)

        # print("loss_out_test: ", loss_out_test)
        # if camera_tensor.grad is not None:
        #     print("gradient after: ", camera_tensor.grad)

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



        ''' preprocessing images beforehand for multi frame tracking '''
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        Wedge = self.ignore_edge_W
        Hedge = self.ignore_edge_H
        # get_samples_sift(Hedge, H-Hedge, Wedge, W-Wedge, batch_size, H, W, fx, fy, cx, cy, gt_depth_prev, gt_color_prev, prev_c2w, gt_depth, gt_color, c2w, idx, self.device, color_list)

        # depth_prev = depth_prev[H0:H1, W0:W1]
        # color_prev = color_prev[H0:H1, W0:W1]

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
        # idx starts at 0 for frame 0
        for idx, gt_color, gt_depth, gt_c2w in pbar:
            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0]}")

            idx = idx[0]
            gt_depth = gt_depth[0]
            gt_color = gt_color[0]
            gt_c2w = gt_c2w[0]
            
            frame_num_d = 5
            print("Size of gt_color tensor:", gt_color.size())

            # append the frame 
            self.color_list.append(gt_color)



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
                # print("estimated_new_cam_c2w: \n", estimated_new_cam_c2w)


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
                    # print("cam_iter: ", cam_iter)

                    # print("current cam_tensor first: ", camera_tensor)

                    self.visualizer.vis(
                        idx, cam_iter, gt_depth, gt_color, camera_tensor, self.c, self.decoders)
                    loss = self.cam_pose_optimization_sift(
                        camera_tensor, gt_color, gt_depth, self.tracking_pixels, optimizer_camera, 
                        prev_camera_tensor, gt_depth_prev, gt_color_prev, idx, self.color_list)
                    # print("batch_Rays_d_prev = ", batch_rays_d_prev)
                    # print("size of batch_rays_d_prev = ", batch_rays_d_prev.size())
                    # print("size of batch_rays_o_prev = ", batch_rays_o_prev.size())

                    if cam_iter == 0:
                        initial_loss = loss
                        # print("initial_loss cor cam iter 0 = ", initial_loss)
                    loss_camera_tensor = torch.abs(gt_camera_tensor.to(device)-camera_tensor).mean().item()
                    if self.verbose:
                        if cam_iter == self.num_cam_iters-1:
                            print(
                                f'Re-rendering loss: {initial_loss:.2f}->{loss:.2f} ' +
                                f'camera tensor error: {initial_loss_camera_tensor:.4f}->{loss_camera_tensor:.4f}')

                    # print("loss: ", loss)   
                    # print("current cam_tensor second: ", camera_tensor)

                    if loss < current_min_loss:
                        current_min_loss = loss
                        print("current min loss: ",current_min_loss)
                    candidate_cam_tensor = camera_tensor.clone().detach()
                # if debug == True:


                # renders and outputs the image
                # every out_num image is rendered and saved in rendered_images
                if (debug2 == True):
                    out_num = 100
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
                



            # if frame index is the one after the first tracking (needs to be after 4th frames) 
            # because frame starts at, first reset to the last input frame is at frame 5
            # input frames are  0 1 2 3 4 - jump to next list with 4 5 6 7 8 9 - jump to next list with 9 10 11 12 13 14
            # meaning when current input index is registered as 5, deletes all up to 4 (without 4) and adds 5 
            # this gets the new list with 4 5 6 ..
            if (idx == frame_num_d-1):
                self.color_list = self.color_list[-1:]
                print("first frame list")
                print("length of frames: ", len(self.color_list))
            # if the color_list is length 6 
            if (len(self.color_list)==frame_num_d+1):
                self.color_list = self.color_list[-1:]
                print("other frame list")

