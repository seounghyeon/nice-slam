import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.common_sift import get_camera_from_tensor
import cv2
# from src.sift import SIFTMatcher


class Visualizer(object):
    """
    Visualize intermediate results, render out depth, color and depth uncertainty images.
    It can be called per iteration, which is good for debugging (to see how each tracking/mapping iteration performs).

    """

    def __init__(self, freq, inside_freq, vis_dir, renderer, verbose, device='cuda:0'):
        self.freq = freq
        self.device = device
        self.vis_dir = vis_dir
        self.verbose = verbose
        self.renderer = renderer
        self.inside_freq = inside_freq
        os.makedirs(f'{vis_dir}', exist_ok=True)

    def vis(self, idx, iter, gt_depth, gt_color, c2w_or_camera_tensor, c,
            decoders):
        """
        Visualization of depth, color images and save to file.

        Args:
            idx (int): current frame index.
            iter (int): the iteration number.
            gt_depth (tensor): ground truth depth image of the current frame.
            gt_color (tensor): ground truth color image of the current frame.
            c2w_or_camera_tensor (tensor): camera pose, represented in 
                camera to world matrix or quaternion and translation tensor.
            c (dicts): feature grids.
            decoders (nn.module): decoders.
        """
        with torch.no_grad():
            if (idx % self.freq == 0) and (iter % self.inside_freq == 0):
                gt_depth_np = gt_depth.cpu().numpy()
                gt_color_np = gt_color.cpu().numpy()
                if len(c2w_or_camera_tensor.shape) == 1:
                    bottom = torch.from_numpy(
                        np.array([0, 0, 0, 1.]).reshape([1, 4])).type(
                            torch.float32).to(self.device)
                    c2w = get_camera_from_tensor(
                        c2w_or_camera_tensor.clone().detach())
                    c2w = torch.cat([c2w, bottom], dim=0)
                else:
                    c2w = c2w_or_camera_tensor

                depth, uncertainty, color = self.renderer.render_img(
                    c,
                    decoders,
                    c2w,
                    self.device,
                    stage='color',
                    gt_depth=gt_depth)
                depth_np = depth.detach().cpu().numpy()
                color_np = color.detach().cpu().numpy()
                depth_residual = np.abs(gt_depth_np - depth_np)
                depth_residual[gt_depth_np == 0.0] = 0.0
                color_residual = np.abs(gt_color_np - color_np)
                color_residual[gt_depth_np == 0.0] = 0.0

                fig, axs = plt.subplots(2, 3)
                fig.tight_layout()
                max_depth = np.max(gt_depth_np)
                axs[0, 0].imshow(gt_depth_np, cmap="plasma",
                                 vmin=0, vmax=max_depth)
                axs[0, 0].set_title('Input Depth')
                axs[0, 0].set_xticks([])
                axs[0, 0].set_yticks([])
                axs[0, 1].imshow(depth_np, cmap="plasma",
                                 vmin=0, vmax=max_depth)
                axs[0, 1].set_title('Generated Depth')
                axs[0, 1].set_xticks([])
                axs[0, 1].set_yticks([])
                axs[0, 2].imshow(depth_residual, cmap="plasma",
                                 vmin=0, vmax=max_depth)
                axs[0, 2].set_title('Depth Residual')
                axs[0, 2].set_xticks([])
                axs[0, 2].set_yticks([])
                gt_color_np = np.clip(gt_color_np, 0, 1)
                color_np = np.clip(color_np, 0, 1)
                color_residual = np.clip(color_residual, 0, 1)
                axs[1, 0].imshow(gt_color_np, cmap="plasma")
                axs[1, 0].set_title('Input RGB')
                axs[1, 0].set_xticks([])
                axs[1, 0].set_yticks([])
                axs[1, 1].imshow(color_np, cmap="plasma")
                axs[1, 1].set_title('Generated RGB')
                axs[1, 1].set_xticks([])
                axs[1, 1].set_yticks([])
                axs[1, 2].imshow(color_residual, cmap="plasma")
                axs[1, 2].set_title('RGB Residual')
                axs[1, 2].set_xticks([])
                axs[1, 2].set_yticks([])
                plt.subplots_adjust(wspace=0, hspace=0)
                plt.savefig(
                    f'{self.vis_dir}/{idx:05d}_{iter:04d}.jpg', bbox_inches='tight', pad_inches=0.2)
                plt.clf()

                if self.verbose:
                    print(
                        f'Saved rendering visualization of color/depth image at {self.vis_dir}/{idx:05d}_{iter:04d}.jpg')
                    


    def vis_rendered(self, idx, iter, gt_depth, gt_color, c2w_or_camera_tensor, c,
            decoders):
        """
        Visualization of depth, color images and save to file.

        Args:
            idx (int): current frame index.
            iter (int): the iteration number.
            gt_depth (tensor): ground truth depth image of the current frame.
            gt_color (tensor): ground truth color image of the current frame.
            c2w_or_camera_tensor (tensor): camera pose, represented in 
                camera to world matrix or quaternion and translation tensor.
            c (dicts): feature grids.
            decoders (nn.module): decoders.
        """
     
        print("vis_rendered is entered\n")
        # print("iter is ", iter)
        # camera iteration starts from 0 so it is 1 lower than division by 10 would give no rest
        # add 1 to give every 10th image
        iter = iter+1

        with torch.no_grad():
            gt_depth_np = gt_depth.cpu().numpy()
            gt_color_np = gt_color.cpu().numpy()
            if len(c2w_or_camera_tensor.shape) == 1:
                bottom = torch.from_numpy(
                    np.array([0, 0, 0, 1.]).reshape([1, 4])).type(
                        torch.float32).to(self.device)
                c2w = get_camera_from_tensor(
                    c2w_or_camera_tensor.clone().detach())
                c2w = torch.cat([c2w, bottom], dim=0)
            else:
                c2w = c2w_or_camera_tensor

            depth, uncertainty, color = self.renderer.render_img(
                c,
                decoders,
                c2w,
                self.device,
                stage='color',
                gt_depth=gt_depth)
            depth_np = depth.detach().cpu().numpy()
            color_np = color.detach().cpu().numpy()
            depth_residual = np.abs(gt_depth_np - depth_np)
            depth_residual[gt_depth_np == 0.0] = 0.0
            color_residual = np.abs(gt_color_np - color_np)
            color_residual[gt_depth_np == 0.0] = 0.0

            # gt_color_np = np.clip(gt_color_np, 0, 1)

            # plt.imshow(color_np, cmap="plasma")
            # plt.axis('off')  # To turn off the axis
            # plt.savefig(f"output_image_plt_{idx}.png", bbox_inches='tight', pad_inches=0)
            # plt.close()

            ########################################################################################################
            # change the np array of the image to cv2 format

            # color is set from 0 to 1 to ensure range of intensity for the pixel is inside this valid range
            color_np = np.clip(color_np, 0, 1)
            # Check if the tensor shape is CxHxW, and if so, transpose it to HxWxC
            if color_np.shape[0] == 3:
                color_np = np.transpose(color_np, (1, 2, 0))

            # If color values are in [0,1], scale to [0,255]
            if color_np.max() <= 1.0:
                color_np = (color_np * 255).astype(np.uint8)

            # Save the image
            # cv2.imwrite("output_image.jpg", color_np)  # Save as JPG
            color_np_bgr = cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)


            # color is set from 0 to 1 to ensure range of intensity for the pixel is inside this valid range
            gt_color_np = np.clip(gt_color_np, 0, 1)
            # Check if the tensor shape is CxHxW, and if so, transpose it to HxWxC
            if gt_color_np.shape[0] == 3:
                gt_color_np = np.transpose(gt_color_np, (1, 2, 0))

            # If color values are in [0,1], scale to [0,255]
            if gt_color_np.max() <= 1.0:
                gt_color_np = (gt_color_np * 255).astype(np.uint8)

            # Save the image
            # cv2.imwrite("output_image.jpg", gt_color)  # Save as JPG
            gt_color_bgr = cv2.cvtColor(gt_color_np, cv2.COLOR_RGB2BGR)
            ########################################################################################################








            cv2.imwrite(f"rendered_images/output_image_cv2_{idx}.png", color_np_bgr)

            # matcher = SIFTMatcher()
            # matched_image, keypoints_1, keypoints_2, matches = matcher.match(idx, gt_color_bgr, color_np_bgr)


            #WHY REPEAT FOR 30 TIMES 


