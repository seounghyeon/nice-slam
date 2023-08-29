import cv2
import torch
from src.utils.Renderer import Renderer
import numpy as np


"""
SIFT Feature Matching Class

- Inputs are index of images
- image1 is query image, image2 is train image
- both images where keypoints should be found and matched
- outputs:  - uv_1      uv coordinates of keypoints in image 1 (in order of best to worst match) 
            - uv_2      uv coordinates of keypoints in image 1 (in order of best to worst match) 
            - index     index of 1D vector of the image
            - matches

"""
class SIFTMatcher:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.brutef = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        self.device = device=torch.device("cuda")


    def match(self, idx, image1in, image2in):
        
        if image2in is None:
            print("\nTHIS IS NONE IN IMAGE2IN\n\n")
            return None
        # detach input images and change them from tensor to cv2 format
        ############################################
        # Detach the tensor from the GPU
        image1in_cpu = image1in.cpu()
        image2in_cpu = image2in.cpu()

        # Convert to NumPy arrays
        np_img1 = image1in_cpu.numpy()
        np_img2 = image2in_cpu.numpy()

        # color is set from 0 to 1 to ensure range of intensity for the pixel is inside this valid range
        np_img1 = np.clip(np_img1, 0, 1)
        np_img2 = np.clip(np_img2, 0, 1)
        # Check if the tensor shape is CxHxW, and if so, transpose it to HxWxC
        print("Shape of np_img2:", np_img2.shape)
        if np_img1.shape[0] == 3:
            np_img1 = np.transpose(np_img1, (1, 2, 0))
        if np_img2.shape[0] == 3:
            np_img2 = np.transpose(np_img2, (1, 2, 0))
        # If color values are in [0,1], scale to [0,255]
        if np_img1.max() <= 1.0:
            np_img1 = (np_img1 * 255).astype(np.uint8)
        if np_img2.max() <= 1.0:
            np_img2 = (np_img2 * 255).astype(np.uint8)
        # Save the images
        image1 = cv2.cvtColor(np_img1, cv2.COLOR_RGB2BGR)
        image2 = cv2.cvtColor(np_img2, cv2.COLOR_RGB2BGR)
        ############################################

        
        debug = True


        # Detect and compute keypoints and descriptors
        keypoints_1, descriptors_1 = self.sift.detectAndCompute(image1, None)
        keypoints_2, descriptors_2 = self.sift.detectAndCompute(image2, None)

        # Create matches using brute force algorithm
        matches = self.brutef.match(descriptors_1, descriptors_2)

        # Sort matches by their distance
        matches = sorted(matches, key=lambda x: x.distance)

        #only save top 100 matches
        matches = matches[:100]

        # matches saved in tensor
        matches_tensor = torch.tensor([(match.queryIdx, match.trainIdx, match.distance) for match in matches], dtype=torch.float32)
        if(debug):
            print("size of matches",matches_tensor.size())


        # for image1 / query image
        W0 = 0
        W1 = image1.shape[1]        #shape[1] is width
        H0 = 0
        H1 = image1.shape[0]        #shape[0] is height

        # both are 2D tensors with position encoded 
        # u_coord for i/width/column
        # v_coord for j/height/row information
        u_coord, v_coord = torch.meshgrid(torch.linspace(
                W0, W1-1, W1-W0).to(self.device), torch.linspace(H0, H1-1, H1-H0).to(self.device))

        u_coord = u_coord.t()  # transpose
        v_coord = v_coord.t()

        if(debug):
            print("u_coord is: ",u_coord.size())
            print(u_coord[:10, :10])                # printing tensor
            print("v_coord is: ",v_coord.size())
            print(v_coord[:10, :10])                # printing tensor

        # reshape both into 1D tensors
        u_coord = u_coord.reshape(-1)
        v_coord = v_coord.reshape(-1)


        u_reshaped_1 = torch.tensor([keypoints_1[mat.queryIdx].pt[0] for mat in matches], dtype=torch.int64, device=self.device)
        v_reshaped_1 = torch.tensor([keypoints_1[mat.queryIdx].pt[1] for mat in matches], dtype=torch.int64, device=self.device)
        uv_1 = torch.stack((u_reshaped_1, v_reshaped_1), dim=1)

        u_reshaped_2 = torch.tensor([keypoints_2[mat.trainIdx].pt[0] for mat in matches], dtype=torch.int64, device=self.device)
        v_reshaped_2 = torch.tensor([keypoints_2[mat.trainIdx].pt[1] for mat in matches], dtype=torch.int64, device=self.device)
        uv_2 = torch.stack((u_reshaped_2, v_reshaped_2), dim=1)

        if(debug):
            print("u_reshaped first 10: ",u_reshaped_1[:10])
            print("v_reshaped first 10: ",v_reshaped_1[:10])
            print("combined tensor keypoints1:", uv_1[:10])
            print("u kp2 first 10: ",u_reshaped_1[:10])
            print("v kp2 first 10: ",v_reshaped_1[:10])
            print("kp2 first 10:", uv_1[:10])
            
            # shows image 2 with the first 10 keypoints of the matches 
            for uv in uv_2[:10]:
                u, v = int(uv[0]), int(uv[1])
                cv2.circle(image2, (u, v), radius=10, color=(0, 255, 0), thickness=-1)  # Draw a green circle
            cv2.imshow('image2', image2)

        # (row(u) * width) + col(v) computes the index of uv coord (in the 1D tensor)
        index_1 = (v_reshaped_1 * W1) + u_reshaped_1
        index_2 = (v_reshaped_2 * W1) + u_reshaped_2
        if(debug):
            print("index size: ", index_1.size())
            print("index first: ", index_1)
            print("index size: ", index_2.size())
            print("index first: ", index_2)
        #?? not needed
        # # Append to each list
        # list_keypoints_1 = list(zip(u_reshaped_1, v_reshaped_1))
        # list_keypoints_2 = [(int(keypoints_2[mat.trainIdx].pt[0]), int(keypoints_2[mat.trainIdx].pt[1])) for mat in matches]

        # Draw and display the first 100 matches
        matched_image = cv2.drawMatches(image1, keypoints_1, image2, keypoints_2, matches[:100], None, flags=2)
        cv2.imwrite(f'/home/shham/Pictures/matched_images/match_{idx}.jpg', matched_image)

        return uv_1, uv_2, index_1, index_2

