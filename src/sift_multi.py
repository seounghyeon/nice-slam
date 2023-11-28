import cv2
import torch
from src.utils.Renderer import Renderer
import numpy as np
import time
"""
SIFT Feature Matching Class

- Inputs are index of images
- image1 is query image, image2 is train image
- both images where keypoints should be found and matched
- outputs:  - uv_1      uv coordinates of keypoints in image 1 (in order of best to worst match) 
            - uv_2      uv coordinates of keypoints in image 2 (in order of best to worst match) 
            - index     indices of matched uv coordinate in 1D tensor
            - matches

"""
class SIFTMatcher:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.brutef = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        self.device = device=torch.device("cuda")
        self.keypoints_list = []  # List to store keypoints for each frame
        self.descriptors_list = []  # List to store descriptors for each frame
        self.uv_tensor_list_2compare = []
        self.indices = []
        self.id_list = []

    def match(self, i, j, idx, image1in, image2in):
        if image1in is None or image1in.numel() == 0:
            # print("\nTHIS IS NONE IN IMAGE1IN no previous image saved up\n\n")
            return None, None, None, None
        # detach input images and change them from tensor to cv2 format
        ############################################
        # Detach the tensor from the GPU

        image1in_cpu = image1in.cpu()
        image2in_cpu = image2in.cpu()
        # Convert to NumPy arrays
        np_img1 = image1in_cpu.numpy()
        np_img2 = image2in_cpu.numpy()
        print("image in in match: ", image1in.size(), image2in.size())  # is 420x580

        # color is set from 0 to 1 to ensure range of intensity for the pixel is inside this valid range
        np_img1 = np.clip(np_img1, 0, 1)
        np_img2 = np.clip(np_img2, 0, 1)
        # Check if the tensor shape is CxHxW, and if so, transpose it to HxWxC
        # print("Shape of np_img2:", np_img2.shape)
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
        cv2.imwrite(f'/home/shham/Code/resize_frame/newcolor/match_{idx-1}.jpg', image1)

        
        debug = False

        # Detect and compute keypoints and descriptors
        keypoints_1, descriptors_1 = self.sift.detectAndCompute(image1, None)
        keypoints_2, descriptors_2 = self.sift.detectAndCompute(image2, None)
        print("keypoint sizes of frame 1 and 2: ", len(keypoints_1), len(keypoints_2))
        # Create matches using brute force algorithm
        matches = self.brutef.match(descriptors_1, descriptors_2)

        # Sort matches by their distance
        matches = sorted(matches, key=lambda x: x.distance)

        #only save top 100 matches
        # matches = matches[:100]

        # matches saved in tensor
        matches_tensor = torch.tensor([(match.queryIdx, match.trainIdx, match.distance) for match in matches], dtype=torch.float32)
        if(debug):
            print("size of matches",matches_tensor.size())


        u_coord = i
        v_coord = j





        """
        UV NEEDS TO ADD 20 because the uv here starts at 0 but in reality it starts at 20
        it was reduced down
        """







        u_reshaped_1 = torch.tensor([keypoints_1[mat.queryIdx].pt[0] for mat in matches], dtype=torch.int64, device=self.device)
        v_reshaped_1 = torch.tensor([keypoints_1[mat.queryIdx].pt[1] for mat in matches], dtype=torch.int64, device=self.device)
        uv_1 = torch.stack((u_reshaped_1, v_reshaped_1), dim=1)
        # print("this is uv_1 size: ", uv_1.size())
        u_reshaped_2 = torch.tensor([keypoints_2[mat.trainIdx].pt[0] for mat in matches], dtype=torch.int64, device=self.device)
        v_reshaped_2 = torch.tensor([keypoints_2[mat.trainIdx].pt[1] for mat in matches], dtype=torch.int64, device=self.device)



        uv_2 = torch.stack((u_reshaped_2, v_reshaped_2), dim=1)
        # print("this is uv_2 size: ", uv_2.size())

        
        
        # print("\nuv_2 in sift: \n", uv_2[:10])
        if(debug):
            # print("u_reshaped first 10: ",u_reshaped_1[:10])
            # print("v_reshaped first 10: ",v_reshaped_1[:10])
            print("combined tensor keypoints1:", uv_1[:10])
            print("u kp2 first 10: ",u_reshaped_1[:10])
            print("v kp2 first 10: ",v_reshaped_1[:10])
            print("kp2 first 10:", uv_1[:10])

        # shows image 2 with the first 10 keypoints of the matches 
        for uv in uv_2[:10]:
            u, v = int(uv[0]), int(uv[1])
            # cv2.circle(image2, (u, v), radius=10, color=(0, 255, 0), thickness=-1)  # Draw a green circle
        #cv2.imshow('image2', image2)


        for uv in uv_1[:10]:
            u, v = int(uv[0]), int(uv[1])
            # cv2.circle(image1, (u, v), radius=10, color=(0, 255, 0), thickness=-1)  # Draw a green circle


        # starts at Wedge is 20
        uv_2 += 20
        # print("\nuv_2 in sift: \n", uv_2[:10])





        # print("IMAGE SHAPE ",image1.shape[0], image1.shape[1])          # (420, 580, 3) with -20 on each side and top and bottom


        # (row(u) * width) + col(v) computes the index of uv coord (in the 1D tensor)
        W1 = image1.shape[1]
        index_1 = (v_reshaped_1 * W1) + u_reshaped_1
        index_2 = (v_reshaped_2 * W1) + u_reshaped_2
        if(debug):
            print("index size: ", index_1.size())
            print("index first: ", index_1)
            print("index size: ", index_2.size())
            print("index second: ", index_2)



        # print("this is index2 in sift: ",index_2[:10])

        # test_u =u_coord[index_2]
        # test_v = v_coord[index_2]

        # print("kp1_sort type and shape:", type(keypoints_2))
        # Draw and display the first 100 matches
        matched_image = cv2.drawMatches(image1, keypoints_1, image2, keypoints_2, matches[:100], None, flags=2)
        cv2.imwrite(f'/home/shham/Pictures/matching_test/match_{idx}.jpg', matched_image)
        # cv2.imwrite(f'/home/seoungham/Pictures/test_img/match_{idx}.jpg', image2)
        uv_1 = uv_1.to(torch.float32)
        uv_2 = uv_2.to(torch.float32)
        # print ("uv_1 printing: ", uv_1)
        # print("uv_1 and uv_2 in match: ", uv_1, uv_2)
        return uv_1, uv_2, index_1, index_2



    # inputs are Nx3 dim tensors with [uv_1, uv_2, index] uv_1 is uv of image 1 and uv_2 uv of image 2
    def compare_ID(self, match_tensor1, match_tensor2):
        # Extract u, v, and ID columns from match_tensor1 and match_tensor2
        u1, v1, id1 = match_tensor1[:, 0], match_tensor1[:, 1], match_tensor1[:, 2]
        u2, v2, id2 = match_tensor2[:, 0], match_tensor2[:, 1], match_tensor2[:, 2]

        # Initialize lists to store matched IDs
        matched_ids = []

        # Iterate through match_tensor1 and find matching elements in match_tensor2
        for i in range(match_tensor1.size(0)):
            # Check if there's a matching element in match_tensor2
            match_indices = (u2 == u1[i]) & (v2 == v1[i])
            if match_indices.any():
                # Get the index of the matching element in match_tensor2
                matching_idx = match_indices.nonzero(as_tuple=True)[0][0]

                # Copy the ID from match_tensor1 to match_tensor2
                id2[matching_idx] = id1[i]
                matched_ids.append(i)

        # Create a new match_tensor2 with updated IDs
        updated_match_tensor2 = torch.stack((u2, v2, id2), dim=1)

        # Calculate the number of matches and unmatched elements
        num_matches = len(matched_ids)
        num_unmatched = match_tensor1.size(0) - num_matches
        # print("match_tensor_1 and tensor2: ", match_tensor1[:100], match_tensor2[:100])
        # Print the results
        print("Number of matches found:", num_matches)
        print("Number of unmatched elements:", num_unmatched)
        print("New ID match_tensor2:")
        # print(updated_match_tensor2)

        return updated_match_tensor2




    # takes 5 frames as the input and uses match() to save the uv_1 and uv_2 coordinates of every consecutively matched 
    # frame in the uv_tensor_list_2compare
    # Add another column to the uv_1 and uv_2 tensors with their respective ID. The ID should be the row number. The first ID should be starting at 1.





    # def add_frames_and_match(self, frames, frame_idx, frame_dist):
    #     # Check if there are enough frames to perform matching
    #     if len(frames) < 2:
    #         print("Insufficient frames for matching")
    #         return

    #     # Initialize the ID counter
    #     id_counter = 1

    #     # Iterate over the frames and perform matching
    #     for idx in range(len(frames) - 1):
    #         frame1 = frames[idx]
    #         frame2 = frames[idx + 1]

    #         # Call the match method to get uv_1, uv_2, and indices
    #         uv_1, uv_2, index_1, index_2 = self.match(idx, idx + 1, id_counter, frame1, frame2)

    #         if uv_1 is not None:
    #             # Add the ID column to uv_1 and uv_2 tensors
    #             uv_1_with_id = torch.cat((uv_1, torch.full((uv_1.size(0), 1), id_counter, dtype=torch.float32, device=self.device)), dim=1)
    #             uv_2_with_id = torch.cat((uv_2, torch.full((uv_2.size(0), 1), id_counter, dtype=torch.float32, device=self.device)), dim=1)

    #             # Append the uv_1 and uv_2 tensors to uv_tensor_list_2compare
    #             self.uv_tensor_list_2compare.append((uv_1_with_id, uv_2_with_id))
    #             self.indices.append((index_1, index_2))

    #     if len(self.uv_tensor_list_2compare) >= 3:
    #         third_match_uv_2 = self.uv_tensor_list_2compare[2][1]  # 3rd match's uv_2 tensor is at index 2
    #         print("UV_2 from the 3rd match (first 10 lines):", third_match_uv_2[:10])
    #     else:
    #         print("Not enough matches in uv_tensor_list_2compare.")

    #     print("Number of frames in uv_tensor_list_2compare:", len(self.uv_tensor_list_2compare))






    def add_frames_and_match(self, i, j, frames, frame_idx, frame_dist, init):
        # Check if there are enough frames to perform matching
        if len(frames) < 2:
            print("Insufficient frames for matching")
            return
        
        # Calculate the starting index for matching
        if init == True:
            start_idx = frame_idx - (frame_dist - 1)
        if init == False:
            start_idx = frame_idx -frame_dist
        print("start and end frame idx, and length of frames: ", start_idx, frame_idx, len(frames))

        # Iterate over the frames and perform matching
        for idx in range(start_idx, frame_idx):
            frame1 = frames[idx - start_idx]
            frame2 = frames[idx + 1 - start_idx]
            print("this is idx inside frame 1: ", idx-start_idx) 
            # Call the match method to get uv_1, uv_2, and indices
            uv_1, uv_2, index_1, index_2 = self.match(i, j, idx, frame1, frame2)


            if uv_1 is None:
                print("uv_1 is None issue in sift_multi")

            
            # initial index column creates a list of indexes starting from 1 to each row
            if init:
                # Add the ID column to uv_1 and uv_2 tensors
                if idx == start_idx:
                    uv_1_with_id = torch.cat((uv_1, (torch.arange(uv_1.size(0)) + 1).view(-1, 1).float().to(self.device)), dim=1)
                    uv_2_with_id = torch.cat((uv_2, (torch.arange(uv_2.size(0)) + 1).view(-1, 1).float().to(self.device)), dim=1)
                else:
                    id_set = 0
                    uv_1_with_id = torch.cat((uv_1, torch.full((uv_1.size(0), 1), id_set, dtype=torch.float32, device=self.device)), dim=1)
                    uv_2_with_id = torch.cat((uv_2, torch.full((uv_2.size(0), 1), id_set, dtype=torch.float32, device=self.device)), dim=1)
                # Append the uv_1 and uv_2 tensors to uv_tensor_list_2compare
                self.uv_tensor_list_2compare.append((uv_1_with_id, uv_2_with_id))
                self.indices.append((index_1, index_2))
                # print("uv_1 and uv_2", uv_1_with_id[:100], uv_2_with_id[:100])

                #printing only uv_1 ten lines
                uv_list_moment = self.uv_tensor_list_2compare[-1]
                uv_1_of_moment = uv_list_moment[0]
                print("in init see if id list uv1: \n", uv_1_of_moment[:10])

                #printing only uv_2 ten lines
                uv_list_moment = self.uv_tensor_list_2compare[-1]
                uv_1_of_moment = uv_list_moment[1]
                print("in init see if id list uv2: \n", uv_1_of_moment[:10])


            # subsequent index column is initialized as 0
            # compare tensors and transfer the index from previous match
            else:
                id_set = 0
                uv_1_with_id = torch.cat((uv_1, torch.full((uv_1.size(0), 1), id_set, dtype=torch.float32, device=self.device)), dim=1)
                uv_2_with_id = torch.cat((uv_2, torch.full((uv_2.size(0), 1), id_set, dtype=torch.float32, device=self.device)), dim=1)
                self.uv_tensor_list_2compare.append((uv_1_with_id, uv_2_with_id))
                self.indices.append((index_1, index_2))
                # print("uv_1 and uv_2", uv_1_with_id[:100], uv_2_with_id[:100])


            print("len(list) ", len(self.uv_tensor_list_2compare))
            # TO DO one more left to do computation for init too
            if len(self.uv_tensor_list_2compare)>1:
                prev_match_tensor = self.uv_tensor_list_2compare[-2][1]
                current_match_tensor = self.uv_tensor_list_2compare[-1][0]
                
                print("previous uv tensor size: ", prev_match_tensor.size())
                print("current uv tensor size: ", current_match_tensor.size())


                # print("current match tensor = ", current_match_tensor[:100])
                # print("prev_match_tensor = ", prev_match_tensor[:100])
                # print("before: cur_uv_tensor = ", current_match_tensor[:5])
                # print("test current uv2: ", uv_1_with_id[:5])
                uv_1_with_id = self.compare_ID(prev_match_tensor, current_match_tensor)
                # print("size of uv_1 new: ", uv_1_with_id.size())
                if uv_1_with_id.size() != current_match_tensor[:, 2].size():
                    print("Sizes are not equal.")
                current_match_tensor[:, 2] = uv_1_with_id[:, 0]



        if len(self.uv_tensor_list_2compare) >= 3:
            third_match_uv_2 = self.uv_tensor_list_2compare[2][1]  # 3rd match's uv_2 tensor is at index 2
            # print("UV_2 from the 3rd match (first 10 lines):", third_match_uv_2[:10])
        else:
            print("Not enough matches in uv_tensor_list_2compare.")

        print("Number of frames in uv_tensor_list_2compare:", len(self.uv_tensor_list_2compare))


        




    # # there are i-1 matches for i frames
    # # there are i-2 connections 

    # def multi_match(self, i , j, idx. images):
    #     for match_num < images.size()-1:
    #         uv_1, uv_2, index_1, index_2 = match(i , j, idx. images[match_num], images[match_num+1])
    #         self.uv_tensor_list_2compare.append[uv_1]
    #         self.uv_tensor_list_2compare.append[uv_2]
    #         self.indices.append(index_1)
    #         self.indices.append(index_2)

    #     for connections < images.size()-2:
            



