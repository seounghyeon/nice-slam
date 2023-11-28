import numpy as np
import torch
import torch.nn.functional as F
from src.sift import SIFTMatcher

# mapper and tracker? need an init for the very first image (get the sift features)
# create sift features for the subsequent images and find the corresponding features
# word from the other thing from file for several images
# get same features and matches over several images

# later create c2w 

debug = False

def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K





def replace_zero_depth(depth_tensor, gt_depth_tensor):
    """
    Replace zero depth values in a 1D depth tensor with a maximum depth value from gt_depth

    Args:
        depth tensor: 1D tensor with the depth values
        max_depth_value (float): Maximum depth value to replace zero depth
        gt_depth_tensor (tensor): tensor of gt depth
    Returns:
        torch.Tensor: updated tensor 1D with max_depth
    """
    max_depth_value = torch.max(gt_depth_tensor)
    # print("max depth value is: ", max_depth_value)
    def_depth = max_depth_value
    # Create a mask for zero depth values
    zero_mask = (depth_tensor == 0)
    
    # Replace zero depths with the maximum depth value
    depth_tensor[zero_mask] = def_depth
    
    return depth_tensor

def ray_to_3D(sbatch_rays_o, sbatch_rays_d, sbatch_gt_depth, gt_depth,  batch_size, sift_feature_size):
    """
    changing 0 depth into max depth
    0 depth means no depth information 
    """
    max_sift = batch_size + sift_feature_size
    s_rays_o = sbatch_rays_o[batch_size:max_sift]
    s_rays_d = sbatch_rays_d[batch_size:max_sift]
    s_depth = sbatch_gt_depth[batch_size:max_sift]
    s_depth     = replace_zero_depth(s_depth, gt_depth)
    # 3D coordinates projected from the previous and current image
    point_3D    = s_rays_o + s_rays_d * s_depth.unsqueeze(1) # output size is [100,3] torch.float32

    return point_3D

def proj_3D_2D(points, W, fx, fy, cx, cy, c2w, device):
    """
    projects 3D points into 2D space at pose given by c2w
    input args:
        - points: torch tensor of 3D points Nx3
        - fx fy cx cy intrinsic camera params
        - c2w camera pose for the image
    output: 
        - uv coordinates (N,2)
    """
    # Define the concatenation tensor for [0, 0, 0, 1]
    concat_tensor = torch.tensor([0, 0, 0, 1], device=device, dtype=c2w.dtype)      # is torch.float32
    # Clone c2w to ensure we don't modify the original tensor

    # Concatenate [0, 0, 0, 1] to the copied tensor
    c2w = torch.cat([c2w, concat_tensor.unsqueeze(0)], dim=0)

    c2w[:3, 1] *= -1.0
    c2w[:3, 2] *= -1.0

    # Calculate the world-to-camera transformation matrix w2c
    w2c = torch.inverse(c2w)

    # Camera intrinsic matrix K
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=c2w.dtype, device=device)

    # Convert points to homogeneous coordinates
    ones = torch.ones_like(points[:, 0], device=device).unsqueeze(1)
    homo_points = torch.cat([points, ones], dim=1).unsqueeze(2)

    # Transform points to camera coordinates
    cam_cord_homo = torch.matmul(w2c, homo_points)

    # Remove the homogeneous coordinate
    cam_cord = cam_cord_homo[:, :3, :]

    cam_cord[:, 0] *= -1

    # Project points to image plane
    uv = torch.matmul(K, cam_cord)

    z = uv[:, -1:] + 1e-5
    uv = uv[:, :2] / z


    # Apply the correct transformation to uv coordinates
    uv[:, 0] = W - uv[:, 0]
    uv[:, 0] -= 2
    # print("these are the points size: \n", points.size())
    # print("uv.size: ", uv.size())
    num_points = points.size(0)
    uv = uv.view(num_points, 2)
    
    return uv



def sample_pdf(bins, weights, N_samples, det=False, device='cuda:0'):
    """
    Hierarchical sampling in NeRF paper (section 5.2).

    """
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    # (batch, len(bins))
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    u = u.to(device)
    # Invert CDF
    u = u.contiguous()
    try:
        # this should work fine with the provided environment.yaml
        inds = torch.searchsorted(cdf, u, right=True)
    except:
        # for lower version torch that does not have torch.searchsorted,
        # you need to manually install from
        # https://github.com/aliutkus/torchsearchsorted
        from torchsearchsorted import searchsorted
        inds = searchsorted(cdf, u, side='right')
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples


def random_select(l, k):
    """
    Random select k values from 0..l.

    """
    return list(np.random.permutation(np.array(range(l)))[:min(l, k)])

# somehow the i j values added +20
def get_rays_from_uv(i, j, c2w, H, W, fx, fy, cx, cy, device):
    """
    Get corresponding rays from input uv.
    c2w backprop works!
    """


    # # Define the range of indices to update
    # start_index = 1000
    # end_index = 1100

    # # Add 20 to the specified range of entries
    # i[start_index:end_index] -= 20
    # j[start_index:end_index] -= 20

    # print("in get_rays_from_uv i (either prev or cur): ", i[1000:1010])
    # print("in get_rays_from_uv j (either prev or cur): ", j[1000:1010])

    # Create a tensor with recovered UV coordinates
    # uv_recovered = torch.stack((u_coord_recovered, v_coord_recovered), dim=1)
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).to(device)
        print("c2w is numpy CHECK IT GRADIENT")
    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(-1, 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


# randomly chooses uv coordinates
# adds sift features and uv coordinates
# color gt image
def select_uv(H0, H1, W0, W1, i, j, n, depth_prev, color_prev, depth_cur, color_cur, frame, device='cuda:0'):
    """
    Select n uv from dense uv.
    color and depth sizes are already smaller
    so is size of i j
    """
    i = i.reshape(-1)
    j = j.reshape(-1)

    k = i.clone()
    l = j.clone()
    #random indices
    # print("test i[0]: ", i[0], i[1], i[-1])
    indices_prev = torch.randint(i.shape[0], (n,), device=device) # generates n random integers between 0 (inclusive) and i.shape[0]
    indices_prev = indices_prev.clamp(0, i.shape[0])

    indices_cur = indices_prev.clone()

    sift_matcher = SIFTMatcher()  # Instantiate the class

    # print("in select uv before match() h0,h1,w0,w1: ", H0, H1, W0, W1)

    uv_prev, uv_cur, index_prev, index_cur = sift_matcher.match(i, j, frame, color_prev, color_cur)
    

    
    # frame_dist = 5

    # # very first init color_list 
    # # frame dist -1 because frame name starts at 0
    # if frame == frame_dist - 1:
    #     init = True
    #     print("init first track \n")
    #     sift_matcher.add_frames_and_match(i, j,color_list, frame, frame_dist, init)

    # # subsequent color lists are larger
    # if len(color_list) == frame_dist+1:
    #     init = False
    #     print("color list is " +str(len(color_list)) +" frames. \n")
    #     sift_matcher.add_frames_and_match(i, j,color_list, frame, frame_dist, init)


    test_u_cur =i[index_cur]
    test_v_cur = j[index_cur]
    # print("\n\nthis is output of the index2 in select_uv:\n ",test_u_cur[:10],test_v_cur[:10])
    # print("\n\nthis is output of the uv_cur in select_uv:\n ",uv_cur[:10])


    # print("this is index2 in sift: ",index_cur[:10])



    if index_prev is not None:
        indices_prev = torch.cat((indices_prev, index_prev), dim=0)
        indices_cur = torch.cat((indices_cur, index_cur), dim=0)
    i = i[indices_prev]  # (n)
    j = j[indices_prev]  # (n)
    # print("\ni size is: ",i.size())
    k = k[indices_cur]
    l = l[indices_cur]

    depth_prev = depth_prev.reshape(-1)
    color_prev = color_prev.reshape(-1, 3)
    depth_prev = depth_prev[indices_prev]  # (n)
    color_prev = color_prev[indices_prev]  # (n,3)

    depth_cur = depth_cur.reshape(-1)
    color_cur = color_cur.reshape(-1, 3)
    depth_cur = depth_cur[indices_cur]  # (n)
    color_cur = color_cur[indices_cur]  # (n,3)

    return  uv_prev, uv_cur, i, j, depth_prev, color_prev, k, l, depth_cur, color_cur








# initiates all the uv coordinates in i and j aswell as depth and color for all pixels
# puts them into select_uv where some pixels are randomly sampled which then form the output tensors
# to add specific uv chosen by sift - edit select_uv
# output i j are the sampled i and j values
# color_cur is the current
def get_sample_uv(H0, H1, W0, W1, n, depth_prev, color_prev, depth_cur, color_cur, frame, device='cuda:0'):
    """
    Sample n uv coordinates from an image region H0..H1, W0..W1

    """
    depth_prev = depth_prev[H0:H1, W0:W1]
    color_prev = color_prev[H0:H1, W0:W1]

    depth_cur = depth_cur[H0:H1, W0:W1]
    color_cur = color_cur[H0:H1, W0:W1]    
    #i is horizontal (width) and j is vertical (height)
    # torch.linspace(W0, W1-1, W1-W0) horizontal u coordinate in a 1D tensor
    # meshgrid takes 2 1D tensor to create 2D tensor
    # i is all horizontal u coordinates in range [W0, W1-1] (W0 starting x, W1-1 ending x coordinate
    # W1-W0 determines the total number of coordinate steps between W0 and W1-1
    i, j = torch.meshgrid(torch.linspace(
        W0, W1-1, W1-W0).to(device), torch.linspace(H0, H1-1, H1-H0).to(device))
    i = i.t()  # transpose
    j = j.t()
    i_test = i.reshape(-1)
    # print("this is W0, H0, W0, W1: ",i,"\n",W0, H0, W1, H1,"\n")

    # considering same input image size for all images i and j are the same for both images (input in select_uv())
    uv_prev, uv_cur, i_prev, j_prev, depth_prev, color_prev, i_cur, j_cur, depth_cur, color_cur = select_uv(H0, H1, W0, W1, i, j, n, depth_prev, color_prev, depth_cur, color_cur, frame, device=device)

    return  uv_prev, uv_cur, i_prev, j_prev, depth_prev, color_prev, i_cur, j_cur, depth_cur, color_cur




# Hedge, H-Hedge, Wedge, W-Wedge, batch_size, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, gt_color_prev, stored_rays, idx, self.device)
# 1 is the previous iteration, 2 is current iteration
def get_samples_sift(H0, H1, W0, W1, n, H, W, fx, fy, cx, cy, depth_prev, color_prev, c2w_prev, depth_cur, color_cur, c2w_cur, frame, device):
    """
    Get n rays from the image region H0..H1, W0..W1.
    c2w is its camera pose and depth/color is the corresponding image tensor.
    """
    # print("\nthis is batch_Rays_d_prev: \n", batch_rays_d_prev)
    # print("in get_samples_sift H0, H1, W0, W1: ", H0, H1, W0, W1)
    # print("image size: ", color_cur.size())

    uv_prev, uv_cur, i_prev, j_prev, sample_depth_prev, sample_color_prev, i_cur, j_cur, sample_depth_cur, sample_color_cur = get_sample_uv(
        H0, H1, W0, W1, n, depth_prev, color_prev, depth_cur, color_cur, frame, device=device)

    rays_o_prev, rays_d_prev = get_rays_from_uv(i_prev, j_prev, c2w_prev, H, W, fx, fy, cx, cy, device)
    rays_o_cur, rays_d_cur = get_rays_from_uv(i_cur, j_cur, c2w_cur, H, W, fx, fy, cx, cy, device)

    return  uv_prev, uv_cur, rays_o_prev, rays_d_prev, sample_depth_prev, sample_color_prev, rays_o_cur, rays_d_cur, sample_depth_cur, sample_color_cur




# quaternions used to compute matrices faster with less memory
# for rotation matrix
def quad2rotation(quad):
    """
    Convert quaternion to rotation in batch. Since all operation in pytorch, support gradient passing.

    Args:
        quad (tensor, batch_size*4): quaternion.

    Returns:
        rot_mat (tensor, batch_size*3*3): rotation.
    """
    bs = quad.shape[0]
    qr, qi, qj, qk = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
    two_s = 2.0 / (quad * quad).sum(-1)
    rot_mat = torch.zeros(bs, 3, 3).to(quad.get_device())
    rot_mat[:, 0, 0] = 1 - two_s * (qj ** 2 + qk ** 2)
    rot_mat[:, 0, 1] = two_s * (qi * qj - qk * qr)
    rot_mat[:, 0, 2] = two_s * (qi * qk + qj * qr)
    rot_mat[:, 1, 0] = two_s * (qi * qj + qk * qr)
    rot_mat[:, 1, 1] = 1 - two_s * (qi ** 2 + qk ** 2)
    rot_mat[:, 1, 2] = two_s * (qj * qk - qi * qr)
    rot_mat[:, 2, 0] = two_s * (qi * qk - qj * qr)
    rot_mat[:, 2, 1] = two_s * (qj * qk + qi * qr)
    rot_mat[:, 2, 2] = 1 - two_s * (qi ** 2 + qj ** 2)
    return rot_mat

# returns a 3x4 matrix like
# [-0.9516, -0.1204,  0.2828,  2.6555]
# [ 0.3073, -0.3925,  0.8669,  2.9814]
# [ 0.0067,  0.9118,  0.4105,  1.3619]
# needs to add 0 0 0 1 to get the 4x4 homogeneous matrix
def get_camera_from_tensor(inputs):
    """
    Convert quaternion and translation to transformation matrix.

    """
    N = len(inputs.shape)
    if N == 1:
        inputs = inputs.unsqueeze(0)
    quad, T = inputs[:, :4], inputs[:, 4:]
    R = quad2rotation(quad)
    RT = torch.cat([R, T[:, :, None]], 2)
    if N == 1:
        RT = RT[0]
    return RT


def get_tensor_from_camera(RT, Tquad=False):
    """
    Convert transformation matrix to quaternion and translation.

    """
    gpu_id = -1
    if type(RT) == torch.Tensor:
        if RT.get_device() != -1:
            RT = RT.detach().cpu()
            gpu_id = RT.get_device()
        RT = RT.numpy()
    from mathutils import Matrix
    R, T = RT[:3, :3], RT[:3, 3]
    rot = Matrix(R)
    quad = rot.to_quaternion()
    if Tquad:
        tensor = np.concatenate([T, quad], 0)
    else:
        tensor = np.concatenate([quad, T], 0)
    tensor = torch.from_numpy(tensor).float()
    if gpu_id != -1:
        tensor = tensor.to(gpu_id)
    return tensor


def raw2outputs_nerf_color(raw, z_vals, rays_d, occupancy=False, device='cuda:0'):
    """
    Transforms model's predictions to semantically meaningful values.

    Args:
        raw (tensor, N_rays*N_samples*4): prediction from model.
        z_vals (tensor, N_rays*N_samples): integration time.
        rays_d (tensor, N_rays*3): direction of each ray.
        occupancy (bool, optional): occupancy or volume density. Defaults to False.
        device (str, optional): device. Defaults to 'cuda:0'.

    Returns:
        depth_map (tensor, N_rays): estimated distance to object.
        depth_var (tensor, N_rays): depth variance/uncertainty.
        rgb_map (tensor, N_rays*3): estimated RGB color of a ray.
        weights (tensor, N_rays*N_samples): weights assigned to each sampled color.
    """

    def raw2alpha(raw, dists, act_fn=F.relu): return 1. - \
        torch.exp(-act_fn(raw)*dists)
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = dists.float()
    dists = torch.cat([dists, torch.Tensor([1e10]).float().to(
        device).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    # different ray angle corresponds to different unit length
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    rgb = raw[..., :-1]
    if occupancy:
        raw[..., 3] = torch.sigmoid(10*raw[..., -1])
        alpha = raw[..., -1]
    else:
        # original nerf, volume density
        alpha = raw2alpha(raw[..., -1], dists)  # (N_rays, N_samples)

    weights = alpha.float() * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(
        device).float(), (1.-alpha + 1e-10).float()], -1).float(), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # (N_rays, 3)
    depth_map = torch.sum(weights * z_vals, -1)  # (N_rays)
    tmp = (z_vals-depth_map.unsqueeze(-1))  # (N_rays, N_samples)
    depth_var = torch.sum(weights*tmp*tmp, dim=1)  # (N_rays)
    return depth_map, depth_var, rgb_map, weights


def get_rays(H, W, fx, fy, cx, cy, c2w, device):
    """
    Get rays for a whole image.

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)
    # pytorch's meshgrid has indexing='ij'
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()  # transpose
    j = j.t()
    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(H, W, 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def normalize_3d_coordinate(p, bound):
    """
    Normalize coordinate to [-1, 1], corresponds to the bounding box given.

    Args:
        p (tensor, N*3): coordinate.
        bound (tensor, 3*2): the scene bound.

    Returns:
        p (tensor, N*3): normalized coordinate.
    """
    p = p.reshape(-1, 3)
    p[:, 0] = ((p[:, 0]-bound[0, 0])/(bound[0, 1]-bound[0, 0]))*2-1.0
    p[:, 1] = ((p[:, 1]-bound[1, 0])/(bound[1, 1]-bound[1, 0]))*2-1.0
    p[:, 2] = ((p[:, 2]-bound[2, 0])/(bound[2, 1]-bound[2, 0]))*2-1.0
    return p
