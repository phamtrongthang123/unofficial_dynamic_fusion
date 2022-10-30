import torch
from os import path
from tqdm import tqdm
import imageio
import cv2
import numpy as np
import open3d as o3d



# Note,this step converts w2c (Tcw) to c2w (Twc)
def load_K_Rt_from_P(P):
    """
    modified from IDR https://github.com/lioryariv/idr
    """
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K


    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()  # convert from w2c to c2w
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class OurDataset(torch.utils.data.Dataset):
    """
    TUM dataset loader, pre-load images in advance
    """

    def __init__(
            self,
            rootdir,
            device,
            near: float = 0.2,
            far: float = 5.,
            img_scale: float = 1.,  # image scale factor
            start: int = -1,
            end: int = -1,
    ):
        super().__init__()
        assert path.isdir(rootdir), f"'{rootdir}' is not a directory"
        self.device = device
        self.c2w_all = []
        self.K_all = []
        self.rgb_all = []
        self.depth_all = []

        # root should be tum_sequence
        data_path = path.join(rootdir, "processed")
        cam_file = path.join(data_path, "cameras.npz")
        print("LOAD DATA", data_path)

        # world_mats, normalize_mat
        cam_dict = np.load(cam_file)
        world_mats = cam_dict["world_mats"]  # K @ w2c

        d_min = []
        d_max = []
        # TUM saves camera poses in OpenCV convention
        for i, world_mat in enumerate(tqdm(world_mats)):
            # ignore all the frames betfore
            if start > 0 and i < start:
                continue
            # ignore all the frames after
            if 0 < end < i:
                break

            intrinsics, c2w = load_K_Rt_from_P(world_mat)
            c2w = torch.tensor(c2w, dtype=torch.float32)
            # read images
            rgb = np.array(imageio.imread(path.join(data_path, "rgb/{:06d}.jpg".format(i)))).astype(np.float32)
            depth = np.array(imageio.imread(path.join(data_path, "depth/{:06d}.png".format(i)))).astype(np.float32)
            depth /= 1000.  # I take our unit is 1 meter, so /1000 to make it go back to 1m
            d_max += [depth.max()]
            d_min += [depth.min()]
            # depth = cv2.bilateralFilter(depth, 5, 0.2, 15)
            # print(depth[depth > 0.].min())
            invalid = (depth < near) | (depth > far)
            depth[invalid] = -1.
            # downscale the image size if needed
            if img_scale < 1.0:
                full_size = list(rgb.shape[:2])
                rsz_h, rsz_w = [round(hw * img_scale) for hw in full_size]
                # TODO: figure out which way is better: skimage.rescale or cv2.resize
                rgb = cv2.resize(rgb, (rsz_w, rsz_h), interpolation=cv2.INTER_AREA)
                depth = cv2.resize(depth, (rsz_w, rsz_h), interpolation=cv2.INTER_NEAREST)
                intrinsics[0, 0] *= img_scale
                intrinsics[1, 1] *= img_scale
                intrinsics[0, 2] *= img_scale
                intrinsics[1, 2] *= img_scale

            self.c2w_all.append(c2w)
            self.K_all.append(torch.from_numpy(intrinsics[:3, :3]).float())
            self.rgb_all.append(torch.from_numpy(rgb))
            self.depth_all.append(torch.from_numpy(depth))
            if i==4:break
        print("Depth min: {:f}".format(np.array(d_min).min()))
        print("Depth max: {:f}".format(np.array(d_max).max()))
        self.n_images = len(self.rgb_all)
        self.H, self.W, _ = self.rgb_all[0].shape

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        return self.rgb_all[idx].to(self.device), self.depth_all[idx].to(self.device), \
               self.c2w_all[idx].to(self.device), self.K_all[idx].to(self.device)

