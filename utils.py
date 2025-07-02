import numpy as np
import torch
import os
import h5py
import json
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image

import IPython
e = IPython.embed

class EpisodicRoboVerseDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicRoboVerseDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, "robot-franka", 'demo_{:04d}'.format(episode_id))
        metadata_path = os.path.join(dataset_path, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        raw_qpos = metadata.get('joint_qpos', [])
        # padding raw_qpos to make it shape (N, 14) with zeros
        raw_qpos = np.array(raw_qpos)

        # padding to 14-dof
        if raw_qpos.shape[1] != 14:
            raw_qpos = np.concatenate([raw_qpos, np.zeros((raw_qpos.shape[0], 14 - raw_qpos.shape[1]))], axis=1)

        ori_qpos = raw_qpos[:-1]
        ori_action = raw_qpos[1:] # require slice off all elements' last element

        if sample_full_episode:
            start_ts = 0
        else:
            start_ts = np.random.choice(len(ori_qpos))

        # get observation at start_ts only
        qpos = ori_qpos[start_ts]
        image_dict = dict()
        for cam_name in self.camera_names:
            image_path = os.path.join(dataset_path, 'rgb_{:04d}.png'.format(start_ts))
            image_dict[cam_name] = np.array(Image.open(image_path))

        action = ori_action[start_ts:]
        action_len = len(action)

        self.is_sim = False
        padded_action = np.zeros([self.norm_stats['max_episode_len'], ori_action.shape[1]], dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.norm_stats['max_episode_len'])
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    max_episode_len = 0
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, "robot-franka", 'demo_{:04d}'.format(episode_idx))
        metadata_path = os.path.join(dataset_path, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        ori_qpos = metadata.get('joint_qpos', [])
        qpos = np.array(ori_qpos[:-1])
        action = np.array(ori_qpos[1:]) # require slice off all elements' last element
        max_episode_len = max(max_episode_len, qpos.shape[0])
        # dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        # with h5py.File(dataset_path, 'r') as root:
        #     qpos = root['/observations/qpos'][()]
        #     qvel = root['/observations/qvel'][()]
        #     action = root['/action'][()]

        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.vstack(all_qpos_data)
    all_action_data = torch.vstack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos, "max_episode_len": max_episode_len}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)
    print(norm_stats)
    # construct dataset and dataloader
    train_dataset = EpisodicRoboVerseDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicRoboVerseDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def unnormalize_image(img_tensor, mean, std):
    img = img_tensor.clone().cpu().numpy()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.transpose(1, 2, 0)  # [H, W, C]
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

def normalize_action(action_tensor, normalizer):
        splits = [7, 21, 7, 21, 2]
        keys = [
            "/action/right_arm/joint_angle/rel",
            "/action/right_hand/joint_angle/rel",
            "/action/left_arm/joint_angle/rel",
            "/action/left_hand/joint_angle/rel",
            "/action/neck/joint_angle/rel",
        ]

        chunks = torch.split(action_tensor, splits, dim=-1)
        norm_chunks = []
        for key, chunk in zip(keys, chunks):
            norm_chunks.append(normalizer[key].normalize(chunk))
        return torch.cat(norm_chunks, dim=-1)

def denormalize_action(action_tensor, normalizer):
        splits = [7, 21, 7, 21, 2]
        keys = [
            "/action/right_arm/joint_angle/rel",
            "/action/right_hand/joint_angle/rel",
            "/action/left_arm/joint_angle/rel",
            "/action/left_hand/joint_angle/rel",
            "/action/neck/joint_angle/rel",
        ]

        chunks = torch.split(action_tensor, splits, dim=-1)
        denorm_chunks = []
        for key, chunk in zip(keys, chunks):
            denorm_chunks.append(normalizer[key].unnormalize(chunk))
        return torch.cat(denorm_chunks, dim=-1)

def normalize_obs_lowdim(lowdim_tensor, normalizer):
        splits = [7, 21, 7, 21, 2]
        keys = [
            "/state/right_arm/joint_angle",
            "/state/right_hand/joint_angle",
            "/state/left_arm/joint_angle",
            "/state/left_hand/joint_angle",
            "/state/neck/joint_angle",
        ]
        chunks = torch.split(lowdim_tensor, splits, dim=-1)
        norm_chunks = [
            normalizer[key].normalize(chunk)
            for key, chunk in zip(keys, chunks)
        ]
        return torch.cat(norm_chunks, dim=-1)

def denormalize_obs_lowdim(lowdim_tensor, normalizer):
        splits = [7, 21, 7, 21, 2]
        keys = [
            "/state/right_arm/joint_angle",
            "/state/right_hand/joint_angle",
            "/state/left_arm/joint_angle",
            "/state/left_hand/joint_angle",
            "/state/neck/joint_angle",
        ]
        chunks = torch.split(lowdim_tensor, splits, dim=-1)
        denorm_chunks = [
            normalizer[key].unnormalize(chunk)
            for key, chunk in zip(keys, chunks)
        ]
        return torch.cat(denorm_chunks, dim=-1)

def normalize_tactile(tactile_tensor, normalizer):
    return normalizer["/observe/tactile/total_force"].normalize(tactile_tensor)

def denormalize_tactile(tactile_tensor, normalizer):
    return normalizer["/observe/tactile/total_force"].unnormalize(tactile_tensor)

def normalize_tactile_next(tactile_tensor, normalizer):
    return normalizer["/observe/tactile/total_force/next"].normalize(tactile_tensor)

def denormalize_tactile_next(tactile_tensor, normalizer):
    return normalizer["/observe/tactile/total_force/next"].unnormalize(tactile_tensor)

def apply_joint_mask(qpos_tensor, mask, start_index):
    B, T, D = qpos_tensor.shape
    mask_tensor = torch.tensor(mask, dtype=qpos_tensor.dtype, device=qpos_tensor.device).view(1, 1, -1)
    qpos_tensor[..., start_index:start_index+len(mask)] *= mask_tensor
    return qpos_tensor

