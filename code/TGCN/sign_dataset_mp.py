import json
import h5py
import math
import os
import random

import numpy as np

import cv2
import torch
import torch.nn as nn

import utils

from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class SignDatasetMP(Dataset):
    def __init__(self, index_file_path, keypoint_path, split_path, split, num_samples, sample_strategy, num_copies=4):
        self.num_samples = num_samples
        self.sample_strategy = sample_strategy
        self.data = []
        self.label_encoder, self.onehot_encoder = LabelEncoder(), OneHotEncoder(categories='auto')
        self.num_copies = num_copies
        self.label_encoder, self.onehot_encoder = LabelEncoder(), OneHotEncoder(categories='auto')

        self.make_dataset(index_file_path, keypoint_path, split_path, split)

    def make_dataset(self, index_file_path, keypoint_path, split_path, split):
        with open(index_file_path, 'r') as f:
            content = json.load(f)
        
        # create label encoder
        glosses = sorted([gloss_entry['gloss'] for gloss_entry in content])

        self.label_encoder.fit(glosses)
        self.onehot_encoder.fit(self.label_encoder.transform(self.label_encoder.classes_).reshape(-1, 1))
        
        with open(split_path, 'r') as j:
            splits = json.load(j)
        
        with h5py.File(keypoint_path, "r") as f:
            for gloss_entry in content:
                _, instances = gloss_entry['gloss'], gloss_entry['instances']
                for instance in instances:
                    if instance['split'] not in split:
                        continue

                    frame_end = instance['frame_end']
                    frame_start = instance['frame_start']
                    video_id = instance['video_id']

                    X = f[video_id][:][:, :162]
                    y = splits[video_id]['action'][0]
                    instance_entry = video_id, frame_start, frame_end, X, y
                    self.data.append(instance_entry)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        video_id, frame_start, frame_end, X, y = self.data[index]
        # X = torch.from_numpy(X)
        poses = []
        if self.sample_strategy == 'rnd_start':
            X = pad(X, self.num_samples)
        elif self.sample_strategy == 'k_copies':
            X = k_copies(X, self.num_samples, self.num_copies)


        return X, y, video_id


def pad(keypoints, total_frames):
    if keypoints.shape[0] < total_frames:
        num_padding = total_frames - keypoints.shape[0]

        if num_padding:
            prob = np.random.random_sample()
            if prob > 0.5:
                pad_keypoint = keypoints[0]
                pad = np.tile(np.expand_dims(pad_keypoint, axis=0), 
                                (num_padding, 1))
                padded_keypoints = np.concatenate([keypoints, pad], axis=0)
            else:
                pad_keypoint = keypoints[-1]
                pad = np.tile(np.expand_dims(pad_keypoint, axis=0), 
                                        (num_padding, 1))
                padded_keypoints = np.concatenate([keypoints, pad], axis=0)
    elif keypoints.shape[0] > total_frames:
        start_f = random.randint(0, keypoints.shape[0] - total_frames - 1)
        padded_keypoints = keypoints[start_f : start_f + total_frames]
    else:
        padded_keypoints = keypoints

    return padded_keypoints

def k_copies(keypoints, num_samples, num_copies):
    num_frames = len(keypoints)
    if num_frames <= num_samples:
        num_pads = num_samples - num_frames
        pad_keypoint = keypoints[-1]
        pad = np.tile(np.expand_dims(pad_keypoint, axis=0), 
                                (num_pads, 1))
        keypoints = np.concatenate([keypoints, pad], axis=0)
        keypoints = np.concatenate([keypoints for i in range(num_copies)], axis=0)
    
    elif num_samples * num_copies < num_frames:
        mid = (len(keypoints)-1) // 2
        half = num_samples * num_copies // 2
        frame_start = mid - half
        frames = []
        for i in range(num_copies):
            frames.append(keypoints[frame_start + i * num_samples : frame_start + i * num_samples + num_samples])
        keypoints = np.concatenate(frames, axis=0)

    else:
        frames = []
        stride = math.floor((num_frames - num_samples) / (num_copies - 1))
        for i in range(num_copies):
            frames.append(keypoints[i * stride : i * stride + num_samples])
        keypoints = np.concatenate(frames, axis=0)
    
    return keypoints

def k_copies_fixed_length_sequential_sampling(frame_start, frame_end, num_samples, num_copies):
    num_frames = frame_end - frame_start + 1

    frames_to_sample = []

    if num_frames <= num_samples:
        num_pads = num_samples - num_frames

        frames_to_sample = list(range(frame_start, frame_end + 1))
        frames_to_sample.extend([frame_end] * num_pads)

        frames_to_sample *= num_copies

    elif num_samples * num_copies < num_frames:
        mid = (frame_start + frame_end) // 2
        half = num_samples * num_copies // 2

        frame_start = mid - half

        for i in range(num_copies):
            frames_to_sample.extend(list(range(frame_start + i * num_samples,
                                               frame_start + i * num_samples + num_samples)))

    else:
        stride = math.floor((num_frames - num_samples) / (num_copies - 1))
        for i in range(num_copies):
            frames_to_sample.extend(list(range(frame_start + i * stride,
                                               frame_start + i * stride + num_samples)))
    
    return frames_to_sample

def rand_start_sampling(frame_start, frame_end, num_samples):
    """Randomly select a starting point and return the continuous ${num_samples} frames."""
    num_frames = frame_end - frame_start + 1

    if num_frames > num_samples:
        select_from = range(frame_start, frame_end - num_samples + 1)
        sample_start = random.choice(select_from)
        frames_to_sample = list(range(sample_start, sample_start + num_samples))
    else:
        frames_to_sample = list(range(frame_start, frame_end + 1))

    return frames_to_sample