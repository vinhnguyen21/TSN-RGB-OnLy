import os
import numpy as np
import PIL
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils import data

class UCF_dataset(data.Dataset):
    def __init__(self, data_path, train_list, label_list, num_segment, transform= None):
        self.data_path = data_path
        self.train_list = train_list
        self.label_list = label_list
        self.num_segment = num_segment
        self.transform = transform

    def _sample_indices(self, len_frame):
        average_duration = (len_frame) // self.num_segment
        if average_duration >0:
            offsets = np.multiply(list(range(self.num_segment)), average_duration) + np.random.randint(average_duration, size = self.num_segment)
        else:
            offsets = np.zeros((self.num_segment),)
        return offsets + 1

    def _read_image(self, video_path, frames):
        images = []

        for index in frames:
            image_path = os.path.join(self.data_path, video_path, 'frame{:06}.jpg'.format(index))
            image = Image.open(image_path).convert('RGB')

            if self.transform:
                image = self.transform(image)            

            images.append(image)

        images = torch.stack(images, dim=0)
        return images

    def __getitem__(self, index):
        video = self.train_list[index]
        max_frame = len(os.listdir(os.path.join(self.data_path, video)))
        frames = self._sample_indices(max_frame)

        #loading frames and labe
        images = self._read_image(video, frames)
        labels = torch.LongTensor([self.label_list[index]-1])

        return images, labels

    def __len__(self):
        return len(self.train_list)
