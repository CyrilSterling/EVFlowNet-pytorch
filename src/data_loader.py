import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from config import configs
from PIL import Image
import os
import numpy as np
import cv2

_MAX_SKIP_FRAMES = 6
_TEST_SKIP_FRAMES = 4
_N_SKIP = 1

class EventData(Dataset):
    """
    args:
    data_folder_path:the path of data
    split:'train' or 'test'
    """
    def __init__(self, data_folder_path, split, count_only=False, time_only=False, skip_frames=False):
        self._data_folder_path = data_folder_path
        self._split = split
        self._count_only = count_only
        self._time_only = time_only
        self._skip_frames = skip_frames
        self.args = configs()
        self.event_data_paths, self.n_ima = self.read_file_paths(self._data_folder_path, self._split)

    def __getitem__(self, index):
        # 获得image_times event_count_images event_time_images image_iter prefix cam
        image_iter = 0
        for i in self.n_ima:
            if index < i:
                break
            image_iter += 1
        image_iter -= 1
        if image_iter % 2 == 0:
            cam = 'left'
        else:
            cam = 'right'
        prefix = self.event_data_paths[image_iter]
        image_iter = index - self.n_ima[image_iter]

        event_count_images, event_time_images, image_times = np.load(prefix + "/" + cam + "_event" +\
             str(image_iter).rjust(5,'0') + ".npy", encoding='bytes', allow_pickle=True)
        event_count_images = torch.from_numpy(event_count_images.astype(np.int16))
        event_time_images = torch.from_numpy(event_time_images.astype(np.float32))
        image_times = torch.from_numpy(image_times.astype(np.float64))

        if self._split is 'test':
            if self._skip_frames:
                n_frames = _TEST_SKIP_FRAMES
            else:
                n_frames = 1
        else:
            n_frames = np.random.randint(low=1, high=_MAX_SKIP_FRAMES+1) * _N_SKIP
        timestamps = [image_times[0], image_times[n_frames]]
        event_count_image, event_time_image = self._read_events(event_count_images, event_time_images, n_frames)

        prev_img_path = prefix + "/" + cam + "_image" + str(image_iter).rjust(5,'0') + ".png"
        next_img_path = prefix + "/" + cam + "_image" + str(image_iter+n_frames).rjust(5,'0') + ".png"

        prev_image = Image.open(prev_img_path)
        next_image = Image.open(next_img_path)

        #transforms
        rand_flip = np.random.randint(low=0, high=2)
        rand_rotate = np.random.randint(low=-30, high=30)
        x = np.random.randint(low=1, high=(event_count_image.shape[1]-self.args.image_height))
        y = np.random.randint(low=1, high=(event_count_image.shape[2]-self.args.image_width))
        if self._split == 'train':
            if self._count_only:
                event_count_image = F.to_pil_image(event_count_image / 255.)
                # random_flip
                if rand_flip == 0:
                    event_count_image = event_count_image.transpose(Image.FLIP_LEFT_RIGHT)
                # random_rotate
                event_image = event_count_image.rotate(rand_rotate)
                # random_crop
                event_image = F.to_tensor(event_image) * 255.
                event_image = event_image[:,x:x+self.args.image_height,y:y+self.args.image_width]
            elif self._time_only:
                event_time_image = F.to_pil_image(event_time_image)
                # random_flip
                if rand_flip == 0:
                    event_time_image = event_time_image.transpose(Image.FLIP_LEFT_RIGHT)
                # random_rotate
                event_image = event_time_image.rotate(rand_rotate)
                # random_crop
                event_image = F.to_tensor(event_image)
                event_image = event_image[:,x:x+self.args.image_height,y:y+self.args.image_width]
            else:
                event_count_image = F.to_pil_image(event_count_image / 255.)
                event_time_image = F.to_pil_image(event_time_image)
                # random_flip
                if rand_flip == 0:
                    event_count_image = event_count_image.transpose(Image.FLIP_LEFT_RIGHT)
                    event_time_image = event_time_image.transpose(Image.FLIP_LEFT_RIGHT)
                # random_rotate
                event_count_image = event_count_image.rotate(rand_rotate)
                event_time_image = event_time_image.rotate(rand_rotate)
                # random_crop
                event_count_image = F.to_tensor(event_count_image)
                event_time_image = F.to_tensor(event_time_image) * 255.
                event_image = torch.cat((event_count_image,event_time_image), dim=0)
                event_image = event_image[...,x:x+self.args.image_height,y:y+self.args.image_width]

            if rand_flip == 0:
                prev_image = prev_image.transpose(Image.FLIP_LEFT_RIGHT)
                next_image = next_image.transpose(Image.FLIP_LEFT_RIGHT)
            prev_image = prev_image.rotate(rand_rotate)
            next_image = next_image.rotate(rand_rotate)
            prev_image = F.to_tensor(prev_image)
            next_image = F.to_tensor(next_image)
            prev_image = prev_image[...,x:x+self.args.image_height,y:y+self.args.image_width]
            next_image = next_image[...,x:x+self.args.image_height,y:y+self.args.image_width]

        else:
            if self._count_only:
                event_image = F.to_tensor(F.center_crop(F.to_pil_image(event_count_image / 255.), 
                                            (self.args.image_height, self.args.image_width)))
                event_image = event_image * 255.
            elif self._time_only:
                event_image = F.to_tensor(F.center_crop(F.to_pil_image(event_time_image), 
                                            (self.args.image_height, self.args.image_width)))
            else:
                event_image = torch.cat((event_count_image / 255.,event_time_image), dim=0)
                event_image = F.to_tensor(F.center_crop(F.to_pil_image(event_image), 
                                            (self.args.image_height, self.args.image_width)))
                event_image[:2,...] = event_image[:2,...] * 255.
            prev_image = F.to_tensor(F.center_crop(prev_image, (self.args.image_height, self.args.image_width)))
            next_image = F.to_tensor(F.center_crop(next_image, (self.args.image_height, self.args.image_width)))

        return event_image, prev_image, next_image, timestamps

    def __len__(self):
        return self.n_ima[-1]

    def _read_events(self,
                     event_count_images,
                     event_time_images,
                     n_frames):
        #event_count_images = event_count_images.reshape(shape).type(torch.float32)
        event_count_image = event_count_images[:n_frames, :, :, :]
        event_count_image = torch.sum(event_count_image, dim=0).type(torch.float32)
        p = torch.max(event_count_image)
        event_count_image = event_count_image.permute(2,0,1)

        #event_time_images = event_time_images.reshape(shape).type(torch.float32)
        event_time_image = event_time_images[:n_frames, :, :, :]
        event_time_image = torch.max(event_time_image, dim=0)[0]

        event_time_image /= torch.max(event_time_image)
        event_time_image = event_time_image.permute(2,0,1)

        '''
        if self._count_only:
            event_image = event_count_image
        elif self._time_only:
            event_image = event_time_image
        else:
            event_image = torch.cat([event_count_image, event_time_image], dim=2)

        event_image = event_image.permute(2,0,1).type(torch.float32)
        '''

        return event_count_image, event_time_image

    def read_file_paths(self,
                        data_folder_path,
                        split,
                        sequence=None):
        """
        return: event_data_paths,paths of event data (left and right in one folder is two)
        n_ima: the sum number of event pictures in every path and the paths before
        """
        event_data_paths = []
        n_ima = 0
        if sequence is None:
            bag_list_file = open(os.path.join(data_folder_path, "{}_bags.txt".format(split)), 'r')
            lines = bag_list_file.read().splitlines()
            bag_list_file.close()
        else:
            if isinstance(sequence, (list, )):
                lines = sequence
            else:
                lines = [sequence]
        
        n_ima = [0]
        for line in lines:
            bag_name = line

            event_data_paths.append(os.path.join(data_folder_path,bag_name))
            num_ima_file = open(os.path.join(data_folder_path, bag_name, 'n_images.txt'), 'r')
            num_imas = num_ima_file.read()
            num_ima_file.close()
            num_imas_split = num_imas.split(' ')
            n_left_ima = int(num_imas_split[0]) - _MAX_SKIP_FRAMES
            n_ima.append(n_left_ima + n_ima[-1])
            
            n_right_ima = int(num_imas_split[1]) - _MAX_SKIP_FRAMES
            if n_right_ima > 0 and not split is 'test':
                n_ima.append(n_right_ima + n_ima[-1])
            else:
                n_ima.append(n_ima[-1])
            event_data_paths.append(os.path.join(data_folder_path,bag_name))

        return event_data_paths, n_ima

if __name__ == "__main__":
    data = EventData('/media/cyrilsterling/D/EV-FlowNet-pth/data/mvsec/', 'train')
    EventDataLoader = torch.utils.data.DataLoader(dataset=data, batch_size=1,shuffle=True)
    it = 0
    for i in EventDataLoader:
        a = i[0][0].numpy()
        b = i[1][0].numpy()
        c = i[2][0].numpy()
        cv2.namedWindow('a')
        cv2.namedWindow('b')
        cv2.namedWindow('c')
        a = a[2,...]+a[3,...]
        print(np.max(a))
        a = (a-np.min(a))/(np.max(a)-np.min(a))
        b = np.transpose(b,(1,2,0))
        c = np.transpose(c,(1,2,0))
        cv2.imshow('a',a)
        cv2.imshow('b',b)
        cv2.imshow('c',c)
        cv2.waitKey(1)