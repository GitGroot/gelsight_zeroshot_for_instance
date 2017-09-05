import os
import cv2
import numpy as np
import random
data_path = '/home/wangfeng/download/gelSight/fuse/data/gel_fold3_224'






class Data:
    def __init__(self,attr_num, path=data_path, target_size=224, mode='gray'):
        self.gelsight_path = path
        self.target_size = target_size
        self.mode = mode
        self.num_step = 5
        self.image_channel = 1 if mode=='gray' else 3
        self.class_attribute_matrix = np.load('../data/class_attribute_matrix.npy')
        # TODO notice that train_label and test_label must be incremental, and start from 0
        self.train_labels = range(80)
        self.test_labels = range(80, 100)

    def get_train_attr_label(self):
        ret = []
        for label in self.train_labels:
            ret.append(self.class_attribute_matrix[label])
        return np.array(ret)

    def get_test_attr_label(self):
        ret = []
        for label in self.test_labels:
            ret.append(self.class_attribute_matrix[label])
        return np.array(ret)


    def S(self,train):
        S = self.get_train_attr_label().T if train else self.get_test_attr_label().T
        return S
        # TODO test for the normalize version
        #return S/[(S[:,i]**2).sum()**0.5 for i in range(S.shape[1])]

    def load_gelsight_data(self):
        video_list = os.listdir(self.gelsight_path)
        random.shuffle(video_list)
        train_videos = []
        train_labels = []
        train_attr = []
        test_videos = []
        test_labels = []
        random_index = range(len(video_list))
        for index in random_index:
            video_folder = video_list[index]
            name = video_folder.split('.')[0]
            class_info = name.split('_')[0].split('F')[1]  # F0001~F0119
            label = eval(class_info.lstrip('0')) - 1
            if label >= 100:
                continue
            frame_list = os.listdir(os.path.join(self.gelsight_path, video_folder))
            frame_list.sort()
            frames = []
            for frame_name in frame_list:
                image = cv2.imread(os.path.join(self.gelsight_path, video_folder, frame_name))
                if self.target_size != 224:
                    image = cv2.resize(image, (self.target_size, self.target_size))
                frames.append(image)
            if label in self.train_labels:
                train_videos.append(frames)
                train_labels.append(self.train_labels.index(label))
                train_attr.append(self.class_attribute_matrix[label])
            elif label in self.test_labels:
                test_videos.append(frames)
                test_labels.append(self.test_labels.index(label))
        return train_videos, train_labels, train_attr, test_videos, test_labels