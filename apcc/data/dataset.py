import torch
import numpy as np
import torch.utils.data as data
import h5py
import os

class MVPSequenceDataset(data.Dataset):
    def __init__(self, 
                 prefix="train", 
                 num_views=4, 
                 random_order=True,
                 data_root="./data/",
                 views_per_object=26):
        super().__init__()

        if prefix=="train":
            filename = 'MVP_Train_CP.h5'
        elif prefix=="val":
            filename = 'MVP_Test_CP.h5'  
        else:
            raise ValueError("ValueError prefix should be [train/val] ")

        self.prefix = prefix
        self.num_views = num_views
        self.random_order = random_order
        self.views_per_object = views_per_object

        file_path = os.path.join(data_root, filename)
        input_file = h5py.File(file_path, 'r')

        self.partials = np.array(input_file['incomplete_pcds'][()])
        self.gt_data = np.array(input_file['complete_pcds'][()])
        self.labels = np.array(input_file['labels'][()])
        print('gt_data shape:',self.gt_data.shape, 'labels shape:', self.labels.shape)

        input_file.close()

        self.num_objects = self.gt_data.shape[0]

    def __len__(self):
        return self.num_objects

    def __getitem__(self, index):
        start = index * self.views_per_object
        end = start + self.views_per_object

        object_views = self.partials[start : end]

        if self.random_order:
            idxs = np.random.choice(self.views_per_object, self.num_views, replace=False)
        else:
            idxs = np.arange(self.num_views)

        seq_partials = torch.from_numpy(object_views[idxs]).float()
        complete = torch.from_numpy(self.gt_data[index]).float()
        label = torch.tensor(self.labels[index]).long()
       
        return label, seq_partials, complete