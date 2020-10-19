import os
import os.path as osp
import glob
import torch

from torch.utils import data
from torchvision import transforms as T
from PIL import Image


class DataSet(data.Dataset):

    def __init__(self, config, img_transform, mode='train'):
        self.img_transform = img_transform
        if mode == 'train':
            self.img_dir = osp.join(config['TRAINING_CONFIG']['IMG_DIR_TRAIN'])
        else:
            self.img_dir = osp.join(config['TRAINING_CONFIG']['IMG_DIR_TEST'])
        #self.img_dir = osp.join(config['TRAINING_CONFIG']['IMG_DIR'], config['TRAINING_CONFIG']['MODE'])
        self.img_size = (config['MODEL_CONFIG']['IMG_SIZE_H'], config['MODEL_CONFIG']['IMG_SIZE_W'], 1)

        self.data_list = glob.glob(os.path.join(self.img_dir, '*.jpg'))
        self.data_list = list(set(self.data_list))

    def __getitem__(self, index):
        target, patient_view, _ = self.data_list[index].split(os.sep)[-1].split('_')
        patient_id = int(patient_view.replace('.jpg', '')[:-4])
        view_id = int(patient_view.replace('.jpg', '')[-4:])
        target_image = Image.open(self.data_list[index])
        target_image = target_image.convert('L')

        if target == 'f':
            target_value = 1
        else:
            target_value = 0
        return int(patient_id), int(view_id), self.img_transform(target_image), torch.LongTensor([target_value])

    def __len__(self):
        return len(self.data_list)


def get_loader(config):

    img_transform = list()
    img_size_h = config['MODEL_CONFIG']['IMG_SIZE_H']
    img_size_w = config['MODEL_CONFIG']['IMG_SIZE_W']

    img_transform.append(T.Resize((img_size_h, img_size_w)))
    img_transform.append(T.ToTensor())
    img_transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    img_transform = T.Compose(img_transform)

    dataset_train = DataSet(config, img_transform, 'train')
    data_loader_train = data.DataLoader(dataset=dataset_train,
                                  batch_size=config['TRAINING_CONFIG']['BATCH_SIZE'],
                                  shuffle=(config['TRAINING_CONFIG']['MODE'] == 'train'),
                                  num_workers=config['TRAINING_CONFIG']['NUM_WORKER'],
                                  drop_last=True)
    dataset_test = DataSet(config, img_transform, 'test')
    data_loader_test = data.DataLoader(dataset=dataset_test,
                                        batch_size=1,
                                        shuffle=(config['TRAINING_CONFIG']['MODE'] == 'train'),
                                        num_workers=config['TRAINING_CONFIG']['NUM_WORKER'],
                                        drop_last=True)
    return data_loader_train, data_loader_test
