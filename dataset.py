from types import DynamicClassAttribute
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch import tensor
import os
from PIL import Image
import utils

# 最好transform传入ToTensor参数
class Set(Dataset):
    def __init__(self,root='tiny_vid',type='JPEG',train=True,transform=ToTensor()):
        super().__init__()
        self.root = root
        self.type = type
        self.train = train
        self.transform = transform
        self.size = 0
        self.paths = []

        for r,d,f in os.walk(self.root):
            for name in f:
                name_split = name.split('.')
                if(name_split[1]==self.type):
                    if(self.train and int(name_split[0])<=150 or ~self.train and int(name_split[0])>150):
                        self.paths.append(os.path.join(r,name))
                        self.size += 1


    def __getitem__(self, index: int):
        if not os.path.isfile(self.paths[index]):
            print(self.paths[index] + 'does not exist!')
            return None
        image = Image.open(self.paths[index])   
        path_split = self.paths[index].split('\\')
        sub_class = path_split[1]
        sub_index = int(path_split[2].split('.')[0])
        label_items = []
        with open(self.root + '\\' + sub_class + '_gt.txt') as f:
            for line in f.readlines()[sub_index-1:sub_index]:
                label_items = line.strip().split(' ')
        coordinate = [float(label_items[i])/128.0 for i in range(1,5)]
        label = self._getclass(sub_class)

        if self.transform:
            image = self.transform(image) 
        coordinate = tensor(coordinate)
        label = tensor(label)

        return image,coordinate,label
    

    def __len__(self):
        return self.size


    def _getclass(self,sub_class):
        return utils.C2N(sub_class)
