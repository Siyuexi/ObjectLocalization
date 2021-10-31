import torch
from torchvision.models.resnet import  resnet18,resnet50
from torchvision import transforms,models
from torch.utils.data import DataLoader,SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import sys

import utils
import model
import dataset

batch_size = 32

transform = transforms.Compose([
    transforms.Resize(256),               # 把图片resize为256*256
    transforms.RandomCrop(224),           # 随机裁剪224*224
    transforms.ToTensor() , # 将图片转换为Tensor,归一化至[0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #标准化
])

# 导入数据集
test_dataset = dataset.Set(train=False,transform=transform)

# 
index_size = len(test_dataset)
indices = range(index_size)
indices_test = indices[round(index_size*0.5):]  

# 
sampler_test = SubsetRandomSampler(indices_test)

# 装载测试集，使用随机采样的原测试集index
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,sampler = sampler_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# net = model.Net(dim_input=512,num_class=5,model=resnet18(pretrained=True),p=0.5,complex=False) 
net = model.Net(dim_input=2048,num_class=5,model=resnet50(pretrained=True),p=0.5,complex=False) 
net.load_state_dict(torch.load('model/resnet50-finetuning-log.pth'))
net = net.to(device)
net.eval() 
test_NoAs = []
test_IoUs = []

with torch.no_grad():
    for data,coordinate,label in test_loader:
        data = data.to(device)
        coordinate = coordinate.to(device)
        label = label.to(device)     

        loc,cla = net(data)        
        test_NoA = utils.NoA(cla,label)
        test_NoAs.append(test_NoA)
        test_IoU = utils.IoU(loc,coordinate)
        test_IoUs.append(test_IoU)
        
# 计算准确率
rights1 = (sum([tup[0] for tup in test_NoAs]), sum([tup[1] for tup in test_NoAs]))
rights2 = (sum([tup[0] for tup in test_IoUs]), sum([tup[1] for tup in test_IoUs]))
right_rate1 = 1.0 * rights1[0].detach().to('cpu').numpy() / rights1[1]
right_rate2 = 1.0 * rights2[0].detach().to('cpu').numpy() / rights2[1]

print("ClassificationTestAccuracy: ",right_rate1,file=sys.stdout)
print("RegressionTestAccuracy: ",right_rate2,file=sys.stdout)