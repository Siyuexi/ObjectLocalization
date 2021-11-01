import torch
from torchvision.models.resnet import  resnet18,resnet50
from torchvision import transforms,models
from torchvision.utils import make_grid
from torch.utils.data import DataLoader,SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import sys,random

import utils
import model
import dataset

batch_size = 16

transform = transforms.Compose([
    transforms.Resize(256),               # 把图片resize
    transforms.ToTensor() , # 将图片转换为Tensor,归一化至[0,1]
])

# 导入数据集
test_dataset = dataset.Set(train=False,transform=transform)

# 测试集前一半划分给验证集了
index_size = len(test_dataset)
indices = range(index_size)
indices_test = indices[round(index_size*0.5):]  

# 从后一半测试集随机采样
sampler_test = SubsetRandomSampler(indices_test)

# 装载测试集，使用随机采样的原测试集index
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,sampler = sampler_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 装载网络参数
# net = model.Net(dim_input=512,num_class=5,model=resnet18(pretrained=True),p=0.5,complex=False) 
net = model.Net(dim_input=2048,num_class=5,model=resnet50(pretrained=True),p=0.5,complex=False) 
net.load_state_dict(torch.load('model/resnet50-finetuning-log.pth'))
net = net.to(device)
net.eval() 
test_NoAs = []
test_IoUs = []

# 测试
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
    
    # 抽前n个图片可视化分类和定位效果
    batch_iter = DataLoader(dataset=test_dataset,batch_size=batch_size,sampler = sampler_test)
    batch = next(iter(batch_iter))

    imgs = (batch[0].permute(0, 2, 3, 1))/255.0
    axes = utils.show_images(imgs, 4, 4, scale=2)
    for ax, coo, label in zip(axes, batch[1], batch[2]):
        utils.show_bboxes(ax, [coo*224], labels=utils.N2C(label), colors=['w'])

    loc,cla = net(batch[0].to(device))
    for ax, coo, label in zip(axes, loc.cpu(), torch.argmax(cla.cpu().t(),dim=0)):
        utils.show_bboxes(ax, [coo*224], labels=utils.N2C(label), colors=['r'])
    plt.show()


# 计算准确率并打印
rights1 = (sum([tup[0] for tup in test_NoAs]), sum([tup[1] for tup in test_NoAs]))
rights2 = (sum([tup[0] for tup in test_IoUs]), sum([tup[1] for tup in test_IoUs]))
right_rate1 = 1.0 * rights1[0].detach().to('cpu').numpy() / rights1[1]
right_rate2 = 1.0 * rights2[0].detach().to('cpu').numpy() / rights2[1]

print("ClassificationTestAccuracy: ",right_rate1,file=sys.stdout)
print("RegressionTestAccuracy: ",right_rate2,file=sys.stdout)

