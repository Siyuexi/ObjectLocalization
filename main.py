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

# log = open('log/resnet18-finetuning-log.txt','wt')
log = open('log/resnet50-finetuning-log.txt','wt')

"""

    Part.1 数据预处理

"""

# 超参数设置
num_epochs = 30   
num_classes = 5
print_period = 5 # print per [print_period] batch
lossmix_period = 0 # mix loss from [lossmix_period] epoch
batch_size = 32  
image_size = 128 
learning_rate = 3e-3
momentum = 9e-1
val_test_rate = 0.5  
L = 1 # lambda
T = 0.7 # threshold of IoU

# 定义图像转换
transform = transforms.Compose([
    transforms.Resize(224),               # 把图片resize
    transforms.ToTensor() , # 将图片转换为Tensor,归一化至[0,1]
])

# 导入数据集
train_dataset = dataset.Set(train=True,transform=transform)
test_dataset = dataset.Set(train=False,transform=transform)

# 装载训练集，随机划分训练批次
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
epoch_size =  len(train_loader.dataset)
print('shape of train set:',epoch_size,file=log,flush=True)
print('shape of train set:',epoch_size,file=sys.stdout)

# 从原测试集中划分出验证集
index_size = len(test_dataset)
indices = range(index_size)
indices_val = indices[:round(index_size*0.5)]
indices_test = indices[round(index_size*0.5):]  

# 使用测试集的index对验证集和测试集随机采样
sampler_val = SubsetRandomSampler(indices_val)
sampler_test = SubsetRandomSampler(indices_test)

# 装载验证集，使用随机采样的原测试集index
validation_loader = DataLoader(dataset=test_dataset,batch_size = batch_size,sampler = sampler_val)
# 装载测试集，使用随机采样的原测试集index
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,sampler = sampler_test)


"""

    Part.2 模型训练与训练集/校验集动态评估

"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device : "+str(device),file=log,flush=True)
print("device : "+str(device),file=sys.stdout)

# 网络初始化与参数记录
# net = model.Net(dim_input=512,num_class=5,model=resnet18(pretrained=True),p=0.5,complex=False) 
net = model.Net(dim_input=2048,num_class=5,model=resnet50(pretrained=True),p=0.5,complex=(False,False)) 
net = net.to(device)
best_model_wts = net.state_dict()

# 分类采用交叉熵 回归采用MSE
criterion_class = nn.CrossEntropyLoss() 
criterion_location = nn.MSELoss()

# 优化器使用SGD
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

record_err1 = [] # 分类错误率记录
record_err2 = [] # 回归错误率记录
best_acc = 0 # 记录最佳acc

""" 训练过程 """
# epoch迭代
for epoch in range(num_epochs):

    #记录
    train_NoAs = [] 
    train_IoUs = []
    
    # batch迭代
    for batch_id, (data,coordinate,label) in enumerate(train_loader):

        # 拷贝数据
        data = data.to(device)
        coordinate = coordinate.to(device)
        label = label.to(device)

        # 模型训练
        net.train()
        
        # 计算Loss
        loc,cla =  net(data) 

        if(epoch>=lossmix_period):
            loss_class = criterion_class(cla, label) 
            loss_location = criterion_location(loc,coordinate)
            loss = loss_class + L*loss_location
        else:
            loss = criterion_class(cla, label) 
        
        # 优化权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算分类精度
        train_NoA = utils.NoA(cla, label)
        train_NoAs.append(train_NoA)

        # 计算定位精度
        train_IoU = utils.IoU(loc,coordinate,threshold=T)
        train_IoUs.append(train_IoU)
        
        """ 可视化：每间隔5个batch做一次validation，并且打印一次精度信息"""
        if batch_id%print_period ==0: 

            # 模型评估(训练集、校验集)
            net.eval() 
            val_NoAs = [] 
            val_IoUs = []
            #迭代已遍历过的训练集数据：
            for (data, coordinate,label) in validation_loader: 

                # 拷贝数据
                data = data.to(device)
                coordinate = coordinate.to(device)
                label = label.to(device)

                # 
                loc,cla = net(data) 

                # 
                val_NoA = utils.NoA(cla, label) 
                val_NoAs.append(val_NoA)

                # 
                val_IoU = utils.IoU(loc,coordinate,threshold=T)
                val_IoUs.append(val_IoU)

                
            # 记录训练集中分类正确样本数与总样本数
            train_r1 = (sum([tup[0] for tup in train_NoAs]), sum([tup[1] for tup in train_NoAs]))
            train_r2 = (sum([tup[0] for tup in train_IoUs]), sum([tup[1] for tup in train_IoUs]))

            # 记录校验集中分类正确样本数与总样本数
            val_r1 = (sum([tup[0] for tup in val_NoAs]), sum([tup[1] for tup in val_NoAs]))
            val_r2 = (sum([tup[0] for tup in val_IoUs]), sum([tup[1] for tup in val_IoUs]))
            
            # 
            train_acc_r1 = 100. * train_r1[0] / train_r1[1]
            train_acc_r2 = 100. * train_r2[0] / train_r2[1]

            val_acc_r1 = 100. * val_r1[0] / val_r1[1]
            val_acc_r2 = 100. * val_r2[0] / val_r2[1]

            checkpoint = 'Epoch [{}/{}]\tBatch [{}/{}]\tSample [{}/{}]\tLoss: {:.6f}\tClaTraAcc: {:.2f}%\tCalValAcc: {:.2f}%\tRegTraAcc: {:.2f}%\tRegValAcc: {:.2f}%'.format(
                epoch+1, num_epochs,
                min(batch_id+print_period, epoch_size//batch_size), epoch_size//batch_size ,
                min((batch_id+print_period) * batch_size,epoch_size), epoch_size,
                loss.item(), 
                train_acc_r1, 
                val_acc_r1,
                train_acc_r2,
                val_acc_r2
                )
            print(checkpoint,file=log,flush=True)
            print(checkpoint,file=sys.stdout)
            val_acc_avg = (val_acc_r1+val_acc_r2)/2
            if(val_acc_avg>best_acc):
                best_acc = val_acc_avg
                best_model_wts = net.state_dict()
            # 记录错误率
            record_err1.append((100 - train_acc_r1.cpu(), 100 - val_acc_r1.cpu()))
            record_err2.append((100 - train_acc_r2.cpu(), 100 - val_acc_r2.cpu()))


# 绘制训练过程的误差曲线，验证集和测试集上的错误率。
plt.figure(figsize = (10,7))
plt.plot(record_err1)
plt.xlabel('Steps')
plt.ylabel('Classification Error rate(%)')
plt.legend(['TrainErr','ValidErr'])
plt.savefig('plot/Classification-Error-rate.jpg')
plt.show()

plt.figure(figsize = (10,7))
plt.plot(record_err2)
plt.xlabel('Steps')
plt.ylabel('Regression Error rate(%)')
plt.legend(['TrainErr','ValidErr'])
plt.savefig('plot/Regression-Error-rate.jpg')
plt.show()

# 保存最佳模型参数
# torch.save(best_model_wts, "model/resnet18-finetuning-log.pth")
torch.save(best_model_wts, "model/resnet50-finetuning-log.pth")


"""

    Part.3 精度测试与测试集静态模型评估

"""

# 模型评估(测试集)
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
        test_IoU = utils.IoU(loc,coordinate,threshold=T)
        test_IoUs.append(test_IoU)

    # 抽前n个图片可视化分类和定位效果
    batch_iter = DataLoader(dataset=test_dataset,batch_size=batch_size,sampler = sampler_test)
    batch = next(iter(batch_iter))

    imgs = (batch[0].permute(0, 2, 3, 1)) / 255.
    axes = utils.show_images(imgs, 4, 4, scale=2)
    for ax, coo, label in zip(axes, batch[1], batch[2]):
        utils.show_bboxes(ax, [coo*224], labels=utils.N2C(label), colors=['w'])

    loc,cla = net(batch[0].to(device))
    for ax, coo, label in zip(axes, loc.cpu(), torch.argmax(cla.cpu().t(),dim=0)):
        utils.show_bboxes(ax, [coo*224], labels=utils.N2C(label), colors=['r'])
    plt.savefig('plot/Test-Sample-Graph.jpg')
    plt.show()
        
# 计算准确率
rights1 = (sum([tup[0] for tup in test_NoAs]), sum([tup[1] for tup in test_NoAs]))
rights2 = (sum([tup[0] for tup in test_IoUs]), sum([tup[1] for tup in test_IoUs]))
right_rate1 = 1.0 * rights1[0].detach().to('cpu').numpy() / rights1[1]
right_rate2 = 1.0 * rights2[0].detach().to('cpu').numpy() / rights2[1]

print("ClassificationTestAccuracy: ",right_rate1,file=log,flush=True)
print("ClassificationTestAccuracy: ",right_rate1,file=sys.stdout)
print("RegressionTestAccuracy: ",right_rate2,file=log,flush=True)
print("RegressionTestAccuracy: ",right_rate2,file=sys.stdout)