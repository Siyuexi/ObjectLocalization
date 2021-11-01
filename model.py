from torch import softmax
from torch.nn import Module,Sequential,Linear,Dropout,ReLU
from torchvision.models.resnet import  resnet18

# 使用现有sota的网络做特征提取，再手动接上两个fc网络
class Net(Module):
    # 默认用pytorch集成的resnet18做特征提取
    def __init__(self,dim_input=512,num_class=5,model=resnet18(pretrained=True),p=0.5,complex=(False,False)):
        super().__init__()
        self.dim= dim_input
        self.num_class = num_class
        self.pre_model = model
        self.complex = complex

        # 特征提取部分
        self.pre_model.fc = Sequential()
        self.feature_layers = self.pre_model 

        # 定位回归
        self.regression_fc = Linear(self.dim,4)
        self.fc_r1 = Linear(self.dim,int(self.dim/4))
        self.fc_r2 = Linear(int(self.dim/4),int(self.dim/16))
        self.fc_r3 = Linear(int(self.dim/16),4)

        # 标签分类
        self.classification_fc = Linear(self.dim,int(self.num_class))
        self.fc_c1 = Linear(self.dim,int(self.dim/4))
        self.fc_c2 = Linear(int(self.dim/4),int(self.dim/16))
        self.fc_c3 = Linear(int(self.dim/16),self.num_class)

        self.dropout = Dropout(p=p)
        self.relu = ReLU(inplace=True)

    def forward(self,i):
        f = self.feature_layers(i)

        # regression
        if(self.complex[0]):            
            v = self.fc_r1(f)
            v = self.dropout(v)
            v = self.relu(v)
            v = self.fc_r2(v)
            v = self.dropout(v)
            v = self.relu(v)
            v = self.fc_r3(v)
        else:
            v = self.regression_fc(f)
        v = self.relu(v)

        # classification
        if(self.complex[1]):
            c = self.fc_c1(f)
            c = self.dropout(c)
            c = self.relu(c)
            c = self.fc_c2(c)
            c = self.dropout(c)
            c = self.relu(c)
            c = self.fc_c3(c)
        else:
            c = self.classification_fc(f)
        c = self.dropout(c)
        c = self.relu(c)
        c = softmax(c,dim=0)

        return v,c
