# 卷积神经网络(CNN)

- 输入层：输入图像等信息
- 卷积层：用来提取图像的底层特征
- 池化层：防止过拟合，将数据维度减小
- 全连接层：汇总卷积层和池化层得到的图像的底层特征和信息
- 输出层：根据全连接层的信息得到概率最大的结果

![LeNet](https://i-blog.csdnimg.cn/blog_migrate/68036225544df1369f698f31b47dfb1d.png)

# 搭建LeNet-5模型并基于CIFAR-10数据集训练

**参考CSDN文章**:

 [用Pytorch搭建LeNet-5模型且基于CIFAR-10数据集训练_lenet模型软件-CSDN博客](https://blog.csdn.net/qq_45445740/article/details/106827428?ops_request_misc=%7B%22request%5Fid%22%3A%2208136AF4-9C59-408B-A91A-4D2A75A0FEE4%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fall.%22%7D&request_id=08136AF4-9C59-408B-A91A-4D2A75A0FEE4&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-4-106827428-null-null.142^v100^control&utm_term=搭建 Lenet 并基于 CIFAR10 训练&spm=1018.2226.3001.4187)

[深度学习入门（搭建Lenet并基于CIFAR10训练）_搭建 lenet 并基于 cifar10 训练-CSDN博客](https://blog.csdn.net/m0_72565814/article/details/140526042?ops_request_misc=%7B%22request%5Fid%22%3A%2208136AF4-9C59-408B-A91A-4D2A75A0FEE4%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fall.%22%7D&request_id=08136AF4-9C59-408B-A91A-4D2A75A0FEE4&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-2-140526042-null-null.142^v100^control&utm_term=搭建 Lenet 并基于 CIFAR10 训练&spm=1018.2226.3001.4187)

#### CIFAR-10数据集 
CIFAR-10数据集由10个类的60000个32x32彩色图像组成，以下是十个类别，每个类有6000个图像。有50000个训练图像和10000个测试图像。
classes = (‘plane’, ‘car’, ‘bird’, ‘cat’,‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’)

#### LeNet-5模型

**1.模型架构**
	LeNet-5模型结构为输入层–卷积层–池化层–卷积层–池化层–全连接层–全连接层–输出，为串联模式，如上图所示
**2.模型特点**

- a.每个卷积层包含3个部分：卷积、池化和非线性激活函数。
- b.使用卷积提前空间特征。
- c.采用降采样（Subsample）的平均池化层（Average Pooling）。
- d.使用双曲正切（Tanh）的激活函数。
- e.最后用MLP作为分类器。

**3.模型的网络介绍**

- 输入：
  - 灰度图像，通道为1，尺寸为1×32×32
- 第一层：卷积层
  - LeNet-5模型接受的输入层大小是1×32x32。卷积层的过滤器的尺寸是5x5，深度（卷积核个数）为6，不使用全0填充，步长为1。则这一层的输出的尺寸为32-5+1=28，深度为6。本层的输出矩阵大小为6×28×28。
- 第二层：池化层
  - 这一层的输入是第一层的输出，是一个6×28x28=4704的节点矩阵。本层采用的过滤器为2x2的大小，长和宽的步长均为2，所以本层的输出矩阵大小为6×14x14。
- 第三层：卷积层
  - 本层的输入矩阵大小为6×14x14，使用的过滤器大小为5x5，深度为16。本层不使用全0填充，步长为1。本层的输出矩阵大小为16×10x10。
- 第四层：池化层
  - 本层的输入矩阵大小是16×10x10，采用的过滤器大小是2x2，步长为2，本层的输出矩阵大小为16×5x5。
- 第五层：全连接层
  - 本层的输入矩阵大小为16×5x5。将此矩阵中的节点拉成一个向量，那么这就和全连接层的输入一样了，本层的输出节点个数为120。
- 第六层：全连接层
  - 本层的输入节点个数为120个，输出节点个数为84个。
- 第七层：全连接层
  - 本层的输入节点为84个，输出节点个数为10个。

***

#### 实例

**1.构建神经网络**

定义一个LeNet()类，继承于nn.Module的父类，这个类中包括两个函数：初始化函数和前向传播函数。初始化函数中：模型使用到的网络层结构；前向传播函数：实现整个前向传播的过程。

```python
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()#调用父类（基类）的构造函数
        # 卷积层1：输入图像深度=3，输出图像深度=16，卷积核大小=5*5，卷积步长=1;16表示输出维度，也表示卷积核个数
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=1)
        # 池化层1：采用最大池化，区域集大小=2*2.池化步长=2
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        # 卷积层2
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1)
        # 池化层2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层1：输入大小=32*5*5，输出大小=120
        self.fc1 = nn.Linear(32*5*5,120)
        # 全连接层2
        self.fc2 = nn.Linear(120,84)
        # 全连接层3
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = F.relu(self.conv1(x))  # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)  # output(16, 14, 14)
        x = F.relu(self.conv2(x))  # output(32, 10, 10)
        x = self.pool2(x)  # output(32, 5, 5)
        x = x.view(-1, 32 * 5 * 5)  # output(32*5*5)
        x = F.relu(self.fc1(x))  # output(120)
        x = F.relu(self.fc2(x))  # output(84)
        x = self.fc3(x)  # output(10)
        return x

```





**2.下载数据集并查看部分数据**

```python
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim # 优化器

'''下载数据集'''
# transforms.Compose()函数将两个函数拼接起来。
# （ToTensor()：把一个PIL.Image转换成Tensor，Normalize()：标准化，即减均值，除以标准差）
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# 训练集：下载CIFAR10数据集，如果没有事先下载该数据集，则将download参数改为True
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=False, transform=transform)
# 用DataLoader得到生成器，其中shuffle：是否将数据打乱；
# num_workers表示使用多进程加载的进程数，0代表不使用多进程
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=0)

# 测试集数据下载
testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 显示图像
def imshow(img):
    # 因为标准化normalize是：output = (input-0.5)/0.5
    # 则反标准化unnormalize是：input = output*0.5 + 0.5
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    # transpose()会更改多维数组的轴的顺序
    # Pytorch中是[channel，height，width],这里改为图像的[height，width，channel]
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 随机获取部分训练数据
dataiter = iter(trainloader)
images, labels = dataiter.next()
# 显示图像
# torchvision.utils.make_grid()将多张图片拼接在一张图中
imshow(torchvision.utils.make_grid(images))
# 打印标签
# str.join(sequence)：用于将序列中的元素以指定的字符str连接成一个新的字符串。这里的str是' '，空格
# %5s：表示输出字符串至少5个字符，不够5个的话，左侧用空格补
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

```

**3.训练模型**

```python
'''训练模型'''
# 有GPU就用GPU跑，没有就用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = LeNet()
net=net.to(device)
# print("该网络共有 {} 个参数".format(sum(x.numel() for x in net.parameters())))
# # 该网络共有 121182 个参数

# 对于多分类问题，应该使用Softmax函数，这里CIFAR10数据集属于多分类，却使用了交叉熵损失函数，
# 是因为进入CrossEntropyLoss()函数内部就会发现其中包含了Softmax函数
loss_function = nn.CrossEntropyLoss() # 使用交叉熵损失函数
# 优化器选择Adam，学习率设为0.001
optimizer = optim.Adam(net.parameters(), lr=0.001)
# 打印查看神经网络的结构
# print(net)
# # LeNet(
# #   (conv1): Conv2d(3, 16, kernel_size=(5, 5), stride=(1, 1))
# #   (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
# #   (conv2): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1))
# #   (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
# #   (fc1): Linear(in_features=800, out_features=120, bias=True)
# #   (fc2): Linear(in_features=120, out_features=84, bias=True)
# #   (fc3): Linear(in_features=84, out_features=10, bias=True)
# # )

for epoch in range(10): # 整个迭代10轮
    running_loss = 0.0 # 初始化损失函数值loss=0
    for i, data in enumerate(trainloader, start=0):
        # 获取训练数据
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device) # 将数据及标签传入GPU/CPU

        # 权重参数梯度清零
        optimizer.zero_grad()

        # 正向及反向传播
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        # 显示损失值
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

```

>**为什么每计算一个batch，就需要调用一次optimizer.zero_grad() 将权重参数梯度清零？**
>是为了进行梯度累加。一定条件下，batchsize越大训练效果越好，梯度累加则实现了batchsize的变相扩大，如果accumulation_steps为8，则batchsize ‘变相’ 扩大了8倍。
>通过这个特性能够变相实现一个很大batch数值的训练。

**4.在测试集中随机预测4张图**

```python
'''预测四张图看看'''
dataiter = iter(testloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

images, labels = images.to(device), labels.to(device)
outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]for j in range(4)))
```

**5.测试模型在测试集上的准确率及10个类别的准确率**

```python
'''测试模型'''
correct = 0
total = 0
# with是一个上下文管理器
# with torch.no_grad()表示其包括的内容不需要计算梯度，也不会进行反向传播，节省内存
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        # torch.max(outputs.data, 1)返回outputs每一行中最大值的那个元素，且返回其索引
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# 打印10个分类的准确率
class_correct = list(0. for i in range(10)) #class_correct=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
class_total = list(0. for i in range(10)) #class_total=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images) # outputs的维度是：4*10
        # torch.max(outputs.data, 1)返回outputs每一行中最大值的那个元素，且返回其索引
        # 此时predicted的维度是：4*1
        _, predicted = torch.max(outputs, 1)
        # 此时c的维度：4将预测值与实际标签进行比较，且进行降维
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

```

**6.存储训练好的权重文件进行预测**

```python
save_path = './Lenet.pth'
torch.save(net.state_dict(), save_path)
```

**预测**

```python
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import LeNet

transform = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = LeNet()
net.load_state_dict(torch.load('Lenet.pth'))

im = Image.open('1.jpg')
im = transform(im)  # [C, H, W]
# 输入pytorch网络中要求的格式是[batch，channel，height，width]，所以这里增加一个维度
im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

with torch.no_grad():
    outputs = net(im)
    predict = torch.max(outputs, dim=1)[1].data.numpy() # 索引即classed中的类别
print(classes[int(predict)])

# 直接打印张量的预测结果
with torch.no_grad():
    outputs = net(im)
    predict = torch.softmax(outputs,dim=1) # [batch，channel，height，width],这里因为对batch不需要处理
print(predict)
```

