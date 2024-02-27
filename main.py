
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# import timm
import torch
import torchattacks
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import AttackMethods
from models.verify import get_model
from Attack import Attack


# 看看配置
print("CUDA is_available:", torch.cuda.is_available())
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

models_path = r'D:\大学\硕士\研究\人工智能安全\Code\Model\npys'

inc_v3 = get_model('tf_inception_v3', models_path).eval().to(device)
inc_v4 = get_model('tf_inception_v4', models_path).eval().to(device)


# 替代模型
substitute_model = inc_v3

# 受害者模型
victim_model = [inc_v4]


# 数据集预处理
transform = transforms.Compose([
    transforms.Resize([299, 299]),
    transforms.ToTensor(),
    # 标准化后无法展示对抗图像
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
# D:\WFZSS\pycharm项目
dataset_root = r'D:\大学\硕士\研究\人工智能安全\Code\Dataset\ImageNet\val_data_1000'
dataset = datasets.ImageFolder(root=dataset_root, transform=transform)
dataloader = DataLoader(dataset, batch_size=5, shuffle=False)


# 实验次数 攻击图片数目
epoch = 1
number = 1000

defend = None

# 对抗攻击
atk = torchattacks.MIFGSM(substitute_model, eps=16/255, alpha=16/255/10, steps=10, decay=1.0)
atk = AttackMethods.RDE_MI_FGSM(substitute_model, eps=16/255, alpha=16/255/10, steps=10, decay=1.0, en=5)
atk = AttackMethods.RDE_VR_MI_FGSM(substitute_model, eps=16/255, alpha=16/255/10, steps=10, decay=1.0, en=5, M=20, beta=16/255/10, u=1.0)

# 开始攻击
start = time.time()
advs = Attack(atk, dataloader, defend, victim_model, epoch=epoch, number=number, device=device)
end = time.time()

print(f"--------------------攻击结束--------------------")

# 看看结果
print(f"{atk.model_name}使用{atk.attack}生成对抗样本")

length = len(victim_model)
acc = [0] * length
min = [1] * length
max = [0] * length

for i in range(epoch):
    for j in range(length):
        acc[j] = advs[i][j][0] / advs[i][j][1]
        if acc[j] < min[j]:
            min[j] = acc[j]
        if acc[j] > max[j]:
            max[j] = acc[j]

    print(f"第{i+1}轮攻击成功率：", end='')
    for k in range(length):
        print(f"%.4f" % (acc[k]), end=' ')
    print('')

print(f"共耗时%.2fmin" % ((end - start) / 60))

print(f"总共攻击了", end=' ')
for k in range(length):
    print(f"{advs[0][k][1]}", end=' ')
print(f"张图片")

print(f"最小攻击成功率/最大攻击成功率：", end='')
for k in range(length):
    print(f"%.4f/%.4f" % (min[k], max[k]), end=' ')

