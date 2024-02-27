import torch
import time
from tqdm import tqdm


# 攻击函数
def Attack(atk, dataloader, defend=None, models=[], epoch=2, number=1000, device=torch.device("cpu")):
    advs = [[] for _ in range(epoch)]
    for i in range(epoch):
        # 模型数
        length = len(models)
        # 记录攻击成功的样本数
        counts = [0] * length
        # 记录已经攻击的样本数
        totals = [0] * length

        print(f"-------------------第{i + 1}轮攻击-------------------")

        # 循环数据集并进行攻击
        start = time.time()
        for j, (images, labels) in enumerate(tqdm(dataloader)):
            # 正在攻击的张数
            T = (j + 1) * dataloader.batch_size

            if T > number:
                break

            images, labels = images.to(device), labels.to(device)
            # showIMAGE(image)

            # 生成对抗样本
            adv_img = atk(images, labels)
            # showIMAGE(adv_img)

            # 防御
            if defend is not None:
                if defend.defend:
                    adv_img = defend(adv_img)
                    # showIMAGE(adv_img)
                    # RS
                    # adv_predict = defend(adv_img)

            with torch.no_grad():
                for k in range(length):
                    # 使用RS则注释下面一行代码
                    adv_output = models[k](adv_img)
                    adv_predict = adv_output.max(1)[1]

                    # 记录
                    k_count = (adv_predict != labels).detach().sum().item()
                    counts[k] = counts[k] + k_count
                    totals[k] = totals[k] + dataloader.batch_size


            if T % 100 == 0:
                print(f"第{i + 1}/{epoch}轮攻击，第{T}/{number}张图片，此时攻击成功率：", end='')
                for k in range(length):
                    print("%.4f " % (counts[k] / totals[k]), end=' ')
                print('')

        end = time.time()
        print(f"第{i + 1}轮耗时%.2fs" % (end - start))

        print(f"第{i + 1}轮攻击成功率：", end='')
        for k in range(length):
            print("%.4f " % (counts[k] / totals[k]), end=' ')
        print('')

        for k in range(length):
            advs[i].append([counts[k], totals[k]])

    return advs


