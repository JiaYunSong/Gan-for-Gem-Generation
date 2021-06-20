# 运行参考
# PS E:\1-课内资料\4-大三下课程\主-人工智能3\实验\06 NLP> python .\draw.py .\IGNORE_THRESH\IGNORE_THRESH=.3.txt

import re
import pprint
import sys
import pandas as pd

name=sys.argv[1]

rg = r'\[Training\] epoch:.* step:.* g_loss:.* d_loss:.* \(f_loss=.* r_loss=.* GP=.*\)'

mix = []

burn = [i for i in rg.replace('.*\\', '.*\\ ').replace('\\', '').split('.* ')]
# print(burn)
with open(name, 'r', encoding='utf8') as f:
    si = f.read().replace('，', ', ')
    for i in re.findall(rg, si):
        i = i.replace('，', ', ')
        for j in burn:
            i = i.replace(j, '')
        i = i.split(' ')

        mix.append([float(j) for j in i])

mix = pd.DataFrame(mix, columns=['Epoch', 'step', 'g_loss', 'd_loss', 'f_loss', 'r_loss', 'GP'])
pprint.pprint(mix)
# exit()
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')


plt.title('Train Result')
ax = sns.lineplot(x="Epoch", y="d_loss", data=mix, label='D-Loss')
ax = sns.lineplot(x="Epoch", y="g_loss", data=mix, label='G-Loss')
# ax = sns.lineplot(x="Epoch", y="f_loss", data=mix, label='fake')
# ax = sns.lineplot(x="Epoch", y="r_loss", data=mix, label='real')
# ax = sns.lineplot(x="Epoch", y="GP", data=mix, label='GP')
ax.set_yscale('log')
# ax.set_xticks(range(0, 10))
# ax.set_yticks(list(range(100, 1000, 100))+list(range(10, 100, 10)))
# ax.set_xlim([-0.5,9.5])
# ax.set_ylim([20,10**3+1000])
# plt.plot(mix[0],mix[1],label='LOSS')
# plt.plot(mix[0],mix[2],label='ACC')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.ylim(0,1)
name=name.replace("\\","_")
plt.savefig(f'./{name}2.png')
# plt.show()
