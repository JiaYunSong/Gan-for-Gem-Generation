# Gan-for-Gem-Generation

使用AI Studio测试DCGAN与WGAN-GP模型，采用宝石数据集



# GAN 宝石生成

### Train & Test

#### 原始

<img src="https://pic.imgdb.cn/item/60ca0b11844ef46bb2d56d6c.jpg" height="500" width="500"/>

#### DCGAN

<img src="https://pic.imgdb.cn/item/60ca3848844ef46bb21c82ab.jpg" height="500" width="500"/>

#### WGAN-GP

<img src="https://pic.imgdb.cn/item/60c71dc7844ef46bb206dda8.jpg" height="500" width="500"/>


* 2021/6/11 - WGAN-GP生成 \[3x96x96\] 宝石图像
* 2021/6/16 - DCGAN生成 \[3x96x96\] 宝石图像
## 数据加载及预处理
### 解压数据


```python
!if [ ! -d ~/data/initial-data ];then mkdir ~/data/initial-data; unzip -oq ~/data/data54865/Gemstones.zip -d ~/data/initial-data; fi
```

### 查找移动所有宝石图像


```python
!if [ ! -d ~/data/all-data ];then mkdir ~/data/all-data; find ~/data/initial-data -name "*.jpg" -type f -exec cp {} ~/data/all-data \; ; fi
```

### 生成低质量与高质量图像


```python
import os
import cv2

al_path = '/home/aistudio/data/all-data'

def get_imgs(size: int):
    datapath = f'/home/aistudio/data/data-{size}'
    if not os.path.exists(datapath):
        os.mkdir(datapath)
        filenames = os.listdir(al_path)
        for filename in filenames:
            fullname = os.path.join(al_path, filename)
            dataname = os.path.join(datapath, filename)
            try:
                img = cv2.imread(fullname, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(dataname, img)
            except:
                ...

get_imgs(96)
get_imgs(256)
```

##  WGAN-GP训练生成：3x96x96
### WGAN-GP 简介

论文地址：[WGAN-GP](https://arxiv.org/pdf/1704.00028.pdf)

1. 这次的模型，我们依然使用了`DCGAN`的网络结构，因为`WGAN-GP`的学习重点不在网络上

2. WGAN 论文发现了 JS 散度导致 GAN 训练不稳定的问题，并引入了一种新的分布距离度量方法：Wasserstein 距离，也叫推土机距离(Earth-Mover Distance，简称 EM 距离)，它表示了从一个分布变换到另一个分布的最小代价

3. 由于前WGAN并没有真正的实现`1-Lipschitz`，只有对任意输入x梯度都小于或等于1的时候，则该函数才是` 1-Lipschitz function`

$$
V(G,D)\approx \max\limits_{D}\{E_{x-P_{data}}[D(x)]-E_{x-P_{generate}}[D(x)]\}-\lambda E_{x-P_{penalty}}[(||\bigtriangledown_x D(x)||-1)^2]
$$

> 也就是想尽办法的让判别器认为真实的图片得到更高的分数所以需要加上负号，让生成的图片得到更低的分数，最后加上一个梯度的惩罚项，对于惩罚项的解释(在惩罚项中希望,如果越是满足`1-Lipschitz function`，惩罚就越少。事实证明这样做的效果是非常好的),现在大量GAN都能看到WGAN-GP的身影，所以它非常的重要

$$
Loss = \operatorname{E}\limits_{x-P_g}[D(x)]-\operatorname{E}\limits_{x-P_r}[D(x)]+\lambda \operatorname{E}\limits_{x-P_{ab}}[\max\{0, ||\bigtriangledown_x D(x)||-1\}]
$$

> 其中有个超参数 $\lambda$ 表示超参数  本文设置为10

### 网络结构
* 1. 使用卷积网络进行下采样
* ![](https://ai-studio-static-online.cdn.bcebos.com/0cb758271fde40b68d51642452f8b18f6b343dd78b8c4fc09c031c0d8db85a09)
* 2. 使用反卷积进行上采样
* ![](https://ai-studio-static-online.cdn.bcebos.com/addd8ef2fd7b402e9d56ea35d98b0666fa77f74feca4438f9af3d38421d01081)
* 取消所有 pooling 层。G 网络中使用微步幅度卷积（fractionally strided convolution）
* 代替 pooling 层，D 网络中使用步幅卷积（strided convolution）代替 pooling 层。 ·在 D 和 G 中均使用 batch normalization
* 去掉 FC 层，使网络变为全卷积网络
* G 网络中使用 ReLU 作为激活函数，最后一层使用 tanh
* D 网络中使用 LeakyReLU 作为激活函数


```python
# 导入包
import os
import cv2
import random
import paddle
import paddle.fluid as fluid
import matplotlib.pyplot as plt
import numpy as np
```

### 训练执行
#### 预设参数


```python
Wg_path = '/home/aistudio/WGAN-GP-model'
d9_path = '/home/aistudio/data/data-96'

use_gpu = True
place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()

batch_size = 10   # 每批次数量
z_num = 100       # 随机量维度

c = 10.0          # 超参数 λ
epsilon = 1e-16   # GP 梯度惩罚项

epochs = 1000-885
train_d = 2
```

#### 构建网络


```python
# 生成器 (-1, z_num, 1, 1) => (3, 96, 96)
def Generator(x, name='G'):
    def deconv(x, num_filters, filter_size=5,padding='SAME', stride=2, act='relu'):
            x = fluid.layers.conv2d_transpose(x, num_filters=num_filters, filter_size=filter_size, stride=stride, padding=padding)
            x = fluid.layers.batch_norm(x, momentum=0.8, act=act)
            return x
    with fluid.unique_name.guard(name+'/'):
        x = fluid.layers.reshape(x, (-1, z_num, 1, 1))
        x = deconv(x, num_filters=512, filter_size=6, stride=1, padding='VALID') # 6
        x = deconv(x, num_filters=256)            #12
        x = deconv(x, num_filters=128)            #24
        x = deconv(x, num_filters=64)             #48
        x = deconv(x, num_filters=3, act='tanh')  #96
    return x

# 判别器 (3, 96, 96) => (1) 
def Discriminator(x, name='D'):
    def conv(x, num_filters, momentum=0.8):
            x = fluid.layers.conv2d(x, num_filters=num_filters, filter_size=5, stride=2, padding='SAME')
            x = fluid.layers.batch_norm(x, momentum=momentum)
            x = fluid.layers.leaky_relu(x, alpha=0.2)
            x = fluid.layers.dropout(x, dropout_prob=0.25)
            return x
    with fluid.unique_name.guard(name+'/'):
        x = conv(x, num_filters=64)             # 48
        x = conv(x, num_filters=128)            # 24
        x = conv(x, num_filters=256)            # 12
        x = conv(x, num_filters=512)            # 6
        x = fluid.layers.pool2d(x, pool_type='avg', global_pooling=True)
        x = fluid.layers.flatten(x)
        x = fluid.layers.fc(x, size=1)
    return x
```

#### 构建优化器
##### 预设


```python
paddle.enable_static()         # 构建静态图
d_program = fluid.Program()    # 生成器
g_program = fluid.Program()    # 判别器
```

##### 判别器优化器实现
1. 生成器的超参数有噪声的数量这里固定为100
2. 通过输入的噪声生成图像(`Generator`)，并将生成的图片给判别器(`Discriminator`),我们给它一个低分
3. 直接将真实的图片给判别器，并给它一个高分，由于优化器是梯度下降的方向，我们想要得到上升那么就给损失一个负号用于得到高分
4. 给定真实图片和生成图片，完成梯度惩罚项GP, 并让他减小损失
5. 训练需要固定生成器，让判别器学习
6. 优化器 beta1=0, beta2=0.9
7. 关于GP梯度惩罚项的实现[参考](https://github.com/PaddlePaddle/models/blob/release/1.7/PaddleCV/gan/trainer/STGAN.py第150行的函数)
8. c为梯度惩罚项的超参数默认为 10.0


```python
# 这里的惩罚项gp是copy于paddle的github
# https://github.com/PaddlePaddle/models/blob/release/1.7/PaddleCV/gan/trainer/STGAN.py第150行的函数
def _interpolate(a, b=None):
    alpha = fluid.layers.uniform_random_batch_size_like(input=a, shape=[a.shape[0]], min=0.0, max=1.0)
    beta = fluid.layers.uniform_random_batch_size_like(input=a, shape=a.shape, min=0.0, max=1.0)
    
    mean = fluid.layers.reduce_mean(a, dim=list(range(len(a.shape))), keep_dim=True)
    input_sub_mean = fluid.layers.elementwise_sub(a, mean, axis=0)
    var = fluid.layers.reduce_mean(
        fluid.layers.square(input_sub_mean),
        dim=list(range(len(a.shape))),
        keep_dim=True)
    
    b = beta * fluid.layers.sqrt(var) * 0.5 + a
    inner = fluid.layers.elementwise_mul((b-a), alpha, axis=0) + a

    return inner
```


```python
with fluid.program_guard(d_program):
    fake_z_1 = fluid.data(name='fake_z', dtype='float32', shape=(None, z_num))    # 传入参数用fluid.data
    img_fake_1 = Generator(fake_z_1)              # 通过生成器生成图片
    fake_ret_1 = Discriminator(img_fake_1)        # 判别器判断好坏
    loss_1 = fluid.layers.mean(fake_ret_1)        # 判别器损失

    parameter_list = [
        var.name for var in d_program.list_vars()
        if fluid.io.is_parameter(var) and var.name.startswith('D')
    ]
    
    fluid.optimizer.AdamOptimizer(learning_rate=1e-4,beta1=0, beta2=0.9).minimize(loss_1, parameter_list=parameter_list)

    img_real_1 = fluid.data(name='img_real', dtype='float32', shape=(None, 3, 96, 96))  # 用fluid.data传入真实的图片
    real_ret_1 = Discriminator(img_real_1)        # 用判别真实的图片
    loss_2 = 1 - fluid.layers.mean(real_ret_1)    # 损失函数
    
    fluid.optimizer.AdamOptimizer(learning_rate=1e-4,beta1=0, beta2=0.9).minimize(loss_2, parameter_list=parameter_list)

    x = _interpolate(img_real_1, img_fake_1)
    pred = Discriminator(x)
    
    a_vars = [
        var.name for var in fluid.default_main_program().list_vars()
        if fluid.io.is_parameter(var) and var.name.startswith("D")
    ]
    
    grad = fluid.gradients(pred, x, no_grad_set=a_vars)[0]
    grad = fluid.layers.reshape(grad, [-1, grad.shape[1] * grad.shape[2] * grad.shape[3]])
    norm = fluid.layers.sqrt(fluid.layers.reduce_sum(fluid.layers.square(grad), dim=1) + epsilon)
    gp = fluid.layers.reduce_mean(fluid.layers.square(norm - 1.0)) * c
    d_loss = loss_1 + loss_2 + gp

    # 限制梯度
    # clip = fluid.clip.GradientClipByValue(min=CLIP[0], max=CLIP[1])

    fluid.optimizer.AdamOptimizer(learning_rate=1e-4,beta1=0, beta2=0.9).minimize(gp, parameter_list=parameter_list)
```

##### 生成器优化的实现
1. 输入噪声给生成器，因为我们想让生成器生成一张真实的图片，所以我们需要固定判别器，让生成器生成一张趋于真实的图片，所以要让他向分数高的方向走，所以也要加个负号


```python
with fluid.program_guard(g_program):
    fake_z_2 = fluid.data(name='fake_z', dtype='float32', shape=(None, z_num))
    img_fake_2 = Generator(fake_z_2)
    test_program = g_program.clone(for_test=True)
    fake_ret_2 = Discriminator(img_fake_2)
    
    avg_loss_2 = 1 - fluid.layers.mean(fake_ret_2)
    
    parameter_list = [
        var.name for var in d_program.list_vars()
        if fluid.io.is_parameter(var) and var.name.startswith('G')
    ]
    
    fluid.optimizer.AdamOptimizer(learning_rate=1e-4, beta1=0, beta2=0.9).minimize(avg_loss_2, parameter_list=parameter_list)
```


```python
exe = fluid.executor.Executor(place=place)
_ = exe.run(fluid.default_startup_program())
```


```python
# 建立快照
try:
    fluid.io.load_params(exe, dirname=Wg_path, main_program=fluid.default_startup_program())
except:
    ...
def get_cur():
    try:
        img_names = np.array([i.strip('.jpg').split('_') for i in os.listdir(Wg_path) if '.jpg' in i]).astype('int')
        assert len(img_names) != 0
    except:
        return [0, 0]
    return np.max(img_names, axis=0)+1
epoch_pro, _ = get_cur()
step_pro = 0
```

#### 数据读取


```python
def get_batch():
    filenames = os.listdir(d9_path)
    random.shuffle(filenames)
    img_list, label_list = [], []
    for i, filename in enumerate(filenames):
        fullname = os.path.join(d9_path, filename)
        img = cv2.imread(fullname, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.*2 - 1
        img = np.transpose(img, (2,0,1))
        img_list.append(img)
        label_list.append(1.)
        if (i+1) % batch_size == 0:
            yield np.array(img_list).astype('float32'), np.array(label_list)
            img_list, label_list = [], []
```

#### 模型训练


```python
for epoch in range(epochs):
    images = None
    epoch += epoch_pro
    for step,(x, y) in enumerate(get_batch()):
        step += step_pro
        fake_z = np.random.uniform(size=(x.shape[0], z_num), low=-1, high=1).astype('float32')
        [loss_1_, loss_2_, gp_] = exe.run(program=d_program, 
                                          feed={'fake_z':fake_z,'img_real':x}, 
                                          fetch_list=[loss_1, loss_2, gp])

        # 生成器训练
        for _ in range(train_d):
            fake_z = np.random.uniform(size=(x.shape[0], z_num), low=-1, high=1).astype('float32')
            [g_loss] = exe.run(program=g_program, 
                               feed={'fake_z':fake_z}, 
                               fetch_list=[avg_loss_2])
        
        # 100次进行预测一次
        if step % 100 == 0:
            loss_1_, loss_2_, gp_, g_loss = map(float, (loss_1_, loss_2_, gp_, g_loss))
            out_str = f'[Training] epoch:{epoch} step:{step} g_loss:{g_loss} d_loss:{loss_1_+loss_2_+gp_} (f_loss={loss_1_} r_loss={loss_2_} GP={gp_})'
            print(out_str)
            with open(f'{Wg_path}_out.txt',"a") as file:
                file.write(out_str+"\n")
            # 快照
            fluid.io.save_params(exe, dirname=Wg_path, main_program=fluid.default_startup_program())    
            fake_z = np.random.uniform(size=(z_num, z_num), low=-1, high=1).astype('float32')
            [pre_im] = exe.run(program=test_program, 
                               feed={'fake_z':fake_z}, 
                               fetch_list=[img_fake_2])
            pre_im = (np.transpose(pre_im, (0, 2, 3, 1))+1) / 2
            
            # 准备一个画布用于存放生成的图片
            images = np.zeros((960, 960, 3))
            for h in range(10):
                for w in range(10):
                    images[h*96:(h+1)*96, w*96:(w+1)*96] = pre_im[(h*10)+w]
            plt.imsave('{}/{}_{}.jpg'.format(Wg_path, epoch, step), images)
        if step == 0 and epoch % 10 == 0:
            plt.imshow(images, cmap='gray')
            plt.show()
```


```python
import imageio
fake_z1 = np.array([[ 0.1628196 ,  0.06020002, -0.2060656 , -0.2008977 ,  0.55397815,  -0.26759636, -0.36206895, -0.59165126, -0.28061983, -0.6222151 ,  -0.96598303, -0.4606206 ,  0.9854551 ,  0.9816093 , -0.32310477,  -0.11815367, -0.5434894 , -0.74145615, -0.8636268 , -0.5834029 ,  -0.07124058, -0.30142292, -0.5333166 , -0.67257905, -0.9680295 ,   0.46814537, -0.6241449 ,  0.13388671,  0.3210726 ,  0.51314497,  -0.1876392 , -0.7934325 ,  0.7071664 ,  0.41295967,  0.76740944,  -0.50664145,  0.08661275, -0.42233503,  0.95415235,  0.34426385,  -0.27813435,  0.21395077,  0.9130151 ,  0.7578369 , -0.6572961 ,  -0.0671403 ,  0.9326817 ,  0.8190068 ,  0.06983604, -0.16990714,   0.71631354, -0.60850775, -0.5150853 , -0.4799691 , -0.9335457 ,  -0.05940403, -0.6704578 ,  0.6399308 ,  0.52119714,  0.31435177,  -0.85910934,  0.40837994, -0.18065603, -0.44286138, -0.59907156,   0.8817434 , -0.8173919 , -0.82570285, -0.23782504, -0.73118174,  -0.7469359 , -0.3491494 , -0.20398776,  0.9803937 ,  0.9116375 ,  -0.2655413 , -0.0376263 , -0.02038905, -0.49076393,  0.9365829 ,  -0.6917846 ,  0.665875  , -0.25417942,  0.90112466,  0.64291835,   0.7206952 ,  0.4192555 ,  0.47273248,  0.7568629 , -0.8213731 ,  -0.7171862 ,  0.67112094,  0.5666802 ,  0.11436127, -0.7256467 ,  -0.320071  , -0.06459896, -0.24214156, -0.51376414, -0.02128822]], dtype='float32')
fake_z2 = np.array([[ 1.73423752e-01, -7.69945920e-01,  3.47309746e-02,  -8.93073976e-01,  3.62380743e-01,  5.61251529e-02,   4.02133048e-01, -5.35719329e-04,  6.83315471e-02,   2.17393041e-01,  8.28237236e-02,  1.52823642e-01,  -9.85875189e-01,  3.83724898e-01,  3.32061827e-01,  -4.95688915e-01,  7.67859519e-01, -1.01901911e-01,   7.21223176e-01,  1.77856877e-01, -3.79434884e-01,  -2.11844251e-01, -2.66274273e-01,  9.90064979e-01,   8.29603255e-01,  3.94291520e-01, -1.16514340e-01,  -8.96508455e-01, -8.47046003e-02,  3.05729099e-02,   8.69425461e-02,  8.26287746e-01,  2.75682479e-01,  -1.33725196e-01,  7.49988407e-02, -4.66008186e-01,  -6.46197319e-01,  4.74357009e-01, -5.04314780e-01,  -3.02102327e-01, -4.18006957e-01,  2.08232135e-01,   1.33660734e-01,  1.63588554e-01,  1.93897516e-01,   5.56772292e-01,  7.61188447e-01,  5.67563713e-01,   7.98536837e-01,  9.79610443e-01,  2.78141707e-01,   6.32494092e-01,  8.37376297e-01,  4.99828428e-01,  -1.59887835e-01, -9.52926099e-01,  8.96628678e-01,   5.41600943e-01, -7.04732239e-01,  9.64725554e-01,   3.96280289e-01, -8.58498752e-01,  2.63351351e-01,   8.67280304e-01, -9.08607617e-02, -5.28445005e-01,  -9.70839798e-01, -5.78085065e-01, -1.11416914e-01,   4.95842934e-01,  4.53393042e-01,  7.17945039e-01,   3.73275667e-01, -5.30356169e-01,  7.64361501e-01,   9.70767558e-01,  1.94362663e-02, -4.35328662e-01,   6.61424458e-01, -4.81950119e-02, -6.80367470e-01,   5.85199833e-01,  9.59802032e-01,  9.28983808e-01,  -7.82506168e-01, -1.04114622e-01, -3.78846288e-01,  -4.60661829e-01,  9.33107734e-01,  6.71302080e-02,  -9.57123935e-01, -3.99602950e-01,  1.06667526e-01,  -2.39341632e-01,  9.59848046e-01, -6.91255629e-01,   9.10927296e-01,  9.03677702e-01,  3.43659163e-01,  -9.69704032e-01]], dtype='float32')
fake_z3 = np.array([[-1.54808953e-01,  3.70797455e-01, -5.50530434e-01,   8.88798118e-01,  1.48261501e-03,  7.54083320e-02,  -3.60852331e-01,  2.43449537e-03,  4.52306420e-01,  -2.68616408e-01,  5.51057532e-02, -8.31197053e-02,  -2.49736071e-01,  6.24021709e-01,  9.88368928e-01,   1.21546052e-01,  4.77205336e-01,  7.82941282e-01,  -1.53581619e-01,  4.02827412e-01, -2.88837820e-01,  -3.87253672e-01, -9.51536000e-02,  1.67471785e-02,  -6.55421197e-01,  2.52013266e-01, -1.93831936e-01,  -9.87818241e-01,  2.24797338e-01, -1.28131136e-01,  -5.52273810e-01, -3.97933125e-01,  8.07264805e-01,  -6.31823361e-01,  2.65025795e-02,  4.88095433e-01,  -5.54655433e-01, -4.67667162e-01, -1.27145112e-01,  -4.99425173e-01,  9.96037185e-01, -9.91724610e-01,  -5.49022555e-02,  5.59283912e-01, -1.05380170e-01,  -6.43494129e-01, -5.18584549e-01,  9.71008182e-01,   3.64938766e-01, -7.27478325e-01, -1.87705070e-01,  -7.46205866e-01, -5.93074322e-01, -3.10427427e-01,  -5.14982700e-01, -9.45145935e-02,  6.63303852e-01,   3.31954628e-01, -7.68493474e-01, -6.98818147e-01,  -7.65405536e-01, -1.23112634e-01,  8.81450176e-01,   3.29640567e-01,  5.57847321e-01,  8.27734172e-01,   9.55396414e-01, -4.73707706e-01, -2.11369112e-01,   9.17738318e-01, -3.17583710e-01,  2.29109004e-01,  -2.48818994e-01, -6.75448358e-01, -3.66140872e-01,  -8.03524792e-01, -2.31833115e-01, -2.44700357e-01,   9.41398084e-01,  6.05898201e-01, -5.25917768e-01,   6.79681301e-01, -8.50292683e-01,  7.23254323e-01,   2.02767387e-01,  4.98614907e-01,  9.75749612e-01,  -2.33748749e-01,  7.47435391e-01,  2.00816706e-01,   1.55958056e-01, -1.13003649e-01,  8.70095789e-01,   2.72289995e-04, -9.05706227e-01,  1.31745622e-01,   4.41222727e-01, -3.08290869e-01, -8.77345204e-01,   7.00753391e-01]], dtype='float32')
fake_z4 = np.array([[-0.19166796,  0.8518012 ,  0.145091  , -0.05993893, -0.45888463,   0.8673928 ,  0.9652282 ,  0.12726055,  0.01244434, -0.93083525,   0.43431133, -0.49930117, -0.55162716,  0.35950926,  0.7037384 ,   0.11401674, -0.28130996, -0.2928921 ,  0.72169   ,  0.16362152,  -0.5016924 ,  0.43633834,  0.11796366,  0.00219617,  0.847646  ,   0.6748598 , -0.32114825, -0.15379265,  0.70928174, -0.8728422 ,   0.31036106, -0.8250491 ,  0.43096024, -0.96281725,  0.10819924,   0.9144131 ,  0.20409475, -0.9040856 ,  0.16202451,  0.4682495 ,  -0.13104072,  0.51667684, -0.7445863 ,  0.9387579 , -0.12740219,   0.17839415, -0.8474835 , -0.28403   ,  0.7451734 ,  0.19540492,   0.49905938, -0.61591697, -0.45006886,  0.7683603 , -0.9711963 ,   0.34337667, -0.22944267,  0.5802888 ,  0.3484021 ,  0.5493261 ,  -0.28516468,  0.02338881,  0.9595401 , -0.10940923, -0.8052722 ,   0.03689744,  0.71858245, -0.43022957,  0.10462559, -0.4443373 ,   0.36294216, -0.04272104, -0.50580406,  0.1556459 ,  0.34158394,   0.921397  , -0.884857  , -0.29005402,  0.9884361 , -0.5774341 ,  -0.30703196,  0.31190208, -0.2624621 ,  0.5999602 , -0.29485407,  -0.85030997, -0.9605466 , -0.5932406 , -0.7367069 , -0.54928774,   0.14493614, -0.36576104, -0.16612104,  0.5380705 , -0.8164843 ,  -0.80553305, -0.37761232,  0.905625  ,  0.67822284, -0.47533378]], dtype='float32')
fake_z5 = np.array([[-0.49124366, -0.7515978 , -0.74474436,  0.33651626, -0.02003454,  -0.49979228, -0.3460132 , -0.1082238 , -0.8632628 ,  0.32383934,   0.57275146,  0.04399857, -0.05217148,  0.1337484 ,  0.5154774 ,  -0.5780535 , -0.5374929 ,  0.3237491 , -0.8140235 , -0.23915705,   0.33803776, -0.97627527,  0.90115136,  0.66156435, -0.16090132,  -0.96128684,  0.97522634,  0.8348675 , -0.51188135, -0.28184426,  -0.6413365 , -0.7100339 ,  0.30043355,  0.42623404, -0.6104447 ,   0.2676714 , -0.4290038 ,  0.2543924 ,  0.30417633, -0.554996  ,  -0.9040517 , -0.48714116, -0.02494111,  0.74440295,  0.16304332,   0.50590354,  0.02005209, -0.27269456,  0.6572092 ,  0.73941165,  -0.8238788 , -0.43116006,  0.79212266,  0.7831773 ,  0.32471797,  -0.43271893,  0.35854793, -0.7456453 ,  0.5968816 , -0.55512017,   0.644918  ,  0.34041697, -0.49072075,  0.9913528 , -0.69197947,  -0.9509921 ,  0.63709044, -0.84002507,  0.5306693 , -0.873428  ,  -0.56814194, -0.31120738, -0.68780464,  0.4997686 ,  0.5911224 ,   0.76103264, -0.8965332 , -0.36284062,  0.96344227,  0.39664373,   0.1157234 ,  0.42655757, -0.9809847 , -0.574825  , -0.81645733,  -0.55916363,  0.44941834,  0.01480339,  0.5341408 , -0.6769514 ,   0.7752634 ,  0.29789975, -0.9451429 ,  0.35611492, -0.4459079 ,  -0.91902405, -0.5986422 , -0.9678276 ,  0.01595495,  0.75727   ]], dtype='float32')
```


```python
sip = 20
num = 0
def app(z1, z2):
    for i in range(sip):
        global num
        num += 1
        step = (z2-z1) / sip
        fake_z = z1 + step*i
        [pre_im] = exe.run(program=test_program, 
                           feed={'fake_z':fake_z}, 
                           fetch_list=[img_fake_2])
        pre_im = ((np.transpose(pre_im, (0, 2, 3, 1))+1) / 2).reshape(96,96,3)
        plt.imshow(pre_im, cmap='gray')
        plt.imsave(f'{Wg_path}_test/{num}.jpg', pre_im)
        if i % 10 == 0:
            plt.show()
app(fake_z1, fake_z2)
app(fake_z2, fake_z3)
app(fake_z3, fake_z4)
app(fake_z4, fake_z5)
app(fake_z5, fake_z1)
```

<img src="https://pic.imgdb.cn/item/60c718ce844ef46bb2c31f14.gif" height="200" width="200"/>

## DCGAN 训练生成：3x96x96
-----------------------------
### DCGAN简介
DCGAN是深层卷积网络与 GAN 的结合，其基本原理与 GAN 相同，只是将生成网络和判别网络用两个卷积网络（CNN）替代。为了提高生成样本的质量和网络的收敛速度，论文中的 DCGAN 在网络结构上进行了一些改进：

* 取消 pooling 层：在网络中，所有的pooling层使用步幅卷积（strided convolutions）(判别器)和微步幅度卷积（fractional-strided convolutions）(生成器)进行替换。
* 加入 batch normalization：在生成器和判别器中均加入batchnorm。
* 使用全卷积网络：去掉了FC层，以实现更深的网络结构。
* 激活函数：在生成器（G）中，最后一层使用Tanh函数，其余层采用 ReLu 函数 ; 判别器（D）中都采用LeakyReLu。

参考[Paddle2.0-通过DCGAN实现人脸图像生成](https://aistudio.baidu.com/aistudio/projectdetail/1086168?channelType=0&channel=0)进行的改进：
* 将Adam优化器beta1参数设置为0.8，具体请参考[原论文](https://arxiv.org/abs/1412.6980)
* 将BatchNorm批归一化中momentum参数设置为0.5
* 将判别器(D)激活函数由elu改为leaky_relu，并将alpha参数设置为0.2
* 生成器输出，判别器输入改为[3,128,128]
* 损失函数选用Softmax_with_cross_entropy

> **参考文献**
> 
> [1] Goodfellow, Ian J.; Pouget-Abadie, Jean; Mirza, Mehdi; Xu, Bing; Warde-Farley, David; Ozair, Sherjil; Courville, Aaron; Bengio, Yoshua. Generative Adversarial Networks. 2014. arXiv:1406.2661 [stat.ML].
> 
> [2] Andrej Karpathy, Pieter Abbeel, Greg Brockman, Peter Chen, Vicki Cheung, Rocky Duan, Ian Goodfellow, Durk Kingma, Jonathan Ho, Rein Houthooft, Tim Salimans, John Schulman, Ilya Sutskever, And Wojciech Zaremba, Generative Models, OpenAI, [April 7, 2016]
> 
> [3] alimans, Tim; Goodfellow, Ian; Zaremba, Wojciech; Cheung, Vicki; Radford, Alec; Chen, Xi. Improved Techniques for Training GANs. 2016. arXiv:1606.03498 [cs.LG].
> 
> [4] Radford A, Metz L, Chintala S. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks[J]. Computer Science, 2015.
> 
> [5]Kingma D , Ba J . Adam: A Method for Stochastic Optimization[J]. Computer ence, 2014.

### DCGAN实现
#### 引入相关包


```python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import paddle
from paddle.io import Dataset
import paddle.fluid as fluid
from paddle.static import InputSpec
```

#### DCGAN网络结构搭建
##### 初始化器

在 DCGAN 论文中，作者指定所有模型权重应从均值为0、标准差为0.02的正态分布中随机初始化。
在paddle.nn中，调用fluid.nn.initializer.Normal实现initialize设置


```python
conv_initializer=paddle.nn.initializer.Normal(mean=0.0, std=0.02)
bn_initializer=paddle.nn.initializer.Normal(mean=1.0, std=0.02)
```

##### 判别器
如上文所述，生成器$D$是一个二进制分类网络，它以图像作为输入，输出图像是真实的（相对应$G$生成的假样本）的概率。输入$Shape$为[3,64,64]的RGB图像，通过一系列的$Conv2d$，$BatchNorm2d$和$LeakyReLU$层对其进行处理，然后通过全连接层输出的神经元个数为2，对应两个标签的预测概率。

* 将BatchNorm批归一化中momentum参数设置为0.5
* 将判别器(D)激活函数leaky_relu的alpha参数设置为0.2

> 输入:  为大小64x64的RGB三通道图片  
> 输出:  经过一层全连接层最后为shape为[batch_size,2]的Tensor


```python
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class Discriminator(paddle.nn.Layer):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_1 = nn.Conv2D(     # 96
            3,64,4,2,1,
            bias_attr=False,weight_attr=paddle.ParamAttr(name="d_conv_weight_1_",initializer=conv_initializer)
            )
        self.conv_2 = nn.Conv2D(     # 48
            64,128,4,2,1,
            bias_attr=False,weight_attr=paddle.ParamAttr(name="d_conv_weight_2_",initializer=conv_initializer)
            )
        self.bn_2 = nn.BatchNorm2D(
            128,
            weight_attr=paddle.ParamAttr(name="d_2_bn_weight_",initializer=bn_initializer),momentum=0.8
            )
        self.conv_3 = nn.Conv2D(     # 24
            128,256,4,2,1,
            bias_attr=False,weight_attr=paddle.ParamAttr(name="d_conv_weight_3_",initializer=conv_initializer)
            )
        self.bn_3 = nn.BatchNorm2D(
            256,
            weight_attr=paddle.ParamAttr(name="d_3_bn_weight_",initializer=bn_initializer),momentum=0.8
            )
        self.conv_4 = nn.Conv2D(     # 12
            256,512,4,2,1,
            bias_attr=False,weight_attr=paddle.ParamAttr(name="d_conv_weight_4_",initializer=conv_initializer)
            )
        self.bn_4 = nn.BatchNorm2D(
            512,
            weight_attr=paddle.ParamAttr(name="d_4_bn_weight_",initializer=bn_initializer),momentum=0.8
            )
        self.conv_5 = nn.Conv2D(     # 6
            512,1,6,1,0,
            bias_attr=False,weight_attr=paddle.ParamAttr(name="d_conv_weight_5_",initializer=conv_initializer)
            )
    
    def forward(self, x):
        x = self.conv_1(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        x = self.conv_4(x)
        x = self.bn_4(x)
        x = F.leaky_relu(x,negative_slope=0.2)
        x = self.conv_5(x)
        x = F.sigmoid(x)
        return x
```

##### 生成器
生成器$G$旨在映射潜在空间矢量$z$到数据空间。由于我们的数据是图像，因此转换$z$到数据空间意味着最终创建具有与训练图像相同大小[3,64,64]的RGB图像。在网络设计中，这是通过一系列二维卷积转置层来完成的，每个层都与$BatchNorm$层和$ReLu$激活函数。生成器的输出通过$tanh$函数输出，以使其返回到输入数据范围[−1,1]。值得注意的是，在卷积转置层之后存在$BatchNorm$函数，因为这是DCGAN论文的关键改进。这些层有助于训练过程中的梯度更好地流动。  

**生成器网络结构**  
![](https://ai-studio-static-online.cdn.bcebos.com/ca0434dd681849338b1c0c46285616f72add01ab894b4e95848daecd5a72e3cb)

* 将$BatchNorm$批归一化中$momentum$参数设置为0.5

> 输入:Tensor的Shape为[batch_size,100]其中每个数值大小为0~1之间的float32随机数  
> 输出:3x64x64RGB三通道图片


```python
class Generator(paddle.nn.Layer):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv_1 = nn.Conv2DTranspose(    # 6
            100,512,6,1,0,
            bias_attr=False,weight_attr=paddle.ParamAttr(name="g_dconv_weight_1_",initializer=conv_initializer)
            )
        self.bn_1 = nn.BatchNorm2D(
            512,
            weight_attr=paddle.ParamAttr(name="g_1_bn_weight_",initializer=bn_initializer),momentum=0.8
            )
        self.conv_2 = nn.Conv2DTranspose(    # 12
            512,256,4,2,1,
            bias_attr=False,weight_attr=paddle.ParamAttr(name="g_dconv_weight_2_",initializer=conv_initializer)
            )
        self.bn_2 = nn.BatchNorm2D(
            256,
            weight_attr=paddle.ParamAttr(name="g_2_bn_weight_",initializer=bn_initializer),momentum=0.8
            )
        self.conv_3 = nn.Conv2DTranspose(    # 24
            256,128,4,2,1,
            bias_attr=False,weight_attr=paddle.ParamAttr(name="g_dconv_weight_3_",initializer=conv_initializer)
            )
        self.bn_3 = nn.BatchNorm2D(
            128,
            weight_attr=paddle.ParamAttr(name="g_3_bn_weight_",initializer=bn_initializer),momentum=0.8
            )
        self.conv_4 = nn.Conv2DTranspose(    # 48
            128,64,4,2,1,
            bias_attr=False,weight_attr=paddle.ParamAttr(name="g_dconv_weight_4_",initializer=conv_initializer)
            )
        self.bn_4 = nn.BatchNorm2D(
            64,
            weight_attr=paddle.ParamAttr(name="g_4_bn_weight_",initializer=bn_initializer),momentum=0.8
            )
        self.conv_5 = nn.Conv2DTranspose(    # 96
            64,3,4,2,1,
            bias_attr=False,weight_attr=paddle.ParamAttr(name="g_dconv_weight_5_",initializer=conv_initializer)
            )
        self.tanh = paddle.nn.Tanh()
    
    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = F.relu(x)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = F.relu(x)
        x = self.conv_4(x)
        x = self.bn_4(x)
        x = F.relu(x)
        x = self.conv_5(x)
        x = self.tanh(x)
        return x

```

#### 模型搭建

##### 参数设定


```python
import paddle.optimizer as optim

output = "/home/aistudio/DCGAN-model/"
output_path = '/home/aistudio/DCGAN-model'

use_gpu = True
device = paddle.set_device('gpu' if use_gpu else 'cpu')

img_dim = 96
lr = 0.0002
epoch = 1000
batch_size = 128
G_DIMENSION = 100
beta1=0.5
beta2=0.999

real_label = 1.
fake_label = 0.
```

##### 搭建网络结构


```python
paddle.disable_static(device)
netD = Discriminator()
netG = Generator()
```

##### 损失函数与优化器设定
选用BCELoss,公式如下:

$Out = -1 * (label * log(input) + (1 - label) * log(1 - input))$


```python
###损失函数
loss = paddle.nn.BCELoss()

optimizerD = optim.Adam(parameters=netD.parameters(), learning_rate=lr, beta1=beta1, beta2=beta2)
optimizerG = optim.Adam(parameters=netG.parameters(), learning_rate=lr, beta1=beta1, beta2=beta2)
```

#### 数据读取


```python
import os
import cv2

data_path = '/home/aistudio/data/data-96/'
paddle.enable_static()
```


```python
class DataGenerater(Dataset):
    def __init__(self, path=data_path):
        super(DataGenerater, self).__init__()
        self.dir = path
        self.datalist = os.listdir(data_path)
        self.image_size = (img_dim,img_dim)
    
    def __getitem__(self, idx):
        return self._load_img(self.dir + self.datalist[idx])

    def __len__(self):
        return len(self.datalist)
    
    def _load_img(self, path):
        try:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2,0,1)).astype('float32')/255*2-1
        except Exception as e:
            print(e)
        return img

train_dataset = DataGenerater()
imgs = paddle.static.data(name='img', shape=[None,3,img_dim,img_dim], dtype='float32')
train_loader = paddle.io.DataLoader(
    train_dataset, 
    places=paddle.CPUPlace(), 
    feed_list = [imgs],
    batch_size=batch_size, 
    shuffle=True,
    num_workers=2,
    use_buffer_reader=True,
    use_shared_memory=False,
    drop_last=True,
    )

# 测试Train_loader
# for batch_id, data in enumerate(train_loader()):
#     plt.figure(figsize=(15,15))
#     try:
#         for i in range(100):
#             image = (np.array(data)[i].transpose((1,2,0))+1)*255/2
#             image = image.astype('int32')
#             plt.subplot(10, 10, i + 1)
#             plt.imshow(image, vmin=-1, vmax=1)
#             plt.axis('off')
#             plt.xticks([])
#             plt.yticks([])
#             plt.subplots_adjust(wspace=0.1, hspace=0.1)
#         plt.suptitle('\n Training Images',fontsize=30)
#         plt.show()
#         break
#     except IOError:
#         print(IOError)
```

#### 模型训练


```python
# 建立快照
try:
    netG.set_state_dict(paddle.load(f"{output_path}/generator.params"))
    netD.set_state_dict(paddle.load(f"{output_path}/discriminator.params"))
except:
    ...
def get_cur():
    try:
        img_names = np.array([i.strip('.png').split('_') for i in os.listdir(output_path) if '.png' in i]).astype('int')
        assert len(img_names) != 0
    except:
        return [0, 0]
    return np.max(img_names, axis=0)+1
epoch_pro, _ = get_cur()
```


```python
###训练过程
paddle.disable_static(device)
losses = [[], []]
if not os.path.exists(output_path):
    os.makedirs(output_path)

for pass_id in range(epoch_pro, epoch):
    for batch_id, data in enumerate(train_loader()):
        #训练判别器 
        optimizerD.clear_grad()
        real_cpu = data
        label = paddle.full((batch_size,1,1,1),real_label,dtype='float32')
        output = netD(real_cpu)
        errD_real = loss(output,label)
        errD_real.backward()
        optimizerD.step()
        optimizerD.clear_grad()

        noise = paddle.randn([batch_size,G_DIMENSION,1,1],'float32')
        fake = netG(noise)
        label = paddle.full((batch_size,1,1,1),fake_label,dtype='float32')
        output = netD(fake.detach())
        errD_fake = loss(output,label)
        errD_fake.backward()
        optimizerD.step()
        optimizerD.clear_grad()

        errD = errD_real + errD_fake
        
        losses[0].append(errD.numpy()[0])
        ###训练生成器
        optimizerG.clear_grad()
        noise = paddle.randn([batch_size,G_DIMENSION,1,1],'float32')
        fake = netG(noise)
        label = paddle.full((batch_size,1,1,1),real_label,dtype=np.float32,)
        output = netD(fake)
        errG = loss(output,label)
        errG.backward()
        optimizerG.step()
        optimizerG.clear_grad()
        
        losses[1].append(errG.numpy()[0])
        if batch_id % 100 == 0:
            msg = f'Epoch ID={pass_id} Batch ID={batch_id} \n\n D-Loss={errD.numpy()[0]} G-Loss={errG.numpy()[0]}'
            print(msg.replace('\n\n', ':'))
            with open(f'{output_path}_out.txt',"a") as file:
                file.write(msg.replace('\n\n', ':')+"\n")

            # 每轮的生成结果
            generated_image = netG(noise).numpy()
            imgs = []
            plt.figure(figsize=(15,15))
            for i in range(100):
                image = (np.array(generated_image)[i].transpose((1,2,0))+1)*255/2
                image = image.astype('int32')
                image = np.where(image > 0, image, 0)
                plt.subplot(10, 10, i + 1)
                plt.imshow(image, vmin=-1, vmax=1)
                plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.suptitle(msg, fontsize=20)
            plt.savefig(f'{output_path}/{pass_id}_{batch_id}.png', bbox_inches='tight')
            plt.close('all')
    paddle.save(netG.state_dict(), f"{output_path}/generator.params")
    paddle.save(netD.state_dict(), f"{output_path}/discriminator.params")
```

##### 绘制 LOSS 变化图


```python
plt.figure(figsize=(15, 6))
x = np.arange(len(losses[0]))
plt.title('Generator and Discriminator Loss During Training')
plt.xlabel('Number of Batch')
plt.plot(x,np.array(losses[0]),label='D Loss')
plt.plot(x,np.array(losses[1]),label='G Loss')
plt.yscale('log')
plt.legend()
plt.savefig(f'{output_path}/Generator and Discriminator Loss During Training.jpg')
plt.show()
```


```python
fake_z = paddle.randn([batch_size,G_DIMENSION,1,1],'float32')
pre_im = netG(fake_z)
image = (np.array(pre_im)[i].transpose((1,2,0))+1)*255/2
image = image.astype('int32')
plt.imshow(image, cmap='gray')
plt.show()
```


```python
sip = 20
num = 0
def app(z1, z2):
    for i in range(sip):
        global num
        num += 1
        step = (z2-z1) / sip
        fake_z = z1 + step*i
        [pre_im] = exe.run(program=test_program, 
                           feed={'fake_z':fake_z}, 
                           fetch_list=[img_fake_2])
        pre_im = ((np.transpose(pre_im, (0, 2, 3, 1))+1) / 2).reshape(96,96,3)
        plt.imshow(pre_im, cmap='gray')
        plt.imsave(f'{output_path}_test/{num}.jpg', pre_im)
        if i % 10 == 0:
            plt.show()
app(fake_z1, fake_z2)
app(fake_z2, fake_z3)
app(fake_z3, fake_z4)
app(fake_z4, fake_z5)
app(fake_z5, fake_z1)
```
