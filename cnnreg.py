#coding: utf-8
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import time
import pylab
import matplotlib.pyplot as plt
import chainer
import chainer.links as L
import os
from PIL import Image
import random
import scipy.stats


gpu_flag = -1

if gpu_flag >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if gpu_flag >= 0 else np

batchsize = 10
val_batchsize=10
n_epoch = 100
tate=125
yoko=25

N =1000
N_test = 1000

# 画像を (nsample, channel, height, width) の4次元テンソルに変換
# MNISTはチャンネル数が1なのでreshapeだけでOK

# plt.imshow(X_train[1][0], cmap=pylab.cm.gray_r, interpolation='nearest')
# plt.show()
#, stride=1,pad=2
model = chainer.FunctionSet(conv1=L.Convolution2D(1,  40, 2),
                            conv2=L.Convolution2D(40, 20,  2),
                            fc6=L.Linear(2320, 516),
                            fc7=L.Linear(516, 208),
                            fc8=L.Linear(208, 1),
                            )
if gpu_flag >= 0:
    cuda.get_device(gpu_flag).use()
    model.to_gpu()

def forward(x_data, y_data, train=True):
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    h = F.max_pooling_2d(F.relu(
        F.local_response_normalization(model.conv1(x))), 4,stride=2)
    h = F.max_pooling_2d(F.relu(
        F.local_response_normalization(model.conv2(h))), 4,stride=2)
    h = F.dropout(F.relu(model.fc6(h)))
    h = F.dropout(F.relu(model.fc7(h)))
    h = F.relu(model.fc8(h))
    #import pdb; pdb.set_trace()
    if train:
        return F.mean_squared_error(h, t)

    else:
        return F.mean_squared_error(h, t), h

optimizer = optimizers.Adam()
optimizer.setup(model)

fp1 = open("accuracy.txt", "w")
fp2 = open("loss.txt", "w")
trainpic = open("train_list.txt", "w")
testpic = open("test_list.txt", "w")
gosa = open("gosa.txt", "w")



fp1.write("epoch\ttest_accuracy\n")
fp2.write("epoch\ttrain_loss\n")

# 訓練ループ
start_time = time.clock()
train_list = []
val_list = []
all_list=[]

def read_image(path):
    image = np.asarray(Image.open("cnndata/a"+path))#.transpose(2, 0, 1)

    return image

def image_count_list(path):
    tuples = []
    for line in open(path):
        pair = line.split(",")
        #tuples.append((os.path.join(".", pair[0]), np.int32(pair[1])))
        tuples.append((str(pair[0]), np.float32(pair[1])))
    return tuples


count_list = image_count_list("cnndata/sumcount.txt")

f = open("cnndata/getdata.txt")
data1 = f.read()
f.close()
lines = data1.split('\n')
for i,w in enumerate(count_list):
    for h in range(1,int(w[1])+1):
        imgn=str(w[0])+"_"+str(h)
        all_list.append((os.path.join(imgn+".png"),np.float32(lines[i])))

random.shuffle(all_list)


for tr in range(0,N):
    path2 , label2=all_list[tr]

    train_list.append((path2,label2))
    trainpic.write(str(all_list[tr]))
    trainpic.write("\n")
    trainpic.flush()


trainpic.close()

for te in range(N,N+N_test):
    path1 , label1=all_list[te]

    val_list.append((path1,label1))
    testpic.write(str(all_list[te]))
    testpic.write("\n")
    testpic.flush()

testpic.close()



for epoch in range(1, n_epoch + 1):


    print "epoch: %d" % epoch

    sum_loss = 0
    count=0
    for i in range(0, N, batchsize):
        if i%100==0:
            print i
        x_batch1 = np.ndarray(
            (batchsize, 1, tate, yoko), dtype=np.float32)
        y_batch1 = np.ndarray((batchsize,), dtype=np.float32)
        batch_pool = [None] * batchsize

        for z in range(batchsize):
            path, label = train_list[count]
            batch_pool[z] = read_image(path)
            x_batch1[z]=batch_pool[z]
            y_batch1[z] = label
            count += 1
        #x_batch2 = x_batch1.reshape(batchsize, 1, insize, insize)
        x_batch = xp.asarray(x_batch1)
        y_batch = xp.asarray(y_batch1).reshape(batchsize,1)
        optimizer.zero_grads()
        loss = forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(y_batch)
    count=0
    print "train mean squared error : %f" % (sum_loss / N)
    fp2.write("%d\t%f\n" % (epoch, sum_loss / N))
    fp2.flush()

    sum_accuracy = 0
    for i in range(0, N_test, val_batchsize):
        val_x_batch = np.ndarray(
            (val_batchsize, 1, tate, yoko), dtype=np.float32)
        val_y_batch = np.ndarray((val_batchsize,), dtype=np.float32)
        val_batch_pool = [None] * val_batchsize

        for zz in range(val_batchsize):
            path, label = val_list[count]
            val_batch_pool[zz] = read_image(path)
            val_x_batch[zz]=val_batch_pool[zz]
            val_y_batch[zz] = label
            count+=1
        x_batch = xp.asarray(val_x_batch)
        y_batch = xp.asarray(val_y_batch).reshape(batchsize,1)

        loss , ans = forward(x_batch, y_batch, train=False)
        sum_accuracy += float(loss.data) * len(val_y_batch)
        #pearson = scipy.stats.pearsonr(acc.data,ans.data)
        for sa in range(val_batchsize):
            gosa.write("%f %f" % (ans.data[sa],val_y_batch[sa]))
            gosa.write("\n")
            gosa.flush()


    count=0
    print "test mean squared error: %f" % (sum_accuracy / N_test)
    fp1.write("%d\t%f\n" % (epoch, sum_accuracy / N_test))
    fp1.flush()

end_time = time.clock()
print end_time - start_time

fp1.close()
fp2.close()
gosa.close()

import cPickle
# CPU環境でも学習済みモデルを読み込めるようにCPUに移してからダンプ
model.to_cpu()
cPickle.dump(model, open("model.pkl", "wb"), -1)