# BPNN

## 简介
阅读 Tom M. Mitchell 的《机器学习》后，根据自己的理解写的最基础的人工神经网络反向传播算法。

## 编译运行
如果本地计算机安装了基本环境，在当前目录执行`make`即可，要求编译器支持`C++ 11`，成功编译后执行` ./bpnn.out`。

## 输出内容说明
输出了训练集的迭代序号以及每次迭代后对应的绝对误差值，最后输出对测试集的测试结果（正确的测试结果数/总的测试样例数）。

## 数据集说明
1. number.data：自己构造的 0 - 9 十个数字的点阵图形，训练集和测试集相同，仅作为编写代码过程中的测试使用，没有代表性；
2.  iris.data：[鸢尾花数据集](http://archive.ics.uci.edu/ml/datasets/Iris) 是机器学习中常用的数据集之一，采用其中一半数据进行训练，另一半数据进行测试，在隐藏层只有2层，学习50次以后，测试结果正确率超过90%，可以通过调节参数获得更高的正确率；
3.  mnist：[MNIST](http://yann.lecun.com/exdb/mnist/)  包含数万张手写的 0 - 9 的数字的图片，数据具有真实性，具体内容请查看上述链接，对60000个训练数据学习后，对测试集的正确率超过90%，可以通过调节参数获得更高的正确率，由于数据文件较大，此处并未包含，可到[MNIST](http://yann.lecun.com/exdb/mnist/)  下载数据集，解压后放到此目录下，打开`main.cpp` 中的`test_mnist()` 函数即可进行测试（读取失败可能是大端小端问题）。

## 训练结果的保存和读取
`BPNN::save(std::ostream&)` 可以用来将学习结果保存到文件，将文件用二进制方式打开，执行此函数即可：
```cpp
BPNN bpnn(a, b, c);
// do some thing
ofstream out;
out.open("bpnn.dat", ios::binary);
bpnn.save(out);
out.close();
```
读取文件同理：
```cpp
BPNN bpnn;
ifstream in;
in.open("bpnn.dat", ios::binary);
bpnn.read(in);
in.close();
// do some thing
```
