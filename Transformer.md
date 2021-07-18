## Transformer

​		正如论文的题目所说的，Transformer中抛弃了传统的CNN和RNN，整个网络结构完全是由Attention机制组成。更准确地讲，Transformer由且仅由self-Attenion和Feed Forward Neural Network组成。一个基于Transformer的可训练的神经网络可以通过堆叠Transformer的形式进行搭建，作者的实验是通过搭建编码器和解码器各6层，总共12层的Encoder-Decoder，并在机器翻译中取得了BLEU值得新高。

作者采用Attention机制的原因是考虑到RNN（或者LSTM，GRU等）的计算限制为是顺序的，也就是说RNN相关算法只能从左向右依次计算或者从右向左依次计算，这种机制带来了两个问题：

1. 时间片 ![[公式]](https://www.zhihu.com/equation?tex=t) 的计算依赖 ![[公式]](https://www.zhihu.com/equation?tex=t-1) 时刻的计算结果，这样限制了模型的并行能力；
2. 顺序计算的过程中信息会丢失，尽管LSTM等门机制的结构一定程度上缓解了长期依赖的问题，但是对于特别长期的依赖现象,LSTM依旧无能为力。

Transformer的提出解决了上面两个问题，首先它使用了Attention机制，将序列中的任意两个位置之间的距离是缩小为一个常量；其次它不是类似RNN的顺序结构，因此具有更好的并行性，符合现有的GPU框架。

### 1. 高层Transformer

​		论文中的验证Transformer的实验室基于机器翻译的，下面我们就以机器翻译为例子详细剖析Transformer的结构，在机器翻译中，Transformer可概括为如图1：

![img](https://pic4.zhimg.com/80/v2-1f6085e214b62d8293b2122a52489bff_720w.jpg)

![img](https://pic1.zhimg.com/80/v2-5a252caa82f87920eadea2a2e93dc528_720w.jpg)

如论文中所设置的，编码器由6个编码block组成，同样解码器是6个解码block组成。与所有的生成模型相同的是，编码器的输出会作为解码器的输入，如图3所示：

![img](https://pic3.zhimg.com/80/v2-c14a98dbcb1a7f6f2d18cf9a1f591be6_720w.jpg)

我们继续分析每个encoder的详细结构：在Transformer的encoder中，数据首先会经过一个叫做‘self-attention’的模块得到一个加权之后的特征向量 ![[公式]](https://www.zhihu.com/equation?tex=Z) ，这个 ![[公式]](https://www.zhihu.com/equation?tex=Z) 便是论文公式1中的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BAttention%7D%28Q%2CK%2CV%29) ：

![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BAttention%7D%28Q%2CK%2CV%29%3D%5Ctext%7Bsoftmax%7D%28%5Cfrac%7BQK%5ET%7D%7B%5Csqrt%7Bd_k%7D%7D%29V+%5Ctag1)

第一次看到这个公式你可能会一头雾水，在后面的文章中我们会揭开这个公式背后的实际含义，在这一段暂时将其叫做 ![[公式]](https://www.zhihu.com/equation?tex=Z) 。

得到 ![[公式]](https://www.zhihu.com/equation?tex=Z) 之后，它会被送到encoder的下一个模块，即Feed Forward Neural Network。这个全连接有两层，第一层的激活函数是ReLU，第二层是一个线性激活函数，可以表示为：

![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BFFN%7D%28Z%29+%3D+max%280%2C+ZW_1+%2Bb_1%29W_2+%2B+b_2+%5Ctag2)

Encoder的结构如图4所示：

![img](https://pic3.zhimg.com/80/v2-89e5443635d7e9a74ff0b4b0a6f31802_720w.jpg)

Decoder的结构如图5所示，它和encoder的不同之处在于Decoder多了一个Encoder-Decoder Attention，两个Attention分别用于计算输入和输出的权值：

1. Self-Attention：当前翻译和已经翻译的前文之间的关系；
2. Encoder-Decnoder Attention：当前翻译和编码的特征向量之间的关系。

![img](https://pic1.zhimg.com/80/v2-d5777da2a84e120846c825ff9ca95a68_720w.jpg)

### 2. 输入编码

下面我们将介绍它的输入数据。如图6所示，首先通过Word2Vec等词嵌入方法将输入语料转化成特征向量，论文中使用的词嵌入的维度为 ![[公式]](https://www.zhihu.com/equation?tex=d_%7Bmodel%7D%3D512) 。



![img](https://pic1.zhimg.com/80/v2-408fcd9ca9a65fdbf9d971cfd9227904_720w.jpg)

在最底层的block中， ![[公式]](https://www.zhihu.com/equation?tex=x) 将直接作为Transformer的输入，而在其他层中，输入则是上一个block的输出。为了画图更简单，我们使用更简单的例子来表示接下来的过程，如图7所示：

![img](https://pic2.zhimg.com/80/v2-ff0f90ebee18dd909999bd3bee38fa45_720w.jpg)

### 3. self-Attention

Self-Attention是Transformer最核心的内容，然而作者并没有详细讲解，下面我们来补充一下作者遗漏的地方。回想Bahdanau等人提出的用Attention\[2\]，其核心内容是为输入向量的每个单词学习一个权重，例如在下面的例子中我们判断it代指的内容，

```text
The animal didn't cross the street because it was too tired
```

通过加权之后可以得到类似图8的加权情况，在讲解self-attention的时候我们也会使用图8类似的表示方式

![img](https://pic2.zhimg.com/80/v2-d2129d06290744ebc12b6f220866b2a5_720w.jpg)

在self-attention中，每个单词有3个不同的向量，它们分别是Query向量（ ![[公式]](https://www.zhihu.com/equation?tex=Q) ），Key向量（ ![[公式]](https://www.zhihu.com/equation?tex=K+) ）和Value向量（ ![[公式]](https://www.zhihu.com/equation?tex=V) ），长度均是64。它们是通过3个不同的权值矩阵由嵌入向量 ![[公式]](https://www.zhihu.com/equation?tex=X) 乘以三个不同的权值矩阵 ![[公式]](https://www.zhihu.com/equation?tex=W%5EQ) ， ![[公式]](https://www.zhihu.com/equation?tex=W%5EK) ， ![[公式]](https://www.zhihu.com/equation?tex=W%5EV) 得到，其中三个矩阵的尺寸也是相同的。均是 ![[公式]](https://www.zhihu.com/equation?tex=512%5Ctimes+64) 。

![img](https://pic2.zhimg.com/80/v2-159cd31e629170e0bade136b91c9de61_720w.jpg)

那么Query，Key，Value是什么意思呢？它们在Attention的计算中扮演着什么角色呢？我们先看一下Attention的计算方法，整个过程可以分成7步：

1. 如上文，将输入单词转化成嵌入向量；
2. 根据嵌入向量得到 ![[公式]](https://www.zhihu.com/equation?tex=q) ， ![[公式]](https://www.zhihu.com/equation?tex=k) ， ![[公式]](https://www.zhihu.com/equation?tex=v) 三个向量；
3. 为每个向量计算一个score： ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7Bscore%7D+%3D+q+%5Ccdot+k) ；
4. 为了梯度的稳定，Transformer使用了score归一化，即除以 ![[公式]](https://www.zhihu.com/equation?tex=%5Csqrt%7Bd_k%7D) ；
5. 对score施以softmax激活函数；
6. softmax点乘Value值 ![[公式]](https://www.zhihu.com/equation?tex=v) ，得到加权的每个输入向量的评分 ![[公式]](https://www.zhihu.com/equation?tex=v) ；
7. 相加之后得到最终的输出结果 ![[公式]](https://www.zhihu.com/equation?tex=z) ： ![[公式]](https://www.zhihu.com/equation?tex=z%3D%5Csum+v) 。

上面步骤的可以表示为图10的形式。

![img](https://pic1.zhimg.com/80/v2-79b6b3c14439219777144668a008355c_720w.jpg)

Query，Key，Value的概念取自于信息检索系统，举个简单的搜索的例子来说。当你在某电商平台搜索某件商品（年轻女士冬季穿的红色薄款羽绒服）时，你在搜索引擎上输入的内容便是Query，然后搜索引擎根据Query为你匹配Key（例如商品的种类，颜色，描述等），然后根据Query和Key的相似度得到匹配的内容（Value)。

self-attention中的Q，K，V也是起着类似的作用，在矩阵计算中，**点积**是计算两个矩阵相似度的方法之一，因此式1中使用了 ![[公式]](https://www.zhihu.com/equation?tex=QK%5ET) 进行相似度的计算。接着便是根据相似度进行**输出的匹配**，这里使用了**加权匹配**的方式，而权值就是query与key的相似度。

### 4. Multi-Head Attention

Multi-Head Attention相当于 ![[公式]](https://www.zhihu.com/equation?tex=h) 个不同的self-attention的集成（ensemble），在这里我们以 ![[公式]](https://www.zhihu.com/equation?tex=h%3D8) 举例说明。Multi-Head Attention的输出分成3步：

1. 将数据 ![[公式]](https://www.zhihu.com/equation?tex=X+) 分别输入到图13所示的8个self-attention中，得到8个加权后的特征矩阵 ![[公式]](https://www.zhihu.com/equation?tex=Z_i%2C+i%5Cin%5C%7B1%2C2%2C...%2C8%5C%7D) 。
2. 将8个 ![[公式]](https://www.zhihu.com/equation?tex=Z_i) 按列拼成一个大的特征矩阵；
3. 特征矩阵经过一层全连接后得到输出 ![[公式]](https://www.zhihu.com/equation?tex=Z) 。

整个过程如图14所示：

![img](https://pic3.zhimg.com/80/v2-c2a91ac08b34e73c7f4b415ce823840e_720w.jpg)

同self-attention一样，multi-head attention也加入了short-cut机制。

### 5. Encoder-Decoder Attention

在解码器中，Transformer block比编码器中多了个encoder-cecoder attention。在encoder-decoder attention中， ![[公式]](https://www.zhihu.com/equation?tex=Q) 来自于解码器的上一个输出， ![[公式]](https://www.zhihu.com/equation?tex=K) 和 ![[公式]](https://www.zhihu.com/equation?tex=V) 则来自于与编码器的输出。其计算方式完全和图10的过程相同。

由于在机器翻译中，解码过程是一个顺序操作的过程，也就是当解码第 ![[公式]](https://www.zhihu.com/equation?tex=k) 个特征向量时，我们只能看到第 ![[公式]](https://www.zhihu.com/equation?tex=k-1) 及其之前的解码结果，论文中把这种情况下的multi-head attention叫做masked multi-head attention。

### 6. 损失层

解码器解码之后，解码的特征向量经过一层激活函数为softmax的全连接层之后得到反映每个单词概率的输出向量。此时我们便可以通过CTC等损失函数训练模型了。

而一个完整可训练的网络结构便是encoder和decoder的堆叠（各 ![[公式]](https://www.zhihu.com/equation?tex=N+) 个， ![[公式]](https://www.zhihu.com/equation?tex=N%3D6) ），我们可以得到图15中的完整的Transformer的结构（即论文中的图1)：

![img](https://pic1.zhimg.com/80/v2-9fb280eb2a69baf5ceafcfa3581aa580_720w.jpg)

### 7. 位置编码

截止目前为止，我们介绍的Transformer模型并没有捕捉顺序序列的能力，也就是说无论句子的结构怎么打乱，Transformer都会得到类似的结果。换句话说，Transformer只是一个功能更强大的词袋模型而已。

为了解决这个问题，论文中在编码词向量时引入了位置编码（Position Embedding）的特征。具体地说，位置编码会在词向量中加入了单词的位置信息，这样Transformer就能区分不同位置的单词了。

那么怎么编码这个位置信息呢？常见的模式有：a. 根据数据学习；b. 自己设计编码规则。在这里作者采用了第二种方式。那么这个位置编码该是什么样子呢？通常位置编码是一个长度为 ![[公式]](https://www.zhihu.com/equation?tex=d_%7Bmodel%7D) 的特征向量，这样便于和词向量进行单位加的操作，如图16。

![img](https://pic3.zhimg.com/80/v2-3e1304e3f8da6cf23cc43ab3927d700e_720w.jpg)图16：Position Embedding

假设`embedding`的维数为4 44，那么实际的`positional encodings`就会类似下图
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201117192230977.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDQ0NDI3MA==,size_16,color_FFFFFF,t_70#pic_center)

论文给出的编码公式如下：

![[公式]](https://www.zhihu.com/equation?tex=PE%28pos%2C+2i%29+%3D+sin%28%5Cfrac%7Bpos%7D%7B10000%5E%7B%5Cfrac%7B2i%7D%7Bd_%7Bmodel%7D%7D%7D%7D%29+%5Ctag3)

![[公式]](https://www.zhihu.com/equation?tex=PE%28pos%2C+2i%2B1%29+%3D+cos%28%5Cfrac%7Bpos%7D%7B10000%5E%7B%5Cfrac%7B2i%7D%7Bd_%7Bmodel%7D%7D%7D%7D%29+%5Ctag4)

在上式中， ![[公式]](https://www.zhihu.com/equation?tex=pos) 表示单词的位置， ![[公式]](https://www.zhihu.com/equation?tex=i) 表示单词的维度。关于位置编码的实现可在Google开源的算法中`get_timing_signal_1d()`函数找到对应的代码。

作者这么设计的原因是考虑到在NLP任务中，除了单词的绝对位置，单词的相对位置也非常重要。根据公式 ![[公式]](https://www.zhihu.com/equation?tex=sin%28%5Calpha%2B%5Cbeta%29+%3D+sin+%5Calpha+cos+%5Cbeta+%2B+cos+%5Calpha+sin%5Cbeta+) 以及![[公式]](https://www.zhihu.com/equation?tex=cos%28%5Calpha+%2B+%5Cbeta%29+%3D+cos+%5Calpha+cos+%5Cbeta+-+sin+%5Calpha+sin%5Cbeta) ，这表明位置 ![[公式]](https://www.zhihu.com/equation?tex=k%2Bp) 的位置向量可以表示为位置 ![[公式]](https://www.zhihu.com/equation?tex=k) 的特征向量的线性变化，这为模型捕捉单词之间的相对位置关系提供了非常大的便利。

### 8 残差

Encoder中值得提出注意的一个细节是，在每个子层中(Self-Attention, Feed Forward），都有残差连接，并且紧跟着layer-normalization。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201117202351822.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDQ0NDI3MA==,size_16,color_FFFFFF,t_70#pic_center)

如果我们可视化向量和layer-norm操作，将如下所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/202011172038137.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDQ0NDI3MA==,size_16,color_FFFFFF,t_70#pic_center)


在Decoder中也是如此，假设两层Encoder+两层Decoder组成Transformer，其结构如下：


![在这里插入图片描述](https://img-blog.csdnimg.cn/20201117203920761.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDQ0NDI3MA==,size_16,color_FFFFFF,t_70#pic_center)

### 9 Decoder

我们已经明白Encoder的大部分概念，现在开始了解Decoder的工作流程。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201117204139702.gif#pic_center)

Encoder接受input sequence转换后的vector，然后最后的Encoder将其转换为K、V这两个被每个Decoder的"encoder-decoder atttention"层来使用，帮助Decoder集中于输入序列的合适位置。

下面的步骤一直重复直到一个特殊符号出现表示解码器完成了翻译输出。
每一步的输出被下一个解码器作为输入。
正如编码器的输入所做的处理，对解码器的输入增加位置向量。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201119141410996.gif#pic_center)

encoder与decoder在Self-Attention层中会有些许不同。Decoder比Encoder多了一个encoder-decoder attention。

在encoder-decoder attention中，Q QQ 来自于与Decoder的上一个输出，K KK和 V VV则来自于与Encoder的输出。其计算方式完全和图10的过程相同。

由于在机器翻译中，解码过程是一个顺序操作的过程，也就是当解码第k kk 个特征向量时，我们只能看到第k − 1 k-1k−1及其之前的解码结果，论文中把这种情况下的multi-head attention叫做masked multi-head attention。

### 10 Linear和Softmax layer

Linear Layer是一个简单的全连接层，将Decoder的最后输出映射到一个logits向量上。

假设模型有1w个单词(输出的词表)从训练集中学习获得。那么logits向量就有1w维，每个值表示某个词的可能倾向值。

softmax则将logits的分数转为概率值(范围[0,1],和为1)，最高值对应的维上是这一步的输出单词。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201119142625297.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDQ0NDI3MA==,size_16,color_FFFFFF,t_70#pic_center)

### 11 训练过程状况

在训练时，模型将经历上述步骤。
假设我们的输出词只有六个(“a”, “am”, “i”, “thanks”, “student”, and “” (short for ‘end of sentence’))。这个输出词典在训练之前的预处理阶段创建。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201119143305527.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDQ0NDI3MA==,size_16,color_FFFFFF,t_70#pic_center)

### 12 损失函数

以一个简单的示例进行讲解，将merci(法语)翻译为thanks。
这意味着输出的概率分布指向单词"thanks",但是由于模型未训练是随机初始化，输出可能并不准确。
根据输出值与实际值进行比较，然后根据后向传播算法进行调参，使得结果更加准确。

如何对比两个概率分布呢？简单采用 cross-entropy或者Kullback-Leibler divergence中的一种。

此次举例非常简单，真实情况下应当以一个句子作为输入。比如，输入是“je suis étudiant”，期望输出是“i am a student”。在这个例子下，我们期望模型输出连续的概率分布满足如下条件：

每个概率分布都与词表同维度。
第一个概率分布对“i”具有最高的预测概率值。
第二个概率分布对“am”具有最高的预测概率值。
一直到第五个输出指向""标记。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201119144826112.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDQ0NDI3MA==,size_16,color_FFFFFF,t_70#pic_center)

经过训练之后希望我们模型的输出如下图所示

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201119144826115.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDQ0NDI3MA==,size_16,color_FFFFFF,t_70#pic_center)

