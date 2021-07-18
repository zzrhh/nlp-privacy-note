## Privacy Risks of General-Purpose Language Models

### 1 介绍

随着深度学习技术在NLP领域的发展，通用语言模型开始广泛应用于NLP的下游任务例如文本分类，问题回答。开始应用于谷歌的搜索引擎。不像传统的静态模型和千层神经模型，它是基于transformer家族的与语言模型（例如谷歌的bert和openAI的GPT-2）。根据官方的教程，这些预训练模型能够用来作为文本特征提取在词嵌入的时候，这些特征能够被再次运用在下游NLP任务当中。

然而，通用语言模型可能会存在捕获词向量当中的敏感信息，这些可能导致敌手的攻击。例如，一个医院与第三方公司合作开发一个病人引导系统。该系统能够自动地分配病人去合适的病床根据病人的症状描述。该三方仅需要医院提供病人的症状描述的向量作为关键信息。然而，由于缺乏对通用语言模型的隐私特性的理解。医院认为分享文本信息会比分享向量信息更危险一些。确实，很难去恢复一些有用的信息从词嵌入当中。然而，我们设计了一个轻量级的有效的攻击线，它证明当拿到一个未被保护的词向量，即使是一个敌手没有掌握多余的知识都能够推测出该领域相关的敏感信息从未知的明文信息当中。

在上面的例子当中，我们的发现表明，一个诚实而且好奇的服务提供者，敌手能够轻易的推测出身份，性别，出生日期等信息

**我们的工作。**我们想要回答这么一个问题，敌手是有可能从未知的明文中推测用户的隐私信息，当敌手仅仅只知道他们提交的词向量的时候。尽管在计算机视觉领域的工作展现出了重构初始照片从预训练的嵌入当中，但是之前并没有NLP方面的任何文章。从我的观点来看，单词的离散性和词汇表的不可见性是两个主要的技术性的挑战来防止重构攻击在词向量上边。**一方面来说**，词的离散型让搜索可能的句子的效率极低，主要是因为学习目标已经不再像视觉案例那样可微，因此基于梯度的方法几乎不能起作用了。**另一方面来说**，因为语言模型是黑盒访问，攻击者是没有单词表的，所以他就很难将索引对应到明文上面。而且即便有单词表，也会因为过大而带来高的计算代价，或者是过小而不能包含一些敏感词汇在明文当中的。

因此，为了解决这些上面的挑战，因此，我们提出直接去重构敏感的信息从词向量当中经由推理。我们发现隐私相关的信息在文本中经常出现在小的片段或者或者出现在关键字当中。因此我们构建了两种不同的攻击方式。**模式重构攻击**，**关键字推理攻击**。模式重构即敌手想要恢复一个详细的包含敏感信息的序列。关键字推理攻击当中，敌手想要去探索是否这些未知的明文包括敏感关键词。只是集中于小的片段，敌手仅需要去推断有限数量的可能性，能够缓解由token离散性造成的优化困难。与此同时，敌手不需要去知道整个词汇表因为他仅关心他所感兴趣的单词。而且，我们的攻击只需要访问云服务器上的语言模型，因此我们只需要一台PC。额外的研究，我们进一步讨论一些结构相关和数据相关的因素，因为这些因素可能影响语言模型的隐私挑战等级。而且，我们也提出了四种初步的对抗措施，**量化**，**差异隐私**，**对抗训练**和**子空间投影**。

总之，四个方面的贡献：

* 发现了通用语言模型的潜在风险
* 设计了通用的攻击模型去发掘用户的隐私
* 提出了第一个系统性的风险评估在通用语言模型当中
* 提出了初步的四种可能的对抗方式

### 2 相关工作



### 4 通用攻击线

#### A. 攻击定义

#### B. 威胁模型

* 假设1，敌手能够访问大量的词向量，而且其中包含敌手感兴趣的敏感信息。
* 假设2，简单起见，敌手知道预训练的词向量是来自哪种模型。
* 假设3，敌手能够访问预训练模型，能够将一句话作为输入然后得到相关的词向量。

#### C. 攻击线

一共划分为4个阶段

##### 1. 第一阶段：准备语料库

因为攻击模型推理敏感信息是在未知的明文里面，一个合适的外部语料库就十分重要。基于明文的不同知识等级，我们认为敌手可以创造外部语料库通过**生成算法**，或者**公共语料库**。After the external corpus is prepared, we apply the algorithm P on each xi ∈ Dext to obtain the label P(xi), which concludes the first stage.

##### 2. 第二阶段：询问语言模型

根据已知模型的信息，敌手能够访问语言模型而且还能保留一些预测信息通过使用在线语言模型服务。At the end of this stage, we have the **training set Dtrain** of the form {(zi ,P(xi))} N i=1, where zi is the embedding corresponding to the sentence xi .

##### 3. 第三阶段：训练攻击模型

因为训练集Dtrain从阶段二获取到了。敌手现在能够训练一个攻击模型g用来推理。 In general, the model is designed as a classifier, which takes the embedding zi as its input and outputs a probabilistic vector g(zi) over all possible values of the sensitive information.

##### 4. 第四阶段：推理

在训练阶段完成后，given the target embedding z, the adversary infers the sensitive information based on the following equation s := A(z) = arg maxi∈{1,2,...,K}[g(z)]i , where [g(z)]i is the value of g(z) at its i-th dimension and K denotes the total number of possible values for s.

### 5 模式重构攻击

这一部分，我们考虑的环境为敌手知道未知明文的生成规则。我们将这一部分作为起点去理解有多少敏感信息被通用语言模型编码在向量里面。

#### A. 攻击定义

模式重构攻击意图恢复一个含有固定格式的明文片段。目标片段可能包含敏感信息例如，出生日期，性别等。

* 假设1，明文的形式是固定的，敌手知道明文的生成规则。

##### 案列1：身份证ID

在中国，身份证ID的规则为：

To demonstrate, we consider the case of citizen ID in China, which consists of the 18 characters (from the vocabulary {0, . . . , 9}), i.e. 6 for the residence code (3000 possibilities), 8 for the birth date (more than 100 × 12 × 30 possibilities) and 4 for extra code (104 possibilities). Consider the adversary wants to recover the exact birth date of the victim via the leaked embedding of his/her citizen ID, we define the mapping P as ：

Pcitizen : |residence|birthday|extra| → |birthday|

##### 案例2：基因组序列

From the disclosed nucleotide, the adversary can further know the gender, race or other privacy-critical information of the victim. For demonstration, we define the mapping P as Pgenome,i : (w1, w2, . . . , wn) → wi . In other words, the nucleotide at position i is assumed to be sensitive.

#### B. 方法

##### 1. 生成外部语料库

知道目标明文的生成规则，敌手可以准备外部语料库经过生成算法。

##### 2. 攻击模型的结构

攻击模型g可以被设计为全连接的神经网络输入维度d然后输出维度|V (wb . . . we)|，代表敏感片段的可能的值的数量。但这个值可能非常大。例如，身份证中的出生日期可能的值有大概40000.为了解决这个问题，我们采取分而治之的想法去重组模型变成小的子攻击，根据敌手对格式的知识。Again on Citizen, we can decompose the attack model gbirth into three sub-attacks, namely year attack gyear, month attack gmonth and day attack gday. Each sub-attack model can be independently implemented with fully-connected neural networks of much smaller size and the total parameter number is largely truncated from O(|V (wb)| × . . . × |V (we)|) to O(|V (wb)|+. . .+|V (we)|). 因此子攻击还能并行。

#### C. 实验设置

##### 1. 系统基准

* 身份证ID：随机生成1000个ID根据生成算法作为明文，然后我们用ID去询问通用模型算法去得到词向量。
* 基因：我们实施了八种基因组分类系统 基于公共基因组数据集的剪接位点预测，称为 HS3D 。所有的基因序列长度为20。我们假设测试集当中的基因序列的词向量泄露了，其中包含1000个样本，这样本有可能有或者没有剪切点位，

##### 2. 攻击实施

* 身份证ID：we implement the year, month and date sub-attacks as three-layer MLPs which respectively contain 400, 25, 200 hidden units with sigmoid activation. The training batch size is set as 128 for each sub-attack.
* 基因：For inference, the attacker inputs the victim’s embedding and the target position and the model outputs the predicted nucleotide type.

#### D. 结果和分析

##### 1. 有用和有效

##### 2. 不同语言模型之间的比较

Facebook’s RoBERTa shows stronger robustness than other language models in both cases. By investigating its design, we find RoBERTa is a re-implementation of Google’s Bert but uses a different byte-level tokenization scheme (i.e., tokenize sentences in the unit of bytes instead of characters or words) [44]. As RoBERTa shows about 50% lower privacy risks than Bert when facing the same attacks, we conjecture the reason is that the byte-level tokenization scheme may make the embeddings less explicit in character-level sensitive information and thus more robust against our attacks. 语言模型的安全性跟训练数据集的大小无关。

##### 3. 其他有趣的发现

模型更容易捕获处在末尾的信息相比于中间的信息，有可能是末尾的信息是沿着最长的路径在模型架构中。

### 6 关键字推理攻击

 #### A. 攻击的定义

we formulate the mapping Pkeyword,k for defining the sensitive information related with keyword k from a sentence x as Pkeyword,k : x → (∃w ∈ x, w == k), where the right side denotes a predicate that yields True if a word w in the sentence x is the target keyword k and otherwise False

##### 案例1：航线审阅

航空公司会在某些时候调查他们的顾客来提高服务。得益于NLP技术，大量的航空调查报告以文本格式的能够自动被处理理解顾客的观点。得益于预训练模型来做特征提取。然而，一旦能够访问这些词向量。敌手能够推测出大量的位置相关的敏感信息。我们能够展现出，敌手能够精确的评估是否一个城市的名字包含在该调查报告里面。

##### 案例2：医药处方

The system is expected to take the patient’s description of the illness to predict which department he/she ought to consult. To form a benchmark system, we concatenate the pretrained language models with an additional linear layer for guiding the patients to 10 different departments。

然而，当敌手仅能访问到词向量的时候，他能确确实实地推测出更多敏感和个人的信息关于病人的。

#### B. 方法

According to the different levels of the adversary’s knowledge on the plain text, the methodology part is divided into white-box and black-box settings, which respectively require the following two assumptions

* 假设1：白盒。敌手能过够得到渐层的语料库
* 假设2：黑盒。敌手没有语料库的任何信息。

##### 1. 白盒攻击



##### 2. 黑盒攻击

To implement the keyword inference attack with no prior knowledge, we propose to first crawl sentences from the Internet to form the external corpus and then transfer the adversarial knowledge of an attack model on the external corpus to the target corpus dynamically. Details are as follows.

##### 2.1 创建外部语料库：

With the aid of the Internet, it is relatively convenient for the adversary to obtain an external corpus from other public corpora. Next, the adversary can generate positive and negative samples via the same word substitution trick we mentioned in the previous part.

##### 2.2 转换敌手的知识：

The model consists of four sub-modules. First, the module E is an encoder which takes the sentence embedding as input and is expected to output a domain-invariant representation zˆ. The hidden representation is followed by two binary classifiers, i.e. Ckeyword and Cdomain. The keyword classifier Ckeyword takes zˆ as input and predicts whether the sentence x contains the keyword k, while the domain classifier Cdomain outputs whether the embedding comes from X0 or X1。

In our implementations, an additional module called gradient reversal layer [9] is fundamental to learn domain-invariant representations and therefore help transfer the adversarial knowledge

For inference, we take Ckeyword ◦E as the attack model g.

#### C. 实验设置

##### 1. 系统基准

* 航线：我们收集航线审阅数据仅保留包含10个具体城市名字的数据集去构成我们的系统数据
* 医院：

##### 2. 标准

##### 3. 攻击实施

* 白盒设置：We study two implementations of the attack model in the white-box setting, namely the linear SVM and the 3-layer MLP with 80 hidden units with sigmoid activations. The batch size is set as 64.
* 黑盒设置：We study three implementations in the black-box setting, namely linear SVM, 3-layer MLP and the DANN-based attacks. The DANN model has 25 dimensional domain-invariant representations and the coefficient λ in Algorithm 1 is set as 1. We use the Adam optimizer with learning rate 0.001 for both the MLP and the DANN-based attacks. The batch size is set as 64.

#### D. 结果和分析

### 7 可能的防御手段

#### 1 rounding

For the first defense, we apply floating-point rounding on each coordinate of the sentence embeddings for obfuscation. Formally, we write the rounding defense as zˆ =rounding(z, r), where the non-negative integer r denotes the number of decimals preserved after rounding.

#### 2 Laplace Mechanism

#### 3 Privacy Preserving Mapping

基于敌手的训练。We borrow the notion of a Privacy Preserving Mapping (PPM) from [57] to denote a mapping Dθ : R d → R d parameterized by θ, which is trained to minimize the effectiveness of an imagined adversary Aψ.

####  4 Subspace Projection

The last defense is especially designed as a countermeasure to keyword inference attacks, inspired by [14] on debiasing word embeddings from gender bias. 即将关键字从通用空间编码到隐私子空间里面。

#### 评估

We evaluate the first three defenses against the pattern reconstruction attack on Genome and all four defenses against the DANN-based keyword inference attack on Medical with a wide range of settings.

According to our preliminary results above, how to balance the elimination of token-level sensitive information from the embeddings and the preservation of the essential information for normal tasks is still an open problem that awaits more in-depth researches. In consideration of the practical threats imposed by our attacks on the applications of general-purpose language models, we highly suggest the exploration of effective defense mechanisms as a future work.

### 8 讨论

#### 危险模型



### 9 结论

In this paper, we design two novel attack classes, i.e., pattern reconstruction attacks and keyword inference attacks, to demonstrate the possibility of stealing sensitive information from the sentence embeddings. We conduct extensive evaluations on eight industry-level language models to empirically validate the existence of these privacy threats. To shed light on future mitigation studies, we also provide a preliminary study on four defense approaches by obfuscating the sentence embeddings to attenuate the sensitive information. To the best of our knowledge, our work presents the first systematic study on the privacy risks of general-purpose language models, along with the possible countermeasures. For the future leverage of the cutting-edge NLP techniques in real world settings, we hope our study can arouse more research interests and efforts on the security and privacy of general-purpose language models.







