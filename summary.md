## summary

### Privacy Risks of General-Purpose Language Models

他们想要回答这么一个问题，敌手是有可能从未知的明文中推测用户的隐私信息，当敌手仅仅只知道他们提交的词向量的时候。尽管在计算机视觉领域的工作展现出了重构初始照片从预训练的嵌入当中，但是之前并没有NLP方面的任何文章。从我的观点来看，单词的离散性和词汇表的不可见性是两个主要的技术性的挑战来防止重构攻击在词向量上边。**一方面来说**，词的离散型让搜索可能的句子的效率极低，主要是因为学习目标已经不再像视觉案例那样可微，因此基于梯度的方法几乎不能起作用了。**另一方面来说**，因为语言模型是黑盒访问，攻击者是没有单词表的，所以他就很难将索引对应到明文上面。而且即便有单词表，也会因为过大而带来高的计算代价，或者是过小而不能包含一些敏感词汇在明文当中的。

因此，为了解决这些上面的挑战，因此，我们提出直接去重构敏感的信息从词向量当中经由推理。我们发现隐私相关的信息在文本中经常出现在小的片段或者或者出现在关键字当中。因此我们构建了两种不同的攻击方式。**模式重构攻击**，**关键字推理攻击**。模式重构即敌手想要恢复一个详细的包含敏感信息的序列。关键字推理攻击当中，敌手想要去探索是否这些未知的明文包括敏感关键词。只是集中于小的片段，敌手仅需要去推断有限数量的可能性，能够缓解由token离散性造成的优化困难。与此同时，敌手不需要去知道整个词汇表因为他仅关心他所感兴趣的单词。而且，我们的攻击只需要访问云服务器上的语言模型，因此我们只需要一台PC。额外的研究，我们进一步讨论一些结构相关和数据相关的因素，因为这些因素可能影响语言模型的隐私挑战等级。而且，我们也提出了四种初步的对抗措施，**量化**，**差分隐私**，**对抗训练**和**子空间投影**。

总之，四个方面的贡献：

* 发现了通用语言模型的潜在风险
* 设计了通用的攻击模型去发掘用户的隐私
* 提出了第一个系统性的风险评估在通用语言模型当中
* 提出了初步的四种可能的对抗方式

### 通用的攻击线

#### A. 攻击定义

we formulate the attack model as A : z → s , where z is the embedding of a target sentence x and s denotes certain type of sensitive information that can be obtained from the plaintext with a publicly-known algorithm P : x → s.

For example, in the above head case, P maps any sentence x to {0, 1}: if the sentence x has word head, then P(x) = 1; otherwise P(x) = 0. This notion will be used in formulating our attack pipeline

#### B. 威胁模型

* 假设1，敌手能够访问大量的词向量，而且其中包含敌手感兴趣的敏感信息。
* 假设2，简单起见，敌手知道预训练的词向量是来自哪种模型。Later in Section VIII, we show this assumption can be easily removed with a proposed learning-based fingerprinting algorithm

* 假设3，敌手能够访问预训练模型，能够将一句话作为输入然后得到相关的词向量。

#### C. 攻击线

一共划分为4个阶段

Our general attack pipeline is divided into four stages. 

**At the first stage**, the adversary prepares an external corpus。Dext := {xi} N i=1 and uses the algorithm P to extract the {P(xi)} N i=1 as labels。the extracted labels usually contain no truly sensitive information。

**At the second stage**, the adversary queries the pretrained language model with each sentence xi ∈ Dext and receives their embeddings {zi} N i=1.

**At the third stage**, the adversary combines the embeddings with the extracted labels to train an attack model A.

**At the final stage**, the adversary uses the well-trained attack model to infer sensitive information s from the target embedding z.

##### 1. 第一阶段：准备语料库

因为攻击模型推理敏感信息是在未知的明文里面，一个合适的外部语料库就十分重要。基于明文的不同知识等级，我们认为敌手可以创造外部语料库通过**生成算法**，或者**公共语料库**。After the external corpus is prepared, we apply the algorithm P on each xi ∈ Dext to obtain the label P(xi), which concludes the first stage.

##### 2. 第二阶段：询问语言模型

第二阶段就是将Dext里面的句子转化成对应的词向量。根据已知模型的信息，敌手能够访问语言模型而且还能保留一些预测信息通过使用在线语言模型服务。At the end of this stage, we have the **training set Dtrain** of the form **{(zi ,P(xi))}** N i=1, where zi is the embedding corresponding to the sentence xi .

##### 3. 第三阶段：训练攻击模型

因为训练集Dtrain从阶段二获取到了。敌手现在能够训练一个攻击模型g用来推理。 In general, the model is designed as a **classifier**, which takes the embedding zi as its input and outputs a probabilistic vector g(zi) over all **possible values of the sensitive information**.

##### 4. 第四阶段：推理

在训练阶段完成后，given the target embedding z, the adversary infers the sensitive information based on the following equation s := A(z) = arg maxi∈{1,2,...,K}[g(z)]i , where [g(z)]i is the value of g(z) at its i-th dimension and K denotes the total number of possible values for s. the adversary considers the value with the highest probability as the most possible value of the sensitive information in the unknown sentence x.

### 模式重构攻击

这一部分，我们考虑的环境为敌手知道未知明文的生成规则。我们将这一部分作为起点去理解有多少敏感信息被通用语言模型编码在向量里面。

#### A. 攻击定义

模式重构攻击意图恢复一个含有固定格式的明文片段。目标片段可能包含敏感信息例如，出生日期，性别等。

* 假设1，明文的形式是固定的，敌手知道明文的生成规则。

##### 案列1：身份证ID

身份证ID的规则为：

To demonstrate, we consider the case of citizen ID in China, which consists of the 18 characters (from the vocabulary {0, . . . , 9}), i.e. 6 for the residence code (3000 possibilities), 8 for the birth date (more than 100 × 12 × 30 possibilities) and 4 for extra code (104 possibilities). Consider the adversary wants to recover the exact birth date of the victim via the leaked embedding of his/her citizen ID, we define the mapping P as ：

Pcitizen : |residence|birthday|extra| → |birthday|

##### 案例2：基因组序列

From the disclosed nucleotide, the adversary can further know the gender, race or other privacy-critical information of the victim. For demonstration, we define the mapping P as Pgenome,i : (w1, w2, . . . , wn) → wi . In other words, the nucleotide at position i is assumed to be sensitive.

#### B. 方法

we denote the set of all possible values for sequence s as V (s).

##### 1. 生成外部语料库

知道目标明文的生成规则，敌手可以准备外部语料库经过生成算法。基本的生成算法可以生成大量的训练样本通过**随机地从可能的值当中取样V (x)**.

##### 2. 攻击模型的结构

攻击模型g可以被设计为全连接的神经网络输入维度d然后输出范围|V (wb . . . we)|，代表敏感片段的可能的值的数量。但这个值可能非常大。例如，身份证中的出生日期可能的值有大概40000.为了解决这个问题，我们采取分而治之的想法去重组模型变成小的子攻击，根据敌手对格式的知识。Again on Citizen, we can decompose the attack model gbirth into three sub-attacks, namely year attack gyear, month attack gmonth and day attack gday. Each sub-attack model can be independently implemented with fully-connected neural networks of much smaller size and the total parameter number is largely truncated from O(|V (wb)| × . . . × |V (we)|) to O(|V (wb)|+. . .+|V (we)|). 而且子攻击还能并行。

#### C. 实验设置

##### 1. 系统基准

* 身份证ID：随机生成1000个ID根据生成算法作为基本事实的明文，然后我们用ID去询问通用模型算法去得到词向量作为攻击对象。
* 基因：我们实施了八种基因组分类系统 基于公共基因组数据集的剪接位点预测，称为 HS3D 。所有的基因序列长度为20。我们假设测试集当中的基因序列的词向量泄露了，其中包含1000个样本，这样本有可能有或者没有剪切点位，

##### 2. 攻击实施

* 身份证ID：we implement the year, month and date sub-attacks as three-layer MLPs which respectively contain 400, 25, 200 hidden units with sigmoid activation. The training batch size is set as 128 for each sub-attack.
* 基因：For inference, the attacker inputs the victim’s embedding and the target position and the model outputs the predicted nucleotide type.

#### D. 结果和分析

##### 1. 有用和有效

##### 2. 不同语言模型之间的比较

Facebook’s RoBERTa shows stronger robustness than other language models in both cases. By investigating its design, we find RoBERTa is a re-implementation of Google’s Bert but uses a different byte-level tokenization scheme (i.e., tokenize sentences in the unit of bytes instead of characters or words) [44]. As RoBERTa shows about 50% lower privacy risks than Bert when facing the same attacks, we conjecture the reason is that the **byte-level tokenization scheme may make the embeddings less explicit** in character-level sensitive information and thus more robust against our attacks. 语言模型的安全性跟训练数据集的大小无关。

##### 3. 其他有趣的发现

模型更容易捕获处在末尾的信息相比于中间的信息，有可能是末尾的信息是沿着最长的路径在模型架构中。也就是说处在边界的信息更容易捕获。

### 关键字推理攻击

 #### A. 攻击的定义

也就是是否关键字K包含在未知的句子x当中。The keyword k can be highly sensitive, which contains indicators for the adversary to further determine e.g., location, residence or illness history of the victim。

we formulate the mapping Pkeyword,k for defining the sensitive information related with keyword k from a sentence x as Pkeyword,k : x → (∃w ∈ x, w == k), where the right side denotes a predicate that yields True if a word w in the sentence x is the target keyword k and otherwise False

不同于模式重构攻击，关键字推理攻击探索关键字是否出现，而不是重构整个序列。

##### 案例1：航线审阅

航空公司会在某些时候调查他们的顾客来提高服务。得益于NLP技术，大量的航空调查报告以文本格式的能够自动被处理理解顾客的观点。得益于预训练模型来做特征提取。然而，一旦能够访问这些词向量。敌手能够推测出大量的位置相关的敏感信息。我们能够展现出，敌手能够精确的评估是否一个城市的名字包含在该调查报告里面。

利用预先训练好的语言模型进行特征提取，可以进一步提高许多现有资源的利用率，意见挖掘系统。

##### 案例2：医药处方

The system is expected to take the patient’s description of the illness to predict which department he/she ought to consult. To form a benchmark system, we concatenate the pretrained language models with an additional linear layer for guiding the patients to 10 different departments。

然而，当敌手仅能访问到词向量的时候，他能确确实实地推测出更多敏感和个人的信息关于病人的。

我们想是为了证明一个对手想要精确地确定病毒的确切发病部位通过推断被害人身体相关疾病的发生概率。

#### B. 方法

According to the different levels of the adversary’s knowledge on the plain text, the methodology part is divided into white-box and black-box settings, which respectively require the following two assumptions

* 假设1：白盒。敌手能过够得到浅层的语料库
* 假设2：黑盒。敌手没有语料库的任何信息。

##### 1. 白盒攻击

Basically, as the adversary has a shadow corpus Dshadow := {(x 0 i )} N i=1 which is sampled from the same distribution as the unknown plain text, he/she can directly use Dshadow as the external corpus Dext and extract the binary label y 0 i = Pkeyword,k(x 0 i ).

接下来，敌手就能使用词向量和标签去训练一个攻击模型。然而，会有一些问题。

**First**, the label set {y 0 i } N i=1 can be highly imbalanced. In other words,有关键字k的句子会非常少相对于这些没有关键字的样本。根据之前的研究，标签的不平衡会导致攻击模型易于过拟合。为了缓解，we propose to randomly replace certain word in the negative samples with the keyword, and we replace the keyword in the positive samples with other random word in the vocabulary (referred to as the word substitution trick). After this operation, the original shadow corpus will be twice enlarged and the samples are balanced in both classes.

**Next**, the shadow corpus after word substitution can still be limited in **size**, i.e., N is small. In this case, we suggest the adversary should implement their attack model with a Support Vector Machine (SVM), which is especially effective for small sample learning [71]. When M is larger than certain threshold (empirically over 103 samples), the adversary can switch to a fully-connected neural network as the attack model, which brings higher attack accuracy.

##### 2. 黑盒攻击

To implement the keyword inference attack with no prior knowledge, we propose to first **crawl** sentences from the Internet to form the **external corpus** and then transfer the adversarial knowledge of an attack model on the external corpus to the target corpus dynamically. Details are as follows.

##### 2.1 创建外部语料库：

With the aid of the Internet, it is relatively convenient for the adversary to obtain an external corpus from other public corpora. Next, the adversary can generate positive and negative samples via the same word substitution trick we mentioned in the previous part.

##### 2.2 转换敌手的知识：

我们发现，当使用现成的MLP或者线性SVM在这些外部语料库训练后精确度不高。我们发现是领域错位导致的这种现象。

为了验证，第一次，我们训练了一个3层的MLP分类器使用外部语料库，该语料库来自餐馆的点评。next，我们绘图发现，As a result, even though the attack model on the public domain (i.e., on restaurant reviews) achieves a near 100% accuracy, its performance is no better than random guess when applied on the private domain (i.e., on medical descriptions).

The model consists of four sub-modules. First, the module E is an encoder which takes the sentence embedding as input and is expected to output a domain-invariant representation zˆ. The hidden representation is followed by two binary classifiers, i.e. Ckeyword and Cdomain. The keyword classifier Ckeyword takes zˆ as input and predicts whether the sentence x contains the keyword k, while the domain classifier Cdomain outputs whether the embedding comes from X0 or X1。

In our implementations, an additional module called gradient reversal layer [9] is fundamental to learn domain-invariant representations and therefore help transfer the adversarial knowledge。梯度反转层通过放大关键词相关特征和去除领域相关信息来正则化隐藏表示zˆ。

For inference, we take Ckeyword ◦E as the attack model g.

#### C. 实验设置

##### 1. 系统基准

* 航线：我们收集航线审阅数据仅，保留包含10个具体城市名字的数据集去构成我们的系统数据
* 医院：我们假定敌手的关键字集包含10个身体相关的词

##### 2. 标准

##### 3. 攻击实施

* 白盒设置：We study two implementations of the attack model in the white-box setting, namely the **linear SVM** and the **3-layer MLP** with 80 hidden units with sigmoid activations. The batch size is set as 64.
* 黑盒设置：We study three implementations in the black-box setting, namely **linear SVM**, **3-layer MLP** and the **DANN-based attacks**. The DANN model has 25 dimensional domain-invariant representations and the coefficient λ in Algorithm 1 is set as 1. We use the Adam optimizer with learning rate 0.001 for both the MLP and the DANN-based attacks. The batch size is set as 64.

#### D. 结果和分析

白盒攻击普遍比黑盒攻击要有效。

### Extracting Training Data from Large Language Models

文章证明，敌手能够执行训练数据提取攻击去恢复个人隐私训练样本，通过询问语言模型。而且大型语言模型比小型模型要更加脆弱。

隐私的泄露以前的观点主要和过度拟合有关（即一个模型的训练错误远小于测试的错误）。因为过拟合会导致模型记住训练集的样本。

然而，过拟合和记忆性之间的联系是错误的。导致很多观点认为完美的LMs并不会泄露关于他们训练数据的信息。因为这些模型经常训练在大量的重复数据消除的数据集仅一个周期，他们几乎不会过拟合。

##### 贡献

我们证明大的语言模型会记忆和泄露个人训练样本。特别的，我们提出了一个简单有效的方法去从语言模型训练集中逐字逐句的提取而且是黑盒访问。

我们的关键观点是，即使我们的训练样本没有显著的更低的错误率比起测试样本（没有过拟合现象），也能证明模型记住了样本。

我们的攻击直接应用在任何语言模型当中，包括训练在敏感数据和非公共数据。我们使用GPT-2模型来作为我们实验的代表。

我们也探讨了策略去减轻隐私的泄露。

##### 隐私攻击

当数据没有被训练在隐私保护算法上面，他们就表现出异常脆弱在应对大量隐私攻击的时候。

***成员推理攻击***：给定一个训练好的模型，敌手能够预测出是否一个特殊的样本被用来训练模型。

***模型逆向攻击***：从一个样本的一部分重构整个典型的“模糊的”样本。

***训练数据提取攻击***：目的是重构逐字的训练样本。跟逆向攻击重构“模糊的”样本有所区别。能够提取一些密码之类的。目前为止，该攻击还被限制在小的LM模型基于学术的训练集或者是敌手有了先验知识再去提取。

##### 保护隐私

一种方法是差分隐私训练技术。然而，这种方式会降低准确度，其导致模型不能捕获长的数据分布。而且增加了训练时间。

#### 威胁模型

##### 敌手的能力

敌手能够黑盒输入输出地访问语言模型。因此允许敌手去计算任意序列的可能性sequences fθ(x1,..., xn)，因此敌手能够获得下一个单词的预测。但是，敌手并不能获得模型里面的权重或者状态。

##### 敌手的目的

去提取模型记住的训练数据。我们并不是要去提取有针对性的目标数据，更多的是不加区别地提取数据。虽然定向的攻击更有伤害性，但是我们的目的是去研究模型记忆数据的普遍能力，而不是创造一种攻击被敌手所使用来得到具体用户的数据。

##### 攻击目标

我们选择GPT-2作为我们攻击的研究。

#### 初始化数据提取攻击

基本步骤

* **生成文本**

  生成一个很大的数据通过无意识的采样从模型当中。

* **预测哪一个输出包含被记忆的文本**

  使用成员推理攻击去筛选哪些是被记忆的文本。

##### 初始化文本生成方案

为了生成文本，我们使用一个包含特殊句首标记和然后以自回归的方式重复地对令牌进行采样从模型。

The first step in our attack is to randomly sample from the language model. Above, we used top-n sampling and conditioned the LM on the start-of-sequence token as input.

局限性就是，前n个从模型中的样本可能生成相同样本很多次。下面，我将介绍两个技术生成更多样性的样本从LM模型当中。

##### Sampling With A Decaying Temperature

​	As described in Section 2.1, an LM outputs the probability of the next token given the prior tokens Pr(xi | x1,..., xi−1). In practice, this is achieved by evaluating the neural network z = fθ(x1,..., xi−1) to obtain the “logit” vector z, and then computing the output probability distribution as y = softmax(z) defined by softmax(z)i = exp(zi)/∑ n j=1 exp(zj). 	One can artificially “flatten” this probability distribution to make the model less confident by replacing the ***output softmax(z) with softmax(z/t)***, for t > 1. Here, t is called the temperature. **A higher temperature causes the model to be less confident and more diverse in its output.**

但是，在整个过程中保持高温生成过程意味着即使采样这个过程可能会产生一个记忆中的例子随机跳出存储输出的路径。因此我们使用 t = 10 and decaying down to t = 1 over a period of the first 20 tokens。 This gives a sufficient amount of time for the model to “explore” a diverse set of prefixes while also allowing it to follow a high-confidence paths that it finds.

##### Conditioning on Internet Text

然而，使用了t仍然会有一些在真实数据当中的后缀没被抽样到。作为最后的策略，我们的第三次抽样策略是喂养模型使用我们从互联网上爬取的前缀。

为了防止我们爬取的数据和模型的训练数据的交叉。我们选择了不同的数据收集过程。

##### 初始化成员推理

Concretely, given a sequence of tokens x1,..., xn, the perplexity is defined as P = exp  − 1 n n ∑ i=1 log fθ(xi |x1,..., xi−1) ! That is, if the perplexity is low, then the model is not very “surprised” by the sequence and has assigned on average a high probability to each subsequent token in the sequence.

生成文本之后，我们需要使用成员推断（Membership Inference）来判断生成文本是否是被记忆的文本。在本文中，作者发现直接运用传统的成员推断存在一定问题：以下两类低质量的生成结果也会被打很高的置信度分数：

* Trivial memorization: 过于普遍常见的内容，例如数字1到100。这些虽然也可能是训练集中被记忆的内容，但意义不大。

* Repeated substrings：语言模型的一种常见智障模式是不断重复输出相同的字符串（例如，“我爱你我爱你我爱你我爱你……”）。作者发现这类文本也容易被打很高的置信度。

为了解决，我们的观点是我们能分类这些没用但又高可能行的样本通过和第二个LM相比较。

Therefore, a natural strategy for finding more diverse and rare forms of memorization is to filter samples where the original model’s likelihood is “unexpectedly high” compared to a second model.

##### Comparing to Other Neural Language Models

对于一个小k，不同的模型训练在不相交集上的数据不太有可能记住相同的数据。一个可选的策略就是采用一个小型的模型训练在相同的数据集上边，因为小型模型的记忆性比较差。

#### 评估记忆性

我们现在来评估多种数据提取方法

##### Methodology

我们刚开始建立三个20000个生成的样本的数据集。每一个256个tokens，使用one of our strategies: 

*  Top-n (§4.1) samples naively from the empty sequence. 
*  Temperature (§5.1.1) increases diversity during sampling. •
*  Internet (§5.1.2) conditions the LM on Internet text

然后筛选这三个数据集根据我们的六个成员推理的标准：

困惑度（perplexity）

Small模型：小型GPT2和大型GPT2的交叉熵比值

Medium模型：中型GPT2和大型GPT2的交叉熵比值

zlib：GPT2困惑度和压缩算法熵的比值

Lowercase：GPT-2模型在原始样本和小写字母样本上的困惑度比例

Window：在最大型GP-2上，任意滑动窗口圈住的50个字能达到的最小困惑度

For each of these 3×6 = 18 configurations, we select 100 samples from among the top-1000 samples according to the chosen metric.9 This gives us 1,800 total samples of potentially memorized content. 

##### 数据删除

为了避免重复计数记忆的内容，我们应用了自动化的模糊的删除步骤，当我们从每一个configuration里面选择100个样本的时候。

给定样本s，定义tri（s）作为s中的三角图，例如my name my name my name” has two trigrams (“my name my” and ”name my name”) ，每一个都重复了两次。

##### 手动搜索评估记忆性

对于1800的选择的样本，我们通过手工搜索的方式判断是否是记忆的内容。Since the training data for GPT-2 was sourced from the public Web, our main tool is Internet searches. We mark a sample as memorized if we can identify a non-trivial substring that returns an exact match on a page found by a Google search

##### 结果

1. The zlib strategy often finds non-rare text (i.e., has a high k-memorization). It often finds news headlines, license files, or repeated strings from forums or wikis, and there is only one “high entropy” sequence this strategy finds. 
2. Lower-casing finds content that is likely to have irregular capitalization, such as news headlines (where words are capitalized) or error logs (with many uppercase words). 
3. The Small and Medium strategies often find rare content. There are 13 and 10 high entropy examples found by using the Small and Medium GPT-2 variants, respectively (compared to just one with zlib).

##### 记忆的内容的例子

##### 个人身份信息

##### URLs

##### Code

##### Unnatural Text

##### Data From Two Sources

##### Removed Content

##### 6.4 提取更长的序列

我们可以扩展许多记忆样本。举个例子，我们从存储库中提取一段源代码在GitHub上。我们可以扩展这个片段来提取整个文件，即1450行逐字源代码。我们可以同时摘录麻省理工学院、知识共享和古腾堡项目许可证。这表明当我们提取了604个记忆中的例子，我们可以扩展其中许多都是记忆内容的更长片段。

##### 6.5 记忆是依赖上下文的

例如，GPT-2会完成前缀3.14159，后25位使用贪婪抽样。但是，我们发现GPT-2知道更多位数因为使用了搜索策略能够提取500位正确的。

有趣的是，提供更多描述性前缀“pi is 3.14159”，贪婪解码提供了前799个位数，比起负载的搜索策略，进一步提供e begins 2.7182818, pi begins 3.14159”, GPT-2 greedily completes the first 824 digits of π。

这个例子表明上下文的重要性。

#### 缓和隐私泄露的策略

##### 差分隐私的训练

##### 控制训练数据

虽然不能手动调整庞大的数据集，然而，有方法去限制所展示的敏感数据，通过调整个人信息使用限制性条款。

##### 限制记忆在下游任务的应用