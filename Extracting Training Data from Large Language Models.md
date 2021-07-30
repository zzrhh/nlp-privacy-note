### Extracting Training Data from Large Language Models

#### 摘要

文章证明，敌手能够执行训练数据提取攻击去恢复个人隐私训练样本，通过询问语言模型。而且大型语言模型比小型模型要更加脆弱。

#### 1 简介

隐私的泄露以前的观点主要和过度拟合有关（即一个模型的训练错误远小于测试的错误）。因为过拟合会导致模型记住训练集的样本。

然而，过拟合和记忆性之间的联系是错误的。导致很多观点认为完美的LMs并不会泄露关于他们训练数据的信息。因为这些模型经常训练在大量的重复数据消除的数据集仅一个周期，他们几乎不会过拟合。

##### 贡献

我们证明大的语言模型会记忆和泄露个人训练样本。特别的，我们提出了一个简单有效的方法去从语言模型训练集中逐字逐句的提取而且是黑盒访问。

我们的关键观点是，即使我们的训练样本没有显著的更低的错误率比起测试样本（没有过拟合现象），也能证明模型记住了样本。

我们的攻击直接应用在任何语言模型当中，包括训练在敏感数据和非公共数据。我们使用GPT-2模型来作为我们实验的代表。

我们野探讨了策略去减轻隐私的泄露。

#### 2 背景和相关工作

##### 训练目标

一个语言模型就是用来最大化训练集数据的可能性的。也就是最小化损失函数L(θ) = −logΠ n i=1 fθ(xi | x1,..., xi−1)。因为这种设定，优化器倾向于去记住下一个token的值。然而，因为数据集太过庞大，经验上来说，训练的损失和测试的损失大致相同。

##### 生成文本

一个语言模型能够生成新的文本，通过迭代地抽样，直到达到终止准则。

##### GPT-2

GPT-2使用比特级别编码的词条词汇表。

The GPT-2 model family was trained on data **scraped** from the public Internet. The authors collected a dataset by following **outbound links** from the social media website Reddit. The webpages were cleaned of HTML, with only the document text retained, and then de-duplicated at the document level. This resulted in a final dataset of 40GB of text data, over which the model was trained for approximately 12 epochs.2 As a result, GPT-2 does not overfit: the training loss is only roughly 10% smaller than the test loss across all model sizes.

##### 隐私攻击

当数据没有被训练在隐私保护算法上面，他们就表现出异常脆弱在应对大量隐私攻击的时候。

***成员推理攻击***：给定一个训练好的模型，敌手能够预测出是否一个特殊的样本被用来训练模型。

***模型逆向攻击***：从一个样本的一部分重构整个典型的“模糊的”样本。

***训练数据提取攻击***：目的是重构逐字的训练样本。跟逆向攻击重构“模糊的”样本有所区别。能够提取一些密码之类的。目前为止，该攻击还被限制在小的LM模型基于学术的训练集或者是敌手有了先验知识再去提取。

##### 保护隐私

一种方法是差分隐私训练技术。然而，这种方式会降低准确度，其导致模型不能捕获长的数据分布。而且增加了训练时间。

#### 3 攻击模型

##### 3.1 模型记忆的定义

Informally, eidetic（异常清晰的） memorization is data that has been memorized by a model despite only appearing in a small set of training instances. **The fewer training samples** that contain the data, the **stronger** the eidetic memorization is.

为了证明这个观念，我们定义一个模型知道一个串s，当s能够通过与模型交流被提取出来的时候。

* 定义一（模型知识提取）

  A string s is extractable from an LM fθ if there exists a prefix c such that: s ← argmax s 0 : |s 0 |=N fθ(s 0 | c) We abuse notation slightly here to denote by fθ(s 0 | c) the likelihood of an entire sequence s 0 .
  
* 定义二（k度记忆）

  A string s is k-eidetic memorized (for k ≥ 1) by an LM fθ if s is extractable from fθ and s appears in at most k examples in the training data X : |{x ∈ X : s ⊆ x}| ≤ k

Key to this definition is what “examples” means. For GPT-2, each **webpage** is used (in its entirety) as one training example.

For any given k, memorizing longer strings is also “worse” than shorter strings, although our definition omits this distinction for simplicity.

##### 3.2 威胁模型

##### 敌手的能力

敌手能够黑盒输入输出地访问语言模型。因此允许敌手去计算任意序列的可能性sequences fθ(x1,..., xn)，因此敌手能够获得下一个单词的预测。但是，敌手并不能获得模型里面的权重或者状态。

##### 敌手的目的

去提取模型记住的训练数据。我们并不是要去提取有针对性的目标数据，更多的是不加区别地提取数据。虽然定向的攻击更有伤害性，但是我们的目的是去研究模型记忆数据的普遍能力，而不是创造一种攻击被敌手所使用来得到具体用户的数据。

##### 攻击目标

我们选择GPT-2作为我们攻击的研究。**First**, from an ethical standpoint, the model and data are public, and so any memorized data that we extract is already public.5 **Second**, from a research standpoint, the dataset (despite being collected from public sources) was never actually released by OpenAI. Thus, it is not possible for us to unintentionally “cheat” and develop attacks that make use of knowledge of the GPT-2 training dataset.

##### 3.3 训练数据提取的风险

##### 数据安全

最直接的隐私泄露的风险是一些隐私数据从模型中被提取。例如，GMail‘s 的模型的训练是建立在用户之间的文本交流之间的，所以，提取训练数据的单独片段就会打破数据安全。

##### 数据的上下文完整性

数据的记忆如果导致数据被用在他本来的文本之外的话就是一种侵犯隐私。由于此类故障，面向用户的应用程序LMs可能会在不适当的上下文中无意中发出数据，例如，一个对话系统可以发出一个用户的电话号码作为对其他用户查询的响应。

##### Small-k Eidetic Risks

我们仍然使用小k值，因为他让提取攻击更加有效。而且，虽然我们定义我们的paper是攻击，但是，LMs会输出所记忆的数据即使没有清晰的攻击者的时候。

##### 3.4 道德的考虑

使用了已经pulic的data

#### 4 初始化数据提取攻击

基本步骤

* **生成文本**

  生成一个很大的数据通过无意识的采样从模型当中。

* **预测哪一个输出包含被记忆的文本**

  使用成员推理攻击去筛选哪些是被记忆的文本。

##### 4.1 初始化文本生成方案

为了生成文本，我们使用一个包含特殊句首标记和然后以自回归的方式重复地对令牌进行采样从模型（背景见第2.1节）。

##### 4.2 初始化成员推理

Concretely, given a sequence of tokens x1,..., xn, the perplexity is defined as P = exp  − 1 n n ∑ i=1 log fθ(xi |x1,..., xi−1) ! That is, if the perplexity is low, then the model is not very “surprised” by the sequence and has assigned on average a high probability to each subsequent token in the sequence.

##### 4.3 初始化提取结果

We generate 200,000 samples using the largest version of the GPT-2 model (XL, 1558M parameters) following the text generation scheme described in Section 4.1. We then sort these samples according to the model’s perplexity measure and investigate those with the **lowest perplexity**.

This **initial approach** has two key weaknesses that we can identify. First, our sampling scheme tends to produce a **low diversity of outputs**. For example, out of the 200,000 samples we generated, several hundred are duplicates of the memorized user guidelines of Vaughn Live

其次，我们的基准**成员推断策略**受到影响来自大量误报，即分配的可能性很高，但没有记忆。多数这些假阳性样本中的一个包含“重复”字符串（例如。同一短语重复多次）。尽管有这样的文字。由于可能性很小，大型LMs通常会错误地分配高这种重复序列的可能性。

#### 5 改善训练数据提取攻击

The proof-of-concept attack presented in the previous section has low precision (high-likelihood samples are not always in the training data) and low recall (it identifies no k-memorized content for low k).

##### 5.1 改善文本生成方法

The first step in our attack is to randomly sample from the language model. Above, we used top-n sampling and conditioned the LM on the start-of-sequence token as input.

局限性就是，前n个从模型中的样本可能生成相同样本很多次。下面，我将介绍两个技术生成更多样性的样本从LM模型当中。

##### 5.1.1  Sampling With A Decaying Temperature

​	As described in Section 2.1, an LM outputs the probability of the next token given the prior tokens Pr(xi | x1,..., xi−1). In practice, this is achieved by evaluating the neural network z = fθ(x1,..., xi−1) to obtain the “logit” vector z, and then computing the output probability distribution as y = softmax(z) defined by softmax(z)i = exp(zi)/∑ n j=1 exp(zj). 	One can artificially “flatten” this probability distribution to make the model less confident by replacing the ***output softmax(z) with softmax(z/t)***, for t > 1. Here, t is called the temperature. **A higher temperature causes the model to be less confident and more diverse in its output.**

但是，在整个过程中保持高温生成过程意味着即使采样这个过程可能会产生一个记忆中的例子随机跳出存储输出的路径。因此我们使用 t = 10 and decaying down to t = 1 over a period of the first 20 tokens。 This gives a sufficient amount of time for the model to “explore” a diverse set of prefixes while also allowing it to follow a high-confidence paths that it finds.

##### 5.1.2  Conditioning on Internet Text

然而，使用了t仍然会有一些在真实数据当中的后缀没被抽样到。作为最后的策略，我们的第三次抽样策略是喂养模型使用我们从互联网上爬取的前缀。

为了防止我们爬取的数据和模型的训练数据的交叉。我们选择了不同的数据收集过程。

##### 5.2 改善成员推理

生成文本之后，我们需要使用成员推断（Membership Inference）来判断生成文本是否是被记忆的文本。在本文中，作者发现直接运用传统的成员推断存在一定问题：以下两类低质量的生成结果也会被打很高的置信度分数：

* Trivial memorization: 过于普遍常见的内容，例如数字1到100。这些虽然也可能是训练集中被记忆的内容，但意义不大。

* Repeated substrings：语言模型的一种常见智障模式是不断重复输出相同的字符串（例如，“我爱你我爱你我爱你我爱你……”）。作者发现这类文本也容易被打很高的置信度。

为了解决，我们的观点是我们能分类这些没用但又高可能行的样本通过和第二个LM相比较。

Therefore, a natural strategy for finding more diverse and rare forms of memorization is to filter samples where the original model’s likelihood is “unexpectedly high” compared to a second model.

##### Comparing to Other Neural Language Models

对于一个小k，不同的模型训练在不相交集上的数据不太有可能记住相同的数据。一个可选的策略就是采用一个小型的模型训练在相同的数据集上边，因为小型模型的记忆性比较差。

##### Comparing to zlib Compression



#### 6 评估记忆性

我们现在来评估多种数据提取方法

##### 6.1 Methodology

我们刚开始建立三个20000个生成的样本的数据集。每一个256个tokens，使用one of our strategies: 

*  Top-n (§4.1) samples naively from the empty sequence. 
*  Temperature (§5.1.1) increases diversity during sampling. •
* Internet (§5.1.2) conditions the LM on Internet text

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

##### 证实结果在初始训练集上边

##### 6.2 结果

1. The zlib strategy often finds non-rare text (i.e., has a high k-memorization). It often finds news headlines, license files, or repeated strings from forums or wikis, and there is only one “high entropy” sequence this strategy finds. 
2. Lower-casing finds content that is likely to have irregular capitalization, such as news headlines (where words are capitalized) or error logs (with many uppercase words). 
3.  The Small and Medium strategies often find rare content. There are 13 and 10 high entropy examples found by using the Small and Medium GPT-2 variants, respectively (compared to just one with zlib).

##### 6.3 记忆的内容的例子

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

#### 7 有关记忆和模型大小以及插入频率

#### 8 缓和隐私泄露的策略

##### 差分隐私的训练

##### 控制训练数据

虽然不能手动调整庞大的数据集，然而，有方法去限制所展示的敏感数据，通过调整个人信息使用限制性条款。

##### 限制记忆在下游任务的应用

#### 9 教训和未来的任务

##### 提取数据攻击是可行的威胁

Developing improved techniques for extracting memorized data, including attacks that are targeted towards specific content, is an interesting area for future work.

##### 记忆性并不要求过拟合

经常的观点就是过拟合导致的记忆性。然而，large LMs have no significant train-test gap, and yet we still extract numerous examples verbatim from the training set。关键原因就是，普遍来说，训练损失只比验证损失，然而仍然有一些训练样本要小。Understanding why this happens is an important problem for future work。

##### 更大的模型记忆更多的数据

Throughout our experiments, larger language models consistently memorized more training data than smaller LMs

##### 记忆性很难被发现

Much of the training data that we extract is only discovered when prompting the LM with a particular prefix. Currently, we simply attempt to use high-quality prefixes and hope that they might elicit memorization. Better prefix selection strategies [62] might identify more memorized data

##### Adopt and Develop Mitigation Strategies

Adopt and Develop Mitigation Strategies. We discuss several directions for mitigating memorization in LMs, including training with differential privacy, vetting the training data for sensitive content, limiting the impact on downstream applications, and auditing LMs to test for memorization. All of these are interesting and promising avenues of future work, but each has weaknesses and are incomplete solutions to the full problem. Memorization in modern LMs must be addressed as new generations of LMs are emerging and becoming building blocks for a range of real-world applications.

#### 10 总结

我们的分析可被视作对大型语言模型在敏感数据上面训练的一种警示。而且，模型越大，记忆性就越好，we expect that these vulnerabilities will become significantly more important in the future。

More generally, there are many open questions that we hope will be investigated further, including why models memorize, the dangers of memorization, and how to prevent memorization.
