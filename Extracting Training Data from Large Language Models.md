### Extracting Training Data from Large Language Models

#### 摘要

文章证明，敌手能够执行训练数据提取攻击去恢复个人隐私训练样本，通过询问语言模型。而且大型语言模型比小型模型要更加脆弱。

#### 1 简介

隐私的泄露主要和过度拟合有关（即一个模型的训练错误远小于测试的错误）。

##### 贡献

我们证明大的语言模型会记忆和泄露个人训练样本。特别的，我们提出了一个简单有效的方法去从语言模型训练集中逐字逐句的提取而且是黑盒访问。

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

训练数据提取攻击目的是重构逐字的训练样本。能够提取一些密码之类的。目前为止，该攻击还被限制在小的LM模型基于学术的训练集或者是敌手有了先验知识再去提取。

##### 保护隐私

一种方法是分布式隐私训练技术。然而，这种方式会降低准确度，其导致模型不能捕获长的数据分布。而且增加了训练时间。

#### 3 攻击模型

##### 3.1 模型记忆的定义

Informally, eidetic memorization is data that has been memorized by a model despite only appearing in a small set of training instances. The fewer training samples that contain the data, the stronger the eidetic memorization is.

* 定义一（模型知识提取）

  

