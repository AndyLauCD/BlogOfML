## 论文研读：基于统计重加权的方法减少通用回复

会议名称：EMNLP2018

文章题目：Towards Less Generic Responses in Neural Conversation Models: A Statistical Re-weighting Method

原文链接：https://link.zhihu.com/?target=https%3A//www.paperweekly.site/papers/2440

一句话概括：  针对开放对话领域的对话多对多关系并且产生通用回复的问题，文章在损失项中引入权重的概念，降低通用回复权重，降低过短或者过长语句的权重。   

### 论文背景

&ensp;&ensp;&ensp;&ensp;神经生成模型在机器翻译中的成功应用，即神经机器翻译（Neural Machine Translation, NMT），激发了研究人员对于**神经对话模型**的热情。目前最常用的框架为Seq2Seq模型，其通常通过极大似然法，最大化回复的概率得到输出结果。但在上述任务中会存在一些问题，其中最严重的的一个是模型经常会产生一个通用的回复（例如，我不知道），而不是一个有意义的特定回答。

&ensp;&ensp;&ensp;&ensp;在开放领域的对话中，我们经常发现对于一个输入$x$，会得到若干意思不一致，但是同样可以接受的回答。如问“你吃饭了吗”，回复“还没”，“不饿”，“刚吃完”，“不急”等等都可以被接受，因此对于$x$到$y$通常是一个一对多甚至多对多的关系，如下图所示：
![file](https://graph.baidu.com/resource/21239ca78e0ef5f8c514a01571835953.png)
&ensp;&ensp;&ensp;&ensp;作者通过这些观察，提出了一种统计重加权的损失函数，减少通用回复。

### 论文方法

&ensp;&ensp;&ensp;&ensp;考虑对于语料库$C$，其对于样本$(\mathbf{x,y})$，损失函数为：
$$
l(\mathbf{x,y},\theta)=-\sum_{t=1}^{T'}logp(y_t|\mathbf{x,y}_{[t-1];}\theta)
$$
&ensp;&ensp;&ensp;&ensp;全样本集的损失函数为：
$$
L(C，\theta)=\sum_{(\mathbf{x,y})\in C}l(\mathbf{x,y},\theta)
$$
&ensp;&ensp;&ensp;&ensp;考虑通用回复出现在很多$\mathbf{x}$对应的回复中，因此，如果我们对于$\mathbf{x}$的两个回复中，如果某个回复比另一个更加通用，他们会具有相同的损失项(根据公式1)，公式2中会包含大量通用回复，导致模型陷入局部最优，即模型更加倾向于产生通用回复。

&ensp;&ensp;&ensp;&ensp;基于上述观察，但是我们应该提高通用回复的损失，降低不通用回复的损失。于是提出下面的损失函数：
$$
l_w(\mathbf{x,y},\theta)=w(\mathbf{y|x},\theta)l(\mathbf{x,y},\theta)
$$

&ensp;&ensp;&ensp;&ensp;在这里，$w(\mathbf{y|x},\theta)$作为一个权重，取值范围为$(0,1]$，对于样本集$C$上的Batch，将其损失函数归一化为：
$$
L(\mathbb{B},\theta)=\frac{\sum_{\mathbf{x,y\in{\mathbb{B}}}}l_w(\mathbf{x,y},\theta)}{\sum_{\mathbf{x,y\in{\mathbb{B}}}}w(\mathbf{y|x})}
$$
&ensp;&ensp;&ensp;&ensp;对于回复，作者总结了两个公共的属性：

&ensp;&ensp;&ensp;&ensp;1.  经常出现在训练语料库中的回复模式往往是通用的。在这里，模式指的是整个句子或n-gram，可以通过回复之间的相似性来描述。 

&ensp;&ensp;&ensp;&ensp;2. 特别长或者特别短的回复都应该避免，太长包含太多特定信息，太短通用回复

&ensp;&ensp;&ensp;&ensp;因此作者设计了权重：
$$
w(\mathbf{y|x},R,C)= \frac{\Phi(\mathbf{y}) }{max_{r\in R}\{\Phi(r)\}}
$$
&ensp;&ensp;&ensp;&ensp;其中$\Phi(\mathbf{y})$指：
$$
\Phi(\mathbf{y})=\alpha\varepsilon(\mathbf{y})+\beta\mathfrak{F}(\mathbf{y})
$$

&ensp;&ensp;&ensp;&ensp;$\varepsilon(\mathbf{y})$为：
$$
\varepsilon(\mathbf{y})=e^{-af\mathbf{(y)}}
$$

&ensp;&ensp;&ensp;&ensp;$\mathfrak{F}(\mathbf{y})$为：
$$
\mathfrak{F}(\mathbf{y})=e^{-c||\mathbf{y}|-|\mathbf{\hat{y}}||}
$$
这里$f(\mathbf{y})$是回复$\mathbf{y}$在所有回复中的出现频次，$\hat y $为所有回复的平均长度，$\{\alpha,\beta,a,c\}$均为超参数。

### 实验结果

&ensp;&ensp;&ensp;&ensp;作者从社交网站爬取了700万对话作为实验，用500作为测试，对句子通顺度，句子相关性，可接受度等方面进行评测，同时对权重的多重设计的有效性进行了评测（只使用频次RWE，长度RWF，都是用RWEF等）结果如下：

![file](https://graph.baidu.com/resource/212c7475f089acb86eb6e01571835974.png)

&ensp;&ensp;&ensp;&ensp;另外作者利用10万测试集统计了常用通用回复的频次，明显看到通用回复变少。

![file](https://graph.baidu.com/resource/2128af00c485e3bb992d401571835989.png)

### 个人总结

&ensp;&ensp;&ensp;&ensp;个人觉得方法还是很有启发性的，通过改变权重，样本原本的分布，以此来达到减少通用回复的目的。

&ensp;&ensp;&ensp;&ensp;但是模型需要顾虑：权重改变改变了样本的分布，这种改变是否合理？噪声点是否因此被放大？在$i.i.d$条件下，人们通用回复说得多是否代表通用回复占比本来就高，这样改变的对话系统是否不符合对话方式？（如在原文中，举例“孟记普洱茶专营店一贴”，通用回复为“我也想喝”，而文章中的模型为“我喜欢喝茶”，是否前者更符合，后者更突兀？）

&ensp;&ensp;&ensp;&ensp;但是这篇文章依旧非常具有启发性，感谢腾讯AILAB，武汉大学，苏州大学的大牛们。





