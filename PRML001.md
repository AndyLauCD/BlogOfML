# PRML第一章读书小结



&ensp;&ensp;&ensp;&ensp;第一章用例子出发，较为简单的引入了概率论、模型、决策、损失、信息论的问题，作为机器学习从业者，读PRML除了巩固已有基础，还受到了很多新的启发，下面将我收到的启发总结如下。

## 1. 多项式曲线拟合问题

多项式拟合问题作为全书的第一个引例，通过此说明了很多关键的概念。

给定一个训练集，训练集由$x$的N次观测组成，记作$\mathbf{x} \equiv\left(x_{1}, \cdots, x_{N}\right)^{T}$，对应了相应的观测值$t$，记作$\mathbf{t} \equiv\left(t_{1}, \cdots, t_{N}\right)^{T}$。**它们拥有了一个内在的规律，这个规律是我们想要学习的**，但是同时独立的观察会被随机噪声所干扰。我们的目标是利用这个训练集预测输入变量的新值，我们需要隐式地发现内在的函数$sin(2\pi x)$，由于**有限的观察和噪声**的，发现这一函数（$sin(2\pi x)$）很难。

**概率论提供了一个框架，用精确的数学形式描述这种不确定性。决策论让我们能够根据合适的标准，利用这种概率的表示，进行最优的预测。**

我们经常用多项式函数进行曲线拟合，即$y(x, \boldsymbol{w})=w_{0}+w_{1} x+w_{2} x^{2}+\ldots+w_{M} x^{M}=\sum_{j=0}^{M} w_{j} x^{j}$，系数的值$w$通过拟合训练数据的方式确定，M作为多项式的阶数是模型对比(model comparison)、模型选择(model selection)的重要问题的一个特例。拟合时，我们通过最小化误差函数（error function）的方法实现，一个简单的最小化误差函数如下：
$$
E(\boldsymbol{w})=\frac{1}{2} \sum_{n=1}^{N}\left\{y\left(x_{n}, \boldsymbol{w}\right)-t_{n}\right\}^{2}
$$


我们发现过小的M拟合效果非常差，而高阶的M完美拟合了数据，但是曲线剧烈震荡，就表达函数$sin(2\pi x)$来说表现很差，这便是**过拟合**。

**我们的目标是通过对新数据的预测实现良好的泛化性**，于是我们考虑一个额外的训练集，生成方式和之前的训练集完全相同，但是包含的噪声不同，对于每个M的选择，我们可以利用误差函数，或者均方根误差（RMS）衡量：
$$
E_{R M S}=\sqrt{2 E\left(\boldsymbol{w}^{*}\right) / N}
$$
N保证了以相同的基础对比不同大小的数据集，平方根保证了$E_{RMS}$与目标变量$t$使用相同的规模和单位进行度量。

我们发现M的值**适中**时，均方根误差较小。M特别大的时候，测试误差很大（即过拟合）。进一步思考这个问题，我们发现，对于一个给定的模型复杂度（M给定），数据集规模增加，过拟合问题变得不那么严重，或者说，数据集规模越大，我们能用来拟合数据的模型就越复杂（灵活）。一个粗略的启发是：**数据点的数量不应该小于模型的可调节参数的若干倍。**我们根据待解决问题的复杂性来选择模型的复杂性，**过拟合现象是极大似然的一个通用属性**，而通过贝叶斯方法，过拟合问题可以被避免。

目前，我们使用**正则化（regularization)**技术控制过拟合， 即增加一个惩罚项，使得系数不会达到一个很大的值，如下例是加入平方惩罚项的误差函数：
$$
\tilde{E}(\boldsymbol{w})=\frac{1}{2} \sum_{n=1}^{N}\left\{y\left(x_{n}, \boldsymbol{w}\right)-t_{n}\right\}^{2}+\frac{\lambda}{2}\|\boldsymbol{w}\|^{2}
$$
正则化后的进行多项式拟合效果就能达到一个理想的值。

之后，作者在重新考察曲线拟合问题时，提到了最大化似然函数和最小化平方和误差函数，而最大化后验概率等价于最小化正则化的误差函数。



## 2. 概率论

文章首先通过简单的例子说明了概率论的基本思想，然后表示了概率论的两条基本规则：

**加和规则sum rule**： $p(X)=\sum_{Y} p(X, Y)$
**乘积规则product rule**： $p(X, Y)=p(Y | X) p(X)$

这两个规则是机器学习全部概率推导的基础。

根据乘积规则，我们得到**贝叶斯定理**：
$$
p(Y | X)=\frac{p(X | Y) p(Y)}{p(X)}=\frac{p(X | Y) p(Y)}{\sum_{Y} p(X | Y) p(Y)}
$$
其中，$p(Y)$称为先验概率($prior$)，即根据先验知识得出的关于变量$Y$的分布，$p(X|Y)$称为似然函数（$likelihood$），$p(X)$为变量$X$的概率，$p(Y|X)$称之为条件概率（给定变量$X$的情况下$Y$的概率，$posterior$，后验概率）。

在连续空间中，一个实值变量$x$的概率落在区间$(x,x+\delta x)$的概率由$p(x)\delta x$给出（$\delta x →0$），那么$p(x)$称为$x$的**概率密度**（probability density），$x$在区间$(a,b)$的概率由下式给出：
$$
p(x \in(a, b))=\int_{a}^{b} p(x) \mathrm{d} x
$$
概率密度是处处大于0且归一化的。

离散变量的期望值（expectation）的定义为：
$$
\mathbb{E}[f]=\sum_{x} p(x) f(x)
$$
连续变量的期望值：
$$
\mathbb{E}[f]=\int p(x) f(x) \mathrm{d} x
$$
方差（variance）的定义：
$$
\operatorname{var}[f]=\mathbb{E}\left[(f(x)-\mathbb{E}[f(x)])^{2}\right]
$$

$$
=\mathbb{E}\left[f(x)^{2}\right]-\mathbb{E}[f(x)]^{2}
$$



它度量了$f(x)$在均值$\mathbb{E}[f(x)]$附近变化性的大小。

协方差（covariance）的定义：
$$
\operatorname{cov}[x, y]=\mathbb{E}_{x, y}[\{x-\mathbb{E}[x]\}\{y-\mathbb{E}[y]\}]
$$

$$
=\mathbb{E}_{x, y}[x y]-\mathbb{E}[x] \mathbb{E}[y]
$$

它表示在多大程度上$x$和$y$会共同变化，如果独立，协方差为0.









### 2.1 概率论之贝叶斯学派和频率学派

&ensp;&ensp;&ensp;&ensp;频率学派试图从**自然**的角度出发，试图直接为**事件**建模，即事件A在独立重复实验中发生的频率趋于极限P，那么这个极限就是事件的概率。

&ensp;&ensp;&ensp;&ensp;贝叶斯学派并不试图刻画**事件**本身，而是从**观察者**角度。贝叶斯学派并不认为**事件本身是随机的**，而是从**观察者知识不完备**这一出发点开始，构造一套贝叶斯概率论的框架下可以对不确定知识作出推断的方法。即不认为**事件本身具有某种客观的随机性**，而只是**观察者不知道事件的结果**。

&ensp;&ensp;&ensp;&ensp;频率学派广泛使用极大似然进行估计，使得似然函数$p(\mathcal{D} | \boldsymbol{w})$达到最大。贝叶斯学派广泛使用先验概率。

&ensp;&ensp;&ensp;&ensp;补充：根据知乎某大佬所言:频率学派和贝叶斯学派最大差别是产生在对参数空间的认知上。频率学派并不关心参数空间的所有细节，而相信数据都是在某个参数值下产生的，所以频率学派从“那个值最有可能是真实值”出发的。有了极大似然和置信区间。贝叶斯学派关心参数空间的每一个值，我们又没有上帝视角，怎么可能知道哪个值是真的，参数空间的每个值都有可能是真实模型使用的值，只是概率不同。

参考：https://www.zhihu.com/question/20587681

### 2.2 高斯分布

&ensp;&ensp;&ensp;&ensp;高斯分布算是模式识别里面的重点难点，在第一章里面简要介绍了其一些简单性质，总结如下：

一元高斯分布：
$$
\mathcal{N}\left(x | \mu, \sigma^{2}\right)=\frac{1}{\left(2 \pi \sigma^{2}\right)^{\frac{1}{2}}} \exp \left\{-\frac{1}{2 \sigma^{2}}(x-\mu)^{2}\right\}
$$


高斯分布满足恒大于0：
$$
\mathcal{N}\left(x | \mu, \sigma^{2}\right)>0
$$
高斯分布是归一化的：
$$
\int_{-\infty}^{\infty} \mathcal{N}\left(x | \mu, \sigma^{2}\right) \mathrm{d} x=1
$$

高斯分布的期望：
$$
\mathbb{E}[x]=\int_{-\infty}^{\infty} \mathcal{N}\left(x | \mu, \sigma^{2}\right) x \mathrm{d} x=\mu
$$
二阶矩：
$$
\mathbb{E}\left[x^{2}\right]=\int_{-\infty}^{\infty} \mathcal{N}\left(x | \mu, \sigma^{2}\right) x^{2} \mathrm{d} x=\mu^{2}+\sigma^{2}
$$
方差：
$$
\operatorname{var}[x]=\mathbb{E}\left[x^{2}\right]-\mathbb{E}[x]^{2}=\sigma^{2}
$$
分布的最大值被称为众数，高斯分布的众数与均值恰好相等。



假定一个观测数据集是独立从高斯分布中抽取（independent and identically distributed， i.i.d.），分布均值$\mu$和方差$\sigma^2$未知。数据集的概率：
$$
p\left(\mathbf{x} | \mu, \sigma^{2}\right)=\prod_{n=1}^{N} \mathcal{N}\left(x_{n} | \mu, \sigma^{2}\right)
$$
**当我们把它看做参数的函数的时候，这就是高斯分布的似然函数**。之后我们利用极大似然法寻找似然函数取得最大值的参数值。同时书中提到了：给定数据集下最大化概率的参数和给定参数的情况下最大化数据集出现的概率是相关的。

高斯分布的最大似然解：$\mu_{M L}=\frac{1}{N} \sum_{n=1}^{N} x_{n}$，$\sigma_{M L}^{2}=\frac{1}{N} \sum_{n=1}^{N}\left(x_{n}-\mu_{M L}\right)^{2}$

高斯分布的极大似然估计对均值的估计是无偏的，对方差的估计是有偏的（低估）。





## 3.模型选择

在曲线拟合中，存在一个最优的多项式阶数。实际情况中，我们可能存在多个控制模型复杂度的参数，同时存在过拟合现象，所以我们需要一个验证集。而数据有限，所以需要交叉验证，S-1组进行训练，1组进行评估，运行S次。但是存在一个问题就是训练本身很耗时。



## 4. 维数灾难

随着维数的变高，我们需要指数级的训练数据。对于高维数据，高斯分布的概率质量集中在薄球壳上。这对我们的模型产生了极大地困难。

## 5. 决策论

1. 最小化错误分类率。把每个点分在后验概率最大的类别中，那么我们分类错误的概率就会最小。
2. 最小化期望损失。损失函数（代价函数）最小。
3. 拒绝选项。对于低于阈值的概率，拒绝作出识别，拒绝决策带来的损失可以放在损失矩阵中。

## 6.信息论

随机变量的熵：$H[x]=-\sum_{x} p(x) \log _{2} p(x)$

熵是传输一个随机变量状态值所需的比特位的下界。

相对熵：$\mathrm{KL}(p \| q)=-\int p(\boldsymbol{x}) \ln \left\{\frac{q(\boldsymbol{x})}{p(\boldsymbol{x})}\right\} \mathrm{d} \boldsymbol{x}$

相对熵也被称之为KL散度，不对称。当且仅当$p=q$时，等号成立。

最小化KL散度等价于最大化似然函数（p为真实分布，q为给定分布）。

互信息：
$$
\begin{aligned}
I[\boldsymbol{x}, \boldsymbol{y}] & \equiv \mathrm{KL}(p(\boldsymbol{x}, \boldsymbol{y}) \| p(\boldsymbol{x}) p(\boldsymbol{y})) \\
&=-\iint p(\boldsymbol{x}, \boldsymbol{y}) \ln \left(\frac{p(\boldsymbol{x}) p(\boldsymbol{y})}{p(\boldsymbol{x}, \boldsymbol{y})}\right) \mathrm{d} \boldsymbol{x} \mathrm{d} \boldsymbol{y}
\end{aligned}
$$
$I[\boldsymbol{x}, \boldsymbol{y}] \geq 0$，当且仅当$x$$y$独立时等号成立。我们可以把互信息看成由于知道$y$值而造成的$x$的不确定性的减少。





## 7. 一些小的知识点：

1. 严格凸函数：每条弦位于函数图像上方，即**二阶导数为正**
2. 变分法
3. 高维空间中，球的大部分体积都聚集在表面附近。
4. 具体化一个连续变量需要大量比特位。

