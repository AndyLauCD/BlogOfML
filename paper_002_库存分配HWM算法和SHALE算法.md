# 库存分配算法之HWM算法和SHALE算法

## 简介

在广告投放系统中，广告通常分为保量交付广告（Guaranteed Delivery，GD，合约广告）和不保量交付（Non-Guaranteed Delivery，NGD，竞价广告）两种。GD广告是提前签好合约的广告，需要将对应的广告投放给特定属性的人群，量不足时会有惩罚。于是出现了一个**问题**：**如何保证GD广告完成投放且保证整个广告系统收益最大？**

另外，不同广告主投放广告目的不一，品牌广告主主要目的是让广告触达尽可能多的目标人群，有些广告主为了点击率和转化率，也有一些诉求介于两者之间。同时，不同广告主的付费方式不同，一般有CPM、CPC、CPA等方式。于是出现了第二个问题：如何分配流量，使得追求不同目标的广告主和采用不同目标的广告主的目标都尽可能的达成？（注：此点出现在文献【3】中，但是笔者认为这个和问题1大同小异，即广告系统收益最大）。

库存分配问题，可以简化为一下二部图（bipartite graph），如图，$s_i$表示用户的一次浏览，$d_i$表示用户的一次广告活动，广告可以是GD，也可以是NGD，他们的连线表示一次流量的分配。流量的分配按照定向投放的方式，即广告投放系统抽象出用户的属性，然后投放对应定向条件的广告。一个广告对应多类用户，一类用户也对应多种广告。

![image-20200402181411282](https://andy-md.oss-cn-beijing.aliyuncs.com/imgs\image-20200402181411282.png)



当流量到达的时候，系统判断流量满足的定向条件，然后选择合法（满足要求）的定向广告展现给用户，完成一次分配。NGD广告按照竞价方式售卖，分配的目标是使广告系统收益最大化，尽可能挣到更多的钱。GD广告首要目标是保量，同时也需要关心广告效果。所以库存分配的目标包括三部分：NGD广告收益最大化，GD广告保量，广告效果最大化。数学语言描述见下图【3】，在此不进行赘述。

![image-20200331161651200](https://andy-md.oss-cn-beijing.aliyuncs.com/imgs\image-20200331161651200.png)



![image-20200331161723178](https://andy-md.oss-cn-beijing.aliyuncs.com/imgs\image-20200331161723178.png)



![image-20200331161741577](https://andy-md.oss-cn-beijing.aliyuncs.com/imgs\image-20200331161741577.png)

![image-20200331161818082](https://andy-md.oss-cn-beijing.aliyuncs.com/imgs\image-20200331161818082.png)

![image-20200331161920190](https://andy-md.oss-cn-beijing.aliyuncs.com/imgs\image-20200331161920190.png)



上述为流量分配问题的目标，在现实中，还需要考虑实时性，即线上计算部分的时间复杂度不宜太高。下面介绍两种库存分配中常用的算法，包括HWM算法和SHALE算法，其优化目标，均为保GD量，未考虑NGD收益和GD收益（SHALE算法通过合约广告重要程度这一参数部分考虑了GD收益）。



## HWM算法

  【2】提出一种算法HMW（High Water Mark Algorithm），HMW通过简单的启发算法解决库存分配的问题，算法主要思想是：首先考虑能满足合约的所有流量，流量越少重要程度越高（$order$更靠前，即更需要提前考虑），然后将流量按照所需量平分，转换为概率。

如下图【3】算法分为两部分，离线计算合约分配概率$\alpha_j$和分配顺序$order_j$，在线部分实时计算各个用户的广告分配概率。看起来很复杂，但是真正理解思想后，发现其实真的思想和复杂度都很简单，是最简单的贪心。



![image-20200331193048204](https://andy-md.oss-cn-beijing.aliyuncs.com/imgs\image-20200331193048204.png)

能看到，HWM算法需要提前预测每一个用户的流量$s_i$，如果预测不准确，就会导致在线分配出现问题，故可以调整分配概率$\alpha_j$，具体见【2,3】。HWM为一种贪心算法，不是最优方案。在线部分时间复杂度为$O(1)$

## HWM算法示例

如图，左侧为到达系统的用户，右侧为广告合约，广告合约来自于三个广告主，分别对应不同的定向条件，且每个合约广告约定了投放量。Supply表示到达系统的用户，他们均有自己的属性和**预估**流量。图中的连线表示用户满足合约的定向条件。根据算法流程，进行计算，流程如下：

![流量分配例图](https://andy-md.oss-cn-beijing.aliyuncs.com/imgs/20200331214130.png)

1. 离线部分

（1）首先计算各个合约的所有用户流量值和：

$S_1=400+400+100=900$

$S_2=100+100=200$

$S_3=400+400+100+100+500+300=1800$

（2）排序，$S_2<S_1<S_3$，故$order_2>order_1>order_3$，在进行后续计算时，按照此顺序计算（即优先分配了可分流量较少的广告，满足其流量要求）。

（3）按照分配顺序进行计算，计算各个合约的概率：

![](https://andy-md.oss-cn-beijing.aliyuncs.com/imgs/20200331221926.png)



（4）如图，对于合约按照顺序计算其分配概率：

首先，对于合约2，根据$\sum_{i \in \Gamma(j)} \min \left\{r_{i}, s_{i} \alpha_{j}\right\}=d_{j}$计算$\alpha_2$：
$$
min\{r_3,s_3*\alpha_j\}+min\{r_4,s_4*\alpha_j\}=d_2
$$

$$
min\{100,100*\alpha_j\}+min\{100,100*\alpha_j\}=200
$$

解得$\alpha_2=1$；

同理，对于合约1，利用合约2计算完的剩余流量(表格中的第三小列)，计算$\alpha_1=\frac{200}{800}=\frac{1}{4}$；

同理，对于合约3，利用合约1计算完的剩余流量（表格中的第五小列），计算$\alpha_3=\frac{1000}{1600}=\frac{5}{8}$。

2. 在线部分

用户4到达时，$l=1$，即取前l个使概率最好不大于1的最大值，故合约2的广告必出，用户1到达时，可以出合约1和3，还可以出NGD广告。

**思考**：可以看到，整个算法的对流量预估敏感，文献【3】通过调整$\alpha_j$和调整预估流量，优化分配算法。这种分配方式也充分考虑了线上计算复杂度，但是，算法没有考虑后续合约如果流量被前面占有之后，导致广告投放无法保量后的情况，（惩罚项），也没有考虑各个广告的重要程度，总而言之，是一种贪婪的分配方式，而不是最优的方案。于是有了SHALE算法。

## SHALE算法及其推导

首先定义SHALE的优化目标：
$$
minimize\frac{1}{2} \sum_{j, i \in \Gamma(j)} s_{i} \frac{V_{j}}{\theta_{i j}}\left(x_{i j}-\theta_{i j}\right)^{2}+\sum_{j} p_{j} u_{j}
$$

$$
\text { s.t. } \quad \sum_{i \in \Gamma(j)} s_{i} x_{i j}+u_{j} \geq d_{j}  \space\space\space \forall j
$$

$$
\sum_{j \in \Gamma(i)} x_{i j} \leq 1  \space\space\space \forall i
$$

$$
x_{i j}, u_{j} \geq 0 \space\space\space \forall i,j
$$

其中，约束1为广告需求约束，约束2为用户流量约束，约束3为非负约束（保证逻辑的正确）。

$s_i$表示$i$类用户的流量，$V_j$表示合约$j$的重要程度；

$\theta_{ij}=\frac{d_j}{S_j}$表示$s_i$中应该分配给广告$j$的理想分配比例，其中$S_{j}=\sum_{i \in \Gamma(j)} s_{i}$表示满足广告$j$的所有可分配流量，$\Gamma(j)$是满足合约$j$的所有用户集合，$d_j$是每个合约的约定投放量；

$x_{ij}$表示算法实际分配量，是需要求解的值；

$p_j$表示每一个合约的惩罚项，流量投递不足时的单位惩罚；

$u_j$表示合约$j$的投递不足的流量。

可以看到，这就是约束条件下的优化问题，分配比例与理想分配比例越近，且投递不足流量越少，则越接近最优结果。利用拉格朗日对偶性，将原始问题（极小极大问题）转换为对偶问题，并且利用KKT条件进行求解。（求解过程见附）

拉格朗日函数：
$$
L(x,u,\alpha,\beta,\gamma,\psi)=\frac{1}{2} \sum_{j, i \in \Gamma(j)} s_{i} \frac{V_{j}}{\theta_{i j}}\left(x_{i j}-\theta_{i j}\right)^2+\sum_{j} p_{j} u_{j}+\sum_j\alpha_j(d_j-u_j-\sum_{i\in\Gamma(j)}s_ix_{ij})
$$

$$
+\sum_i\beta_i(s_i-s_i\sum_{j\in\Gamma(i)}x_{ij})+\sum_{i,j}\gamma_{i j}(-x_{ij})+\sum_j\psi_{j}(-u_{j})
$$

原始问题和对偶问题的解的充分必要条件是满足KKT条件，即：

$\nabla_xL(x,u,\alpha,\beta,\gamma,\psi)=0$

$\nabla_uL(x,u,\alpha,\beta,\gamma,\psi)=0$

$\nabla_\alpha L(x,u,\alpha,\beta,\gamma,\psi)=0$

$\nabla_\beta L(x,u,\alpha,\beta,\gamma,\psi)=0$

$\nabla_\gamma L(x,u,\alpha,\beta,\gamma,\psi)=0$

$\nabla_\psi L(x,u,\alpha,\beta,\gamma,\psi)=0$

$\forall j\space\space\alpha_j(d_j-u_j-\sum_{i\in\Gamma(j)}s_ix_{ij})=0$

$\forall i \space\space\beta_i(s_i-s_i\sum_{j\in\Gamma(i)}x_{ij})=0$

$\forall i,j\space\space\gamma_{i j}(-x_{ij})=0$

$\forall j\space\space\space\psi_{j}(-u_{j})=0$

$\alpha,\beta,\gamma,\psi\ge0$

根据KKT条件，求解得：                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
$$
\forall i,j \space\space  s_{i} \frac{V_{j}}{\theta_{i j}}\left(x_{i j}-\theta_{i j}\right) -s_i\alpha_j+s_i\beta_i-\gamma_{ij}=0
$$

$$
\forall i, p_j-\alpha_j-\psi_j=0
$$

$$
\forall j \space \alpha_j=0 \text{ or }u_j+\sum_{i\in\Gamma(j)}s_ix_{ij}=d_j
$$

$$
\forall i \space \space \beta_{i}=0 \text { or } \sum_{j \in \Gamma(i)} s_{i} x_{i j}=s_{i}
$$

$$
\forall i,j \space \space \space \gamma_{i j}=0 \text { or } x_{i j}=0
$$

$$
\forall j \space \space \space \psi_{j}=0 \text { or } u_{j}=0
$$

$$
\alpha,\beta,\gamma,\psi\ge0
$$

解第一个公式，得
$$
x_{i j}=\theta_{i j}\left(1+\frac{\alpha_{j}-\beta_{i}+\gamma_{i j} / s_{i}}{V_{j}}\right)
$$
同时，$x_{i,j}$和$\gamma_{i,j}$必有一个为0，当$(1+\frac{\alpha_{j}-\beta_{i} }{V_{j}})$小于0时，$\gamma_{ij}$会增加，使得$x_{ij}=0$，所以：
$$
x_{ij}=max\{ 0, \theta_{i j}\left(1+\frac{\alpha_{j}-\beta_{i} }{V_{j}}\right) \}=g_{ij}(\alpha_j-\beta_i)
$$
​                                                                        

同时，根据第二个公式，$\alpha_j=p_j-\psi_j$，且$\psi \ge0$，得出$\alpha_{j} \leq p_{j}$。另外，$\psi_{j}=0 \text { or } u_{j}=0$，对应的即为$\alpha_j=p_j$或$\sum_{i \in \Gamma(j)} s_{i} x_{i j} \geq d_{j}  $，

同时，根据第三个公式，$\alpha_j=0$或$u_j+\sum_{i\in\Gamma(j)}s_ix_{ij}=d_j$。

$\alpha\in(0,p)$时，$\psi \neq0$,所以$u=0$，所以$\sum_{i\in\Gamma(j)}s_ix_{ij}=d_j$，$\alpha = 0$时，$\sum_{i \in \Gamma(j)} s_{i} x_{i j} \geq d_{j}  $(因为$u_j=0$)
$$
\sum_{i\in\Gamma(j)}s_i x_{ij}=\sum_{i\in\Gamma(j)}s_i g_{ij}(-\beta_i)=\sum_{i\in\Gamma(j)}s_i     max\{ 0, \theta_{ij}  \frac{V_j-\beta_j}{V_j}\}
$$

又因为$\sum_i s_i \theta_{ij}=d_j$，所以$\sum_{i\in\Gamma(j)}s_i x_{ij} \le d_j$,所以

$\alpha_{j}^{*}=p_{j} \text { or } \sum_{i \in \Gamma(j)} s_{i} x_{i j}^{*}=0$

整理结果如下：

1. 最优解由$x_{i j}^{*}=g_{i j}\left(\alpha_{j}^{*}-\beta_{i}^{*}\right)$得出。
2. $\forall j，0 \leq \alpha_{j}^{*} \leq p_{j}$，具体地，$\alpha_{j}^{*}=p_{j} \text { or } \sum_{i \in \Gamma(j)} s_{i} x_{i j}^{*}=0$
3. $\forall i,\beta_{i} \geq 0$，具体地，$\beta_{i}=0 \text { or } \sum_{j \in \Gamma(i)} x_{i j}^{*}=1$



## SHALE算法流程

根据上一节推导结果，文章中利用坐标下降法进行求解，主要进行如下操作：

### **初始化**

初始化所有的$\alpha_j=0$.

### **第一步**

重复这一步，直到退出

1. 对于每一个用户$i$，求解满足$\sum_{j \in \Gamma(i)} g_{i j}\left(\alpha_{j}-\beta_{i}\right)=1$的$\beta_i$，若小于0或无解，则$\beta_i=0$
2. 对于每一个合约$j$,q求解满足$\sum_{i \in \Gamma(j)} s_{i} g_{i j}\left(\alpha_{j}-\beta_{i}\right)=d_{j}$的$\alpha_j$，若大于$p_j$或无解，则$\alpha_j=p_j$

### **第二步**

1. 初始化$\tilde{s}_{i}=1，\forall i$

2. 对于每一个用户$i$，求解满足$\sum_{j \in \Gamma(i)} g_{i j}\left(\alpha_{j}-\beta_{i}\right)=1$的$\beta_i$，若小于0或无解，则$\beta_i=0$

3. 对于每一个合约$j$，用HWM中方法分配顺序队列，然后：

   a. 寻找满足$\sum_{i \in \Gamma(j)} \min \left\{\tilde{s}_{i}, s_{i} g_{i j}\left(\zeta_{j}-\beta_{i}\right)\right\}=d_{j}$的$\zeta_{j}$，若无解，则$\zeta_{j}=\infty$

   b. 对于每一个满足合约$j$的用户$i$，更新$\tilde{s}_{i} = \tilde{s}_{i}-\min \left\{\tilde{s}_{i}, s_{i} g_{i j}\left(\zeta_{j}-\beta_{i}\right)\right\}$

### 输出

对于每一个合约$j$，输出$\alpha_{j}$ 和$\zeta_{j}$。

### 在线分配

当流量到来时，根据合约的输出$\alpha_{j}$ 和$\zeta_{j}$，进行以下分配：

1. 设置$r_i=1$，计算$\beta_i$：$\sum_{j \in \Gamma(i)} g_{i j}\left(a_{j}-\beta_{i}\right)=1$

2. 对于所有的合约，按照分配顺序，计算

   $x_{i j}=\min \left\{r_{i}, g_{i j}\left(\zeta_{j}-\beta_{i}\right)\right.$，更新$r_i=r_i-x_{ij}$

3. 挑选合约

### 完成分配

在第一步中，通过假设$\beta$正确迭代$\alpha$，然后假设$\alpha$正确迭代$\beta$，论文【1】中的定理表明，第一步中的解收敛，具体地，$\varepsilon>0 .$在 $\frac{1}{\varepsilon} n \max _{j}\left\{p_{j} / V_{j}\right\}$ 迭代后，输出 $\alpha$在误差内 $\varepsilon$近似。在第二步中，利用和HWM相似的方法计算了$\zeta_{j}$，通过第一步计算的$\alpha$计算了$\beta$，同时注意到，分配的流量需要知道并且不能超过$s_i$。

## 参考文献

1. Bharadwaj V , Chen P , Ma W , et al. SHALE: An Efficient Algorithm for Allocation of Guaranteed Display Advertising[J]. 2012.
2. Chen P , Ma W , Mandalapu S , et al. Ad Serving Using a Compact Allocation Plan[J]. 2012.
3. 张亚东《在线广告 互联网广告系统的架构和算法》
4. https://mp.weixin.qq.com/s/2VejGdsZrCxB8pD1y7Bqhg