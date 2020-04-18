# [论文]Bid optimization by multivariable control in display advertising

目标：广告预算约束和CPC约束下的出价优化。

思路：文章首先根据约束和目标进行建模，目标为给定CPC约束和预算约束，最大化会话数量。文章借助线性规划的primal_dual方法，将约束问题转化为对偶问题，并加入超参数$p,q,r_i$，通过互补松弛条件，寻找最优解满足的条件。之后巧妙地设计了竞价价格$bid^*_i=\frac{CTR_I \times CVR_i}{p^*+q^*} + \frac{q^* \times CTR_I \times C}{p^*+q^*}$。同时，RTB环境波动，所以必须要实时反馈调整出价策略，通过分析发现$p、q$分别影响预算和CPC，但是也相互影响，建立independent的PID控制系统会导致系统将相互影响的部分作为噪声，于是作者又分析设计了基于权重的多参数调控系统，解决了参数耦合的问题。

借助线性规划的primal_dual方法，给定双重约束，最大化会话数量。同时为了应对RTB环境波动问题，本文提出利用实时反馈对出价策略进行调整，并基于对最优出价策略的分析，设计了多参数调控系统。为了解决多参数调控系统中参数之间的耦合影响，本文对环境进行建模，并通过部署解耦模块优化系统调控能力。在真实数据集上评估了本文方法的有效性。

关键点：

1. 假设了CTR预估和CVR预估的准确

2. 根据对偶理论，最小值问题的任意可行解都是其对偶问题最优值的一个上界

3. bid for impression 通过加入CTR，转换为了bid for click。

4. c_bid不过原点，分析是，根据CPC约束，出价策略尝试赢得一些便宜的广告机会，以降低整体CPC，以此赢得高CPC的广告。

5. $p$直接影响预算，$q$直接影响CPC。但是$p、q$也相互影响，所以独立PID控制系统会把这种相互影响的数据当做噪声。作者通过推导，利用训练集寻找权重$\alpha,\beta$，使得：$\left[\begin{array}{c}u_{p}^{\prime}(t) \\ u_{q}^{\prime}(t)\end{array}\right]=\left[\begin{array}{ccc}\alpha & 1-\alpha \\ 1-\beta & \beta\end{array}\right]\left[\begin{array}{c}u_{p}(t) \\ u_{q}(t)\end{array}\right]$

   

   

   
