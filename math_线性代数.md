# LINEAR ALGEBRA 线性代数听课笔记

Gilbert Strang的线性代数听课笔记

# 1. 线性代数

1. $(AB)^{T}=B^TA^T$

2. 任何矩阵和他的置换矩阵相乘，得到的新矩阵是对称矩阵

   证明：
   $$
   (R^TR)^T=R^TR^{TT}=R^TR
   $$
   
3. 置换矩阵的逆也是置换矩阵

4. LU分解中，L（lower）对角线是1，且元素就是消元的系数，U是消元后的矩阵

5. $R^2$子向量空间的三种情况:(1)本身(2)过（0,0）的直线（3）（0,0）本身

6. $Ax=b$有解，当且仅当b属于A的列空间


# 7 .求解$AX=0$：主变量，特解

矩阵的秩（rank）：矩阵主元数量

$AX=0$，消元LU分解，变成$UX=0$，找出主变量“pivot variables”，主变量所在列称为主列，其他列称为自由列，自由列的值任取，然后解主列。求得的特解（special solutions），乘以$c$找到一个无限延伸的直线，再重新取自由列的值，回代（back substitution），求得另外一个特解。组合获得零空间（Null space）。

0空间所包含的正好是特解的线性组合。

如果矩阵为$m*n$,$rank=r$,表示主变量的个数等于$r$，自由变量的个数$n-r$，$n-r$个自由变量的值随意选取。

RREF 简化行阶梯矩阵（reduced row echelon form）意味着U可以继续化简，向上化简，主元上下全是0，主元是1，（MATLAB命令RREF），它用最简的形式包含了所有了信息。典型形式如下，I为单位阵，F为自由矩阵
$$
R=\left[\begin{array}{ll}
{I} & {F} \\
{0} & {0}
\end{array}\right]
$$

求解$RX=0$，构造一个零空间（nullspace），特解$RN=0$，零空间特解$N=\left[\begin{array}{l}
{-F} \\{I} \end{array}\right]$,

举例,求零空间：
$$
A=\left[\begin{array}{lll}
{1} & {2} & {3} \\
{2} & {4} & {6} \\
{2} & {6} & {8} \\
{2} & {8} & {10}
\end{array}\right]
$$

$$
=\left[\begin{array}{lll}
{1} & {2} & {3} \\
{0} & {0} & {0} \\
{0} & {2} & {2} \\
{0} & {4} & {4}
\end{array}\right]
$$

$$
→\left[\begin{array}{lll}
{1} & {2} & {3} \\
{0} & {2} & {2} \\
{0} & {0} & {0} \\
{0} & {0} & {0}
\end{array}\right]
$$

设置自由变量等于1： 
$$
x=\left[\begin{array}{c}
{-1} \\
{-1} \\
{1}
\end{array}\right]
$$
**所以零空间是一条直线：**
$$
x=c\left[\begin{array}{c}
{-1} \\
{-1} \\
{1}
\end{array}\right]
$$
A的RREF矩阵：
$$
 \left[\begin{array}{lll}
{1} & {0} & {1} \\
{0} & {1} & {1} \\
{0} & {0} & {0} \\
{0} & {0} & {0}
\end{array}\right]
$$
利用上面的$RN=0$，解得零空间：
$$
X= \left[\begin{array}{c}
{-F} \\
{I}
\end{array}\right]
$$

# 8. 求解Ax=b：可解性和解的结构

求解齐次方程组AX=b。

举例，增广矩阵（augmented matrix）$[A|b]$：
$$
\left[\begin{array}{lllll}
{1} & {2} & {2} & {2} & {b_{1}} \\
{2} & {4} & {6} & {8} & {b_{2}} \\
{3} & {6} & {8} & {10} & {b_{3}}
\end{array}\right]
$$
消元： 
$$
\left[\begin{array}{lll}
{1} & {2} & {2} & {2} & {b_{1}} \\
{0} & {0} & {2} & {4} & {b_{2}-2 b_{1}} \\
{0} & {0} & {2} & {4} & {b_{3}-3 b_{1}}
\end{array}\right]
$$

$$
\left[\begin{array}{lll}
{1} & {2} & {2} & {2} & {b_{1}} \\
{0} & {0} & {2} & {4} & {b_{2}-2 b_{1}} \\
{0} & {0} & {0} & {0} & {b_{3}-b_{2}-b_1}
\end{array}\right]
$$

最后一行必须为0，即$b_3-b_2-b_1=0$

**可解性solvability：**

AX=b仅当b在A的列空间中时成立。

或 如果A各行的线性组合得到0行，b的线性组合必然等于0.

**寻找AX=b的所有解**

第一步，寻找上方方程的一个特解，自由变量为0，解出主变量，求得一个特解，视频中为：
$$
x_p=\left[\begin{array}{c}
{-2} \\
{0} \\
{3 / 2} \\
{0}
\end{array}\right]
$$
第二步，加上nullspace的任意$x$，依旧为解:
$$
x=x_p+x_n
$$
即所有解为特解+零空间的一组基向量。

更一般的情况：秩为$r$的$m×n$矩阵A，有解的情况

1. 满秩：$r=n<m$，自由变量个数为0，其中$R=\left[\begin{array}{l}
   {I} \\
   {0}
   \end{array}\right]$ ，零空间N(A)只包含0向量，如果有解即为特解（0个或1个解），0个出现是因为消元完毕后最下面会出现0行，而右侧未必为0。
2. 满秩：$r=m<n$，自由变量个数$n-r$，$R=[I F]$，对于任意$b$，都有解（无数个）。
3. 满秩：$m=n=r$，得到的是一个可逆矩阵，消元后的RREF=$I$，必定有唯一解。
4. $r<m, r<n$ ，$R=R=\left[\begin{array}{ll}
   {I} & {F} \\
   {0} & {0}
   \end{array}\right]$，（0个或者无穷个解）

# 9. 线性相关性、基、维数

假设A是一个$m×n$矩阵，且$m<n$，求解$Ax=0$，由于存在自由变量，所以一定有无数解。

## 线性无关性（independent）

如果不存在结果为零向量的组合（除了全部系数为0），向量$x_1,x_2,x_3，\dots,x_n$，则向量 无关。

key:零向量和任何向量线性相关。

向量$v_1,v_2,\dots,v_n$构成矩阵$A$，当他们的零空间只包含零向量时，则**线性无关($rank=n$)**

若存在$Ac=0$，且$c!=0$，则**线性相关($rank<n$)**

## 基（basis）

线性无关

生成整个空间

## 维数（dimension）

基的个数

# 10. 四个基本子空间

$A$ $m$行$n$列

列空间$C(A)$ $R^m$  维数 $r$

零空间$N(A)$ $R^n$ 维数$n-r$

行空间$C(A^T) $$R^n$ 维数$r$

左零空间$N(A^T)$ $R^m$ 维数$m-r$

$E[A I]=[R E]$

$EA=R$

在第二章，A可逆，所以$R=I$，故$E=A^{-1}$



# 11. 矩阵空间、秩1矩阵和小世界图

矩阵空间，和向量空间一样，作加法或者数乘。

$3×3$矩阵的基：9个只有一个位置为1的矩阵。

上三角矩阵的基：6个

对称矩阵的基：6个

对角矩阵的基：3个

秩为1的矩阵：
$$
A=UV^T
$$





# 12. 图和网络

本节主要通过网络的概念解析了线性代数的矩阵，线性相关和图中冗余边的关系。

对于一个4个点，5条边的网络，简化为5x4矩阵。

秩为：节点数-1

回路个数=边的个数-rank=边的个数-（节点数-1)=m-n+1

即**欧拉公式**：节点数量-边的数量+回路个数=1



# 14. 正交向量与子空间

**结论：**

$R^n$：行空间$C(A^T) $维数$r$  正交于 零空间$N(A)$  维数$n-r$

$R^m$ ：列空间$C(A)$  维数 $r$正交于左零空间$N(A^T)$  维数$m-r$

**正交orthogonal**： $X^Ty=0$ 等价于$\|x\|^{2}+\|y\|^{2}=\|x+y\|^{2}$

# 15.  子空间投影

二维空间下：一个向量$b$在$a$上的投影：$p=a \frac{a^{T} b}{a^{T} a}$

定义$P= \frac{aa^t}{a^t a }$

$rank(P)=1$

$P^T=P$

$P^2=P$

N维：
$$
\hat{x}=(A^TA)^{-1}A^Tb
$$

$$
p=A \hat{x}=A(A^T A )^{-1}A^T b
$$

$$
P=A (A^TA)^{-1}A^T
$$

$$
P^T=P
$$

$$
P^2=P
$$







# 16. 投影矩阵和最小二乘

**如果A各列线性无关，那么$A^TA$可逆。即（$A^TAx=0$的$x$解只有0）**

# 17.  正交矩阵和Gram-Schmidt正交化

**正交矩阵**

标准正交向量:

$q_{i}^{\top} q_{j}=\left\{\begin{array}{ll}
0 & \text { if } i \neq j \\
1 & \text { if } i=j
\end{array}\right.$

标准正交矩阵：$Q^TQ=I$

若Q为方阵，则$Q^T=Q^{-1}$

利用Q的投影矩阵：$P=QQ^T$

**Gram-Schmidt正交化**

对于任意两个向量$a,b$，求正交向量$A,B$

A=a

$B=b-\frac{A^{T} b}{A^T{A}} A$

$C = c-\frac{A^{\prime} c}{A^{\prime} A} A-\frac{B^{\prime} c}{B^{\prime} B} B$

$q_1 = \frac{A}{||A||}$

$q_2 = \frac{B}{||B||}$

$q_3 = \frac{C}{||C||}$

$Q = [q_1, q_2, q_3]$

矩阵 A

$A = Q R$

（$R$为上三角矩阵）

$R=Q^T A$

# 18. 行列式及其性质

矩阵可逆等价于行列式不为0

**行列式性质**

1. $|I| = 1$；

2. 交换行，行列式的符号相反；

3. $$
   \left|\begin{array}{ccc}
   t a & tb \\
   c & d
   \end{array}\right|
   = t\left|\begin{array}{ccc}
   a & b \\
   c & d
   \end{array}\right|
   $$


$$
   \left|\begin{array}{ccc}
   a + a' & b + b' \\
   c & d
   \end{array}\right|
   = \left|\begin{array}{ccc}
   a & b \\
   c & d
   \end{array}\right|
   
   + 
   \left |
   \begin {array} {ccc}
   a' & b' \\
   c  & d
   \end {array}
   
   
   \right |
$$

4. 两行相等，行列式的值为0

5. 从行$k$减去行$i$的$l$倍，行列式不改变。

6. 若有一行为0，行列式为0.

7. 上下三角矩阵的行列式为对角线元素的乘积。

8. 行列式为0等价于矩阵为奇异矩阵。行列式不为0，等价于矩阵可逆。

9. $det(AB) = det(A) det(B)$

   $det A^2 = (detA) ^ 2$

   $det(2A) = 2^ n det A$

10. $det A ^ T = det A$







# 19.  行列式公式和代数余子式



cofactor of $a_{ij}$:$C_{ij}$ 剩余$n-1$矩阵组成的行列式

当（$i+j$）偶数时，符号为正，否则为负

代数余子式行列式公式：

$det A = a_{11}C_{11} + \dots+a_{1n}C_{1n}$









# 20.  克拉默法则、逆矩阵、体积 

**高斯亚尔当求逆**：$[A| I]$进行线性变换，变成$[I|A^{-1}]$

**行列式求解逆矩阵**：$A^{-1} = \frac{1}{detA} C^T$

**克拉默法则**

https://mp.weixin.qq.com/s?subscene=23&__biz=MzIyMTU0NDMyNA==&mid=2247490302&idx=1&sn=10336d8d71fa23579925305537e92035&chksm=e83a7015df4df903ccc33f874e38f3399a15ea94c5eb18a439ab10d0d1f05ca8b690750ce23c&scene=7#rd

如果有$n$个未知数，$n$个方程组成的线性方程组，它的行列式不等于0，即
$$
|A|=\left|\begin{array}{ccc}
a_{11} & \cdots & a_{1 n} \\
\vdots & & \vdots \\
a_{n 1} & \cdots & a_{n n}
\end{array}\right| \neq 0
$$

则方程组有唯一解：
$$
x_{1}=\frac{\left|A_{1}\right|}{|A|} \quad x_{2}=\frac{\left|A_{2}\right|}{|A|} \quad \ldots \quad x_{n}=\frac{\left|A_{n}\right|}{|A|}
$$

其中$|A_j|$是把矩阵$A$中第$j$列替换为$b$后的$n$阶矩阵：
$$
A_{j}=\left[\begin{array}{ccccccc}
a_{11} & \cdots & a_{1, j-1} & b_{1} & a_{1, j+1} & \cdots & a_{1 n} \\
\vdots & & \vdots & \vdots & \vdots & & \vdots \\
a_{n 1} & \cdots & a_{n, j-1} & b_{n} & a_{n, j+1} & \cdots & a_{n n}
\end{array}\right]
$$

行列式的值等于矩阵构成的空间的体积

# 21. 特征值和特征向量

$Ax = \lambda x$

矩阵的迹：主对线元素之和

特征值之和等于矩阵对角线之和（迹）

特征值相乘等于矩阵行列式

对称矩阵的特征值为实数

# 22. 对角化和A的幂

$A$有$n$个线性无关的特征向量，组成$S$
$$
AS = [\lambda_1x_1,\lambda_2x_2, \dots, \lambda_n x_n] = [x_1, x_2, \dots ,x_n] \left|\begin{array}{ccc}
\lambda_{1} & 0 & \cdots & 0 \\
\vdots & & & \vdots \\
0 & 0 & \cdots & \lambda_{n}
\end{array}\right| = S \Lambda
$$

$$
S^{-1}AS = \Lambda
$$

$$
A=S\Lambda S^{-1}
$$



当所有的$\lambda$不同，必然存在$n$个线性无关的特征向量（且可对角化）。

如果特征值存在重复，可能存在$n$个线性无关特征向量。

$u_{k+1}=Au_k$

$u_0=c_1x_1 + c_2x_2+ \dots + c_n x_n$

$Au_0 = c_1 \lambda_1 x_1 + c_2 \lambda_2 x_2 + \dots$

$u_k = A^k u_0 = c_1 \lambda^k_1 x_1 + c_2 \lambda^k_2 x_2 \dots$

斐波那契数列通项的求解。

# 23. 微分方程和exp(At)



$\frac{du}{dt} = Au$



1. 稳定性：$u(t)->0$，当所有的特征值小于0
2. 稳态：$u(t)->$某值，$\lambda = 0$且其他$\lambda <0$
3. 爆炸：$\lambda>0$




$$
\frac{du}{dt} =A u
$$

令$u = Sv$，$S$为特征向量集合，
$$
\frac{du}{dt} = S \frac{dv}{dt} =ASv
$$

$$
\frac{dv}{dt} = S^{-1}ASv=\Lambda v
$$

$$
v(t)= e^{\Lambda t}v(0)
$$

$$
u(t)=S e^{\Lambda t} S ^ {-1} u(0) = e ^{At} u(0)
$$



因为：
$$
e^x = \sum (\frac{x^n}{n!})
$$

$$
\frac{1}{1-x} = \sum ({x^n})
$$

所以
$$
e^{A t}=I+A t+\frac{(A t)^{2}}{2}+\frac{(A t)^{3}}{6}+\dots
$$

$$
(I-At)^{-1}=I+A t+(A t)^{2}+(A t)^{3}+\dots
$$

所以：
$$
e ^{At} = S e^{\Lambda t} S ^ {-1} 
$$





# 24.马尔可夫矩阵;.傅立叶级数

**马尔科夫矩阵**：所有值$\ge0$，一列总和$=1$

马尔科夫矩阵必有一个特征值为1，且其他特征值的绝对值$\le1$

$u_{k+1}=Au_k$，其中$A$为马尔科夫矩阵

$u_k = A^k u_0 = c_1 \lambda^k_1 x_1 + c_2 \lambda^k_2 x_2 \dots$





**傅里叶级数**

标准正交基：$q_1, \dots, q_n$

任意向量$v$都可以由标准正交基计算得：
$$
v = x_1q_1 + x_2q_2 + \dots + x_n q_n
$$

$$
q^T_1v = x_1
$$

 傅里叶级数
$$
f(x)=a_{0}+a_{1} \cos x+b_{1} \sin x+a_{1} \cos 2 x+b_{2} \sin 2 x+\dots
$$
向量正交：

$v^tw = v_1w_1+ \dots + v_n w_n$

函数正交：

$f^tg = \int f(x) g(x) dx$

傅里叶级数在无穷的正交函数上展开，每一项的系数$a$可以通过和相应项积分获得。

# 25. 复习二



# 26. 对称矩阵及正定性

**对称矩阵**：$A = A^T$

**特征值：实数**

特征向量：垂直

**$A=S \Lambda S^{-1} = Q \Lambda Q^T$**

主元的符号和特征值相同

对称矩阵的逆的特征值为原矩阵的倒数

**正定矩阵**（对称矩阵）：

特征值全为正；主元全为正，所有子行列式为正。



# 27. 复数矩阵和快速傅里叶变换

复数矩阵求模不能用$z^T z$，因为$i$平方为复数，共轭复数矩阵$\bar z ^T z$

$Z^HZ$

复数对称矩阵：$A^H=A$

复数正交矩阵，酉矩阵：$Q^HQ=I$

傅里叶矩阵：
$$
(F_n)_{ij} = w^{ij} \space \space \space i,j=0,\dots,n-1
$$

$$
T_{n}=\left[\begin{array}{lll}
1 & 1 & 1 & \dots\\
1 & w & w^{2} & \dots\\
1 & w^{2} & w^4 & \dots\\ 
\dots & \dots & \dots & \dots\\
\end{array}\right]
$$

$$
w^n=1
$$

$$
w=e^{i2\pi/n}
$$



# 28.  正定矩阵和最小值

**正定矩阵**（对称矩阵）：

1. 特征值全为正；
2. 主元全为正;
3. 所有子行列式为正
4. $x^TAx>0$

在微积分中，一个函数存在极小值：一阶导数为0，二阶导数$>0$

在线性代数中，一系列自变量最小值：二阶导数矩阵为正定矩阵

$A^TA$为（半）正定矩阵



# 29. 相似矩阵和若尔当形

$A^TA$为正定矩阵

$A$为正定矩阵，$B$也为，则$A+B$也是。

$A$和$B$相似，指对于某些$M$，$B=M^{-1}AM$

相似矩阵有同样的特征值。



**bad case**: 具有相同的特征值，如4:

一小类（相似矩阵只包含自己）
$$
\left [ \begin {array}{ccc}
 4 & 0
\\
0 & 4
\end{array}
\right]
$$
另外一大类，所有其他特征值为4和4的，不能被对角化，若尔当型
$$
\left[ \begin{array} {ccc} 
4 & 1 \\
0 & 4
\end{array}
\right]
$$
每一个方阵$A$均相似于若尔当矩阵$J$







# 30. 奇异值分解

所有的矩阵都可以奇异值分解。行空间\*矩阵->列空间\*矩阵

$A=U\Sigma V^T$

$A^TA = V(\Sigma ^ T \Sigma) V^T$

$AA^T = U(\Sigma ^ T \Sigma) U^T$

先确定$v$的符号，然后通过$Av_i=\sigma_i u_i$确定







# 31. 线性变换及对应矩阵

$$
\begin{aligned}
&T(V+w)=T(v)+T(w)\\
&T(c v)=C T(v)\\
&T (c v+d w)=c T(v)+d T(w)
\end{aligned}
$$

投影是一个线性变换；

平移整个平面不是一个线性变换；

求绝对值不是一个线性变换；

旋转是一个线性变换;求导是线性变换。



寻找线性变换的矩阵$A$，输入$v_1,v_2,\dots,v_n$,输出$w_1,w_2,\dots, w_m$:

对第一列，$=a_{11} w_{1}+a_{21} w_{2}+\dots+a_{m 1} w_m$







# 32. 基变换和图像压缩

1. 傅里叶基和小波基
2. 快速，有良好压缩性

**基变换**：变换基，每个矩阵都发生变化，新旧矩阵相似，$B=M^{-1}AM$

# 33. 复习3

1. 怎样的矩阵拥有正交特征向量：对称阵，反对称阵，正交矩阵，满足$A^TA=AA^T$
2. 可对角化的条件：特征值全不相同，或者，特征值相同的特征向量不同。
3. 对称矩阵的特征：特征值全为实数
4. 半正定，正定矩阵的特征：特征值为正的对称矩阵
5. 马尔科夫矩阵的特征：特征值=1，其他小于等于1
6. 投影矩阵：$P^2=P$
7. SVD分解对任意矩阵成立
8. 相似矩阵特征值相同
9. 正交矩阵的特征绝对值为1

# 34.左右逆和伪逆

长方形矩阵是没有逆的， 需要用到左逆，右逆，伪逆的概念。

若$r=m<n$ ，则$AA^T(AA^T)^{-1}=I$，则右逆$A^T(AA^T)^{-1}$

若$r=n<m$，则$(A^TA)^{-1}A^TA=I$，则左逆$(A^TA)^{-1}A^T$

伪逆，一种方式：SVD求解，$A=U\Sigma V^T$，则$A=V\Sigma^{+}U^T$



# 35. 期末复习

# 36. 后记

前前后后，历时几个多月，终于把线性代数重新刷了一遍，距离上一次认真学习线性代数已经有了7年之久，线性代数基本思路都已经忘了七七八八，无奈在工作中还经常碰到，比如梯度爆炸，特征值，马尔科夫，图，等等，遂重新学习。听这个教授的课，生动且深入浅出，真的可以重新理解线性代数，我知道特征值，却不知道为什么要去求特征值，我知道行列式，却不知道为什么去求行列式，应试教育的可悲之处就是知其然不知其所以然，看似有用，其实什么都不理解。感谢。

此篇读书报告乃听课时随手记得笔记，不全，只希冀便于后续的理解，一些概念忘了的时候也可以翻阅



