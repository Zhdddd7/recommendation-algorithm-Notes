# Recommendation System
I am studying the basic ideas in recommendation systems.Here is some note.
## Collaborative Filtering
### the basic idea
1. based on user's behavior, choose similar user and recommend their preferences
### The typical CF algorithms
1. UserCF
2. ItemCF
### Jaccard coefficient
 $$ sim_{uv}=\frac{|N(u) \cap N(v)|}{|N(u)| \cup|N(v)|} $$
### cosine similarity
 $$ sim_{uv}=\frac{|N(u) \cap N(v)|}{\sqrt{|N(u)|\cdot|N(v)|}} $$
### Pearson similarity
$$ sim(u,v)=\frac{\sum_{i\in I}(r_{ui}-\bar r_u)(r_{vi}-\bar r_v)}{\sqrt{\sum_{i\in I }(r_{ui}-\bar r_u)^2}\sqrt{\sum_{i\in I }(r_{vi}-\bar r_v)^2}} $$
### UserCF
  $$ R_{\mathrm{u}, \mathrm{p}}=\bar{R}{u} + \frac{\sum{\mathrm{s} \in S}\left(w_{\mathrm{u}, \mathrm{s}} \cdot \left(R_{s, p}-\bar{R}{s}\right)\right)}{\sum{\mathrm{s} \in S} w_{\mathrm{u}, \mathrm{s}}} $$
### Swing
$$s(i,j)=\sum\limits_{u\in U_i\cap U_j} \sum\limits_{v \in U_i\cap U_j}w_uw_v \frac{1}{\alpha+|I_u \cap I_v|}$$
### Matrix Factorization
矩阵分解算法将 $m\times n$ 维的共享矩阵 $R$ ，分解成 $m \times k$ 维的用户矩阵 $U$ 和 $k \times n$ 维的物品矩阵 $V$ 相乘的形式。其中，$m$是用户数量， $n$ 是物品数量， $k$ 是隐向量维度， 也就是隐含特征个数。 + 这里的隐含特征没有太好的可解释性，需要模型自己去学习。   
一般而言， $k$ 越大隐向量能承载的信息内容越多，表达能力也会更强，但相应的学习难度也会增加。所以，我们需要根据训练集样本的数量去选择合适的数值，在保证信息学习相对完整的前提下，降低模型的学习难度。在分解得到用户矩阵和物品矩阵后，若要计算用户$u$对物品$i$的评分，公式如下:   
$$ \operatorname{Preference}(u, i)=r_{u i}=p_{u}^{T} q_{i}=\sum_{k=1}^{K} p_{u, k} q_{i,k} $$

### Funk SVD
For a random init/hidden P/Q matrix.   
$$ \operatorname{SSE}=\sum_{u, i} e_{u i}^{2}=\sum_{u, i}\left(r_{u i}-\sum_{k=1}^{K} p_{u,k} q_{i,k}\right)^{2} $$
In a iteration optimizing process, we will be using SGD:
$$ \frac{\partial}{\partial p_{u,k}} S S E=\frac{\partial}{\partial p_{u,k}}\left(\frac{1}{2}e_{u i}^{2}\right) =e_{u i} \frac{\partial}{\partial p_{u,k}} e_{u i}=e_{u i} \frac{\partial}{\partial p_{u,k}}\left(r_{u i}-\sum_{k=1}^{K} p_{u,k} q_{i,k}\right)=-e_{u i} q_{i,k} $$
 $$ \frac{\partial}{\partial q_{i,k}} S S E=\frac{\partial}{\partial q_{i,k}}\left(\frac{1}{2}e_{u i}^{2}\right) =e_{u i} \frac{\partial}{\partial q_{i,k}} e_{u i}=e_{u i} \frac{\partial}{\partial q_{i,k}}\left(r_{u i}-\sum_{k=1}^{K} p_{u,k} q_{i,k}\right)=-e_{u i} p_{u,k} $$


 $$ \min_{\boldsymbol{q}^{}, \boldsymbol{p}^{}} \frac{1}{2} \sum_{(u, i) \in K}\left(\boldsymbol{r}{\mathrm{ui}}-p{u}^{T} q_{i}\right)^{2}+\lambda\left(\left|p_{u}\right|^{2}+\left|q_{i}\right|^{2}\right) $$
### Bias SVD
$$ \begin{aligned} \min_{q^{*}, p^{*}} \frac{1}{2} \sum_{(u, i) \in K} &\left(r_{u i}-\left(\mu+b_{u}+b_{i}+q_{i}^{T} p_{u}\right)\right)^{2} \ &+\lambda\left(\left||p_{u}|\right|^{2}+\left||q_{i}|\right|^{2}+b_{u}^{2}+b_{i}^{2}\right) \end{aligned} $$
the gradients are:   
$ \frac{\partial}{\partial b_{i}}SSE=-e_{ui}+\lambda b_{i} \ $   
$ \frac{\partial}{\partial b_{u}}SSE=-e_{ui}+\lambda b_{u} \ $

## References
 [Fun-Rec推荐算法](https://github.com/datawhalechina/fun-rec)
