# Likelihood Ratio Scoring

The BA-LR (Binary Attribute Likelihood Ratio) method introduced in [Imen Ben-Amor's thesis](https://github.com/LIAvignon/BA-LR) consists in replacing typical speaker embeddings by vectors of binary attributes. Theses attributes are supposed to be independent; therefore a speaker identification scoring can be computed based on activation statistics of the attributes over a reference population. One of the hypothesis of the model is that each attribute encodes a specific caracteristic shared among a subset of speakers. It is therefore more than just a binarized version of speaker embeddings.

By replacing the usual scoring in the high dimension space of speaker embeddings by a score that can be decomposed according to each binary attribute, the BA-LR method offers increased explicability.

We propose two models for LLR scoring of binary attribute speaker embeddings. For the first model, each speaker either possesses or does not possess each attribute. For the second model, each speaker is instead assigned a probability of possessing or not possessing each attribute. The first model is therefore a special case of the second. The second model is more generic and is based on standard statistical methods.

## Notation

We write $n$ the number of binary attributes. The activations of the attributes are written as $ba_i \in \{ 0, 1\}$. A recording $x$ is represented by the series of binary attribute activations $ x = [ ba_0, ..., ba_i, ..., ba_n ] $.

For a set $X$ of $p$ recordings, we define the number of activations $a_i = \sum_{j=1}^p 1_{ba_i^j = 1}$ and non activations $n_i = \sum_{j=1}^p 1_{ba_i^j = 0} = p - a_i$ of the attribute $ba_i$ for the recordings of $X$.

To estimate the parameters of the scoring model, we have a training set of $k$ speakers. For each speaker $l \in [1, k]$, we have a set $X_l$ of $p_l$ recodings.

$$
X_l = [(a_1^l, n_1^l), .., (a_i^l, n_i^l), .., (a_k^l, n_k^k)]
$$

During inference for speaker verification, we have a set of enrollment recordings $X_e$ and a set of test recordings $X_t$.

$$
X_e = [(a_1^e, n_1^e), .., (a_i^e, n_i^e), .., (a_n^e, n_n^e)]\\
X_t = [(a_1^t, n_1^t), .., (a_i^t, n_i^t), .., (a_n^t, n_n^t)]
$$

The hypothesis of independence between the attributes implies that the likelihood ratio can be factorized according to the attributes:

$$
LR(X_e, X_t) = \prod_{i=1}^n LR_i(X_e, X_t) =  \prod_{i=1}^n LR_i((a_i^e, n_i^e), (a_i^t, n_i^t))
$$

## Reference LLR scoring

The reference scorers from [Imen Ben-Amor's thesis](https://github.com/LIAvignon/BA-LR) model the distribution of attributes over a reference population with three parameters: *typicality*, *drop-out* and *drop-in*.

The typicality $T_i$ of a binary attribute $b_i$ is the frequency of speaker pairs sharing the attribute in the reference population.

The drop-out $Dout_i$ of attribute $b_i$ is defined as the probability of the attribute disappearing from the profile. This can be due either to a failure to detect the attribute when it is actually present in a recording, or to the attribute being genuinely absent from the recording despite being part of the speaker's profile.

The drop-in is the probability of encountering noise leading to a false detection of the attribute in a recording. It is considered independent of the attribute.

Two formulas for computing the LLR values for the trials are given: *DNA* and *Speech*.

### Parameter estimation

Each speaker either possesses or does not possess each attribute. Thus, for each speaker $l$ and each attribute, we define the variable $y^i_l \in {0, 1}$ which is equal to 1 if the speaker possesses the attribute and 0 otherwise. $y^i_l$ is a latent variable that is never observed.

The model approximates the latent variable $y^i_l$ with the speaker's profile $P_l(BA_i)$.

$$
P_l(BA_i) = 1_{a_i^l > 0}
$$

This gives us an estimator of the typicality corresponding to the proportion of speaker pairs who both have a profile equal to 1.

$$
\hat{T_i} = \frac{N_c(P(BA_i)=1)}{N_c}
$$

where $N_c$ is the number of speaker pairs and $N_c(P(BA_i)=1)$ is the number of speaker pairs who both have a profile equal to 1. If we write $K$ the number of speakers and $K_i^1 = \sum_{l=1}^K 1_{P_l(BA_i) = 1} = \sum_{j=1}^K 1_{a_i^l > 0}$ the number of speakers having a profile equal to 1, we have:

$$
\hat{T_i} = \frac{K_i^1 (K_i^1 -1)}{K (K - 1)}
$$

The dropout estimator $\hat{Dout_i}$ relies on the the profile $P_l(BA_i)$ also. We estimate the dropout for each speaker having the attribute in its profile $\hat{Dout_i^l}$ as well as an average over the speakers.

$$
\hat{Dout_i^l} = \frac{n_i^l}{a_i^l + n_i^l} \\
\hat{Dout_i} =  \frac{\\sum_{l = 1}^K \hat{Dout_i^l}  1_{P_l(BA_i) =1}}{K_i^1}
$$


The drop-in probability $P_{din}$ is assumed to depend on the typicality of the attribute and on a parameter $Din$ that is independent of the attribute.

$$
P_{din, i} = Din \times T_i
$$

The parameter $Din$ is selected so as to minimize the speaker verification error (measured in terms of Cllr).

### DNA scoring

This scoring is based on the following assumptions:

* drop-in and drop-out occur only in test recordings;
* the profile established from enrollment recordings is accurate;
* the drop-in probability is represented by $P_{din, i} = Din \times T_i$, whereas typicality does not affect the probability of no drop-in.

The value of the likelihood ratio depends solely on the enrollment and test profiles. **This scoring assumes there are only one enrollment recording and one test recording**.

$$
LR_i((a_i^e, n_i^e), (a_i^t, n_i^t)) = LR_i(P_e(BA_i), P_t(BA_i))
$$

Thus,

* $LR_i(0,1) = \frac{Dout_i}{T_i}$
* $LR_i(0,0) = \frac{1}{T_i ( 1 - Din + Dout_i)}$
* $LR_i(1,1) = \frac{1}{T_i ( 1 - Dout_i + Din T_i)}$
* $LR_i(1,0) = \frac{Din T_i}{T_i ( Din T_i + 1  - Din)}$

We make the cases $01$ and $10$ equal with the following transformation $LR_i(0,1) = LR_i(1,0) = \frac{LR_i(0,1)+ LR_i(1,0)}{2}$.

### Speech scoring

This scoring is based on the following assumptions:

* drop-in and drop-out can occur in both enrollment and test recordings;
* drop-in and drop-out events in enrollment recordings are independent of those in test recordings;
* the drop-in probability is represented by $P_{din, i} = Din \times T_i$, whereas typicality does not affect the probability of no drop-in.

The value of the likelihood ratio still depends solely on the enrollment and test profiles. **This scoring assumes there are only one enrollment recording and one test recording**.

$$
LR_i((a_i^e, n_i^e), (a_i^t, n_i^t)) = LR_i(P_e(BA_i), P_t(BA_i))
$$

Thus,

* $LR_i(0,1) = \frac{ (1 -Din ) Din T_i + Dout_i (1 - Dout_i) }{ T_i ( (1-Din) Din T_i + Dout_ i (1 - Dout_i)  + 1 + Din T_i Dout_i ) } $
* $LR_i(0,0) = \frac{1 + Dout_i^2}{T_i ( 2 Dout_i (1 - Din) + Dout_i^2 + (1 - Din)^2)} $
* $LR_i(1,1) = \frac{1 + (Din T_i)^2 }{T_i ( 2 Din T_i (1 - Dout_i) + ( Din T_i)^2 + (1 - Dout_i )^2 )} $
* $LR_i(1,0) = \frac{ (1- Din) Din T_i + Dout_i (1 - Dout_i)}{ T_i ( (1-Din) Din T_i + Dout_i (1- Dout_i) + 1 + Din T_i Dout_i)  } $

We observe that $LR_i(0,1) = LR_i(1,0)$.

## Maximum Likelihood Ratio Scorer

The Maximum Likelihood Ratio scorer has the same objectives as the reference model but is based on standard statistical methods. The MaxLLR scorer also models the distribution of attributes over a reference population with the same three parameters (typicality, drop-out and drop-in), but instead of assuming that each speaker either possesses or does not possess each attribute, it assigns each speaker a probability of possessing or not possessing each attribute.

To avoid confusion, the parameters (typicality, drop-out and drop-in) are renamed ($f$, $p$ and $q$).

For an attribute $b_i$, the population frequency $f_i$ is defined as the probability that a speaker possesses the attribute. By definition, $f_i = \mathbb{E}[y^i_l]$.
We denote $p_i$ as the activation frequency of the attribute in a recording for speakers who possess the attribute, and $q_i$ as the activation frequency of the attribute in a recording for speakers who do not possess the attribute.

### Parameter estimation

We propose here an estimation based on the maximum likelihood principle. Since the likelihood expression involves the latent variables $y^l_i$, maximizing the likelihood requires the use of the Expectation-Maximization method.

For a speaker $l$, the likelihood of the attribute activations conditioned on $y^l$ is given by:

$$
\mathbb{L}((a^{l}, n^{l}) \mid y^l) = p^a (1-p)^n \cdot \mathbf{1}_{y^l = 1} +  q^a (1-q)^n \cdot \mathbf{1}_{y^l = 0}
$$

We denote $\theta = (f, p, q)$. The update of $\theta$ is given by:

$$
\theta_{m+1} = \arg\max_{\theta} \sum_{l=1}^k \mathbb{E}_{y^l \mid (a^l, n^l), \theta_m}\left[ \log\left( \mathbb{L}((a^l, n^l), y^l \mid \theta_m) \right) \right]
$$

We denote $\tilde{f}^l = \mathbb{P}(y^l = 1 \mid (a^l, n^l), \theta_m)$, the posterior probability that speaker $l$ possesses the attribute. Thus, we have:

$$
\theta_{m+1} = \arg\max_{\theta} \sum_{l=1}^k \left[ \tilde{f}^l \log( \mathbb{P}((a^l , n^l), y^l = 1 \mid \theta_m)) + (1-\tilde{f}^l) \log( \mathbb{P}((a^l , n^l), y^l = 0 \mid \theta_m)) \right]
$$

We can determine $\tilde{f}^l$:

$$
\tilde{f}^l = \mathbb{P}(y^l=1 \mid (a^l , n^l), \theta_m) = \frac{\mathbb{P}((a^l,n^l), y^l=1 \mid \theta_m)}{ \mathbb{P}( (a^l, n^l) \mid \theta_m)}
$$

$$
\tilde{f}^l = \frac{ f \cdot \mathbb{P}((a^l, n^l) \mid y^l=1, \theta_m) }{ f \cdot \mathbb{P}((a^l, n^l) \mid y^l=1, \theta_m) + (1-f) \cdot \mathbb{P}((a^l, n^l) \mid y^l=0, \theta_m) }
$$

By substituting $\tilde{f}^l$ into the expression for the log-likelihood and differentiating with respect to the parameters $f, p$, and $q$, we obtain the parameter update formulas:

$$
\hat{f} = \frac{\sum_l \tilde{f}^l}{k}
$$

$$
\hat{p} = \frac{\sum_l a^l \tilde{f}^l}{\sum_l (a^l + n^l) \tilde{f}^l}
$$

$$
\hat{q} = \frac{\sum_l a^l (1-\tilde{f}^l)}{\sum_l (a^l + n^l)(1- \tilde{f}^l)}
$$

The initialization of the EM algorithm can be crucial. For now, we naively initialize the algorithm with $f_0 = 0.5$, $p_0 = 0.9$, and $q_0 = 0.1$. **Indeed, the interpretation of $f$ implies that $p > q$.**

### Scoring

We define the likelihood of the observations based on the model parameters:

$$
p((a, n) | f, p, q) = f p^a (1-p)^n + (1-f) q^a (1-q)^n
$$

This allows us to define the likelihood ratio:

$$
LR((a^e, n^e), (a^t, n^t)) = \frac{p((a^e + a^t, n^e + n^t) | f, p, q)}{p((a^e, n^e) | f, p, q) \cdot p((a^t, n^t) | f, p, q)}
$$

When $n^e + a^e = n^t + a^t = 1$, we have the following simplified formulas:

* $LR(0,0) = LR((0,1),(0,1)) = \frac{f (1-p)^2 + (1-f) (1-q)^2}{(f (1-p) + (1-f) (1-q))^2}$
* $LR(0,1) = LR(1,0) = LR((0,1),(1,0)) = \frac{f p (1-p) + (1-f) q (1-q)}{(f (1-p) + (1-f)(1-q)) \cdot (f p + (1-f) q)}$
* $LR(1,1) = LR((1,0),(1,0)) = \frac{f p^2 + (1-f) q^2}{(f p + (1-f) q)^2}$
