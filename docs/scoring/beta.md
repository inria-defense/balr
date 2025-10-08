# Beta-Bernouilli Scoring

In this probabilistic modeling of speakers, each speaker $l$ is characterized by the probability $p^i_l \in [0,1]$ of activating attribute $ba_i$. $p^i_l$ is a latent variable that is never observed.

We define $f$ as the probability density corresponding to the probability $p$ of activating the attribute for a speaker. $f$ can be estimated either from a parametric family or using the empirical distribution.

For a Beta-Bernouilli model, $f$ is parameterized by a Beta distribution with parameters $\alpha$ and $\beta$:

$$
f(p \mid \alpha, \beta) = \frac{p^{\alpha -1} (1-p)^{\beta -1}}{B(\alpha, \beta)}
$$

where $B$ is the Beta function. This distribution is chosen because it is the conjugate prior of the binomial distribution, which simplifies likelihood computations.

## Parameter estimation

The parameters $\alpha$ and $\beta$ can be estimated by maximum likelihood from the activation and non-activation statistics of the attributes per speaker. The formulas are given in [the work of Thomas Minka](https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf). Minka provides the maximum likelihood estimate for Dirichlet multinomial distributions. The Beta-Bernoulli scoring is a special case of a Dirichlet-Multinomial model (with K=2 parameters). The Beta-Bernouilli implementation is done directly using the Dirichlet-Multinomial model.

## Scoring

The choice of a Beta distribution for $f$ allows the likelihoods to be computed explicitly.

$$
L(a, n)  = \int_{p=0}^1  \binom{a+n}{a} p^a (1-p)^n f(p) dp = \frac{B(\alpha+a, \beta+n)}{B(\alpha, \beta)}
$$

which allows us to derive the likelihood ratio

$$
LR((a^e, n^e) , (a^t, b^t) ) = \frac{B(\alpha+a^e +a^t, \beta+n^e + n^t) B(\alpha, \beta) }{B(\alpha+a^e, \beta+n^e) B(\alpha+a^t, \beta+n^t)}
$$
