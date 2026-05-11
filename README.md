# Research on the Approximate Solution of the Ruin Variable in the Risk Model

A method combining the data-driven deep neural network (DNN) approach and the Newton-Cotes formula is used to discretise the continuous function first, thereby calculating the finite-time survival probability through the DNN.

WANG Guanxu (Gwanook), Shenzhen, CHN, 2023

## Classical Compound Poisson Risk Model

The surplus process under the classical compound Poisson risk model is defined as

$$
U(t)=u+ct-\sum_{i=1}^{N_t}X_i .
$$

where $u\ge 0$ is the insurer’s initial capital; $c$ is the rate of premium income per unit time; $\{N_t,t\ge 0\}$ denotes the number of claims up to time $t$, which is assumed to be a Poisson process with Poisson rate $\lambda$; and $\{X_i,i=1,2,\ldots\}$ is a sequence of independent and identically distributed positive random variables that are independent of $\{N_t,t\ge 0\}$. The random variable $\{X_i\}_{i=1}^{\infty}$ represents the amount of the $i$-th claim, which has the distribution function

$$
P(x)=1-\bar P(x)=\Pr(X\le x),\qquad x\ge 0,
$$

and the density function $p(x)$. The positive loading condition is assumed, whereby

$$
c=(1+\theta)\lambda E(X),\qquad \theta>0,
$$

which ensures that the average income is greater than the average claim so that ruin is not guaranteed.

The aggregate claims are defined as

$$
S_t=\sum_{i=1}^{N_t}X_i .
$$

They have the mass point

$$
\Pr(S_t=0)=e^{-\lambda t}
$$

at $0$, and the density $f(x,t)$ for $x>0$, given by

$$
f(x,t)=\sum_{n=1}^{\infty}\frac{(\lambda t)^n e^{-\lambda t}}{n!}p^{*n}(x),
$$

where $p^{*n}$ denotes the density function of

$$
Y_n=X_1+X_2+\cdots+X_n .
$$

Then

$$
F(x,t)=\Pr(S_t\le x)
$$

satisfies

$$
F(x,t)=e^{-\lambda t}+\int_0^x f(\xi,t)d\xi,\qquad x\ge 0.
$$

In the classical risk model, the time of ruin is defined as

$$
T_u=\inf ( t\ge 0:U(t)\le 0 \mid U(0)=u ),
$$

with $T_u=\infty$ if $U(t)\ge 0$ for $t\ge 0$. The finite-time ruin probability is defined as

$$
\psi(u,t)=\Pr(T_u < t\mid U(0)=u),
$$

and the finite-time survival probability is denoted as

$$
\phi(u,t)=1-\psi(u,t).
$$

It is well known that $\phi(u,t)$ satisfies the following PIDE:

$$
\frac{\partial}{\partial t}\phi(u,t)=c\frac{\partial}{\partial u}\phi(u,t)-\lambda \phi(u,t)+\lambda\int_0^u \phi(u-x,t)p(x)dx,
$$

with

$$
\phi(u,0)=1.
$$

## DNN Methodology

In this section, DNN methodology is introduced and then used to solve the PIDE above. The idea of this method is to construct a function $\phi_\theta$ using neural networks that satisfies the original PIDE. More intuitively, $\phi_\theta$ is an approximation of the analytic solution of Equation the PIDE above, and it represents a function that is realized by a neural network with parameters $\theta$.

The parameters $\theta$ include weight matrices $W_i$ and bias vectors $b_i$, $i=1,2,\ldots,M$, where $M$ represents the depth of the neural network structure, that is, the number of hidden layers. DNNs give the best approximation of the original function by minimizing the loss function.

Thus, for any $u$ and $t$, the goal is to make the following equation true:

$$
\frac{\partial}{\partial t}\phi_\theta(u,t)=c\frac{\partial}{\partial u}\phi_\theta(u,t)-\lambda \phi_\theta(u,t)+\lambda\int_0^u \phi_\theta(u-x,t)p(x)dx.
$$

For an initialized neural network with initial parameters $\theta$, since the parameters $\theta$ are randomly given, Equation $(4)$ may not hold for every $u$ and $t$. If that is the case, it results in a data error $E_\theta(u,t)$:

$$
E_\theta(u,t):=\frac{\partial}{\partial t}\phi_\theta(u,t)+\mathcal{N}[u,t,\phi_\theta],
$$

where $\mathcal{N}[\cdot]$ is a nonlinear time-dependent differential operator that represents the network function,

$$
\mathcal{N}[u,t,\phi_\theta]=-c\frac{\partial}{\partial u}\phi_\theta(u,t)+\lambda\phi_\theta(u,t)-\lambda\int_0^u \phi_\theta(u-x,t)p(x)dx.
$$

The aim is to train the neural network to find an optimal parameter $\theta^\ast$. Hence, the error of all training data points should be minimized under the corresponding $\phi_{\theta^\ast}$ of a given parameter.

Therefore, we introduce the loss function $L_\theta(X)$. According to the optimization theory, we know that in the DNN method, the optimal parameter $\theta=\theta^\ast$ can be obtained by minimizing the loss function $L_\theta(X)$:

$$
L_\theta(X):=L_\theta^E(X^E)+L_\theta^I(X^I),
$$

$$
\theta^\ast=\arg\min L_\theta(X).
$$

where $X$ denotes the collection of the training data. The loss function $L_\theta$ consists of two parts: the mean square error of the data points within the region and the mean square error of the data points satisfying the initial condition.

### Mean Square Error of the Data Points Within the Region

$$
L_\theta^E(X^E):=\frac{1}{D_E}\sum_{i=1}^{D_E}\left|E_\theta(u_i^E,t_i^E)\right|^2,
$$

in a training data set

$$
X^E:=\{(u_i^E,t_i^E)\}_{i=1}^{N_E}
\subset [0,\infty)\times(0,\infty).
$$

### Mean Squared Error with Respect to the Initial Condition

$$
L_\theta^I(X^I):=\frac{1}{D_I}\sum_{i=1}^{D_I}\left|\phi_\theta(u_i^I,t_i^I)-\phi_I(u_i^I)\right|^2,
$$

in a number of points

$$
X^I:=\{(u_i^I,t_i^I)\}_{i=1}^{N_I}\subset [0,\infty)\times\{0\},
$$

where $\phi_\theta$ is the neural network approximation of the solution

$$
\phi:[0,\infty)\times[0,\infty)\to\mathbb{R}.
$$

$\phi_I$ denotes the initial value function.

Note that the training data $X$ consists of all data points extracted in the time direction $t$ and space direction $u$. In order to minimize the loss function $L_\theta(X)$ to obtain an approximate solution for Equation $(3)$, DNNs require a further differentiation to evaluate the differential operators $\partial_t\phi_\theta$ and $\mathcal{N}[\phi_\theta]$.

Thus, $L_\theta(X)$ shares the same parameters as the original network $\phi_\theta$. Both types of derivatives can be easily determined through automatic differentiation with machine learning libraries, such as TensorFlow or PyTorch.

It is worth noting that the convolution term in neural network training makes the calculation extremely complicated. Therefore, in order to solve this problem, we use the general Simpson’s rule to discretize the convolution term on the right side of Equation $(4)$.

Letting

$$
g_\theta(x,t)=\phi_\theta(u-x,t)p(x),
$$

we have the following discrete form:

$$
\int_0^u \phi_\theta(u-x,t)p(x)dx\approx\frac{u}{6N_u}\left[g_\theta(0,t)+4\sum_{k=1}^{N_u}g_\theta((2k-1)\delta_u,t)+2\sum_{k=1}^{N_u-1}g_\theta(2k\delta_u,t)+g_\theta(u,t)\right],
$$

where

$$
N_u=\frac{u}{2\delta_u}
$$

and $\delta_u>0$ is a given value.

Then, the network function $\mathcal{N}[u,t,\phi_\theta]$ can be written as

$$
\mathcal{N}[u,t,\phi_\theta]\approx-c\frac{\partial}{\partial u}\phi_\theta(u,t)+\lambda\phi_\theta(u,t)-\lambda\frac{u}{6N_u}\left[g_\theta(0,t)+4\sum_{k=1}^{N_u}g_\theta((2k-1)\delta_u,t)+2\sum_{k=1}^{N_u-1}g_\theta(2k\delta_u,t)+g_\theta(u,t)\right].
$$

## Markov-Modulated Risk Model

In the Markov-modulated risk model, the surplus process is affected by an external environmental process, denoted by $\{J(t);t\geq 0\}$. It is assumed that this process is a homogeneous, irreducible, and recurrent Markov process on the finite state space

$$
\mathcal{E}=\{1,2,\cdots,m\}.
$$

The process has an intensity matrix

$$
Q=(q_{ij})_{m\times m},
$$

where

$$
q_{ii}:=-q_i,\qquad i\in\mathcal{E}.
$$

It also has a stationary distribution

$$
\vec{\pi}=(\pi_1,\pi_2,\cdots,\pi_m).
$$

Let $N(t)$ denote the number of claims occurring in the time interval $(0,t]$. If $J(s)=i$ for all $s$ in a small interval $(t,t+h]$, then $N(t+h)-N(t)$ represents the number of claims occurring in this interval and follows a Poisson distribution with parameter $\lambda_i>0$. Therefore, the process $\{N(t);t\geq 0\}$ is a Markov-modulated Poisson process. Equivalently, it can be regarded as a Poisson process driven by the external environmental process $\{J(t);t\geq 0\}$.

For $i\in\mathcal{E}$, let $\{U_i(t)\}_{t\geq 0}$ denote the surplus process in the classical risk model with premium rate $c_i$, Poisson intensity $\lambda_i$, and individual claim-size distribution $F_i$. Then the surplus process $\{U(t);t\geq 0\}$ in the Markov-modulated risk model is given by

$$
U(t)=u+\sum_{i=1}^{m}\int_{0}^{t}I(J(s)=i)dU_i(s),\qquad t\geq 0,
\qquad (13)
$$

where $u\geq 0$ is the initial surplus and $I(\cdot)$ denotes the indicator function. The positive loading condition is assumed to hold, namely,

$$
\sum_{i=1}^{m}\pi_i(c_i-\lambda_i\mu_i)>0.
$$

According to the semi-Markov model for the Seal-type integral equation proposed by Reinhard et al. (1984) [15], the Markov-modulated risk model corresponding to Equation $(3)$ can be written as

$$
\frac{\partial}{\partial t}\phi_{ij}(u,t)=c_i\frac{\partial}{\partial u}\phi_{ij}(u,t)-(q_i+\lambda_i)\phi_{ij}(u,t)+\lambda_i\int_{0}^{u}\phi_{ij}(u-x,t)p_i(x)dx+q_i\sum_{k=1}^{m}h_{ik}\phi_{kj}(u,t),
$$

where the transition probability matrix is

$$
H=(h_{ij})_{m\times m}.
$$

## Markov Chain Monte Carlo Simulation

The basic idea of Markov Chain Monte Carlo, or MCMC, simulation is to construct a Markov chain whose stationary distribution is the target distribution. A Markov chain is a mathematical model describing a sequence of random variables in which each variable depends only on the immediately preceding variable. The stationary distribution of a Markov chain is the distribution to which the chain converges after running for a sufficiently long period of time.

First, a target distribution is specified. This is the distribution that we aim to approximate. Then, a Markov chain is constructed such that the target distribution is its stationary distribution. One of the most commonly used methods for constructing such a chain is the Metropolis-Hastings algorithm. This algorithm generates a new proposed state from the current state and then accepts or rejects the proposed state according to a probabilistic acceptance rule. The acceptance rule is designed to ensure that the Markov chain converges to the desired target distribution.

Once the Markov chain has been constructed, it can be used to generate a large number of samples from the target distribution. These samples can then be used to estimate various characteristics of the distribution, such as the mean, variance, or higher-order moments. MCMC simulation is particularly useful when the target distribution is high-dimensional or has a complex structure, making direct sampling by other methods difficult.

The Metropolis-Hastings algorithm is commonly used to generate samples from a given target distribution. The procedure is as follows.

### Step 1: Initialization

Choose an initial state $x_0$.

### Step 2: Generation of a Proposed State

Given the current state $x_t$, generate a new proposed state $x'$ from a proposal distribution

$$
q(x'\mid x_t).
$$

### Step 3: Acceptance Probability

Compute the acceptance probability

$$
\alpha=
\min\left(
1,
\frac{p(x')q(x_t\mid x')}{p(x_t)q(x'\mid x_t)}
\right).
$$

Here, $p(x)$ denotes the target distribution, and $q(x'\mid x_t)$ denotes the probability density function of proposing $x'$ given the current state $x_t$.

### Step 4: Acceptance or Rejection

Accept the proposed state $x'$ with probability $\alpha$. With probability $1-\alpha$, reject the proposed state and remain at the current state $x_t$.

### Step 5: State Transition

If the proposed state is accepted, set

$$
x_{t+1}=x'.
$$

Otherwise, set

$$
x_{t+1}=x_t.
$$

### Step 6: Iteration

Repeat Steps 2 to 5 until the specified number of samples has been generated or until the chain has converged to the target distribution.

Since the acceptance probability $\alpha$ is not always equal to $1$, not every proposed state will be accepted. However, by appropriately adjusting the proposal distribution, the acceptance rate can be made close to approximately $50\%$. This helps the Markov chain explore the state space effectively while maintaining sampling efficiency.
