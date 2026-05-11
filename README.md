# Research on the Approximate Solution of the Ruin Variable in the Risk Model

A method combining the data-driven deep neural network (DNN) approach and the Newton-Cotes formula is used to discretise the continuous function first, thereby calculating the finite-time survival probability through the DNN.

WANG Guanxu (Gwanook), Shenzhen, CHN, 2023

## Classical Compound Poisson Risk Model

The surplus process under the classical compound Poisson risk model is defined as

$$
U(t)=u+ct-\sum_{i=1}^{N_t}X_i .
\tag{1}
$$

where \(u\ge 0\) is the insurer’s initial capital; \(c\) is the rate of premium income per unit time; \(\{N_t,t\ge 0\}\) denotes the number of claims up to time \(t\), which is assumed to be a Poisson process with Poisson rate \(\lambda\); and \(\{X_i,i=1,2,\ldots\}\) is a sequence of independent and identically distributed positive random variables that are independent of \(\{N_t,t\ge 0\}\).

The random variable \(\{X_i\}_{i=1}^{\infty}\) represents the amount of the \(i\)-th claim, which has the distribution function

$$
P(x)=1-\bar P(x)=\Pr(X\le x),\qquad x\ge 0,
$$

and the density function \(p(x)\). The positive loading condition is assumed, whereby

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

at \(0\), and the density \(f(x,t)\) for \(x>0\), given by

$$
f(x,t)=\sum_{n=1}^{\infty}\frac{(\lambda t)^n e^{-\lambda t}}{n!}p^{*n}(x),
\tag{2}
$$

where \(p^{*n}\) denotes the density function of

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
T_u=\inf\{t\ge 0:U(t)\le 0\mid U(0)=u\},
$$

with \(T_u=\infty\) if \(U(t)\ge 0\) for \(t\ge 0\). The finite-time ruin probability is defined as

$$
\psi(u,t)=\Pr(T_u<t\mid U(0)=u),
$$

and the finite-time survival probability is denoted as

$$
\phi(u,t)=1-\psi(u,t).
$$

It is well known that \(\phi(u,t)\) satisfies the following PIDE:

$$
\frac{\partial}{\partial t}\phi(u,t)
=
c\frac{\partial}{\partial u}\phi(u,t)
-\lambda \phi(u,t)
+\lambda\int_0^u \phi(u-x,t)p(x)dx,
\tag{3}
$$

with

$$
\phi(u,0)=1.
$$

## DNN Methodology

In this section, DNN methodology is introduced and then used to solve Equation \((3)\). The idea of this method is to construct a function \(\phi_\theta\) using neural networks that satisfies the original PIDE. More intuitively, \(\phi_\theta\) is an approximation of the analytic solution of Equation \((3)\), and it represents a function that is realized by a neural network with parameters \(\theta\).

The parameters \(\theta\) include weight matrices \(W_i\) and bias vectors \(b_i\), \(i=1,2,\ldots,M\), where \(M\) represents the depth of the neural network structure, that is, the number of hidden layers. DNNs give the best approximation of the original function by minimizing the loss function.

Thus, for any \(u\) and \(t\), the goal is to make the following equation true:

$$
\frac{\partial}{\partial t}\phi_\theta(u,t)
=
c\frac{\partial}{\partial u}\phi_\theta(u,t)
-\lambda \phi_\theta(u,t)
+\lambda\int_0^u \phi_\theta(u-x,t)p(x)dx.
\tag{4}
$$

For an initialized neural network with initial parameters \(\theta\), since the parameters \(\theta\) are randomly given, Equation \((4)\) may not hold for every \(u\) and \(t\). If that is the case, it results in a data error \(E_\theta(u,t)\):

$$
E_\theta(u,t)
:=
\frac{\partial}{\partial t}\phi_\theta(u,t)
+
\mathcal{N}[u,t,\phi_\theta],
\tag{5}
$$

where \(\mathcal{N}[\cdot]\) is a nonlinear time-dependent differential operator that represents the network function,

$$
\mathcal{N}[u,t,\phi_\theta]
=
-c\frac{\partial}{\partial u}\phi_\theta(u,t)
+\lambda\phi_\theta(u,t)
-\lambda\int_0^u \phi_\theta(u-x,t)p(x)dx.
$$

The aim is to train the neural network to find an optimal parameter \(\theta^\ast\). Hence, the error of all training data points should be minimized under the corresponding \(\phi_{\theta^\ast}\) of a given parameter.

Therefore, we introduce the loss function \(L_\theta(X)\). According to the optimization theory, we know that in the DNN method, the optimal parameter \(\theta=\theta^\ast\) can be obtained by minimizing the loss function \(L_\theta(X)\):

$$
L_\theta(X):=L_\theta^E(X^E)+L_\theta^I(X^I),
$$

$$
\theta^\ast=\arg\min L_\theta(X).
\tag{6}
$$

where \(X\) denotes the collection of the training data. The loss function \(L_\theta\) consists of two parts: the mean square error of the data points within the region and the mean square error of the data points satisfying the initial condition.

### Mean Square Error of the Data Points Within the Region

$$
L_\theta^E(X^E)
:=
\frac{1}{D_E}
\sum_{i=1}^{D_E}
\left|E_\theta(u_i^E,t_i^E)\right|^2,
\tag{7}
$$

in a training data set

$$
X^E:=\{(u_i^E,t_i^E)\}_{i=1}^{N_E}
\subset [0,\infty)\times(0,\infty).
$$

### Mean Squared Error with Respect to the Initial Condition

$$
L_\theta^I(X^I)
:=
\frac{1}{D_I}
\sum_{i=1}^{D_I}
\left|\phi_\theta(u_i^I,t_i^I)-\phi_I(u_i^I)\right|^2,
\tag{8}
$$

in a number of points

$$
X^I:=\{(u_i^I,t_i^I)\}_{i=1}^{N_I}
\subset [0,\infty)\times\{0\},
$$

where \(\phi_\theta\) is the neural network approximation of the solution

$$
\phi:[0,\infty)\times[0,\infty)\to\mathbb{R}.
$$

\(\phi_I\) denotes the initial value function.

Note that the training data \(X\) consists of all data points extracted in the time direction \(t\) and space direction \(u\). In order to minimize the loss function \(L_\theta(X)\) to obtain an approximate solution for Equation \((3)\), DNNs require a further differentiation to evaluate the differential operators \(\partial_t\phi_\theta\) and \(\mathcal{N}[\phi_\theta]\).

Thus, \(L_\theta(X)\) shares the same parameters as the original network \(\phi_\theta\). Both types of derivatives can be easily determined through automatic differentiation with machine learning libraries, such as TensorFlow or PyTorch.

It is worth noting that the convolution term in neural network training makes the calculation extremely complicated. Therefore, in order to solve this problem, we use the general Simpson’s rule to discretize the convolution term on the right side of Equation \((4)\).

Letting

$$
g_\theta(x,t)=\phi_\theta(u-x,t)p(x),
$$

we have the following discrete form:

$$
\int_0^u \phi_\theta(u-x,t)p(x)dx
\approx
\frac{u}{6N_u}
\left[
g_\theta(0,t)
+
4\sum_{k=1}^{N_u}g_\theta((2k-1)\delta_u,t)
+
2\sum_{k=1}^{N_u-1}g_\theta(2k\delta_u,t)
+
g_\theta(u,t)
\right],
\tag{9}
$$

where

$$
N_u=\frac{u}{2\delta_u}
$$

and \(\delta_u>0\) is a given value.

Then, the network function \(\mathcal{N}[u,t,\phi_\theta]\) can be written as

$$
\mathcal{N}[u,t,\phi_\theta]
\approx
-c\frac{\partial}{\partial u}\phi_\theta(u,t)
+\lambda\phi_\theta(u,t)
-\lambda
\frac{u}{6N_u}
\left[
g_\theta(0,t)
+
4\sum_{k=1}^{N_u}g_\theta((2k-1)\delta_u,t)
+
2\sum_{k=1}^{N_u-1}g_\theta(2k\delta_u,t)
+
g_\theta(u,t)
\right].
$$
