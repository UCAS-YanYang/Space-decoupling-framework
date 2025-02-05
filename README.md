## Space-decoupling framework: optimization on $\mathcal{M}_h$



This is the code to reproduce the experiments in the following paper:

> *A space-decoupling framework for optimization on bounded-rank matrices with orthogonally invariant constraints*
>
> Yan Yang, Bin Gao, and Ya-xiang Yuan
>
> <https://arxiv.org/abs/2501.13830>



We aim to tackle the low-rank optimization problem with additional constraints $X\in\mathcal{H}:=\left\\{X:h(X)=0\right\\}$,

```math
\mathop{\mathrm{min}}\limits_{X\in\mathbb{R}^{m\times n}} \ \ f(X)\ \ \mathrm{s.\,t.}\ \mathrm{rank}(X)\le r,\ \ h(X)=0.
```



We turn to the reformulated Riemannian optimization problem as follows,

```math
\mathop{\mathrm{min}}\limits_{x\in\mathcal{M}_h} \ \ \bar{f}(x):= f\circ \phi(x),
```

where the proposed $(\mathcal{M}_h,\phi)$ parameterizes $\mathbb{R}^{m\times n}\cap\mathcal{H}$ in the sense that $\phi(\mathcal{M}_h)=\mathbb{R}^{m\times n}\cap\mathcal{H}$; more details are referred to the article.



### Dependencies

+ Matlab (Release 9.7.0)

  +  [Manopt (8.0)](https://www.manopt.org/)

+ Python (Release 3.8.10)

  + gym 0.26
  + numpy 1.24
  + PyTorch 1.13
  + wandb 0.19  (We rely on "Weights & Biases" for visualization and monitoring)
  
  

### Get Started

#### Spherical data fitting

```math
\mathop{\mathrm{min}}\limits_{x\in\mathcal{M}_h} \ \ \ \frac{1}{2} \| P_{\Omega} (\phi(x)-A) \|_{\mathrm{F}}^2
```

+ Run `Test_fitting_exact.m` for testing with the exact rank parameter
+ Run `Test_fitting_overestimate.m` for testing with the over-estimated rank parameter



#### Low-rank reinforcement learning

```math
\mathop{\mathrm{min}}\limits_{x\in\mathcal{M}_h} \ \ \ -J(\pi(\phi(x)))
```

+ Run `Test_RL_pendulum.py` to evaluate the algorithm in the "pendulum" environment
+ Run `Test_RL_car.py` to evaluate the algorithm in the "mountain car" environment



### Authors

- Yan Yang (AMSS, China)



### Copyright

Copyright (C) 2025, Yan Yang, Bin Gao, Ya-xiang Yuan.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/