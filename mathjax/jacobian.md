**Derivative Matrix**

If *f* : $\Bbb R^D \to \Bbb R^M$ is a differentiable function, then the derivative is an  *M* x *D* matrix given by 

$$\mathbf J f = \begin{bmatrix} \frac{\partial f_0}{\partial w_0} & \frac{\partial f_0}{\partial w_1} & \cdots & \frac{\partial f_0}{\partial w_{D-1}}\\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial w_0} & \frac{\partial f_m}{\partial w_2} & \cdots & \frac{\partial f_m}{\partial w_{D-1}}\end{bmatrix}$$ 

where $ f(w_0, w_2, ... , w_{D-1}) = (f_0(w_0, ... w_{D-1}), f_1(w_0, ..., w_{D-1})..., f_{M-1}(w_0,...,w_{D-1}))$

**Matrix Version of Chain Rule**

If *f* : $\Bbb R^m \to  \Bbb R^p $ and *g* : $\Bbb R^n \to \Bbb R^m$ are differentiable functions and the composition *f* $\circ$ *g* is defined then $$\mathbf D(f \circ g) = \mathbf Df \mathbf Dg$$  

Note: (*f* $\circ$ *g*)$(w_0, w_2,...,w_{D-1})$ = *f* [*g* ($w_0,w_2,...,w_{D-1})]$


$
\begin{align*}
Jf(Y) &=
\begin{bmatrix} 
\frac{\partial f_0}{\partial y_0}&\cdots&\frac{\partial f_0}{\partial y_{M-1}}\\ 
\vdots&&\vdots
\\
\frac{\partial f_{M-1}}{\partial y_0} & \cdots & \frac{\partial f_{M-1}}{\partial y_{M-1}}
\end{bmatrix}
\\
Jg(W^{T})&=\begin{bmatrix} 
\frac{\partial g_0}{\partial w_0}&\cdots&\frac{\partial g_0}{\partial w_{D-1}}\\ 
\vdots&&\vdots\\ 
\frac{\partial g_{N-1}}{\partial w_0}&\cdots&\frac{\partial g_{N-1}}{\partial w_{D-1}} 
\end{bmatrix}
\end{align*}
$