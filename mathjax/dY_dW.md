$
\begin{align*}
\text{--------------------------------------------------------------------------------} \\
\text{dy/dw} \\
\text{--------------------------------------------------------------------------------} \\
\text{This is dY00/w00} \\
\frac{\partial y_{(n=0,m=0)}}{\partial w_{(m=0,d=0)}} \rightarrow x_{(n=0,d=0)} \quad
\frac{\partial y_{(n=0,m=1)}}{\partial w_{(m=0,d=0)}} \rightarrow 0 \\
\\
\text{This is dY00/w01} \\
\frac{\partial y_{(n=0,m=0)}}{\partial w_{(m=0,d=1)}} \rightarrow x_{(n=0,d=1)} \quad
\frac{\partial y_{(n=0,m=1)}}{\partial w_{(m=0,d=1)}} \rightarrow 0
\\
\text{This is dY00/w02} \\
\frac{\partial y_{(n=0,m=0)}}{\partial w_{(m=0,d=2)}} \rightarrow x_{(n=0,d=2)} \quad
\frac{\partial y_{(n=0,m=1)}}{\partial w_{(m=0,d=2)}} \rightarrow 0
\\
\text{This is dY01/w00} \\
\frac{\partial y_{(n=0,m=0)}}{\partial w_{(m=1,d=0)}} \rightarrow 0 \quad
\frac{\partial y_{(n=0,m=1)}}{\partial w_{(m=1,d=0)}} \rightarrow x_{(n=0,d=0)} \\
\\
\text{This is dY01/w01} \\
\frac{\partial y_{(n=0,m=0)}}{\partial w_{(m=1,d=1)}} \rightarrow 0 \quad
\frac{\partial y_{(n=0,m=1)}}{\partial w_{(m=1,d=1)}} \rightarrow x_{(n=0,d=1)}
\\
\text{This is dY01/w02} \\
\frac{\partial y_{(n=0,m=0)}}{\partial w_{(m=1,d=2)}} \rightarrow 0 \quad
\frac{\partial y_{(n=0,m=1)}}{\partial w_{(m=1,d=2)}} \rightarrow x_{(n=0,d=2)}
\\
\text{--------------------------------------------------------------------------------} \\
\text{dy/dw} as matrix \\
\text{--------------------------------------------------------------------------------} \\
\begin{bmatrix}
\begin{bmatrix}
\frac{\partial y_{(n=0,m=0)}}{\partial w_{(m=0,d=0)}} & \frac{\partial y_{(n=0,m=1)}}{\partial w_{(m=0,d=0)}}
\end{bmatrix} \\
\begin{bmatrix}
\frac{\partial y_{(n=0,m=0)}}{\partial w_{(m=0,d=1)}} & \frac{\partial y_{(n=0,m=1)}}{\partial w_{(m=0,d=1)}} \\
\end{bmatrix} \\
\begin{bmatrix} 
\frac{\partial y_{(n=0,m=0)}}{\partial w_{(m=0,d=2)}} & \frac{\partial y_{(n=0,m=1)}}{\partial w_{(m=0,d=2)}} \\
\end{bmatrix}
\end{bmatrix}
\end{align*}
$