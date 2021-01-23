$
\begin{align*}
\frac{\partial Y}{\partial W^{T}} 
&=
\quad
    \begin{bmatrix}
        \begin{matrix}
            \frac{\partial Y}{\partial w_{(m=0,d=0)}} \\
            \frac{\partial Y}{\partial w_{(m=0,d=1)}} \\
            \frac{\partial Y}{\partial w_{(m=0,d=2)}} 
        \end{matrix} 
        &
        \begin{matrix}
            \frac{\partial Y}{\partial w_{(m=1,d=0)}} \\
            \frac{\partial Y}{\partial w_{(m=1,d=1)}} \\
            \frac{\partial Y}{\partial w_{(m=1,d=2)}} 
        \end{matrix}
    \end{bmatrix}
\\
&=
\quad
    \begin{bmatrix}
        \begin{matrix}
            \begin{bmatrix}
                \frac{\partial y_{(n=0,m=0)}}{\partial w_{(m=0,d=0)}} & \frac{\partial y_{(n=0,m=1)}}{\partial w_{(m=0,d=0)}}
            \end{bmatrix}
            \\
            \begin{bmatrix}
                \frac{\partial y_{(n=0,m=0)}}{\partial w_{(m=0,d=1)}} & \frac{\partial y_{(n=0,m=1)}}{\partial w_{(m=0,d=1)}} 
            \end{bmatrix}
            \\
            \begin{bmatrix} 
                \frac{\partial y_{(n=0,m=0)}}{\partial w_{(m=0,d=2)}} & \frac{\partial y_{(n=0,m=1)}}{\partial w_{(m=0,d=2)}} 
            \end{bmatrix}
        \end{matrix}
        &
        \begin{matrix}
            \begin{bmatrix}
                \frac{\partial y_{(n=0,m=0)}}{\partial w_{(m=1,d=0)}} & \frac{\partial y_{(n=0,m=1)}}{\partial w_{(m=1,d=0)}}
            \end{bmatrix}
            \\
            \begin{bmatrix}
                \frac{\partial y_{(n=0,m=0)}}{\partial w_{(m=1,d=1)}} & \frac{\partial y_{(n=0,m=1)}}{\partial w_{(m=1,d=1)}} 
            \end{bmatrix} 
            \\
            \begin{bmatrix} 
                \frac{\partial y_{(n=0,m=0)}}{\partial w_{(m=1,d=2)}} & \frac{\partial y_{(n=0,m=1)}}{\partial w_{(m=1,d=2)}} 
            \end{bmatrix}
        \end{matrix}
    \end{bmatrix}
\\
&=
\quad
    \begin{bmatrix}
        \begin{matrix}
            \begin{bmatrix}
                \frac{\partial y_{(n=0,m=0)}}{\partial w_{(m=0,d=0)}} & 0
            \end{bmatrix}
            \\
            \begin{bmatrix}
                \frac{\partial y_{(n=0,m=0)}}{\partial w_{(m=0,d=1)}} & 0
            \end{bmatrix}
            \\
            \begin{bmatrix} 
                \frac{\partial y_{(n=0,m=0)}}{\partial w_{(m=0,d=2)}} & 0
            \end{bmatrix}
        \end{matrix}
        &
        \begin{matrix}
            \begin{bmatrix}
                0 & \frac{\partial y_{(n=0,m=1)}}{\partial w_{(m=1,d=0)}}
            \end{bmatrix}
            \\
            \begin{bmatrix}
                0 & \frac{\partial y_{(n=0,m=1)}}{\partial w_{(m=1,d=1)}} 
            \end{bmatrix} 
            \\
            \begin{bmatrix} 
                0 & \frac{\partial y_{(n=0,m=1)}}{\partial w_{(m=1,d=2)}} 
            \end{bmatrix}
        \end{matrix}
    \end{bmatrix}
\\
&=
\quad
    \begin{bmatrix}
        \begin{matrix}
            \begin{bmatrix}
            x_{(n=0,d=0)} & 0
            \end{bmatrix}
            \\
            \begin{bmatrix}
            x_{(n=0,d=1)} & 0
            \end{bmatrix}
            \\
            \begin{bmatrix} 
            x_{(n=0,d=2)} & 0
            \end{bmatrix}
        \end{matrix}
        &
        \begin{matrix}
            \begin{bmatrix}
            0 & x_{(n=0,d=0)}
            \end{bmatrix}
            \\
            \begin{bmatrix}
            0 & x_{(n=0,d=1)}
            \end{bmatrix}
            \\
            \begin{bmatrix} 
            0 & x_{(n=0,d=2)}
            \end{bmatrix}
        \end{matrix}
    \end{bmatrix}
\end{align*}
$