The general flow when get a new depth map 
```mermaid
graph LR 
A[estimate W for all nodes] --> B[fusion all voxel] --> C[update, insert nodes]
```

### Estimate 
Input: Depth map $D_{t}$ 
Process: 
For each $u \in \mathbb{R}^2 = \Omega$ (pixel space)
$$
\operatorname{Data}\left(\mathcal{W}, \mathcal{V}, D_{t}\right) \equiv \sum_{u \in \Omega} \psi_{\text {data }}\left(\hat{\mathbf{n}}_{u}^{\top}\left(\hat{\mathbf{v}}_{u}-\mathbf{v l}_{\tilde{u}}\right)\right)
$$
where
$$
\mathbf{v}(u) \in \mathbb{R}^3 \\ 
\tilde{\mathbf{T}}^{u}=\mathcal{W}(\mathbf{v}(u)) \in \mathbb{R}^{4 \times 4} = \textbf{SE}(3) \\ 
\hat{\mathbf{v}}_{u}=\tilde{\mathbf{T}}^{u} \mathbf{v}(u) \in \mathbb{R}^3 \\ 
\hat{\mathbf{n}}_{u}=\tilde{\mathbf{T}}^{u} \mathbf{n}(u) \in \mathbb{R}^3 \\ 
\tilde{u}=\pi\left(\mathbf{K} \hat{\mathbf{v}}_{u}\right) \in \mathbb{R}^2 \\ 
\left[\mathbf{v l}(\tilde{u})^{\top}, 1\right]^{\top}=\mathbf{K}^{-1} D_{t}(\tilde{u})\left[\tilde{u}^{\top}, 1\right]^{\top} \in \mathbb{R}^4 \\ 
\mathbf{v l}(\tilde{u}) \in \mathbb{R}^3 \\ 
\psi_{\text {data }}() \text{:  Tukey penalty function}
$$ 

(Tukey loss [What is the Tukey loss function? | R-bloggers](https://www.r-bloggers.com/2021/04/what-is-the-tukey-loss-function/))
Note that we need W here, with input $x_c = \mathbf{v}(u)$, and a set of center nodes ($n$ deformation nodes) $\mathcal{N}_{\mathbf{w a r p}}^{t}=\left\{\mathbf{d g}_{v}, \mathbf{d g}_{w}, \mathbf{d g}_{s e 3}\right\}_{t}$ where each node $i$ has a position in the canonical frame $\mathbf{d g}_{v}^{i} \in$ $\mathbb{R}^{3}$, its associated transformation $\hat{\mathbf{q}_{i c}} = \operatorname{get\_quaternion}(\mathbf{T}_{i c}=\mathbf{d g}_{s e 3}^{i})$ , and a radial basis weight $\mathbf{d g}_{w}$ that controls the extent of the transformation $\mathbf{w}_{i}\left(x_{c}\right)=\exp \left(-\left\|\mathbf{d g}_{v}^{i}-x_{c}\right\|^{2} /\left(2\left(\mathbf{d g}_{w}^{i}\right)^{2}\right)\right)$. Each radius parameter $\mathbf{d g}_{w}^{i}$ is set to ensure the node's influence overlaps with neighbouring nodes, dependent on the sampling sparsity of nodes = which means it is set manually. 
note that $\left[\mathbf{v l}(\tilde{u})^{\top}, 1\right]^{\top}=\mathbf{K}^{-1} D_{t}(\tilde{u})\left[\tilde{u}^{\top}, 1\right]^{\top} \in \mathbb{R}^4$ should be $\mathbf{v l}(\tilde{u})^{\top}=\mathbf{K}^{-1} D_{t}(\tilde{u})\left[\tilde{u}^{\top}, 1\right]^{\top} \in \mathbb{R}^3$ because the shape doesn't match, and later the author also use it is R^3, not R^4.

The formula of W: 
$$
\mathcal{W}_{t}\left(x_{c}\right)=\mathbf{T}_{l w} S E 3\left(\mathbf{D Q B}\left(x_{c}\right)\right) \\ 
\mathbf{T}_{l w} \in \textbf{SE}(3) = ??? \text{ (explicit warped model to live camera transform)} \\ 
S E 3(\cdot)\text{ (get transformation matrix from quaternion)} \\ 
\mathbf{D Q B}\left(x_{c}\right) \equiv \frac{\sum_{k \in N\left(x_{c}\right)} \mathbf{w}_{k}\left(x_{c}\right) \hat{\mathbf{q}}_{k c}}{\left\|\sum_{k \in N\left(x_{c}\right)} \mathbf{w}_{k}\left(x_{c}\right) \hat{\mathbf{q}}_{k c}\right\|} \\ 
$$
### Fusion 
The input is voxel $\mathbf{x} \in \mathrm{S} \subset \mathbb{N}$, 
we then compute $x_c = \operatorname{dc}(\mathbf{x}) \in \textbf{S} \in \mathbb{R}^3$  and psdf value 
$$
\operatorname{psdf}\left(x_{c}\right)=\left[\mathbf{K}^{-1} D_{t}\left(u_{c}\right)\left[u_{c}^{\top}, 1\right]^{\top}\right]_{z}-\left[x_{t}\right]_{z},
$$
where 
$$
\left(x_{t}^{\top}, 1\right)^{\top}=\mathcal{W}_{t}\left(x_{c}\right)\left(x_{c}^{\top}, 1\right)^{\top} \\ 
u_{c}=\pi\left(\mathbf{K} x_{t}\right) \\ 
D_{t}: \mathbb{R^2}  \mapsto \mathbb{R} \\ 
\mathbf{K} \in \mathbb{R}^{3 \times 3} \text{camera intrinsic matrix} \\ 
{[]}_z \text{get z component} \\
$$
After this, we check the threshold and update 

$$
\mathcal{V} (\mathbf{x})_{t}= \begin{cases}{\left[\mathrm{v}^{\prime}(\mathbf{x}), \mathrm{w}^{\prime}(\mathbf{x})\right]^{\top},} & \text { if } \operatorname{psdf}(\operatorname{dc}(\mathbf{x}))>-\tau \\ \mathcal{V}(\mathbf{x})_{t-1}, & \text { otherwise }\end{cases}
$$
where 
$$
\begin{aligned}
\mathrm{v}^{\prime}(\mathbf{x}) &=\frac{\mathrm{v}(\mathbf{x})_{t-1} \mathrm{w}(\mathbf{x})_{t-1}+\min (\rho, \tau) w(\mathbf{x})}{\mathrm{w}(\mathbf{x})_{t-1}+w(\mathbf{x})} \\
\rho &=\operatorname{psdf}(\mathbf{d c}(\mathbf{x})) \\
\mathrm{w}^{\prime}(\mathbf{x}) &=\min \left(\mathrm{w}(\mathbf{x})_{t-1}+w(\mathbf{x}), w_{\max }\right)
\end{aligned}
$$
note 
$$
w(\mathbf{x}) = ???? \propto \frac{1}{k} \sum_{i \in N\left(x_{c}\right)}\left\|\mathbf{d g}_{w}^{i}-x_{c}\right\|_{2}
$$
Làm sao mà radius weight R đem đi trừ R3 được, này author ghi sai tiếp hay gì? 

### After fusion, we update new node
After Fusion, we extract the canonical frame as mesh $\hat{\mathcal{V}}_c$ = vertices + faces 
That is one input, and another one is the set of nodes $\mathcal{N}_{\text{warp}}$ 
Now, we compute the distance from each vertex $v_c \in \hat{\mathcal{V}}_c$ to its supporting nodes, and when 
$$
\min _{k \in N\left(v_{c}\right)}\left(\frac{\left\|\mathbf{dg}_{v}^{k}-v_{c}\right\|}{\mathbf{d g}_{w}^{k}}\right) \geq 1
$$
we conclude $v_c$ is unsupported. Call set $\textbf{V}_c = \{\text{unsupported } v_c\}$. We sample a sample set $\tilde{\textbf{V}}$ by using radius search averaging. 
- The position $\tilde{\mathbf{d g}}_{v}$ of that set = {$x_c$, distance from each other is at least $\epsilon$}
- Each node center  $\mathbf{d g}_{v}^{*} \in \tilde{\mathbf{d g}}_{v}$, the transformation $\mathbf{dg}_{s e 3}^{*} \leftarrow \mathcal{W}_{t}\left(\mathbf{dg}_{v}^{*}\right)$. This creates the set of transformation matrices $\tilde{\mathbf{d g}}_{s e 3}$.
- I don't know how to compute $\tilde{\mathbf{d g}}_{w}$ though, paper said nothing.


Then the set of node at frame $t$ is:
$\mathcal{N}_{\text {warp }}^{t}=\mathcal{N}_{\text {warp }}^{t-1} \cup\left\{\tilde{\mathbf{d g}}_{v}, \tilde{\mathbf{d g}}_{s e 3}, \tilde{\mathbf{d g}}_{w}\right\}$