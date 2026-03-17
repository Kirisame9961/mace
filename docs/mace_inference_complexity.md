# MACE 推理（能量/力）时间复杂度分析（可溯源版）

> 目标：给出可代入的复杂度表达式，并把每一项都对应到项目中的**具体位置（文件 + 函数/代码段）**。

## 0) 符号与代码变量一一对应

- `N`：节点数（原子数）
- `E`：边数（邻居对）
- `L`：interaction 层数（`num_interactions`）
- `B`：径向基维度（`num_bessel`）
- `d_edge`：边属性（球谐）维度，随 `max_ell` 增长
- `d_tp_w`：TP 权重输出维度（`self.conv_tp.weight_numel`）
- `h_r`：径向 MLP 隐藏宽度（`radial_MLP`）
- `C`：隐藏标量通道规模（与 `hidden_irreps` 中标量通道同量级）
- `nu`：相关阶（`correlation`）

---

## 1) 总体复杂度表达式（energy）

\[
T_{energy}=T_{prep}+T_{edge\_feat}+\sum_{l=1}^{L}(T_{inter}^{(l)}+T_{prod}^{(l)})+T_{readout}
\]

其中每项如下。

### 1.1 预处理与邻接构图 `T_prep`

ASE 调用路径中，`MACECalculator.calculate -> _atoms_to_batch -> AtomicData.from_config -> get_neighborhood` 构建 `edge_index` 与 `shifts`。

- 复杂度记为：\(T_{prep}=O(N+E)\)（邻域搜索具体算法在 `matscipy.neighbour_list`）
- 固定 cutoff 与稳定密度下：`E≈N·k`

### 1.2 边特征 `T_edge_feat`

1) 球谐：`edge_attrs = self.spherical_harmonics(vectors)`，按边计算
\[
T_{sh}=O(E\cdot d_{edge})
\]

2) 径向：`RadialEmbeddingBlock.forward` 中 `cutoff_fn(edge_lengths)` + `bessel_fn(edge_lengths)`，输出维度 `B`
\[
T_{rad}=O(E\cdot B)
\]

因此：
\[
T_{edge\_feat}=O(E\cdot(d_{edge}+B))
\]

### 1.3 单层 interaction `T_inter^(l)`

以 `RealAgnosticInteractionBlock.forward` 为基准：

- `linear_up(node_feats)`：节点线性
- `conv_tp_weights(edge_feats)`：每条边一个 MLP，输入 `B`，输出 `d_tp_w`
- `conv_tp(...)`：按边 TP
- `scatter_sum(...)`：边到点聚合
- `linear(message)` + `skip_tp(message, node_attrs)`：节点侧后处理

写成：
\[
\begin{aligned}
T_{inter}^{(l)}=O(&N\cdot d_{in}^{(l)}d_{up}^{(l)}
+E\cdot \text{MLP}(B,h_r,d_{tp\_w}^{(l)})
+E\cdot \kappa_{tp}^{(l)}
+E \\
&+N\cdot d_{msg}^{(l)}d_{out}^{(l)}
+N\cdot \kappa_{skip}^{(l)})
\end{aligned}
\]

通常主导项在 `E` 相关部分（边 MLP + TP + scatter）。

### 1.4 单层 product `T_prod^(l)`

`EquivariantProductBasisBlock.forward` 调 `self.symmetric_contractions(...)`；而 `SymmetricContraction/Contraction` 内部由多组 `einsum` 图构成，张量阶与 `nu` 直接相关。

抽象为：
\[
T_{prod}^{(l)}=O(N\cdot \kappa_{sc}(C,\nu,l_{max}))
\]

`nu` 增大会显著放大 contraction 常数（通常非线性增长）。

### 1.5 readout `T_readout`

`MACE.forward` 中 `readout(...)` 后 `scatter_sum(...)` 到图级能量，近似：
\[
T_{readout}=O(N\cdot d_{ro})
\]

---

## 2) 合并表达式（可直接引用）

\[
\begin{aligned}
T_{energy}=&\ O(N+E)+O(E(d_{edge}+B)) \\
&+\sum_{l=1}^{L} O\Big(
E[\text{MLP}(B,h_r,d_{tp\_w}^{(l)})+\kappa_{tp}^{(l)}+1]
+N[d_{in}^{(l)}d_{up}^{(l)}+d_{msg}^{(l)}d_{out}^{(l)}+\kappa_{skip}^{(l)}+\kappa_{sc}^{(l)}]
\Big) \\
&+O(N\cdot d_{ro})
\end{aligned}
\]

在 `E~N·k`（`k` 近常数）时，化为
\[
T_{energy}=O\big(N\cdot L\cdot F_{layer}(B,C,l_{max},\nu)\big)
\]
即对 `N` 近线性，但常数因子受 TP 路径、`nu`、irreps 影响很大。

---

## 3) 力 / 应力 / Hessian

### 3.1 Force

`get_outputs -> compute_forces` 通过 `torch.autograd.grad(outputs=[energy], inputs=[positions])` 求导：
\[
T_{force}=T_{energy}+T_{backward}\approx \alpha\,T_{energy}
\]
经验上 `α` 常见 2~4。

### 3.2 Stress/Virials

`get_outputs -> compute_forces_virials` 对 `(positions, displacement)` 双输入求梯度：
\[
T_{force+stress}\sim O(T_{force})\quad(常数更大)
\]

### 3.3 Hessian

`get_outputs -> compute_hessians_vmap` 对 `forces.view(-1)`（约 `3N` 分量）做 VJP：
\[
T_{hessian}\gg T_{force},\ \text{可近似看作 } O(N^2\cdot\text{局部反传成本})
\]

---

## 4) 全量“表达式项 -> 代码位置”映射（文件名+具体段落）

> 以下按表达式中的每一类项列出**可溯源位置**。

### A. `T_prep`（构图/数据准备）

1. 计算入口（ASE）：
   - 文件：`mace/calculators/mace.py`
   - 段落：`MACECalculator.calculate(...)` 中 `batch_base = self._atoms_to_batch(atoms)`
2. 批数据构建：
   - 文件：`mace/calculators/mace.py`
   - 段落：`_atoms_to_batch(...)` 中 `AtomicData.from_config(...)`
3. 邻接构图：
   - 文件：`mace/data/atomic_data.py`
   - 段落：`AtomicData.from_config(...)` 中 `edge_index, shifts, ... = get_neighborhood(...)`
4. 邻域搜索实现：
   - 文件：`mace/data/neighborhood.py`
   - 段落：`get_neighborhood(...)` 中 `neighbour_list(...)` 与输出 `edge_index = np.stack((sender, receiver))`

### B. `T_edge_feat`（球谐+径向）

1. 球谐计算：
   - 文件：`mace/modules/models.py`
   - 段落：`MACE.forward(...)` 中 `edge_attrs = self.spherical_harmonics(vectors)`
2. 径向 embedding 调用：
   - 文件：`mace/modules/models.py`
   - 段落：`edge_feats, cutoff = self.radial_embedding(...)`
3. 径向内部细节：
   - 文件：`mace/modules/blocks.py`
   - 段落：`RadialEmbeddingBlock.forward(...)`
   - 关键代码：`cutoff = self.cutoff_fn(edge_lengths)`、`radial = self.bessel_fn(edge_lengths)`、`return radial * cutoff, ...`

### C. `T_inter^(l)`（每层 interaction）

1. 层循环入口：
   - 文件：`mace/modules/models.py`
   - 段落：`for i, (interaction, product) in enumerate(zip(self.interactions, self.products))`
2. 具体 interaction 实现（典型）：
   - 文件：`mace/modules/blocks.py`
   - 段落：`RealAgnosticInteractionBlock._setup / forward`
3. 对应项映射：
   - `N·d_in·d_up`：`node_feats = self.linear_up(node_feats)`
   - `E·MLP(...)`：`tp_weights = self.conv_tp_weights(edge_feats)`；其输出维度来自 `_setup` 的 `self.conv_tp.weight_numel`
   - `E·κ_tp`：`mji = self.conv_tp(node_feats[edge_index[0]], edge_attrs, tp_weights)`
   - `E` 聚合：`message = scatter_sum(src=mji, index=edge_index[1], ...)`
   - 节点后处理：`message = self.linear(message)`、`message = self.skip_tp(message, node_attrs)`

### D. `T_prod^(l)`（product/symmetric contraction）

1. product 调用：
   - 文件：`mace/modules/models.py`
   - 段落：循环内 `node_feats = product(node_feats=node_feats, sc=sc, node_attrs=node_attrs_slice)`
2. product block：
   - 文件：`mace/modules/blocks.py`
   - 段落：`EquivariantProductBasisBlock.forward(...)` 中 `self.symmetric_contractions(...)`
3. contraction 内部（`nu` 来源）：
   - 文件：`mace/modules/symmetric_contraction.py`
   - 段落：`Contraction.__init__(..., correlation: int, ...)` 与 `for i in range(correlation, 0, -1)`
   - 计算核心：多处 `torch.einsum(...)` 图优化与执行（`graph_opt_main`, `contractions_weighting`, `contractions_features`）

### E. `T_readout` 与能量聚合

1. readout：
   - 文件：`mace/modules/models.py`
   - 段落：`for i, readout in enumerate(self.readouts)` 与 `node_es = readout(...)`
2. 聚合：
   - 文件：`mace/modules/models.py`
   - 段落：`energy = scatter_sum(node_es, data["batch"], ...)`、`total_energy = torch.sum(contributions, dim=-1)`

### F. `T_force / T_force+stress / T_hessian`

1. 从 forward 进入导数路径：
   - 文件：`mace/modules/models.py`
   - 段落：`forces, virials, stress, hessian, edge_forces = get_outputs(...)`
2. force：
   - 文件：`mace/modules/utils.py`
   - 段落：`compute_forces(...)` 的 `torch.autograd.grad(outputs=[energy], inputs=[positions])`
3. force+stress/virials：
   - 文件：`mace/modules/utils.py`
   - 段落：`compute_forces_virials(...)` 的 `inputs=[positions, displacement]`
4. hessian：
   - 文件：`mace/modules/utils.py`
   - 段落：`compute_hessians_vmap(...)`（`forces.view(-1)` + `torch.vmap(get_vjp, ...)`）以及回退 `compute_hessians_loop(...)`

---

## 5) 实务结论（针对性能分析）

1. 固定 cutoff 下，`E~N·k`，energy 近似线性于 `N`。
2. 真正决定常数的是：`L`、TP 路径（`κ_tp`）、`nu` 导致的 contraction 开销（`κ_sc`）。
3. force 通常是 energy 的常数倍；hessian 成本远高于 force。
4. 优化优先级通常是：减少 `L`、降低 `nu`、约束 irreps/TP 路径、降低平均邻居数 `k`。
