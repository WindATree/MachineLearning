from MLcode.label_propagation import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# 定义一个优化后的多层 GNN 模型
class OptimizedGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.5):
        super(OptimizedGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.layers.append(GCNConv(hidden_dim, output_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

def labelPropagation_GNN(X: matCoo, y_label: mat, y_pred: mat, y_res: mat,
                          hidden_dim: int = 32, num_layers: int = 4, max_iter: int = 200, lr: float = 0.005, weight_decay: float = 5e-4):
    # X: 稀疏矩阵，包含样本的相似性数据
    # y_label: 包含已标记样本的标签，未标记样本的标签为 -1
    # y_pred: 预测结果，算法最终输出
    # y_res: 修正后的预测结果
    # hidden_dim: GNN 隐藏层的维度
    # num_layers: GNN 的层数
    # max_iter: 最大迭代次数（训练的 epoch 数）
    # lr: 学习率
    # weight_decay: 权重衰减，用于正则化

    n_samples = X.n  # 样本数量
    n_classes = y_label.m  # 类别数量

    # 构建 PyG 数据对象
    edge_index = torch.tensor([[elem.row, elem.col] for elem in X.elem], dtype=torch.long).t()
    edge_weight = torch.tensor([elem.v for elem in X.elem], dtype=torch.float)

    # 初始化节点特征和标签
    x = torch.eye(n_samples, dtype=torch.float)  # 用单位矩阵表示每个样本的特征
    y = torch.tensor(y_label.v, dtype=torch.float)  # 标签矩阵

    # 为有标签的样本构造 mask
    mask = (y_label.v != -1).any(axis=1)
    train_mask = torch.tensor(mask, dtype=torch.bool)
    train_labels = torch.argmax(y[train_mask], dim=1)

    # 创建 PyG 数据
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)

    # 初始化 GNN 模型
    model = OptimizedGNN(input_dim=n_samples, hidden_dim=hidden_dim, output_dim=n_classes, num_layers=num_layers, dropout=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 训练模型
    model.train()
    for epoch in range(max_iter):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[train_mask], train_labels)  # 使用负对数似然损失函数
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == max_iter - 1:
            print(f"Epoch {epoch}/{max_iter}, Loss: {loss.item():.4f}")

    # 推断未标记样本的标签
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        y_pred_np = out.argmax(dim=1).numpy()

    # 赋值预测结果
    y_pred.createmat(n_samples, 1)  # 创建一个新的矩阵来存储预测结果
    for i in range(n_samples):
        y_pred.v[i, 0] = y_pred_np[i]

    print("Optimized multi-layer GNN-based label propagation complete.")

    # 修正预测结果
    rectify(X, y_label, y_pred, y_res)  # 使用 rectify 函数修正预测结果