from label_propagation import *  # 导入标签传播模块
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# 处理邻接矩阵
def compute_adjacency_matrix(dense_matrix, k=10):
    """ 通过K近邻法得到图的邻接矩阵 """
    adjacency_matrix = kneighbors_graph(dense_matrix, n_neighbors=k, mode='connectivity', include_self=True)
    return torch.tensor(adjacency_matrix.toarray(), dtype=torch.float32)

# 插补稀疏矩阵
def impute_missing_values(dense_matrix):
    """ 插补稀疏矩阵中的缺失值 """
    imputer = SimpleImputer(strategy='mean')
    dense_matrix_imputed = imputer.fit_transform(dense_matrix)
    return dense_matrix_imputed

# 归一化处理
def normalize_adj(adj_matrix):
    """ 对邻接矩阵进行归一化处理 """
    D = torch.diag(torch.pow(adj_matrix.sum(1), -0.5))
    return torch.matmul(torch.matmul(D, adj_matrix), D)

# 基于PCA降维
def pca_reduce_features(dense_matrix, n_components=50):
    """ 使用PCA进行降维 """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(dense_matrix)


class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        self.weight = nn.Parameter(torch.rand(in_features, out_features))
        self.bias = nn.Parameter(torch.rand(out_features))

    def forward(self, adj, x):
        """ 基本的图卷积操作: x' = D^(-1/2) * A * D^(-1/2) * x """
        support = torch.matmul(x, self.weight)  # xW
        output = torch.matmul(adj, support)  # A * xW
        return output + self.bias


class LabelPropagationGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LabelPropagationGCN, self).__init__()
        self.gc1 = GraphConvolutionLayer(input_dim, hidden_dim)
        self.gc2 = GraphConvolutionLayer(hidden_dim, output_dim)

    def forward(self, adj, x):
        x = F.relu(self.gc1(adj, x))
        x = self.gc2(adj, x)
        return F.log_softmax(x, dim=-1)


def label_propagation(trueLabel, dense_matrix, y_label, num_classes=11, epochs=200, learning_rate=0.01):
    # 数据预处理
    dense_matrix_imputed = impute_missing_values(dense_matrix)
    adj = compute_adjacency_matrix(dense_matrix_imputed)
    adj = normalize_adj(torch.tensor(adj, dtype=torch.float32))

    # 获取训练数据和标签
    labels = torch.tensor(y_label, dtype=torch.float32)  # 转为torch tensor
    true_labels = torch.tensor(trueLabel, dtype=torch.long)  # 真实标签
    mask = (labels == -1)  # 识别缺失的标签，mask为True表示缺失

    # 使用PCA降维
    dense_matrix_reduced = pca_reduce_features(dense_matrix_imputed, n_components=100)
    dense_matrix_reduced = torch.tensor(dense_matrix_reduced, dtype=torch.float32)

    # 模型定义
    model = LabelPropagationGCN(input_dim=dense_matrix_reduced.shape[1], hidden_dim=256, output_dim=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 在训练时，使用已知标签并将未知标签部分mask掉
        output = model(adj, dense_matrix_reduced)

        # 计算损失：交叉熵损失，只计算已知标签部分
        # 确保索引的维度是正确的，这里我们用mask去选择出未标记的样本
        loss = F.cross_entropy(output[~mask], true_labels[~mask])

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    # 推断阶段：根据标签传播得到最终类别
    model.eval()
    with torch.no_grad():
        output = model(adj, dense_matrix_reduced)
        _, predicted = torch.max(output, dim=1)
        return predicted.cpu().numpy()

# # 测试函数
# trueLabel = np.random.randint(0, 11, 3000)  # 假设的真实标签
# dense_matrix = np.random.rand(3000, 100)  # 假设的表达矩阵
# y_label = np.random.choice([-1, 0, 1, 2], size=(3000, 11))  # 假设的已知标签（部分为-1）
#
# predicted_labels = label_propagation(trueLabel, dense_matrix, y_label)
# print(predicted_labels)
