from label_propagation import *  # 导入标签传播模块
import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 神经网络模块
import torch.optim as optim  # 导入优化器模块
import numpy as np  # 导入 NumPy 库，用于处理数组和矩阵
import scipy.sparse as sp  # 导入稀疏矩阵模块（此处未使用，但可能用于其他操作）

# 定义 GCN 层，用于图卷积神经网络（Graph Convolutional Network）
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        # 初始化 GCN 层，输入和输出特征维度
        self.fc = nn.Linear(in_features, out_features)  # 使用线性层进行特征变换

    def forward(self, adj_matrix, features):
        # 前向传播函数：计算图卷积
        # 将邻接矩阵与特征矩阵相乘，再通过线性层转换特征
        return self.fc(torch.matmul(adj_matrix, features))


# 定义 GAN 标签传播模型，结合生成对抗网络（GAN）与标签传播
class GANLabelPropagation(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GANLabelPropagation, self).__init__()
        # 初始化模型结构
        # 定义两层图卷积神经网络
        self.gcn1 = GCNLayer(input_dim, hidden_dim)  # 第一层：从输入到隐层
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)  # 第二层：从隐层到输出层
        self.gcn2 = GCNLayer(hidden_dim, output_dim)  # 第三层：从隐层到输出层
        self.criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数

        # 定义 GAN 中的判别器部分
        self.discriminator = nn.Sequential(
            nn.Linear(output_dim, 64),  # 第一层全连接
            nn.ReLU(),  # 激活函数
            nn.Linear(64, 128),  # 第二层全连接
            nn.ReLU(),  # 激活函数
            nn.Linear(128, 1),  # 第三层全连接
            nn.Sigmoid()  # Sigmoid 激活函数，输出为概率
        )

    def forward(self, adj_matrix, features):
        # 前向传播函数：首先通过图卷积层进行标签传播
        x = torch.relu(self.gcn1(adj_matrix, features))  # 经过第一层 GCN 后进行 ReLU 激活
        x = self.gcn2(adj_matrix, x)  # 经过第二层 GCN
        return x  # 返回最终的输出（预测的标签分布）

    def discriminator_loss(self, y_real, y_fake):
        # 判别器的损失函数，计算真实样本与伪造样本的二元交叉熵损失
        real_loss = torch.mean(torch.log(y_real))  # 真实样本的损失
        fake_loss = torch.mean(torch.log(1 - y_fake))  # 伪造样本的损失
        return -(real_loss + fake_loss)  # 返回总的判别器损失

    def generator_loss(self, y_fake):
        # 生成器的损失函数，计算生成器生成伪造样本的损失
        return -torch.mean(torch.log(y_fake))  # 生成器的损失

    def label_propagation(self, adj_matrix, y_label, epochs=10, lr=0.01, device='cuda'):
        # 标签传播过程：通过优化模型使得标签得以传播
        optimizer = optim.Adam(self.parameters(), lr=lr)  # 使用 Adam 优化器

        # 将标签数据转化为张量，并处理缺失标签（例如，-1 表示缺失标签）
        mask = (y_label != -1).astype(np.float32)  # 创建掩码，标记已知标签的位置
        y_label_tensor = torch.tensor(y_label, dtype=torch.float32).to(device)  # 标签转为张量
        y_label_tensor = y_label_tensor * torch.tensor(mask, dtype=torch.float32).to(device)  # 根据掩码处理标签

        # 初始化特征矩阵，这里使用单位矩阵作为初始特征（可以根据需要修改）
        features = torch.eye(adj_matrix.shape[0]).to(device)

        # 训练过程
        for epoch in range(epochs):
            self.train()  # 设置模型为训练模式
            optimizer.zero_grad()  # 清零梯度

            # 前向传播
            outputs = self(adj_matrix.to(device), features)  # 计算模型输出

            # 计算损失
            # 计算已知标签的损失（标签传播损失）
            labeled_loss = self.criterion(outputs[mask == 1], y_label_tensor[mask == 1])

            # 更新 GAN 判别器
            y_real = self.discriminator(outputs)  # 判别器判断真实输出
            y_fake = self.discriminator(outputs.detach())  # 判别器判断伪造输出（不更新梯度）
            d_loss = self.discriminator_loss(y_real, y_fake)  # 判别器损失
            g_loss = self.generator_loss(y_fake)  # 生成器损失

            # 总损失：标签传播损失 + GAN 损失
            total_loss = labeled_loss + d_loss + g_loss

            # 反向传播
            total_loss.backward()
            optimizer.step()  # 更新模型参数

            # 每10轮输出一次损失
            if epoch % 10 == 0:
                print(f'Epoch {epoch}/{epochs}, Loss: {total_loss.item()}')

        # 返回预测结果和精炼后的标签
        with torch.no_grad():
            self.eval()  # 设置模型为评估模式
            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()  # 获取预测的标签
            y_res = torch.argmax(outputs, dim=1).cpu().numpy()  # 获取精炼后的标签

        return y_pred, y_res  # 返回预测结果和精炼后的标签


# 标签传播的包装函数
def labelPropagation(dense_matrix, y_label, epochs=100, lr=0.01, device='cuda'):
    # 将输入的稠密矩阵转换为张量，并移至指定设备（如 GPU）
    adj_matrix = torch.tensor(dense_matrix, dtype=torch.float32).to(device)
    input_dim = adj_matrix.shape[0]  # 获取输入特征的维度（即矩阵的行数）

    # 创建 GAN 标签传播模型
    model = GANLabelPropagation(input_dim=input_dim, hidden_dim=64, output_dim=y_label.shape[1]).to(device)

    # 运行标签传播过程
    y_pred, y_res = model.label_propagation(adj_matrix, y_label, epochs=epochs, lr=lr, device=device)
    return y_pred, y_res  # 返回预测标签和精炼后的标签
