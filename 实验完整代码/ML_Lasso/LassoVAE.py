import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scanpy as sc
import random
import torch
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import pickle

class VAE(nn.Module):
    def __init__(self, input_size, output_size, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, h_dim)
        self.batch_norm1 = torch.nn.BatchNorm1d(h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.batch_norm2 = torch.nn.BatchNorm1d(h_dim)
        self.fc5 = nn.Linear(h_dim, output_size)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = self.batch_norm1(h)
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        h = self.batch_norm2(h)
        return torch.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var


def Gen_TrainSet(h5adFile='D:/desktop/ML_Lasso/data/dataset1.h5ad'):
    adata = sc.read_h5ad(h5adFile)
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor='seurat', inplace=True)
    sc.pp.pca(adata, n_comps=50, use_highly_variable=True, svd_solver='arpack')
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    candidates = pd.DataFrame(index=adata.obs_names)
    print("For small resolutions")
    can_num = 0
    for res in range(1, 10, 2):
        lores = res * 0.1
        sc.tl.leiden(adata, key_added='clusters', resolution=lores)
        tmp_can = np.array(list(map(lambda x: np.array(adata.obs['clusters'] == x), adata.obs['clusters'].unique()))).T
        tmp_can = pd.DataFrame(tmp_can, index=adata.obs_names)
        candidates = pd.concat([candidates, tmp_can], axis=1, ignore_index=True)
        uni_num = len(adata.obs['clusters'].unique())
        print("unique candidates:{}".format(uni_num))
        can_num += uni_num
    print("For large resolutions")
    for res in [1, 2, 5, 10]:
        sc.tl.leiden(adata, key_added='clusters', resolution=res)
        tmp_can = np.array(list(map(lambda x: np.array(adata.obs['clusters'] == x), adata.obs['clusters'].unique()))).T
        tmp_can = pd.DataFrame(tmp_can, index=adata.obs_names)
        candidates = pd.concat([candidates, tmp_can], axis=1, ignore_index=True)
        uni_num = len(adata.obs['clusters'].unique())
        print("unique candidates:{}".format(uni_num))
        can_num += uni_num
    c = list(candidates.columns)
    random.shuffle(c)
    candidates = candidates.reindex(columns=c)
    print("All candidates Number: {}".format(can_num))

    # 计算细胞之间的余弦相似度，得到一个 (3000, 3000) 的矩阵
    similarity_matrix = cosine_similarity(candidates)

    # 将相似度矩阵转换为稀疏矩阵（CSR格式）
    sparse_matrix = csr_matrix(similarity_matrix)

    # 将稀疏矩阵保存为文件
    with open("adjacency_matrix.pkl", "wb") as f:
        pickle.dump(sparse_matrix, f)

    print("邻接矩阵已保存为稀疏矩阵格式。")
    return adata, candidates


def Gen_maskSet(candidate: pd.DataFrame, errRate=0.20):
    sele_can = candidate[candidate == True]
    cell_len = len(sele_can)
    mask_can = candidate.copy()
    errArray = random.sample(range(cell_len), int(cell_len * errRate))
    for cell in errArray:
        mask_can.loc[sele_can.index[cell]] = ~mask_can.loc[sele_can.index[cell]]
    return mask_can


def train_VAE(candidates, h_dim=400, z_dim=40, num_epochs=5, learning_rate=1e-3, batch_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VAE(input_size=len(candidates), output_size=len(candidates), h_dim=h_dim, z_dim=z_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        kl_divs = []
        mask_x = torch.from_numpy(
            np.array([Gen_maskSet(candidates.iloc[:, i], errRate=0.05) for i in range(len(candidates.columns))],
                     dtype=np.float32))
        origin_x = torch.from_numpy(np.array(candidates, dtype=np.float32).T)
        mask_x = mask_x.to(device)

        reconst_x, mu, log_var = model(mask_x)
        reconst_loss = F.binary_cross_entropy(reconst_x, origin_x, reduction='mean')
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        kl_divs.append(kl_div)

        print("Epoch[{}/{}] Reconst Loss: {}, KL Div: {}"
              .format(epoch, num_epochs, reconst_loss.item(), kl_div.item()))

    with torch.no_grad():
        test_x = torch.from_numpy(
            np.array([Gen_maskSet(candidates.iloc[:, i], errRate=0.05) for i in range(len(candidates.columns))],
                     dtype=np.float32))
        out, _, _ = model(test_x)
        npout = out.cpu().numpy().T
        npout = pd.DataFrame(npout)

        prob = candidates.sum(axis=0) / len(candidates)
        for i in range(len(npout.columns)):
            sorted_arr = np.sort(np.array(npout[i]))
            index = int((len(sorted_arr) - 1) * (1 - prob.iloc[i])) + 1
            flag = sorted_arr[index]
            npout[i] = npout[i].apply(lambda x: 1 if x >= flag else 0)

        # 根据用户输入的细胞索引返回预测的相似细胞
        def find_similar_cells(user_indices, npout, threshold=0.8):
            npout=np.array(npout)
            # 提取user_indices对应的特征
            user_features = npout[user_indices]

            # 计算所有细胞样本与输入细胞样本之间的余弦相似度
            similarities = cosine_similarity(npout, user_features)

            # 计算每个细胞与输入细胞的最大相似度
            max_similarities = np.max(similarities, axis=1)

            # 找出相似度超过阈值的细胞索引
            similar_cells = np.where(max_similarities >= threshold)[0]

            return similar_cells

        return find_similar_cells(user_indices=[23, 36, 60, 113, 136, 237, 245, 326, 369, 471], npout=npout)


def do():
    adata, candidates = Gen_TrainSet()
    similar_cells = train_VAE(candidates)
    print(f"预测相似细胞索引为：{similar_cells}")


if __name__ == '__main__':
    do()
