# 评估GAN算法的鲁棒性
import h5py  # 用于读取HDF5格式的文件
import scipy.sparse  # 用于稀疏矩阵的处理
import time  # 用于计算程序运行时间
import random  # 用于生成随机数
import numpy as np  # 用于数值计算
import pandas as pd  # 用于数据操作和处理
from scipy.sparse import csr_matrix

import GAN as GAN  # TODO: 这里需要引入你封装好的GAN函数 (标签传播算法)
# import pairpotlpa as GAN
from sklearn.metrics.cluster import adjusted_rand_score  # 导入ARI（调整兰德指数）用于评估聚类效果
from scipy.sparse import coo_matrix

import scanpy as sc  # 用于单细胞RNA-seq数据分析的包

# 创建一个空的DataFrame，保存结果数据
df = pd.DataFrame(columns=['MR', 'ARI_o', 'ARI_r', 'Time'])

# 记录开始时间
start = time.time()

# 读取HDF5文件 'dataset1.h5ad'
with h5py.File('../data/dataset1.h5ad', 'r') as f:
    # 从文件中获取连接矩阵的数据
    group = f['obsp']['connectivities']

    # 获取矩阵的数据、索引和指针
    data = group['data'][:]
    indices = group['indices'][:]
    indptr = group['indptr'][:]

    # 获取矩阵的形状信息
    shape = (f['obsp']['connectivities'].attrs['shape'][0], f['obsp']['connectivities'].attrs['shape'][1])

    # 构建稀疏矩阵（CSR格式）
    mat = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)

# 转换为COO格式（坐标格式），以便后续处理
coo = mat.tocoo()

# 获取稀疏矩阵的行、列索引和数据
rows = coo.row
cols = coo.col
data = coo.data
csr = csr_matrix((data, (rows, cols)), shape=(rows.shape[0], cols.shape[0]))

# 创建一个稀疏矩阵
sparse_matrix = coo_matrix((data, (rows, cols)))

# 将稀疏矩阵转换为稠密矩阵
dense_matrix = sparse_matrix.toarray()  # 或者使用 sparse_matrix.todense()
# 再次读取HDF5文件，获取观察数据
with h5py.File('../data/dataset1.h5ad', 'r') as h5file:
    obs_group = h5file['obs']

    # 如果'annotation'下有'codes'，使用'codes'，否则使用'annotation'
    if "codes" in obs_group['annotation']:
        mat = obs_group['annotation']['codes'][:]
    else:
        mat = obs_group['annotation'][:]

# 初始化一个字典，将mat中的唯一值映射到一个整数ID
val = {}
for i in mat:
    if i not in val:
        val[i] = len(val)

# k1初始化为1.0，表示在GAN过程中标签传播的权重
k1 = 1.0

# 开始循环，逐步减少k1的值来测试算法的鲁棒性
while True:
    for _ in range(50):  # 循环100次进行评估
        # 初始化GAN输入矩阵X，大小为mat的行数×行数
        X = GAN.matCoo(mat.shape[0], mat.shape[0])
        for i in range(len(data)):
            X.append(rows[i], cols[i], data[i])

        # 初始化标签矩阵y_label和y_new，用于存储标签信息
        y_label = GAN.mat(mat.shape[0], len(val))
        y_new = GAN.mat(mat.shape[0], len(val))

        # 随机选择10%的节点作为初始标签
        random_list = random.sample(range(mat.shape[0]), int(mat.shape[0] * 0.1))
        select_list = np.zeros(mat.shape[0])  # 选择的节点标记

        # 设置y_label矩阵的初始标签
        y_label.setneg()  # 设置所有标签为负
        for i in random_list:
            select_list[i] = 1
        for i in range(mat.shape[0]):
            if select_list[i]:
                y_label.editval2(i, val[mat[i]])

        # case-ori：记录处理时间并进行标签传播
        start_time = time.perf_counter()

        # 初始化预测标签矩阵y_pred和最终结果矩阵y_res
        y_pred = GAN.mat(mat.shape[0], len(val))
        y_res = GAN.mat(mat.shape[0], len(val))

        # 进行标签传播
        # start1 = time.time()
        GAN.dataProcess(y_label, y_new, k1, (1 - k1), 0)  # 处理数据
        # end1 = time.time()
        # print(start1-end1)
        # y_pred_fzy,y_res_fzy=GAN.labelPropagation(csr,y_new.v, y_pred.v, y_res.v)  # 执行GAN算法
        # y_pred_fzy, y_res_fzy = GAN.labelPropagation(csr,y_new.v)

        y_pred_fzy, y_res_fzy = GAN.labelPropagation(dense_matrix, y_new.v)

        # 计算执行时间
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # 计算ARI（调整兰德指数）- 原始预测结果
        res_arr = np.zeros(mat.shape[0])
        for i in range(mat.shape[0]):
            # res_arr[i] = y_pred.getval(i, 0)
            res_arr=y_pred_fzy
        out_arr = np.array(mat)  # 将真实标签转为数组
        ari_o = adjusted_rand_score(out_arr, res_arr)  # 计算ARI
        # ari_o = adjusted_rand_score(out_arr.detach().numpy(), res_arr.detach().numpy())

        # case-rectified：修正后的标签传播结果
        res_arr = np.zeros(mat.shape[0])
        for i in range(mat.shape[0]):
            # res_arr[i] = y_res.getval(i, 0)
            res_arr = y_res_fzy

        # 计算修正后的ARI
        ari_r = adjusted_rand_score(out_arr, res_arr)
        # ari_r = adjusted_rand_score(out_arr.detach().numpy(), res_arr.detach().numpy())

        # 保存当前实验的结果
        item = [round((1 - k1), 2), ari_o, ari_r, round(execution_time, 5)]
        print(item)
        df.loc[len(df)] = item  # 将结果加入DataFrame

    # 减少k1的值，逐步调整标签传播的权重
    k1 -= 0.05
    print(k1)
    if (k1 < 0):  # 当k1小于0时，停止循环
        break

# 将所有实验结果保存为TSV文件
df.to_csv("GAN_mistake.tsv", sep='\t', header=True, index=True)

# 计算并输出总的程序运行时间
end = time.time()
print("time :{}".format(end - start))
