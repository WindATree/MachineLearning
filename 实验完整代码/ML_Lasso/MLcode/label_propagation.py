import bisect
import math
import random
import time
from copy import deepcopy
from typing import List, Optional, Dict
from scipy.sparse import csr_matrix
import numpy as np
from typing import List
from copy import deepcopy




class Elem:
    def __init__(self, row: int, col: int, v: float):
        # 初始化元素，指定行列和对应值
        self.row = row
        self.col = col
        self.v = v

    def __lt__(self, other):
        # 比较当前元素与另一个元素或整数的大小
        if isinstance(other, Elem):
            return (self.row, self.col) < (other.row, other.col)
        elif isinstance(other, int):
            return self.row < other
        return NotImplemented

    def __repr__(self):
        # 返回元素的字符串表示
        return f"Elem(row={self.row}, col={self.col}, v={self.v})"


class matCoo:
    def __init__(self, n0: int = 0, m0: int = 0):
        # 初始化稀疏矩阵（COO格式），指定行数、列数和元素列表
        self.elem: List[Elem] = []
        self.n: int = n0
        self.m: int = m0
        self.totalElements: int = 0

    def createmat(self, n0: int, m0: int):
        # 创建稀疏矩阵，重新设置行数、列数，并清空元素
        self.n = n0
        self.m = m0
        self.totalElements = 0
        self.elem.clear()

    def matTimes(self, c: float):
        # 对矩阵的所有元素乘以常数c
        if c == 0:
            self.elem.clear()
            self.n = 0
            self.m = 0
            self.totalElements = 0
            return
        for e in self.elem:
            e.v *= c

    def append(self, n0: int, m0: int, val: float):
        # 向稀疏矩阵中添加新元素
        if self.isZero(val):
            return
        new_elem = Elem(n0, m0, val)
        self.elem.append(new_elem)
        self.totalElements += 1

    def assign(self, other: 'matCoo'):
        # 复制另一个稀疏矩阵的元素及属性
        self.elem = deepcopy(other.elem)
        self.n = other.n
        self.m = other.m
        self.totalElements = other.totalElements


    @staticmethod
    def isZero(a: float) -> bool:
        # 判断值是否接近零
        return abs(a) < 1e-10

    def sortElems(self):
        # 对元素进行排序
        self.elem.sort()

    def __repr__(self):
        # 返回稀疏矩阵的字符串表示
        return f"matCoo(n={self.n}, m={self.m}, totalElements={self.totalElements}, elem={self.elem})"


class mat:
    def __init__(self, n0: int = 1, m0: int = 1):
        # 初始化矩阵，指定行数、列数及矩阵值
        self.n: int = n0
        self.m: int = m0
        self.v = np.zeros((n0, m0))  # 使用 NumPy 来管理矩阵数据
        self.createmat(n0, m0)

    def createmat(self, n0: int, m0: int):
        # 创建矩阵，设置行列数并初始化矩阵元素为0
        self.n = n0
        self.m = m0
        self.v = np.zeros((n0, m0))

    def matTimes(self, c: float):
        # 对矩阵的所有元素乘以常数c
        # for i in range(self.n):
        #     for j in range(self.m):
        #         self.v[i,j] *= c
        self.v *= c

    def findDiff(self, other: 'mat') -> float:
        # Check if dimensions match
        if self.n != other.n or self.m != other.m:
            return -1.0

        # Use NumPy to calculate the element-wise absolute difference and sum it up
        return np.sum(np.abs(self.v - other.v))

    def assign(self, other: 'mat'):
        # 复制另一个矩阵的值和大小
        self.n = other.n
        self.m = other.m
        self.v = deepcopy(other.v)

    def getval(self, x: int, y: int) -> float:
        # 获取矩阵指定位置的值
        return float(self.v[x,y])

    def editval(self, x: int, y: int, val: float):
        # 修改矩阵指定位置的值
        self.v[x,y] = val

    def setneg(self):
        # 将矩阵的所有元素设置为-1
        self.v = np.full((self.n, self.m), -1.0)

    def editval2(self, x: int, y: int):
        # 将矩阵指定行的所有值置为0，并将指定列的值设为1
        for i in range(self.m):
            self.v[x,i] = 0.0
        self.v[x,y] = 1.0

    def __repr__(self):
        # 返回矩阵的字符串表示
        return f"mat(n={self.n}, m={self.m}, v={self.v})"


def matMultiply_x1_x2_res(x1: matCoo, x2: mat, res: mat):
    # 计算稀疏矩阵x1与稠密矩阵x2的乘积，并将结果存储在res中
    n = x1.n
    m = x2.m
    res.createmat(n, m)
    x1.sortElems()

    # 创建行索引的列表，以进行类似于lower_bound的操作
    rows = [elem.row for elem in x1.elem]

    # 遍历稀疏矩阵的每一行
    for i in range(n):
        # 找到当前行在x1.elem中的起始位置
        p = bisect.bisect_left(rows, i)
        # 缓存行的非零元素，避免重复查找
        row_elements = []

        # 遍历该行的所有非零元素
        while p < x1.totalElements and x1.elem[p].row == i:
            row_elements.append(x1.elem[p])
            p += 1

        # 对每一列进行计算
        for j in range(m):
            sum_val = 0.0
            # 计算该行与矩阵x2对应列的乘积
            for elem in row_elements:
                sum_val += elem.v * x2.v[elem.col][j]
            res.v[i, j] = sum_val




# def matMultiply_x1_x2_res_mat(x1: mat, x2: mat, res: mat):
#     # 计算稠密矩阵x1与x2的乘积，并将结果存储在res中
#     n = x1.n
#     m = x2.m
#     k = x1.m
#     res.createmat(n, m)
#     for i in range(n):
#         for j in range(m):
#             for t in range(k):
#                 res.v[i,j] += x1.v[i,t] * x2.v[t][j]


def matMultiply_x1_x2_res_mat(x1: mat, x2: mat, res: mat):
    # 转换为 NumPy 数组
    x1_np = np.array(x1.v)
    x2_np = np.array(x2.v)

    # 使用 NumPy 的矩阵乘法
    res.v = np.dot(x1_np, x2_np)

def dataProcess(y_old: mat, y_new: mat, preserved: float = 0.8, changed: float = 0.1, masked: float = 0.1):
    # 数据处理，生成新矩阵 y_new
    np.random.seed()  # 初始化随机数生成器
    n0 = y_old.n
    m0 = y_old.m
    y_new.createmat(n0, m0)
    y_new.setneg()

    # 生成一个[0, 1)的随机数矩阵，用于后续判断每行的数据更新方式
    r_matrix = np.random.uniform(0, 1, (n0, 1))  # n0 x 1 matrix for row-based operations

    # 创建掩码：判断哪些元素需要被保留，哪些需要更改，哪些需要被掩盖
    preserve_mask = r_matrix < preserved
    change_mask = (r_matrix >= preserved) & (r_matrix < preserved + changed)
    mask_masked = ~preserve_mask & ~change_mask

    # 保留原值：直接复制对应行
    y_new.v[preserve_mask.flatten(), :] = y_old.v[preserve_mask.flatten(), :]

    # 将值设置为-1：掩码处理
    y_new.v[mask_masked.flatten(), :] = -1

    # 随机更改某些值：改变数据的行，随机选择一个列
    change_indices = np.where(change_mask.flatten())[0]
    for i in change_indices:
        valid_columns = np.setdiff1d(np.arange(m0), np.where(y_old.v[i, :] == 1)[0])
        if len(valid_columns) > 0:
            sd = np.random.choice(valid_columns)
            y_new.v[i, :] = 0.0  # Reset the entire row to 0
            y_new.v[i, sd] = 1.0  # Set the random column to 1

    return y_new


def rectify(x: matCoo, y_label: mat, y_ori: mat, y_new: mat):
    """
    根据给定的预测结果 y_ori 和标签 y_label，通过修正的逻辑将 y_new 中的标签更新为在特定区间内出现最多的标签。
    :param x:
    :param y_label:
    :param y_ori:
    :param y_new:
    :return:
    """
    # 修正预测结果
    # 将 y_ori 的值赋给 y_new
    y_new.assign(y_ori)

    # 对输入矩阵 x 进行元素排序
    x.sortElems()

    # 创建一个字典 p 用于记录每个标签出现的次数
    p: Dict[int, int] = {}

    # 遍历所有样本
    for i in range(y_label.n):
        # 只处理标签值不为 -1 的样本
        if y_label.v[i, 0] != -1:
            # 使用二分查找找到当前样本所在区间的开始和结束位置
            p1 = bisect.bisect_left([elem.row for elem in x.elem], i)
            p2 = bisect.bisect_left([elem.row for elem in x.elem], i + 1)

            # 清空字典 p，准备记录标签出现频次
            p.clear()

            # 将当前样本的原始标签作为初始值加入字典 p
            p[y_ori.v[i, 0]] = 1
            maxx = 1  # 记录出现次数最多的标签的频次
            maxxj = y_ori.v[i, 0]  # 记录出现次数最多的标签

            # 遍历当前样本所在的区间，更新标签的频次
            for j in range(p1, p2):
                # 获取当前元素对应的标签值
                val = y_ori.v[x.elem[j].col][0]

                # 更新标签出现次数
                if val in p:
                    p[val] += 1
                else:
                    p[val] = 1

                # 如果当前标签出现次数大于最大次数，则更新最大次数及对应标签
                if p[val] > maxx:
                    maxx = p[val]
                    maxxj = val

            # 将出现次数最多的标签赋值给 y_new
            y_new.v[i, 0] = maxxj


def labelPropagation(X: matCoo, y_label: mat, y_pred: mat, y_res: mat, alpha: float = 0.5, max_iter: int = 1000):
    # 标签传播算法，用于半监督学习
    # X: 稀疏矩阵，包含样本的相似性数据
    # y_label: 包含已标记样本的标签，未标记样本的标签为 -1
    # y_pred: 预测结果，算法最终输出
    # y_res: 修正后的预测结果
    # alpha: 调节相似度计算的超参数，控制相似度衰减速度
    # max_iter: 最大迭代次数，防止无限循环

    n_samples = X.n  # 样本数量
    n_classes = y_label.m  # 类别数量
    a=X.elem[-1].row
    diff = 0.0  # 标签传播过程中的变化量

    # 初始化 Y 为 y_label 的深拷贝，避免直接修改原标签
    Y = deepcopy(y_label)

    # 计算相似度矩阵 W
    W = matCoo(n_samples, n_samples)  # 创建稀疏矩阵 W，存储样本之间的相似度
    for elem in X.elem:
        row = elem.row  # 获取样本的行索引
        col = elem.col  # 获取样本的列索引
        val = elem.v  # 获取该位置的值，代表样本之间的某种相似性
        dist = val * val  # 计算距离的平方，通常距离越小，相似度越高
        similarity = math.exp(-alpha * dist)  # 使用高斯核函数计算相似度
        W.append(row, col, similarity)  # 将相似度添加到矩阵 W 中
    W.totalElements = len(W.elem)  # 记录 W 中的元素数量

    # 初始化用于计算的矩阵
    Y_old = mat()  # 存储上一次迭代的标签矩阵
    Y_new = mat()  # 存储当前迭代的标签矩阵

    start2 = time.time()  # 记录开始时间，计算总耗时

    # 迭代标签传播过程
    for iter_num in range(max_iter):
        Y_old.assign(Y_new)  # 将当前 Y_new 的值赋给 Y_old，保存上一次的标签
        matMultiply_x1_x2_res(W, Y, Y_new)  # 使用相似度矩阵 W 和标签矩阵 Y 进行矩阵乘法，更新 Y_new

        # 遍历所有样本，更新标签
        for i in range(n_samples):
            # 计算当前样本的标签值总和，并进行归一化
            row_sum = np.sum(Y_new.v[i, :])
            if row_sum != 0:
                Y_new.v[i, :] /= row_sum  # 归一化标签概率分布

            # 检查该样本是否有先验标签
            has_prior = False
            for j in range(n_classes):
                if y_label.v[i, j] != -1:  # 如果标签不为 -1，说明该样本有先验标签
                    Y_new.v[i, j] = y_label.v[i, j]  # 将先验标签赋值给 Y_new
                    has_prior = True
                    break  # 找到先验值后退出内层循环

            # 如果没有先验标签，则更新 Y
            if not has_prior:
                Y.v[i, :] = Y_new.v[i, :]  # 用 Y_new 更新 Y 中的标签值

        # 计算 Y_old 和 Y_new 之间的差异
        diff = Y_old.findDiff(Y_new) / n_samples  # 使用绝对值差异计算
        if iter_num > 0 and diff < 1e-5:  # 如果两次迭代结果差异很小，则提前终止迭代
            break

    end2 = time.time()  # 记录结束时间
    print(end2 - start2)  # 输出迭代过程的总时间

    # 赋值预测结果
    y_pred.createmat(n_samples, 1)  # 创建一个新的矩阵来存储预测结果
    for i in range(n_samples):
        max_index = 0  # 初始化最大值索引
        max_value = Y.v[i, 0]  # 初始化最大值
        for j in range(1, n_classes):
            # 找到当前样本预测标签概率最大的类别
            if Y.v[i, j] > max_value:
                max_value = Y.v[i, j]
                max_index = j
        y_pred.v[i, 0] = max_index  # 将预测的类别赋值给 y_pred

    # 修正预测结果
    rectify(X, y_label, y_pred, y_res)  # 使用 rectify 函数修正预测结果


def interpolate_features(matrix: matCoo, range_radius: int = 1) -> None:
    """
    对稀疏矩阵中的元素进行特征插值，增强其表达能力。

    参数:
    matrix (matCoo): 待插值的稀疏矩阵。
    range_radius (int): 插值范围半径，定义插值值计算时的邻域大小。
    """
    # 收集现有元素的位置
    occupied_positions = {(elem.row, elem.col) for elem in matrix.elem}

    # 创建一个新的元素列表，用于存放插值后的结果
    interpolated_elems = []

    # 遍历矩阵中每一个可能的位置，寻找需要插值的空位置
    for i in range(matrix.n):
        for j in range(matrix.m):
            if (i, j) not in occupied_positions:
                # 在范围内寻找邻近元素
                neighbor_values = []
                for di in range(-range_radius, range_radius + 1):
                    for dj in range(-range_radius, range_radius + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < matrix.n and 0 <= nj < matrix.m:
                            for elem in matrix.elem:
                                if elem.row == ni and elem.col == nj:
                                    neighbor_values.append(elem.v)
                                    break

                # 如果找到足够的邻近值，就计算加权平均值
                if neighbor_values:
                    interpolated_value = sum(neighbor_values) / len(neighbor_values)
                    interpolated_elems.append(Elem(i, j, interpolated_value))

    # 将新插值的元素添加到原有的矩阵中
    matrix.elem.extend(interpolated_elems)
    matrix.totalElements += len(interpolated_elems)
    matrix.sortElems()
