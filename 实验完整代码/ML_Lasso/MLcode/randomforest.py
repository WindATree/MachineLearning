from label_propagation import *  # 导入标签传播模块
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class RandomForestLabelPropagation:
    def __init__(self, n_estimators=100, max_iter=10):
        self.n_estimators = n_estimators  # 随机森林树的数量
        self.max_iter = max_iter  # 最大迭代次数

    def labelPropagation(self, dense_matrix, y_label):
        """
        使用随机森林进行标签传播

        :param dense_matrix: (3000, 3000) 类型的稠密图矩阵，表示样本之间的关联
        :param y_label: (3000, 11) 类型的标签矩阵，已标记样本的标签，-1表示未标记样本
        :return: y_pred, y_res - 初始预测结果和最终修正后的预测结果
        """
        # 获取已标记和未标记样本的索引
        labeled_idx = np.where(np.sum(y_label != -1, axis=1) > 0)[0]
        unlabeled_idx = np.where(np.sum(y_label == -1, axis=1) > 0)[0]

        # 提取已标记样本的标签和特征
        X_train = dense_matrix[labeled_idx]
        y_train = y_label[labeled_idx]

        # 使用One-Hot编码处理标签
        y_train = np.argmax(y_train, axis=1)  # 将标签转换为单一标签值（0-10）

        # 创建随机森林分类器
        rf = RandomForestClassifier(n_estimators=self.n_estimators)

        # 初始化预测结果和修正结果
        y_pred = np.full((dense_matrix.shape[0],), -1)  # 初始预测结果为-1
        y_res = np.full((dense_matrix.shape[0],), -1)  # 最终修正结果

        # 初始训练
        rf.fit(X_train, y_train)

        # 获取初始预测结果
        y_pred[labeled_idx] = y_train
        y_pred[unlabeled_idx] = rf.predict(dense_matrix[unlabeled_idx])

        # 进行标签更新
        for i in range(self.max_iter):
            # 对未标记样本进行预测
            y_res[unlabeled_idx] = rf.predict(dense_matrix[unlabeled_idx])

            # 如果有未标记样本，更新标签
            y_label[unlabeled_idx, :] = -1  # 将未标记样本的标签清空
            for idx in unlabeled_idx:
                label = y_res[idx]
                y_label[idx, label] = 1  # 更新未标记样本的标签

            # 重新训练模型，加入更新后的标签
            X_train = dense_matrix[np.where(np.sum(y_label != -1, axis=1) > 0)[0]]
            y_train = y_label[np.where(np.sum(y_label != -1, axis=1) > 0)[0]]
            y_train = np.argmax(y_train, axis=1)

            rf.fit(X_train, y_train)

            # 获取新的预测结果
            y_pred = y_res.copy()  # 使用修正后的预测结果

        return y_pred, y_res
