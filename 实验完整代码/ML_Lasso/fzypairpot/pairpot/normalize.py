# 将所有的输入数据标准化成统一格式，其中标准化格式如下：
# 单细胞数据
# -表达矩阵
# --下采样得到3000个细胞
# --2000个与空转交集的高变基因
# -低维表示
# --PCA前两维表示
# --UMAP2维和3维表示
# --TSNE2维表示
# -细胞类型标注
# --原始数据自带标注
# --细胞类型marker基因
# --KEGG结果
# --GSEA结果
# --GO结果
# -邻接矩阵
# --k近邻矩阵
import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
from .AUCell import *
import MENDER
from .db import panglaoDB
import cell2location
from cellphonedb.src.core.methods import cpdb_statistical_analysis_method

# 读取10X Genomics格式的数据并返回AnnData对象
def input_adata_10X(sample):
    # 读取稀疏矩阵数据（.mtx.gz文件），并进行转置，使基因作为行，细胞作为列
    adata = sc.read_mtx(sample+'/matrix.mtx.gz')
    adata = adata.T
    # 读取条形码文件（barcodes.tsv.gz），并将列命名为'barcodes'
    bar = pd.read_csv(sample+'/barcodes.tsv.gz', header=None)
    # 读取特征文件（features.tsv.gz），并设定列名为'ID', 'name', 'type'
    fea = pd.read_csv(sample+'/features.tsv.gz', header=None, sep='\t')
    bar.columns = ['barcodes']
    fea.columns = ['ID', 'name', 'type']
    # 将AnnData的obs_names设置为条形码数据
    adata.obs_names = bar.iloc[:,0]
    # 确保obs_names唯一
    adata.obs_names_make_unique()
    # 将AnnData的var_names设置为基因名
    adata.var_names = fea.iloc[:,1]
    # 将特征信息添加到AnnData的var属性中
    adata.var = fea
    # 确保var_names唯一
    adata.var_names_make_unique()
    # 返回处理后的AnnData对象
    return adata

# 读取10X Genomics H5格式的数据并返回AnnData对象
def input_adata_10Xh5(sample):
    # 读取10X H5格式的数据
    adata = sc.read_10x_h5(sample)
    # 将基因名和细胞名转为大写字母
    adata.var_names = [s.upper() for s in adata.var_names]
    adata.obs_names = [s.upper() for s in adata.obs_names]
    # 确保obs_names和var_names唯一
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    # 返回处理后的AnnData对象
    return adata

# 读取H5AD格式的数据并返回AnnData对象
def input_adata_h5ad(sample):
    # 读取H5AD格式的数据
    adata = sc.read_h5ad(sample)
    # 将基因名和细胞名转为大写字母
    adata.var_names = [s.upper() for s in adata.var_names]
    adata.obs_names = [s.upper() for s in adata.obs_names]
    # 确保obs_names和var_names唯一
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    # 返回处理后的AnnData对象
    return adata

# 读取文本格式的数据并返回AnnData对象
def input_adata_txt(sample):
    # 读取文本格式的数据
    adata = sc.read_text(sample)
    # 确保obs_names和var_names唯一
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    # 返回处理后的AnnData对象
    return adata

# 读取CSV格式的数据并返回AnnData对象
def input_adata_csv(sample):
    # 读取CSV格式的数据
    adata = sc.read_text(sample, delimiter=',')
    # 确保obs_names和var_names唯一
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    # 返回处理后的AnnData对象
    return adata

# 合并多个样本的AnnData对象
def concat_adata(samples, sampleNames, inputFunc=input_adata_10Xh5):
    # 创建一个空列表，用于存储每个样本的AnnData对象
    adatas = []
    # 遍历每个样本，读取并处理数据
    for i in range(len(sampleNames)):
        adata = inputFunc(samples[i])  # 读取数据
        # 过滤掉基因数少于5的细胞和计数少于30的细胞
        sc.pp.filter_cells(adata, min_genes=5)
        sc.pp.filter_cells(adata, min_counts=30)
        adatas.append(adata)
    # 如果有多个样本，计算这些样本的共同基因集合
    if len(adatas) > 0:
        intersection_var = set(adatas[0].var_names)
        for a in adatas[1:]:
            intersection_var &= set(a.var_names)
        common_vars = list(intersection_var)
        # 对每个样本，只保留共同的基因
        for i in range(len(adatas)):
            adatas[i] = adatas[i][:, common_vars]
    # 合并所有样本数据，并标记样本来源
    adata_concat = adatas[0].concatenate(adatas[1:], batch_categories=sampleNames)
    # 返回合并后的AnnData对象
    return adata_concat

# 数据预处理，包括基因过滤、细胞过滤、标准化等
def pp(adata:ad.AnnData):
    # 筛选出线粒体基因，计算每个细胞线粒体基因的表达占比
    mito_genes = adata.var_names.str.startswith('MT')
    mt_frac = np.sum(
        adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1)
    adata.obs['mt_frac'] = np.array(mt_frac)
    # 过滤掉基因数少于5的细胞样本
    sc.pp.filter_cells(adata, min_genes=5)
    # 过滤掉在少于5个细胞中表达的基因
    sc.pp.filter_genes(adata, min_cells=5)
    # 过滤掉计数少于30的细胞样本
    sc.pp.filter_cells(adata, min_counts=30)
    # 过滤掉线粒体基因和核糖体基因
    rp_genes = adata.var_names.str.startswith('RP')
    mt_genes = adata.var_names.str.startswith('MT')
    adata = adata[:, ~(rp_genes + mt_genes)]
    # 过滤掉线粒体基因占比大于20%的细胞
    adata = adata[adata.obs['mt_frac'] < 0.2]
    # 保存原始数据
    adata.layers['Raw'] = adata.X
    # 进行总数标准化
    sc.pp.normalize_total(adata)
    # 进行对数转换
    sc.pp.log1p(adata)
    # 选择高变基因
    sc.pp.highly_variable_genes(adata,min_disp=0.5, n_top_genes=2000)
    # 返回预处理后的AnnData对象
    return adata

# 聚类分析
def clu(adata, key_added="leiden-1", n_neighbors=50, n_pcs=30, rep='X_pca_harmony', do_har=True, max_iter=20, resolution=1, do_scrublet=True, har_key='batch'):
    # 如果需要进行双重细胞过滤
    if do_scrublet:
        n0 = adata.shape[0]
        print("{0} Cell number: {1}".format(key_added, n0))
        sc.external.pp.scrublet(adata)
        adata = adata[adata.obs['predicted_doublet']==False,:].copy()
        print("{0} Cells retained after scrublet, {1} cells removed.".format(adata.shape[0], n0-adata.shape[0]))
    else:
        print("Ignoring processing doublet cells...")
    # 进行PCA降维
    sc.pp.pca(adata, svd_solver='arpack', use_highly_variable=True)
    # 如果需要使用Harmony进行批次效应校正
    if do_har and len(adata.obs[har_key].cat.categories) > 1:
        sc.external.pp.harmony_integrate(adata, key=har_key, max_iter_harmony=max_iter)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=rep)
    else:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    # 进行UMAP降维可视化
    sc.tl.umap(adata)
    # 进行t-SNE降维可视化
    sc.tl.tsne(adata)
    # 进行Leiden聚类
    sc.tl.leiden(adata, key_added=key_added, resolution=resolution)
    # 绘制UMAP图
    sc.pl.umap(adata, color=key_added, legend_fontoutline=True, palette=sc.pl.palettes.default_20, legend_loc="on data")
    # 返回聚类后的AnnData对象
    return adata

# MENDER聚类分析
def mender(adata:ad.AnnData, key_added="mender"):
    # 设置MENDER的参数
    scale = 6
    radius = 15
    msm = MENDER.MENDER(
        adata,
        batch_obs='batch',  # 批次信息
        ct_obs='leiden-1'   # 聚类标签
    )
    msm.prepare()
    # 设置MENDER算法的参数
    msm.set_MENDER_para(
        n_scales=scale,
        nn_mode='radius',
        nn_para=radius,
    )
    # 运行MENDER算法并进行聚类
    msm.run_representation_mp()
    msm.run_clustering_normal(-0.5)
    # 将聚类结果保存到adata.obs中
    adata.obs[key_added] = np.array(msm.adata_MENDER.obs['MENDER'])
    adata.obs['annotation'] = np.array(msm.adata_MENDER.obs['MENDER'])
    # 返回带有MENDER聚类信息的AnnData对象
    return adata


# 进行基因表达排序与细胞类型注释的函数
def rank(adata, organs,
         method="AUCell",
         top=0.05,
         alpha=10e-40,
         n_jobs=16,
         clu_key='leiden-1',
         test_func=mannwhitneyu,
         ):
    # 对每个细胞的基因表达进行排序，并且提取前top%的基因
    adata = AUCell_buildRankings(adata, top=top)

    # 查找目标器官相关的潜在细胞类型
    # 使用panglaoDB数据库过滤出在指定器官中的细胞类型，并进行排序
    celltype = pd.unique(panglaoDB[panglaoDB['organ'].isin(organs)]['cell type'].dropna())
    celltype.sort()

    # 使用UCell算法进行细胞类型的分配与注释
    adata = AUCell_UCAssign(adata,
                            db=panglaoDB,  # 使用panglaoDB作为细胞类型数据库
                            celltype=celltype,  # 细胞类型列表
                            alpha=alpha,  # alpha值，用于UCell的调节
                            n_jobs=n_jobs,  # 并行计算的线程数
                            clu_key=clu_key,  # 聚类标签键
                            test_func=test_func)  # 使用的统计检验方法（默认为Mann-Whitney U检验）

    return adata  # 返回注释后的AnnData对象


# 根据给定的注释字典对AnnData对象进行注释
def anno(adata: ad.AnnData, annoDict: dict):
    # 默认将所有细胞标注为"Unknown"
    adata.obs['annotation'] = 'Unknown'

    # 根据annoDict字典中的细胞类型信息进行注释
    for key in annoDict.keys():
        # 更新细胞类型为字典中对应的key
        adata.obs.loc[adata.obs['leiden-1'].isin(annoDict[key]), 'annotation'] = key

    return adata  # 返回注释后的AnnData对象


# 进行标记基因的识别与可视化
def marker(adata, groupby="annotation", method='wilcoxon'):
    # 使用给定的分组信息（默认是'annotation'列）进行标记基因的排名
    sc.tl.rank_genes_groups(adata, groupby=groupby, method=method)

    # 计算并绘制基于分组的树状图（基于基因的排名）
    sc.tl.dendrogram(adata, groupby=groupby)

    # 绘制标记基因的Dot Plot
    sc.pl.rank_genes_groups_dotplot(adata, groupby=groupby)

    return adata  # 返回带有标记基因分析结果的AnnData对象


# 使用Cell2Location进行单细胞数据的空间转录组学分析（回归模型部分）
def Cell2Location_rg_sc(adata_sc, max_epoches=250, batch_size=2500, train_size=1, lr=0.002,
                        num_samples=1000, use_gpu=True):
    from cell2location.models import RegressionModel

    # 设置AnnData对象以便用于回归模型
    cell2location.models.RegressionModel.setup_anndata(adata=adata_sc,
                                                       # 细胞类型标签
                                                       labels_key='annotation')

    # 创建并训练回归模型
    mod = RegressionModel(adata_sc)

    # 训练模型，训练过程中不使用验证集（train_size=1）
    mod.train(max_epochs=max_epoches, batch_size=batch_size, train_size=train_size, lr=lr)

    # 导出训练后的细胞丰度（后验分布的总结）
    adata_sc = mod.export_posterior(
        adata_sc, sample_kwargs={'num_samples': num_samples, 'batch_size': batch_size, 'use_gpu': use_gpu}
    )

    # 导出每个簇的表达估计
    if 'means_per_cluster_mu_fg' in adata_sc.varm.keys():
        inf_aver = adata_sc.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                                             for i in adata_sc.uns['mod']['factor_names']]].copy()
    else:
        inf_aver = adata_sc.var[[f'means_per_cluster_mu_fg_{i}'
                                 for i in adata_sc.uns['mod']['factor_names']]].copy()

    # 设置列名称为因子名称
    inf_aver.columns = adata_sc.uns['mod']['factor_names']
    return inf_aver  # 返回每个簇的基因平均表达量


# 使用Cell2Location进行空间数据的转录组学分析（空间部分）
def Cell2Location_rg_sp(adata_sp, inf_aver, N_cells_per_location=30, detection_alpha=20,
                        max_epoches=5000, batch_size=None, train_size=1, lr=0.002,
                        num_samples=1000, use_gpu=True):
    # 执行空间转录组学分析
    # 查找共同的基因并筛选出这些基因的数据
    intersect = np.intersect1d(adata_sp.var_names, inf_aver.index)
    adata_sp = adata_sp[:, intersect].copy()
    inf_aver = inf_aver.loc[intersect, :].copy()

    # 设置AnnData对象以便进行Cell2Location模型的训练
    cell2location.models.Cell2location.setup_anndata(adata=adata_sp, batch_key="batch")

    # 创建并训练Cell2Location模型
    mod = cell2location.models.Cell2location(
        adata_sp, cell_state_df=inf_aver,
        N_cells_per_location=N_cells_per_location,  # 每个位置的细胞数
        detection_alpha=detection_alpha  # 检测的超参数
    )

    # 训练模型
    mod.train(max_epochs=max_epoches, batch_size=batch_size, train_size=train_size, lr=lr, use_gpu=use_gpu)

    # 导出后验分布的细胞丰度数据
    adata_sp = mod.export_posterior(
        adata_sp, sample_kwargs={'num_samples': num_samples, 'batch_size': mod.adata.n_obs, 'use_gpu': use_gpu}
    )

    # 添加细胞丰度的5%分位数，作为细胞丰度的可信度指标
    adata_sp.obs[adata_sp.uns['mod']['factor_names']] = adata_sp.obsm['q05_cell_abundance_w_sf']
    return adata_sp  # 返回带有空间细胞丰度数据的AnnData对象


# 运行Cell2Location模型的整体流程，包括单细胞与空间数据的结合
def Cell2Location_run(adata_sc, adata_sp, sc_max_epoches=250, sc_batch_size=2500, sc_train_size=1, sc_lr=0.002,
                      sc_num_samples=1000,
                      N_cells_per_location=30, detection_alpha=20,
                      sp_max_epoches=5000, sp_batch_size=2500, sp_train_size=1, sp_lr=0.002, sp_num_samples=1000,
                      use_gpu=True):
    # 将基因名改为ENSEMBL形式
    adata_sc.var['SYMBOL'] = adata_sc.var.index
    adata_sp.var['SYMBOL'] = adata_sp.var_names
    adata_sp.var['gene_name'] = adata_sp.var_names

    # 细胞过滤的参数
    cell_count_cutoff = 5
    cell_percentage_cutoff2 = 0.03
    nonz_mean_cutoff = 1.12

    adata_sc = adata_sc.copy()
    adata_sp = adata_sp.copy()
    adata_sc.X = adata_sc.layers['Raw'].astype('int')
    adata_sp.X = adata_sp.layers['Raw'].astype('int')

    # 细胞过滤
    selected = cell2location.utils.filtering.filter_genes(adata_sc,
                                                          cell_count_cutoff=cell_count_cutoff,
                                                          cell_percentage_cutoff2=cell_percentage_cutoff2,
                                                          nonz_mean_cutoff=nonz_mean_cutoff)

    adata_sc = adata_sc[:, selected]  # 只保留选中的基因

    # 训练单细胞回归模型
    inf_aver = Cell2Location_rg_sc(adata_sc, sc_max_epoches, sc_batch_size, sc_train_size, sc_lr, sc_num_samples,
                                   use_gpu)

    # 训练空间回归模型
    adata_sp = Cell2Location_rg_sp(adata_sp, inf_aver, N_cells_per_location=N_cells_per_location,
                                   detection_alpha=detection_alpha, max_epoches=sp_max_epoches,
                                   batch_size=sp_batch_size, train_size=sp_train_size, lr=sp_lr,
                                   num_samples=sp_num_samples, use_gpu=use_gpu)

    # 提取并返回最终的细胞丰度数据
    weight = adata_sp.obsm['q05_cell_abundance_w_sf']
    weight.columns = adata_sp.uns['mod']['factor_names']
    return weight  # 返回细胞丰度数据


# 运行CellphoneDB进行细胞间通讯分析
def CellphoneDB_run(adata_path, cpdb_file_path="resources/cellphonedb.zip", output_path="resources/cpdb",
                    counts_data='hgnc_symbol'):
    # 读取AnnData对象
    adata = sc.read_h5ad(adata_path)
    meta = pd.DataFrame(adata.obs['annotation'].copy())
    meta = meta.reset_index()
    meta.columns = ['Cell', 'cell_type']
    meta.to_csv("resources/meta.txt", sep='\t', index=False)

    # 调用CellphoneDB的统计分析方法
    cpdb_res = cpdb_statistical_analysis_method.call(
        cpdb_file_path=cpdb_file_path,
        meta_file_path="resources/meta.txt",
        counts_file_path=adata_path,
        counts_data=counts_data,
        output_path=output_path
    )

    return cpdb_res  # 返回CellphoneDB分析结果

# 空转数据
# -表达矩阵
# --原始in_tissue细胞数
# --2000个与空转交集的高变基因
# -低维表示
# --PCA前两维表示
# --UMAP2维和3维表示
# --TSNE2维表示
# --空间位置2维表示
# -细胞类型解卷积
# --CARD解卷积
# --cell2location解卷积
# --空间区域marker基因
# --KEGG结果
# --GSEA结果
# --GO结果
# -邻接矩阵
# --k近邻矩阵

# 联合嵌入数据
# 单细胞和多个空转样本联合嵌入UMAP2维、3维表示(降采样到5k)
# 联合嵌入邻接矩阵（降采样到5k）两次标签传播算法

