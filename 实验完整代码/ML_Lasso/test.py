import fzypairpot.pairpot as pt
import anndata as ad
import scanpy as sc

adata = ad.read_h5ad('./data/newdataset.h5ad')
# sc.pp.neighbors(adata)
# UMAP可视化
# sc.tl.umap(adata)  # 计算UMAP，它在保留局部结构的同时，尝试保留全局结构。基于流形学习和拓扑数据分析，旨在捕捉数据的全局和局部结构，并能够更有效地处理大规模数据。
# sc.pl.umap(adata, color='annotation', title='UMAP of Cells',legend_loc='on data')  # 按照注释着色
#  t-SNE可视化
sc.tl.tsne(adata)  # 计算t-SNE，线性降维t-SNE通过计算数据点之间的相似度，并尽量保持高维空间中相似的数据点在低维空间中的相对位置。
sc.pl.tsne(adata, color='annotation', title='Original data image')  # 按照注释着色
del adata
for i in range(1, 10):
    # Corresponding adata can be downloaded in MetaInfo or use the code as following.
    adata = ad.read_h5ad('./data/dataset1.h5ad')
    # 打开文件并读取第一行
    with open(f'./data/test/{i}.txt', 'r') as f:
        # 读取第一行并解析为列表
        first_line = f.readline().strip()  # 去除首尾空格或换行符
        lassoed_index = eval(first_line)  # 使用 eval 将字符串转换为列表

    refined_index = pt.lassoView(lassoed_index, adata, do_correct=True, function="anndata")
    print(refined_index)

    # 创建一个新的 'highlight' 列
    adata.obs['annotation'] = list(adata.obs['annotation'])
    adata.obs['annotation'].iloc[refined_index] = "Chosen_Cell"
    adata.obs['annotation'].iloc[lassoed_index] = "Chosen_Cell"
    adata.obs['annotation'] = adata.obs['annotation'].astype("category")
    # 绘制 UMAP 图，'highlight' 用于区分 'refined' 和 'other' 细胞
    # sc.pl.umap(adata, color='highlight', title='Highlighted Refined Cells', palette=['gray', 'red'])
    # sc.tl.umap(adata)  # 计算UMAP，它在保留局部结构的同时，尝试保留全局结构。基于流形学习和拓扑数据分析，旨在捕捉数据的全局和局部结构，并能够更有效地处理大规模数据。
    # sc.pl.umap(adata, color='annotation', title='UMAP of Cells',legend_loc='on data')  # 按照注释着色

    #  t-SNE可视化
    sc.tl.tsne(adata)  # 计算t-SNE，线性降维t-SNE通过计算数据点之间的相似度，并尽量保持高维空间中相似的数据点在低维空间中的相对位置。
    sc.pl.tsne(adata, color='annotation', title=f'File {i} Cell Selection Image')  # 按照注释着色
    del adata