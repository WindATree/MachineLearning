import numpy as np
import scanpy as sc
import pandas as pd
import random
from collections import Counter
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
import h5py
import scipy.sparse
import time
import random
import numpy as np
import pkg_resources
import sys
import os
import anndata as ad
# lpa_so_path = pkg_resources.resource_filename('pairpot', 'label_propagation.cpython-310-x86_64-linux-gnu.so')
# print(lpa_so_path)
# sys.path.append(os.path.dirname(lpa_so_path))
# import label_propagation as GNN
import MLcode.GNN as GNN

from fzypairpot.pairpot.normalize import *


def EagerRefine(selected, file):
    adata = sc.read_h5ad(file)
    obs = adata.obs.copy()
    obs = obs.reset_index()
    selectedObs = obs.iloc[selected, :]
    selectedClu = np.unique(selectedObs['leiden'])
    refinedObs = obs.loc[obs['leiden'].isin(selectedClu), :]
    return list(refinedObs.index)


def LazyRefine(selected, file):
    adata = sc.read_h5ad(file)
    obs = adata.obs.copy()
    obs = obs.reset_index()
    obsCounter = Counter(obs['leiden'])

    selectedObs = obs.iloc[selected, :]
    selectedCounter = Counter(selectedObs['leiden'])

    selectedClu = []
    for key in selectedCounter:
        if selectedCounter[key] / obsCounter[key] > 0.5:
            selectedClu.append(key)

    refinedObs = obs.loc[obs['leiden'].isin(selectedClu), :]
    return list(refinedObs.index)


def GNNRefine(adata, selected, function="anndata", use_model=LabelPropagation, do_correct=True):
    mat = 1
    if function == "anndata":
        mat = adata.obsp['connectivities']
        if not scipy.sparse.issparse(mat):
            mat = scipy.sparse.csr_matrix(mat)
    elif function == "h5adfile":
        with h5py.File(adata, 'r') as f:
            group = f['obsp']['connectivities']

            data = group['data'][:]
            indices = group['indices'][:]
            indptr = group['indptr'][:]
            shape = (f['obsp']['connectivities'].attrs['shape'][0], f['obsp']['connectivities'].attrs['shape'][1])

            mat = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
    else:
        print('No this function, use "anndata" or "h5adfile" instead.')
        return
    coo = mat.tocoo()

    rows = coo.row
    cols = coo.col
    data = coo.data

    if function == "anndata":
        obs_col = 'annotation'
        if obs_col not in adata.obs:
            obs_col = 'leiden-1'

        if "codes" in adata.obs[obs_col]:
            mat = adata.obs[obs_col]['codes'].values
        else:
            mat = adata.obs[obs_col].values
    elif function == "h5adfile":
        with h5py.File(adata, 'r') as h5file:
            obs_group = h5file['obs']
            obs_col = 'annotation'
            if obs_col not in obs_group:
                obs_col = 'leiden-1'

            if "codes" in obs_group[obs_col]:
                mat = obs_group[obs_col]['codes'][:]
            else:
                mat = obs_group[obs_col][:]
    else:
        print('No this function, use "anndata" or "h5adfile" instead.')
        return
    val = {}

    for i in np.unique(mat):
        val[i] = len(val)
    val[len(val)] = len(val)
    X = GNN.matCoo(mat.shape[0], mat.shape[0])
    for i in range(len(data)):
        X.append(rows[i], cols[i], data[i])

    y_label = GNN.mat(mat.shape[0], len(val))
    random_list = random.sample(range(mat.shape[0]), int(mat.shape[0] * 0.1))
    select_list = np.zeros(mat.shape[0])
    y_label.setneg()
    select_list[random_list] = 1

    # add selected item
    select_list[selected] = 1
    selected_val = len(val) - 1

    mat_list = mat.tolist()
    for t in range(len(selected)):
        mat_list[selected[t]] = selected_val
    mat = pd.Categorical(mat_list)
    for i in range(mat.shape[0]):
        if select_list[i]:
            y_label.editval2(i, val[mat[i]])
    y_pred = GNN.mat(mat.shape[0], len(val))
    y_new = GNN.mat(mat.shape[0], len(val))
    GNN.labelPropagation_GNN(X, y_label, y_pred, y_new)
    y_res = np.zeros(mat.shape[0])
    if do_correct:
        for i in range(mat.shape[0]):
            y_res[i] = y_new.getval(i, 0)
    else:
        for i in range(mat.shape[0]):
            y_res[i] = y_pred.getval(i, 0)
    y_res = pd.Series(y_res)
    y_res = y_res[y_res == selected_val]
    return list(y_res.index)


def Gen_maskSet(candidate: pd.DataFrame, errRate=0.20):
    sele_can = candidate[candidate == True]
    cell_len = len(sele_can)
    mask_can = candidate.copy()
    errArray = random.sample(range(cell_len), int(cell_len * errRate))
    for cell in errArray:
        print(sele_can.index[cell])
        mask_can.loc[sele_can.index[cell]] = not mask_can.loc[sele_can.index[cell]]
    return mask_can


def train_GNN(candidate, adata, use_model=LabelPropagation, errRate=0.05):
    X = adata.obsm['X_pca']
    y = Gen_maskSet(candidate, errRate)
    model = use_model().fit(X, y)
    y_pred = model.predict(X)

    def acc(y_true, y_pred):
        return np.sum(np.equal(y_true, y_pred)) / len(y_true)

    print("acc:{}".format(acc(candidate, y_pred)))

