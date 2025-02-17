# Machine Learning
## Interactive Similar Cell Prediction Based on Semi-Supervised Learning

### Background
Single-cell sequencing data analysis involves high-throughput sequencing of genomes and transcriptomes at the single-cell level, aiming to elucidate the heterogeneity among cells within a sample with precision. In essence, single-cell sequencing data represents a high-dimensional sparse matrix where rows correspond to cells and columns to genes. The workflow for single-cell sequencing data encompasses quality control, data normalization, dimensionality reduction, cell clustering, cell type annotation, and visualization.

However, the critical step of cell clustering often fails to reflect the true cell types due to varying resolutions, necessitating manual interactive refinement. Manual selection inevitably leads to the omission of similar cells or the misselection of incorrect cells, thus necessitating an interactive method for predicting similar cells to remedy this deficiency.

### Algorithmic Principle
The Label Propagation Algorithm (LPA), a graph-based semi-supervised learning algorithm, is commonly utilized for node classification and clustering analysis of graph data. This algorithm primarily operates by propagating labels among various nodes and gradually determining the classification, ultimately partitioning the nodes in the graph into several clusters or categories.

Lasso-View, contributed by the PairPot database, is a method based on the LPA algorithm to infer user-interested cell groups. In PairPot's analysis of single-cell data, each cell serves as a node in the graph. Once users manually select a subset of cells, Lasso-View can identify the user-interested cell subgroup within milliseconds.

During the data integration phase, Pairpot generates a probability transition matrix for each dataset. In the online analysis phase, users select cells to create a new cell type U. The label propagation algorithm iteratively generates the probability of all cells belonging to cell type U. This process, optimized with C++, achieves millisecond-level response times. The results are adjusted using the K-nearest neighbors method.

### Model Evaluation Method
Evaluating semi-supervised learning is often challenging due to the inability to generate an ample amount of user-selected cells and their corresponding true values. Most of the initial annotations can be masked, and the remaining small portion of annotation information can be used to infer all cell type annotations, which are then compared with the initial annotations to assess the model's performance.

To align with practical scenarios, some erroneous labels can be artificially added to the remaining small portion of annotation information. In such cases, all cell type annotations are inferred, and the artificially added erroneous labels are identified. Lasso-View uses an evaluation method where 90% of the initial annotations are masked, and erroneous labels (0-95%) are added to the remaining 10% of the labels. After inferring all cell type annotations, the Adjusted Rand Index (ARI) is calculated compared to the initial annotations.

### Evaluation Metric ARI
The Adjusted Rand Index (ARI) is an evaluation method used to measure the similarity between clustering results and true classifications. It calculates the score by comparing the pairwise sample similarities between clustering results and true classifications, with a matching range of [-1,1]. The `adjusted_rand_score` function in `sklearn` can be utilized for calculation.

Two types of ARI need to be calculated in the experiment:
1. Original ARI_o (the similarity between the predicted labels using LPA and the true labels)
2. Corrected ARI_r (the similarity between the corrected label propagation results and the true labels)

### Final Algorithm Assessment
1. Accuracy: The ARI value when the error rate is 0
2. Robustness: The error rate when ARI > 0.7

### Work Completed in This Assignment
In this assignment, we completed a semi-supervised learning task based on machine learning and label propagation, performing classification and prediction on single-cell RNA-seq data and graph-structured data. We utilized the standard LPA label propagation algorithm for cell group prediction. Additionally, we integrated improved LPA algorithms, Graph Neural Networks (GNN), and Generative Adversarial Networks (GAN) to optimize the LPA model.

### Experimental Results
Although the ARI value at an error rate of 0 for the optimized models using improved LPA, GNN, and GAN was significantly higher than that of the baseline method, the model robustness was somewhat reduced, and the anti-interference capability was weak. We conducted an analysis on the reduced robustness of the optimized models. The increased complexity of neural network models raises the risk of overfitting as complex models are more prone to fitting noise in the training data or being overly reliant on specific data distributions, thereby diminishing generalization capabilities. This explains the faster decline in the ARI metric for our neural network models as the error rate increases. Moreover, label propagation's long-distance dependency requires reliance on multi-hop neighbor relationships within the graph. However, distant propagation is easily affected by low-quality neighbors, and GNNs, by stacking multiple graph convolutions to capture long-distance dependencies, can lead to cumulative errors at each propagation layer, thereby impacting the final prediction outcomes. These are aspects that necessitate our reflection.
