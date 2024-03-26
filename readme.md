# Run Example
```shell
python train_scCDCG.py --dataname 'Meuro_human_Pancreas_cell' --num_class 9 --epochs 200 --foldername 'logger_folder' --gpu 0
```



# scCDCG
scCDCG, a clustering model based on deep cut-informed graph for scRNA-seq data. See details in our paper: "scCDCG: Efficient Deep Structural Clustering for single-cell RNA-seq via Deep Cut-informed Graph Embedding" published in DASFAA2024（CCF-B） https://www.XXX.
In conclusion, our study introduces scCDCG, an innovative framework for the efficient and accurate clustering of single-cell RNA sequencing (scRNA-seq) data. scCDCG successfully navigates the challenges of high-dimension and high-sparsity through a synergistic combination of a graph embedding module with deep cut-informed techniques, a self-supervised learning module guided by optimal transport, and an autoencoder-based feature learning module. Our extensive evaluations on six datasets confirm scCDCG's superior performance over seven established models, marking it as a transformative tool for bioinformatics and cellular heterogeneity analysis. Looking forward, we aim to extend scCDCG's capabilities to integrate multi-omics data, enhancing its applicability in more complex biological contexts. Additionally, further exploration into the interpretability of the clustering results generated by scCDCG will be crucial for providing deeper biological insights and facilitating its adoption in clinical research settings. This future work will continue to expand the frontiers of scRNA-seq data analysis and its impact on understanding the complexities of cellular systems.


# Requirements
We implement scCDCG in Python 3.7 based on PyTorch (version 1.12+cu113).

Keras --- 2.4.3
njumpy --- 1.19.5
pandas --- 1.3.5
Scanpy --- 1.8.2
torch --- 1.12.0


Please note that if using different versions, the results reported in our paper might not be able to repeat.

# The raw data
Setting data_file to the destination to the data (stored in h5 format, with two components X and Y, where X is the cell by gene count matrix and Y is the true labels), n_clusters to the number of clusters.
In order to ensure the accuracy of the experimental results, we conducted more than 10 times runs on all the datasets and reported the mean and variance of these running results, reducing the result bias caused by randomness and variability, so as to obtain more reliable and stable results. Hyperparameter settings for all datasets can be found in the code.
The final output reports the clustering performance, here is an example on Meuro_human_Pancreas_cell scRNA-seq data:

Final: ACC= 0.9265, NMI= 0.8681, ARI= 0.9137

The raw data used in this paper can be found:https://github.com/XPgogogo/scCDCG/tree/master/datasets

# Contact
Ping Xu:
xuping098@gmail.com; xuping@cnic.cn
