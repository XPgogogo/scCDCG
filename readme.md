## Run Example
```shell
python train_scCDCG.py --dataname 'neural_2100' --num_class 10 --epochs 200 --foldername 'logger_folder' --gpu 0
```



# scCDCG
scCDCG, a clustering model based on deep cut-informed graph for scRNA-seq data. See details in our paper: "singel-cell RNA-seq Clustering via Deep Cut-informed Graph" published in Briefings in Bioinformatics https://www.XXX.


# requirements
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
The final output reports the clustering performance, here is an example on Maayan_Mouse_Pancreas_cell_1 scRNA-seq data:

Final: ACC= 0.8264, NMI= 0.7766, ARI= 0.8060

The raw data used in this paper can be found:https://github.com/XPgogogo/scCDCG/tree/master/datasets

# Contact
Ping Xu:
xuping098@gmail.com; xuping@cnic.cn
