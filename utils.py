# -*- encoding: utf-8 -*-

import torch
import math
import numpy as np
import scipy.sparse as sp
from scipy.special import iv
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import recall_score, precision_score

################################################################################
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

######################################## Evaluation ########################################
def best_map(y_true, y_pred):
    """
    https://github.com/jundongl/scikit-feature/blob/master/skfeature/utility/unsupervised_evaluation.py
    Permute labels of y_pred to match y_true as much as possible
    """
    if len(y_true) != len(y_pred):
        print("y_true.shape must == y_pred.shape")
        exit(0)

    label_set = np.unique(y_true)
    num_class = len(label_set)

    G = np.zeros((num_class, num_class))
    for i in range(0, num_class):
        for j in range(0, num_class):
            s = y_true == label_set[i]
            t = y_pred == label_set[j]
            G[i, j] = np.count_nonzero(s & t)

    A = linear_assignment(-G)
    new_y_pred = np.zeros(y_pred.shape)
    for i in range(0, num_class):
        new_y_pred[y_pred == label_set[A[1][i]]] = label_set[A[0][i]]
    return new_y_pred.astype(int), label_set[A[1]], label_set[A[0]]

def evaluation(y_true, y_pred):
    y_pred_, label_original, label_truth = best_map(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred_)
    f1_macro = f1_score(y_true, y_pred_, average='macro')
    # f1_micro = f1_score(y_true, best_map(y_true, y_pred), average='micro')
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    # print('origi label', label_original)
    # print('truth label', label_truth)
    # print('recall', recall_score(y_true, y_pred_, average=None))
    # print('precision', precision_score(y_true, y_pred_, average=None))
    return acc, nmi, ari, f1_macro

######################################## vMF ########################################
def pdf_norm(dim, kappas):
    numerator = torch.pow(kappas, dim/2 -1)
    denominator = torch.pow(torch.mul(torch.pow(torch.ones_like(kappas)*2*math.pi, dim/2), iv(dim/2 -1, kappas)), -1)
    return torch.mul(numerator, denominator)

def A_d(dim, kappas):
    numerator = iv(dim/2, kappas)
    denominator = torch.pow(iv(dim/2 -1, kappas), -1)
    return torch.mul(numerator, denominator)

def estimate_kappa(dim, kappas):
    r = A_d(dim, kappas)
    numerator = dim*r - torch.pow(r, 3)
    denominator = torch.pow(1 - torch.pow(r, 2), -1)
    return torch.mul(numerator, denominator)

######################################## Visual ########################################
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.family'] = 'Times New Roman'
def visual(num_class, h, y, c, pred_q, save_path_truth, save_path_pred_q):
    h = np.vstack((h, c))
    # h_ = TSNE(n_components=2, init='pca', random_state=0, early_exaggeration=30).fit_transform(h)
    pca = PCA(n_components=2)
    h_ = pca.fit_transform(h)
    h = h_[:-c.shape[0]]
    c = h_[-c.shape[0]:]

    # # h_ = TSNE(n_components=2, init='pca', random_state=0, early_exaggeration=30).fit_transform(h)
    # pca = PCA(n_components=2)
    # h = pca.fit_transform(h)
    # c = pca.fit_transform(c)

    fig, ax = plt.subplots()
    # plt.xlim(-1.25, 1.25)
    # plt.ylim(-1.25, 1.25)
    for index, color in zip(range(num_class), ['tab:blue', 'tab:green', 'tab:orange', 'tab:pink', 'tab:purple', 'yellow', 'navy', 'black', 'tan', 'cyan']):
        mask = (y[:]==index)
        axis_0 = h[:, 0][mask]
        axis_1 = h[:, 1][mask]
        ax.scatter(axis_0, axis_1, c=color, label='cluster '+str(index), s=10, alpha=1, edgecolors='none')
    # ax.grid(True)
    plt.axis('off')
    # ax.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    plt.savefig(save_path_truth, bbox_inches='tight')
    
    # fig, ax = plt.subplots()
    # # plt.xlim(-1.25, 1.25)
    # # plt.ylim(-1.25, 1.25)
    # for index, color in zip(range(num_class), ['tab:blue', 'tab:green', 'tab:orange', 'tab:pink', 'tab:purple', 'yellow', 'navy', 'black', 'tan', 'cyan']):
    #     mask = (pred_p[:]==index)
    #     axis_0 = h[:, 0][mask]
    #     axis_1 = h[:, 1][mask]
    #     ax.scatter(axis_0, axis_1, c=color, label='cluster '+str(index), s=10, alpha=1, edgecolors='none')
    #     ax.scatter(c[index, 0], c[index, 1], c=color, label='center '+str(index), s=100, alpha=1, edgecolors='black')
    # # ax.grid(True)
    # ax.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    # plt.savefig(save_path_pred_p, bbox_inches='tight')

    fig, ax = plt.subplots()
    # plt.xlim(-1.25, 1.25)
    # plt.ylim(-1.25, 1.25)
    for index, color in zip(range(num_class), ['tab:blue', 'tab:green', 'tab:orange', 'tab:pink', 'tab:purple', 'yellow', 'navy', 'black', 'tan', 'cyan']):
        mask = (pred_q[:]==index)
        axis_0 = h[:, 0][mask]
        axis_1 = h[:, 1][mask]
        ax.scatter(axis_0, axis_1, c=color, label='cluster '+str(index), s=10, alpha=1, edgecolors='none')
        ax.scatter(c[index, 0], c[index, 1], c=color, label='center '+str(index), s=100, alpha=1, edgecolors='black')
    # ax.grid(True)
    plt.axis('off')
    # ax.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
    plt.savefig(save_path_pred_q, bbox_inches='tight')



    ####################################### SciPy ########################################
def obj_func(target, pred):
    target = target.reshape(pred.shape[0], pred.shape[1])
    loss = -np.mean(target * np.log(pred))
    return loss

def grad_func(target, pred):
    gradient = -np.log(pred)
    return np.ravel(gradient)

def cons_row(i, shape0, shape1):  
    return {'type':'eq', 'fun': lambda x: np.sum(x.reshape(shape0, shape1), axis=1)[i] - 1}  
def cons_col(j, shape0, shape1):
    return {'type':'eq', 'fun': lambda x: np.sum(x.reshape(shape0, shape1), axis=0)[j] - shape0/shape1} 
def cons_positive(k):
    return {'type':'ineq', 'fun': lambda x: x[k]}
def cons_orthogonal(j1, j2, shape0, shape1):
    return {'type':'eq', 'fun': lambda x: np.dot(x.reshape(shape0, shape1).T, x.reshape(shape0, shape1))[j1][j2]}

def re_assignment(pred):
    num_node = pred.shape[0]
    num_class = pred.shape[1]
    cons_1 = list(map(cons_row, list(range(num_node)), [num_node for i in range(num_node)], [num_class for i in range(num_node)]))
    cons_2 = list(map(cons_col, list(range(num_class)), [num_node for i in range(num_class)], [num_class for i in range(num_class)]))
    cons_3 = list(map(cons_positive, list(range(num_node*num_class))))
    cons_4 = list(map(cons_orthogonal, np.nonzero(np.eye(num_class)-1)[0].tolist(), np.nonzero(np.eye(num_class)-1)[1].tolist(), [num_node for i in range(num_class*(num_class-1))], [num_class for i in range(num_class*(num_class-1))]))
    cons = cons_1 + cons_2 + cons_3

    init_target = np.ravel(np.ones_like(pred)/num_class)

    res = minimize(fun=obj_func, x0=init_target, args=pred, jac=grad_func, constraints=cons)
    return res.success, res.x.reshape(num_node, num_class)

####################################### Greenhorn ########################################
def dist_pho(a, b):
    return b - a + a * np.log(a/b)

def greenkhorn(pred):
    num_node = pred.shape[0]
    num_class = pred.shape[1]
    p = np.power(pred, 1).T

    row = np.ones(num_node)
    col = np.ones(num_class)*(num_node/num_class)

    x = np.ones_like(row)
    y = np.ones_like(col)

    for index in range(1000):
        max_i = np.argmax(dist_pho(row, np.sum(p, axis=1)))
        max_j = np.argmax(dist_pho(col, np.sum(p, axis=0)))
        
        print(dist_pho(row[max_i], torch.sum(q, dim=1)[max_i]), dist_pho(col[max_j], torch.sum(q, dim=0)[max_j]))
        if dist_pho(row[max_i], torch.sum(q, dim=1)[max_i]) > dist_pho(col[max_j], torch.sum(q, dim=0)[max_j]) :
            x[max_i] = x[max_i] + row[max_i] / torch.sum(q, dim=1)[max_i]
        else:
            y[max_j] = y[max_j] + col[max_j] / torch.sum(q, dim=0)[max_j]
        q = torch.mm(torch.mul(p, torch.exp(x).unsqueeze(1)), torch.diag(torch.exp(y)))
    print(torch.sum(q, dim=1), torch.sum(q, dim=0))
    return q


##### DEC target distribution ########
def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    """
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


##### laplace matrix
def get_laplace_matrix(tensor_matrix):
    A = np.array(tensor_matrix)
    D = A.sum(axis=1)
    L_matrix = np.diag(D**(-0.5)).dot(A.dot(np.diag(D**(-0.5))))
    L_matrix = torch.tensor(L_matrix,dtype=torch.float)
    # print("L_matrix",torch.isnan(L_matrix))
    # labels_count = L_matrix.unique(return_counts=True)
    # print("label_count", torch.isnan(L_matrix).int().sum())
    return torch.nan_to_num(L_matrix)







###################################
# 为了处理单细胞数据而添加的

import json
import functools
import operator
import collections
import jgraph
import numpy as np
import scipy.sparse
import tqdm

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def in_ipynb():  # pragma: no cover
    try:
        # noinspection PyUnresolvedReferences
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True   # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def smart_tqdm():  # pragma: no cover
    if in_ipynb():
        return tqdm.tqdm_notebook
    return tqdm.tqdm


def with_self_graph(fn):
    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        with self.graph.as_default():
            return fn(self, *args, **kwargs)
    return wrapped


# Wraps a batch function into minibatch version
def minibatch(batch_size, desc, use_last=False, progress_bar=True):
    def minibatch_wrapper(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            total_size = args[0].shape[0]
            if use_last:
                n_batch = np.ceil(
                    total_size / float(batch_size)
                ).astype(np.int)
            else:
                n_batch = max(1, np.floor(
                    total_size / float(batch_size)
                ).astype(np.int))
            for batch_idx in smart_tqdm()(
                range(n_batch), desc=desc, unit="batches",
                leave=False, disable=not progress_bar
            ):
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, total_size)
                this_args = (item[start:end] for item in args)
                func(*this_args, **kwargs)
        return wrapped_func
    return minibatch_wrapper


# Avoid sklearn warning
def encode_integer(label, sort=False):
    label = np.array(label).ravel()
    classes = np.unique(label)
    if sort:
        classes.sort()
    mapping = {v: i for i, v in enumerate(classes)}
    return np.array([mapping[v] for v in label]), classes


# Avoid sklearn warning
def encode_onehot(label, sort=False, ignore=None):
    i, c = encode_integer(label, sort)
    onehot = scipy.sparse.csc_matrix((
        np.ones_like(i, dtype=np.int32), (np.arange(i.size), i)
    ))
    if ignore is None:
        ignore = []
    return onehot[:, ~np.in1d(c, ignore)].tocsr()


class CellTypeDAG(object):

    def __init__(self, graph=None, vdict=None):
        self.graph = jgraph.Graph(directed=True) if graph is None else graph
        self.vdict = {} if vdict is None else vdict

    @classmethod
    def load(cls, file):
        if file.endswith(".json"):
            return cls.load_json(file)
        elif file.endswith(".obo"):
            return cls.load_obo(file)
        else:
            raise ValueError("Unexpected file format!")

    @classmethod
    def load_json(cls, file):
        with open(file, "r") as f:
            d = json.load(f)
        dag = cls()
        dag._build_tree(d)
        return dag

    @classmethod
    def load_obo(cls, file):  # Only building on "is_a" relation between CL terms
        import pronto
        ont = pronto.Ontology(file)
        graph, vdict = jgraph.Graph(directed=True), {}
        for item in ont:
            if not item.id.startswith("CL"):
                continue
            if "is_obsolete" in item.other and item.other["is_obsolete"][0] == "true":
                continue
            graph.add_vertex(
                name=item.id, cell_ontology_class=item.name,
                desc=str(item.desc), synonyms=[(
                    "%s (%s)" % (syn.desc, syn.scope)
                 ) for syn in item.synonyms]
            )
            assert item.id not in vdict
            vdict[item.id] = item.id
            assert item.name not in vdict
            vdict[item.name] = item.id
            for synonym in item.synonyms:
                if synonym.scope == "EXACT" and synonym.desc != item.name:
                    vdict[synonym.desc] = item.id
        for source in graph.vs:
            for relation in ont[source["name"]].relations:
                if relation.obo_name != "is_a":
                    continue
                for target in ont[source["name"]].relations[relation]:
                    if not target.id.startswith("CL"):
                        continue
                    graph.add_edge(
                        source["name"],
                        graph.vs.find(name=target.id.split()[0])["name"]
                    )
                    # Split because there are many "{is_infered...}" suffix,
                    # falsely joined to the actual id when pronto parses the
                    # obo file
        return cls(graph, vdict)

    def _build_tree(self, d, parent=None):  # For json loading
        self.graph.add_vertex(name=d["name"])
        v = self.graph.vs.find(d["name"])
        if parent is not None:
            self.graph.add_edge(v, parent)
        self.vdict[d["name"]] = d["name"]
        if "alias" in d:
            for alias in d["alias"]:
                self.vdict[alias] = d["name"]
        if "children" in d:
            for subd in d["children"]:
                self._build_tree(subd, v)

    def get_vertex(self, name):
        return self.graph.vs.find(self.vdict[name])

    def is_related(self, name1, name2):
        return self.is_descendant_of(name1, name2) \
            or self.is_ancestor_of(name1, name2)

    def is_descendant_of(self, name1, name2):
        if name1 not in self.vdict or name2 not in self.vdict:
            return False
        shortest_path = self.graph.shortest_paths(
            self.get_vertex(name1), self.get_vertex(name2)
        )[0][0]
        return np.isfinite(shortest_path)

    def is_ancestor_of(self, name1, name2):
        if name1 not in self.vdict or name2 not in self.vdict:
            return False
        shortest_path = self.graph.shortest_paths(
            self.get_vertex(name2), self.get_vertex(name1)
        )[0][0]
        return np.isfinite(shortest_path)

    def conditional_prob(self, name1, name2):  # p(name1|name2)
        if name1 not in self.vdict or name2 not in self.vdict:
            return 0
        self.graph.vs["prob"] = 0
        v2_parents = list(self.graph.bfsiter(
            self.get_vertex(name2), mode=jgraph.OUT))
        v1_parents = list(self.graph.bfsiter(
            self.get_vertex(name1), mode=jgraph.OUT))
        for v in v2_parents:
            v["prob"] = 1
        while True:
            changed = False
            for v1_parent in v1_parents[::-1]:  # Reverse may be more efficient
                if v1_parent["prob"] != 0:
                    continue
                v1_parent["prob"] = np.prod([
                    v["prob"] / v.degree(mode=jgraph.IN)
                    for v in v1_parent.neighbors(mode=jgraph.OUT)
                ])
                if v1_parent["prob"] != 0:
                    changed = True
            if not changed:
                break
        return self.get_vertex(name1)["prob"]

    def similarity(self, name1, name2, method="probability"):
        if method == "probability":
            return (
                self.conditional_prob(name1, name2) +
                self.conditional_prob(name2, name1)
            ) / 2
        # if method == "distance":
        #     return self.distance_ratio(name1, name2)
        raise ValueError("Invalid method!")  # pragma: no cover

    def count_reset(self):
        self.graph.vs["raw_count"] = 0
        self.graph.vs["prop_count"] = 0  # count propagated from children
        self.graph.vs["count"] = 0

    def count_set(self, name, count):
        self.get_vertex(name)["raw_count"] = count

    def count_update(self):
        origins = [v for v in self.graph.vs.select(raw_count_gt=0)]
        for origin in origins:
            for v in self.graph.bfsiter(origin, mode=jgraph.OUT):
                if v != origin:  # bfsiter includes the vertex self
                    v["prop_count"] += origin["raw_count"]
        self.graph.vs["count"] = list(map(
            operator.add, self.graph.vs["raw_count"],
            self.graph.vs["prop_count"]
        ))

    def best_leaves(self, thresh, retrieve="name"):
        subgraph = self.graph.subgraph(self.graph.vs.select(count_ge=thresh))
        leaves, max_count = [], 0
        for leaf in subgraph.vs.select(lambda v: v.indegree() == 0):
            if leaf["count"] > max_count:
                max_count = leaf["count"]
                leaves = [leaf[retrieve]]
            elif leaf["count"] == max_count:
                leaves.append(leaf[retrieve])
        return leaves


class DataDict(collections.OrderedDict):

    def shuffle(self, random_state=np.random):
        shuffled = DataDict()
        shuffle_idx = None
        for item in self:
            shuffle_idx = random_state.permutation(self[item].shape[0]) \
                if shuffle_idx is None else shuffle_idx
            shuffled[item] = self[item][shuffle_idx]
        return shuffled

    @property
    def size(self):
        data_size = set([item.shape[0] for item in self.values()])
        assert len(data_size) == 1
        return data_size.pop()

    @property
    def shape(self):  # Compatibility with numpy arrays
        return [self.size]

    def __getitem__(self, fetch):
        if isinstance(fetch, (slice, np.ndarray)):
            return DataDict([
                (item, self[item][fetch]) for item in self
            ])
        return super(DataDict, self).__getitem__(fetch)


def densify(arr):
    if scipy.sparse.issparse(arr):
        return arr.toarray()
    return arr


def empty_safe(fn, dtype):
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)
    return _fn


decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)
encode = empty_safe(np.vectorize(lambda _x: str(_x).encode("utf-8")), "S")
upper = empty_safe(np.vectorize(lambda x: str(x).upper()), str)
lower = empty_safe(np.vectorize(lambda x: str(x).lower()), str)
tostr = empty_safe(np.vectorize(str), str)