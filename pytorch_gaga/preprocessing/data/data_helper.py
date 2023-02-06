import copy
import dgl
import numpy as np
import torch
from dgl import function as fn
from scipy import sparse as sp
from sklearn import preprocessing

from data import fraud_dataset


def normalize(feats, train_nid, dtype=np.float32):
    """Normalizing node features.
    Reference: <https://github.com/PonderLY/PC-GNN/blob/main/src/utils.py>

    :param feats:
    :param train_nid:
    :param dtype:
    :return:
    """
    train_feats = feats[train_nid]
    scaler = preprocessing.StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)
    return feats.astype(dtype)


def row_normalize(mx, dtype=np.float32):
    """Row-normalize sparse matrix
    Reference: <https://github.com/williamleif/graphsage-simple>

    :param mx:
    :param dtype:
    :return:
    """
    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx.astype(dtype)


def load_graphs(dataset_name='amazon', raw_dir='~/.dgl/', train_size=0.4, val_size=0.1,
                seed=717, norm=True, force_reload=False, verbose=True) -> dict:
    """Loading dataset from dgl's FraudDataset.
    这里的设计目前是冗余且不必要的,可以直接使用dgl的异构图来处理.
    为了兼容后期的数据集, 将每一张图单独处理
    """
    if dataset_name in ['amazon', 'yelp', 'mimic']:
        fraud_data = fraud_dataset.FraudDataset(dataset_name, train_size=train_size, val_size=val_size,
                                                random_seed=seed, force_reload=force_reload)
    # elif dataset_name in ['BF10M']:
    #     fraud_data = baidu_dataset.BaiduFraudDataset(dataset_name, raw_dir=raw_dir,
    #                                                  train_size=train_size, val_size=val_size,
    #                                                  random_seed=seed, force_reload=force_reload)
    # dgl 下一个版本可以取消下面的注释
    # fraud_data = dgl.data.FraudDataset(dataset_name, train_size=train_size, val_size=val_size,
    #                                    random_seed=seed, force_reload=force_reload)
    g = fraud_data[0]

    # Feature tensor dtpye is float64, change it to float32
    if norm and (dataset_name not in ['BF10M']):
        # train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
        # h = normalize(g.ndata['feature'], train_nid, dtype=np.float32)
        h = row_normalize(g.ndata['feature'], dtype=np.float32)
        g.ndata['feature'] = torch.from_numpy(h)
    else:
        g.ndata['feature'] = g.ndata['feature'].float()

    # label shape is (n,1), reshape it to be (n, )
    lb = g.ndata['label'].squeeze().long()
    g.ndata['label'] = lb

    graphs = {}
    for etype in g.etypes:
        graphs[etype] = g.edge_type_subgraph([etype])

    # By default, the returned homogeneous graph will not have any node features.
    # Reloading homogeneous graphs from <.mat> file due to the bug in <dgl.metis_partition>
    # Or, using <to_simple> to remove redundant and duplicate edges.
    # name_dict = {"yelp": "YelpChi.mat", "amazon": "Amazon.mat"}
    # mat_path = os.path.join('~/.dgl', '{}/{}'.format(dataset_name, name_dict[dataset_name]))
    # mat = loadmat(mat_path)
    # graphs['homo'] = dgl.from_scipy(mat['homo'])
    g_homo = dgl.to_homogeneous(g)
    # remove duplicate edges
    graphs['homo'] = dgl.to_simple(g_homo)
    for key, value in g.ndata.items():
        graphs['homo'].ndata[key] = value

    return graphs


def calc_weight(g):
    """Compute row_normalized(D^(-1/2)AD^(-1/2)).
    Reference: <>

    :param g: DGLGraph.
        The homogeneous graph is used for calculating normalized edge weights.
    :return: FloatTensor.
        Edge weights.
    """
    with g.local_scope():
        # @todo (yuchen) 这里原本是 $\hat A = D^{-1/2}AD^{-1/2}$, 假设A=I,后期改一下
        # Computing D^(-0.5)*D(-1/2), assuming A is Identity
        g.ndata["in_deg"] = g.in_degrees().float().pow(-0.5)
        g.ndata["out_deg"] = g.out_degrees().float().pow(-0.5)
        g.apply_edges(fn.u_mul_v("out_deg", "in_deg", "weight"))

        # Row-normalize weight
        g.update_all(fn.copy_e("weight", "msg"), fn.sum("msg", "norm"))
        g.apply_edges(fn.e_div_v("weight", "norm", "weight"))
        return g.edata["weight"]


def preprocess(args, g, features):
    """Pre-compute the average of n-th hop neighbors.
    Reference: <>

    :param g: DGLGraph.
        A homogeneous graph with specific single relation "net_xxx".
    :param features: FloatTensor.
        The input node features.
    :param args: dict.
        Arguments used for preprocessing multi-hop (donated as n_hops) averaged features.
    :return: list
        A list that contains [0, args['n_hops']] hop's averaged node features.
        hop_feat_list = [feat_0, feat_1,...,feat_R]
    """
    # g = dgl.to_homogeneous(g)
    with torch.no_grad():
        g.edata["weight"] = calc_weight(g)
        g.ndata["feat_0"] = features
        for hop in range(1, args['n_hops'] + 1):
            g.update_all(fn.u_mul_e(f"feat_{hop - 1}", "weight", "msg"),
                         fn.sum("msg", f"feat_{hop}"))
        hop_feat_list = []
        for hop in range(args['n_hops'] + 1):
            hop_feat_list.append(g.ndata.pop(f"feat_{hop}"))
        return hop_feat_list


def mask_nodes(nids, label_rate=0.5, seed=717):
    index = np.arange(nids.shape[0])
    index = np.random.RandomState(seed).permutation(index)
    unmasked_idx = index[:int(label_rate * len(index))]
    masked_idx = index[int(label_rate * len(index)):]
    unmasked_nid = nids[unmasked_idx]
    masked_nid = nids[masked_idx]

    return unmasked_nid, masked_nid
    
    
def add_label_emb(unmasked_nid, features, labels):
    n_nodes = features.shape[0]
    pos_nid, neg_nid = pos_neg_split(unmasked_nid, labels[unmasked_nid])
    padding_feat = torch.zeros((n_nodes,3))
    padding_feat[:, -1]  = 1
    padding_feat[unmasked_nid, -1] = 0

    if pos_nid.shape != torch.Size([0]):
        padding_feat[pos_nid, 0] = 1

    if neg_nid.shape != torch.Size([0]):
        padding_feat[neg_nid, 1] = 1

    new_feat = torch.cat([features, padding_feat], dim=1)

    return new_feat 
    
    
def prepare_data(args):
    """Preparing training data.

    :param args: dict
        Arguments for loading datasets and pre-computing multi-hop neighbours' node features.
    :return: tuple
        Training data.
        feat_list is a list that contains $|relations|$ hop_feat_lists.
        [[feat_0, feat_1,...,feat_R],  // hop-0
         [feat_0, feat_1,...,feat_R],  // hop-1
         ...,
         [feat_0, feat_1,...,feat_R]]  // hop-R
    """
    graphs = load_graphs(dataset_name=args['dataset'], raw_dir=args['base_dir'],
                         train_size=args['train_size'], val_size=args['val_size'],
                         seed=args['seed'], norm=args['norm_feat'],
                         force_reload=args['force_reload'])

    # MR-Graphs share same {feat,label,mask}, here we can load homo_g
    g = graphs['homo']

    # Processing mask
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    train_nid = torch.nonzero(train_mask, as_tuple=True)[0]
    val_nid = torch.nonzero(val_mask, as_tuple=True)[0]
    test_nid = torch.nonzero(test_mask, as_tuple=True)[0]

    # Processing labels
    n_classes = 2
    in_feats = g.ndata['feature'].shape[1]
    labels = g.ndata['label'].squeeze().long()

    # Pre-computing node features for each relation.
    feat_list = []
    for k, vg in graphs.items():
        # @todo minic dataset是rel开头的
        if k.startswith('net') or k.startswith('rel'):
            feats = preprocess(args, vg, vg.ndata['feature'].float())
            feat_list.append(feats)

    print(f"[Global] Dataset <{args['dataset']}> Overview\n"
          f"\tEntire (postive/total) {torch.sum(labels):>6} / {labels.shape[0]:<6}\n"
          f"\tTrain  (postive/total) {torch.sum(labels[train_nid]):>6} / {labels[train_nid].shape[0]:<6}\n"
          f"\tValid  (postive/total) {torch.sum(labels[val_nid]):>6} / {labels[val_nid].shape[0]:<6}\n"
          f"\tTest   (postive/total) {torch.sum(labels[test_nid]):>6} / {labels[test_nid].shape[0]:<6}\n")

    return feat_list, labels, in_feats, n_classes, train_nid, val_nid, test_nid


def load_batch(batch, feat_list, device='cpu'):
    """Loading a subset of features for each relation as a batch.

    :param batch: Tensor
        Node's nids for loading node features as a batch.
    :param feat_list: list
        List of hop_feat_lists that contains the averaged multi-hop node features.
    :param device: str
    :return: list
        A list contains batched node features for each relation (asuming there are $P$ relations).
        [batch_feat_list_0, batch_feat_list_1, ..., batch_feat_list_P]
    """
    batch_feat_list = []
    for hop_feat_list in feat_list:
        batch_feats = [feat[batch] for feat in hop_feat_list]
        batch_feat_list.append(batch_feats)

    batch_feat_list = [torch.stack(feat) for feat in batch_feat_list]
    batch_feats = torch.cat(batch_feat_list, dim=0)
    # if len(batch_feats.shape) == 2:
    #     batch_feats = batch_feats.unsqueeze(1)

    return batch_feats.to(device)


# deprecated
def _pos_neg_split(nids, labels):
    """Split positive and negtive nodes in array nids .
    @todo 大规模性能太差需要改进
    :param nids: FloatTensor
    :param labels: LongTensor
    :return: tuple (LongTensor, LongTensor)
    """
    # nids = nids.cpu().tolist()
    pos_nids = []
    neg_nids = []
    for nid in nids:
        if labels[nid] == 1:
            pos_nids.append(nid.item())
        else:
            neg_nids.append(nid.item())
    # torch.int64
    pos_nids = torch.tensor(pos_nids)
    neg_nids = torch.tensor(neg_nids)
    return pos_nids, neg_nids


def pos_neg_split(nids, labels):
    pos_idx = torch.where(labels == 1)[0]
    neg_idx = torch.where(labels == 0)[0]

    # 特殊判断孤立点的情况
    pos_nids = nids[pos_idx] if min(pos_idx.shape) != 0 else torch.LongTensor([])
    neg_nids = nids[neg_idx] if min(neg_idx.shape) != 0 else torch.LongTensor([])

    return pos_nids, neg_nids


def under_sample(pos_nids, neg_nids, scale=1):
    """Under-sample the negative nodes based on scale.

    :param pos_nids: LongTensor
    :param neg_nids: LongTensor
    :param scale: float
    :param seed: int
    :return:
    """
    index = np.arange(neg_nids.shape[0])
    index = np.random.RandomState().permutation(index)
    N = min(int(pos_nids.shape[0] * scale), neg_nids.shape[0])
    index = index[0: N]
    neg_sampled = neg_nids[index]
    sampled_nids = torch.cat((pos_nids, neg_sampled))

    return sampled_nids
