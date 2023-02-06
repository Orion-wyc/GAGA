"""
预处理数据集, 获取每个点的0~K跳邻居结点, 每一跳邻居分组聚合(mean).
给定一个具有 $R$ 个关系的图, 分组数为 $P$, 跳数为 $K$, 则对于每个
点, 处理后获得的序列长度为 $seq_len = R \times (P \times K + 1)$,
特征维度feat_dim $d$ 保持不变.
流程：
1. 读取数据集,获得图结构和特征、标签、train/val/test划分方式
2. 按照论文中的分组聚合方式处理数据，保存序列为numpy.memmap
   shape=(n_nodes, seq_len, d).
3. 设计FeatureSequence类，可以包含以下成员
   步骤2的特征, 标签信息, 数据集nid划分train/val/test

Tips: seq分为两组norm与no_norm, 即在group_aggregation时每个分组上使用
      $\frac{1}{\sqrt { |h_g| }}$ 归一化

"""

import argparse
import os
import numpy as np
import dgl
import torch
import time
from tqdm import tqdm

import data_utils


def save_sequence(args, data):
    pass


def graph2seq(args, graph_data):
    group_loader = data_utils.GroupFeatureSequenceLoader(graph_data)
    g = graph_data.graph
    labels = graph_data.labels
    train_nid = graph_data.train_nid
    val_nid = graph_data.val_nid
    test_nid = graph_data.test_nid
    n_classes = graph_data.n_classes

    feat_dim = graph_data.feat_dim
    n_relations = graph_data.n_relations
    n_groups = n_classes + 1
    n_hops = len(args['fanouts'])
    n_nodes = g.num_nodes()

    seq_len = n_relations * (n_hops * n_groups + 1)

    all_nid = g.nodes()
    file_dir = os.path.join(args['save_dir'], args['dataset'])
    os.makedirs(file_dir, exist_ok=True)

    infos = np.array([feat_dim, n_classes, n_relations], dtype=np.int64)
    
    info_name = f"{args['dataset']}_infos_" \
                f"{args['train_size']}_{args['val_size']}_{args['seed']}.npz"
    info_file = os.path.join(file_dir, info_name)
    print(f"Saving infos to {info_file}")
    np.savez(info_file, label=labels.numpy(), train_nid=train_nid, val_nid=val_nid, 
             test_nid=test_nid, infos=infos)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='graph2seq')
    parser.add_argument('--dataset', type=str, default='amazon',
                        help='Dataset name, [amazon, yelp, BF10M]')

    parser.add_argument('--train_size', type=float, default=0.4,
                        help='Train size of nodes.')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Val size of nodes.')
    parser.add_argument('--seed', type=int, default=717,
                        help='Collecting neighbots in n hops.')

    parser.add_argument('--norm_feat', action='store_true', default=False,
                        help='Using group norm, default False')
    parser.add_argument('--grp_norm', action='store_true', default=False,
                        help='Using group norm, default False')
    parser.add_argument('--force_reload', action='store_true', default=False,
                        help='Using group norm, default False')
    parser.add_argument('--add_self_loop', action='store_true', default=False,
                    help='add self-loop to all the nodes')

#     parser.add_argument('--n_hops', type=int, default=1,
#                         help='Collecting neighbots in n hops.')
    parser.add_argument('--fanouts', type=int, default=[-1], nargs='+',
                        help='Sampling neighbors, default [-1] means full neighbors')

    # TODO (yuchen) 后面添加 fanouts 版本
    parser.add_argument('--base_dir', type=str, default='~/.dgl',
                        help='Directory for loading graph data.')
    parser.add_argument('--save_dir', type=str, default='seq_data',
                        help='Directory for saving the processed sequence data.')
    # BF10M base_dir = /home/work/wangyuchen09/Projects/fraud_detection/dataset/fraud_20211209
    args = vars(parser.parse_args())
    # 转化fanoutsw为int list
#     fanouts = args['fanouts'].split(',')
#     args['fanouts'] = [int(n) for n in fanouts]
    print(args)
    tic = time.time()
    data = data_utils.prepare_data(args)
    graph2seq(args, data)
    toc = time.time()
    print(f"Elapsed time={toc -tic:.2f}(s)")
