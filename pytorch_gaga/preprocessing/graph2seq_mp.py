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

import multiprocessing as mp
import argparse
import os
import numpy as np
import dgl
import torch
from tqdm import tqdm
import time

import data_utils


def save_sequence(args, data):
    pass


def graph2seq(pid, args, st, ed, nids, graph_data, sequence_array):
    seq_loader = data_utils.GroupFeatureSequenceLoader(graph_data, fanouts=args['fanouts'],
                                                       grp_norm=args['grp_norm'])
    nids = torch.from_numpy(nids)
    # 可以改成多进程
    seq_feat = seq_loader.load_batch(nids, pid=pid)
    sequence_array[st:ed] = seq_feat.numpy()


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
    parser.add_argument('--save_dir', type=str, default='mp_output',
                        help='Directory for saving the processed sequence data.')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Using n processes.')
    # BF10M base_dir = /home/work/wangyuchen09/Projects/fraud_detection/dataset/fraud_20211209
    args = vars(parser.parse_args())
    print(args)

    # 读取data
    graph_data = data_utils.prepare_data(args, add_self_loop=args['add_self_loop'])

    # 基本数据
    g = graph_data.graph
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

    # 聚合序列文件
    flag_1 = 'grp_norm' if args['grp_norm'] else 'no_grp_norm'
    flag_2 = 'norm_feat' if args['norm_feat'] else 'no_norm_feat'
#     flag_3 = 'self_loop' if args['add_self_loop'] elsr 'no_self_loop'
    file_name = f"{args['dataset']}_{flag_1}_{flag_2}_{n_hops}_" \
                f"{args['train_size']}_{args['val_size']}_{args['seed']}.npy"
    seq_file = os.path.join(file_dir, file_name)
    print(f"Saving seq_file to {seq_file}")
    sequence_array = np.memmap(seq_file, dtype=np.float32, mode='w+', shape=(n_nodes, seq_len, feat_dim))
        
    procs = []
    n_workers = args['n_workers']

    nids = g.nodes().numpy()
    block_size = nids.shape[0] // n_workers + 1
    
    tic = time.time()
    
    for pid in range(n_workers):
        st = pid * block_size
        ed = min((pid + 1) * block_size, n_nodes)
        p = mp.Process(target=graph2seq, args=(pid, args, st, ed, nids[st:ed], graph_data, sequence_array))
        procs.append(p)
        p.start()     
    
    for p in procs:
        p.join()

    sequence_array.flush()
    
    toc = time.time()
    print(f"Elapsed Tiem = {toc -tic:.2f}(s)")
