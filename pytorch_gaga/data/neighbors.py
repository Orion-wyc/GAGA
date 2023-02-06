import torch
import dgl
import numpy as np
import os
import copy
import pickle


class MultiHopFullNeighborSampler:
    """Sampler that collects all the neighbors within R hops.
    """
    store_args_list = ['dt_name']
    log_domain = 'MultiHopFullNeighborSampler'

    def __init__(self, graphs: dict, n_hops, store=False, **kwargs):
        self.store_kwargs = {}
        if store:
            for k, v in kwargs.items():
                if k in self.store_args_list:
                    self.store_kwargs[k] = v
            assert 'dt_name' in self.store_kwargs, \
                "If store=True, 'data_name' should be specified. (e.g. dt_name='yelp')"

        self.graphs = copy.deepcopy(graphs)

        # Remove homo graphs
        if 'homo' in self.graphs.keys():
            self.graphs.pop('homo')

        self.relations = list(self.graphs.keys())

        self.n_hops = n_hops
        self.store = store
        self.mr_neighbors_dict = {}

        self.init()

    def init(self):
        # If subgraph file exists, directly load from disk.
        # Otherwize, create file and stored sampled subgraphs.
        if not self.store:
            for k, v in self.graphs.items():
                print(f'[{self.log_domain}] Sampling on relation {k}')
                self.mr_neighbors_dict[k] = self._k_hop_neighbors(k, self.n_hops)
        else:
            DATA_DIR = os.path.dirname(__file__)
            fn_dir = os.path.join(DATA_DIR, f'../subgraphs/{self.store_kwargs["dt_name"]}')

            # Check if all pickle objects are available
            if self._check_path(fn_dir):
                for k in self.relations:
                    fn_path = os.path.join(fn_dir, f'{self.n_hops}_hops_{k}.pkl')
                    print(f'[{self.log_domain}] Loading subgraphs\' nids from {fn_path} for relation {k}')

                    with open(fn_path, 'rb') as f:
                        nb_list = pickle.load(f)

                    self.mr_neighbors_dict[k] = nb_list
            else:
                os.makedirs(fn_dir, exist_ok=True)
                for k, v in self.graphs.items():
                    fn_path = os.path.join(fn_dir, f'{self.n_hops}_hops_{k}.pkl')
                    print(f'[{self.log_domain}] Sampling on relation {k}')

                    nb_list = self._k_hop_neighbors(k, self.n_hops)
                    self.mr_neighbors_dict[k] = nb_list
                    with open(fn_path, 'wb') as f:
                        pickle.dump(nb_list, f)

    def _check_path(self, fn_dir):
        for k in self.graphs.keys():
            fn_path = os.path.join(fn_dir, f'{self.n_hops}_hops_{k}.pkl')
            if not os.path.exists(fn_path):
                return False

        return True

    def _k_hop_neighbors(self, relation, k_hop=0):
        rel_g = self.graphs[relation]
        all_nids = rel_g.nodes()

        neighbor_list = []
        for nid in all_nids:
            nbs = nid.item()
            hop_nb_list = [nbs]
            for i in range(k_hop):
                nbs = rel_g.in_edges(nbs)[0]
                hop_nb_list.extend(nbs.unique().tolist())

            neighbor_list.append(hop_nb_list)

        return neighbor_list

    def load_neighbors(self, relation: str):
        assert relation in self.relations, f"No relation named {relation}"
        return self.mr_neighbors_dict[relation]
