## Description

The PaddlePaddle and PyTorch implementation of GAGA (**Label Information Enhanced Fraud Detection against Low Homophily in Graphs**, WWW '23).

**Abstract:** Node classification is a substantial problem in graph-based fraud detection. Many existing works adopt Graph Neural Networks (GNNs) to enhance fraud detectors. While promising, currently most GNN-based fraud detectors fail to generalize to the low homophily setting. Besides, label utilization has been proved to be significant factor for node classification problem. But we find they are less effective in fraud detection tasks due to the low homophily in graphs. In this work, we propose GAGA, a novel Group AGgregation enhanced TrAnsformer, to tackle the above challenges. Specifically, the group aggregation provides a portable method to cope with the low homophily issue. Such an aggregation explicitly integrates the label information to generate distinguishable neighborhood information. Along with group aggregation, an attempt towards end-to-end trainable group encoding is proposed which augments the original feature space with the class labels. Meanwhile, we devise two additional learnable encodings to recognize the structural and relational context. Then, we combine the group aggregation and the learnable encodings into a Transformer encoder to capture the semantic information. Experimental results clearly show that GAGA outperforms other competitive graph-based fraud detectors by up to 24.39% on two trending public datasets and a real-world industrial dataset from Baidu. Even more, the group aggregation is demonstrated to outperform other label utilization methods (e.g., C&S, BoT/UniMP) in the low homophily setting.



![](https://github.com/Orion-wyc/GAGA/blob/master/images/gaga_overview.png)

## Reproduction Tutorial

1. Setup.

   - Two public dataset YelpChi and Amazon will be downloaded automatically at first run. Alternatively, you can download both datasets from [Github](https://github.com/YingtongDou/CARE-GNN).

   - Requirements

     Details in `gaga_env.yaml`, [conda](https://docs.conda.io/en/latest/) (an open source package management tool) is recommended.

     ```bash
     conda env create -f gaga_env.yaml
     ```

2. Prepare sequence data.

   ```bash
   cd preprocessing
   
   # dataset spliting
   python dataset_split.py --dataset yelp  --save_dir seq_data --train_size 0.4 --val_size 0.1
   python dataset_split.py --dataset amazon  --save_dir seq_data --train_size 0.4 --val_size 0.1
   
   # preprocess feature sequence
   python graph2seq_mp.py --dataset yelp --fanouts -1 -1  --save_dir seq_data --train_size 0.4 --val_size 0.1 --n_workers 8 --add_self_loop --norm_feat
   python graph2seq_mp.py --dataset amazon --fanouts -1 -1  --save_dir seq_data --train_size 0.4 --val_size 0.1 --n_workers 8 --add_self_loop --norm_feat
   ```

3. Run `main_transformer.py`

   ```bash
   python main_transformer.py --config configs/yelpchi_paper.json --gpu 0  --log_dir logs --early_stop 100
   python main_transformer.py --config configs/amazon_paper.json --gpu 1  --log_dir logs --early_stop 100
   ```



## Acknowledgements

GAGA is inspired by the recent success of graph-based fraud detectors (i.e. [CARE-GNN](https://github.com/YingtongDou/CARE-GNN.), [PC-GNN](https://github.com/PonderLY/PC-GNN), [RioGNN](https://github.com/safe-graph/RioGNN), etc.) and label utilization in node classification tasks (i.e. [BoT](https://arxiv.org/abs/2103.13355), [UniMP](https://www.ijcai.org/proceedings/2021/0214.pdf), etc.). We also thank the authors for sharing their codes.
