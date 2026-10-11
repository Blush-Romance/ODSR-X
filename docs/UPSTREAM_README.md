# ODSR：Douban 0.189761 对应的三个数据集实验

三个实验使用相同的六个模型与训练源文件。每个数据集的实际启动参数及最佳结果如下：

| 数据集 | Recall@20 | NDCG@20 | 最佳轮次 | 实验代码和启动配置 |
|---|---:|---:|---:|---|
| Douban-Book | 0.189761 | 0.172059 | 10 | [根目录](run.sh) |
| Epinions | 0.086273 | 0.049644 | 4 | [experiments/epinions](experiments/epinions) |
| Yelp2018 | 0.080804 | 0.053348 | 3 | [experiments/yelp](experiments/yelp) |

Yelp 实际运行使用 GPU 候选激活重计算入口 `train_gpu_memory.py` 来控制显存；该入口已与实验源文件一起保留。两项新增实验均固定取 8 个社交邻居，从随机初始化训练。

以下为根目录 Douban 实验的说明；Epinions 和 Yelp 的运行方式见各自目录中的 README。


本仓库保存 2026-10-09 Douban-Book 实验实际使用的完整可读代码、启动脚本、生效参数和复现说明。该实验从随机初始化训练，第 10 轮获得最佳 Recall@20 **0.189761**。

| 指标 | 本次实验 | 原 RTX 4080 参考实验 |
|---|---:|---:|
| Recall@10 | 0.134551 | 0.134064 |
| Recall@20 | **0.189761** | 0.189176 |
| NDCG@10 | 0.165671 | 0.165515 |
| NDCG@20 | 0.172059 | 0.172108 |

## 完整源文件

`source/` 包含 `main.py`、`ODSR.py`、`ODSR_base.py`、`ODSR_legacy.py`、`common.py`、`utils.py`。六个文件与实验实际使用的源文件逐字节一致，包含可读的骨干实现，无外部冻结 `.pyc` 依赖。`run.sh` 也保留原始训练命令。完整生效配置在 `experiment_config.json`，实际结果和环境摘要在 `result_summary.json`。

## 准备数据与环境

公共仓库提供代码及复现配置。训练数据、历史模型和原始运行日志保留在本地。

将原实验的 Douban-Book 数据划分放在以下位置；`dataset_checksums.json` 提供校验值：

```text
dataset/douban-book/train.txt
dataset/douban-book/test.txt
dataset/douban-book/trust.txt
```

0.189761 实验环境：Python 3.9.25、PyTorch 2.8.0+cu128、CUDA runtime 12.8、NumPy 2.0.2、SciPy 1.13.1，GPU 为 RTX 5070 Ti。

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==2.0.2 scipy==1.13.1
sha256sum -c SHA256SUMS
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHON_BIN=python bash run.sh
```

新训练结果写入 `outputs/best.pth` 和 `outputs/training.log`。执行脚本会更新 `outputs/` 内文件。原 RTX 4080 参考环境为 Python 3.9.25、PyTorch 2.7.1+cu118、NumPy 1.24.4、SciPy 1.10.1。不同环境重新训练可能产生数值差异；上述指标来自原完整实验记录。

## 参数与逻辑

| 参数 | 值 |
|---|---:|
| seed | 2026 |
| learning rate | 0.002 |
| reg / ssl_reg / eps | 0.00001 / 0.5 / 0.8 |
| batch_size / dropout | 2048 / 0.01 |
| embedding dimension / graph layers | 64 / 3 |
| SSL temperature | 0.2 |
| max_epoch / patience | 100 / 50 |
| eval_interval | 1 |
| eval_batch / eval_item_batch | 64 / 512 |
| 用户历史上限 / Top-L | 64 / 8 |
| 邻居数 | 8 |
| 邻居历史上限 / Top-L | 16 / 4 |
| 候选校正范围 | backbone Top-512 未见物品 |
| candidate_loss_weight | 0.2 |
| hard_pool_size / hard_topk | 64 / 8 |

邻居按社交边权排序固定取前 8 个，同权按邻居 ID 排序。候选表示由可学习 Q/K/V 检索历史物品生成；每个邻居先对最多 16 条历史计算，再取 Top-4。本实验使用未截断的骨干 sigmoid 门控、完整历史检索，无 79 的两阶段粗筛。负采样排除训练历史，使用独立 NumPy RNG，并保持每行负样本互异。

评估使用完整物品目录并屏蔽训练交互，报告 Recall/NDCG@10、@20；最优模型由 Recall@20 选择。

## 评估训练所得模型

```bash
python source/main.py --data_root dataset --dataset douban-book --device cuda \
  --ckpt outputs/best.pth --eval_batch 64 --eval_item_batch 512
```

## 验证

六个源文件和启动脚本与原实验校验值一致；本地原最佳模型通过严格参数加载，并在 CPU 上为两个用户生成有限的完整目录评分。此检查未重新计算全数据集指标。
