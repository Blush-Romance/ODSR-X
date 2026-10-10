# Yelp2018：Douban 0.189761 对应代码版本

本实验的六个 Python 源文件与仓库根目录的 Douban 0.189761 实验逐字节一致。保存此实验实际使用的源码快照、启动参数和结果摘要。

最佳轮次：**3**；Recall@20：**0.080804**。

| 指标 | 值 |
|---|---:|
| Recall@10 | 0.048999 |
| Recall@20 | 0.080804 |
| NDCG@10 | 0.041954 |
| NDCG@20 | 0.053348 |

参数：lr=0.001、reg=0.0、ssl_reg=0.43、eps=0.54、dropout=0.17、batch_size=2048、seed=2026、max_epoch=100、patience=50、eval_interval=1、eval_batch=64、eval_item_batch=512。

模型设置沿用 Douban 对应版本：按社交边权固定取前 8 个邻居，同权按邻居 ID 排序；用户历史上限 64、Top-8，邻居历史上限 16、Top-4；骨干门控不截断，无两阶段历史粗筛。

从随机初始化训练，不加载历史模型。本次 Yelp 实际运行使用 `train_gpu_memory.py` 的非重入 checkpoint 重计算候选激活，保留原始 batch=2048，不使用 CPU offload。必须保留此入口才能匹配保存的实验运行方式；原有小批次损失与梯度一致性检查记录在 `gpu_memory_validation.json`。

## 运行

在仓库根目录安装原环境依赖（见根 README），将对应数据划分放在 `dataset/yelp2018/train.txt`、`test.txt`、`trust.txt`；校验值见本目录 `dataset_checksums.json`。

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHON_BIN=python bash experiments/yelp/run.sh
```

可通过 `DATA_ROOT=/path/to/dataset` 指定包含 `yelp2018` 子目录的数据根目录，通过 `OUTPUT_DIR=/path/to/new-output` 指定输出目录。默认写入 `experiments/yelp/outputs/`，再次执行会更新该目录的模型和日志。

`run_config.json` 记录运行方式和显式参数；`experiment_config.json` 记录主程序解析后的全部生效配置；`result_summary.json` 记录实际完成实验的最佳指标。公共仓库提供源码及复现配置，训练数据、模型和原始日志保留在本地。

## 发布前检查

六个源文件与 Douban 快照一致；启动脚本展开的参数与原实验实际命令一致。`validation.json` 记录本地历史模型的严格加载及两个用户完整目录评分检查；此检查不重新训练，也不重新计算全数据集指标。
