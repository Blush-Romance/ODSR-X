# ODSR 重构与性能提升 — 项目进度

## 当前阶段与模式
MODE B（Step-0 零成本探针）。用户 2026-10-10 授权：可彻底重构、换/缝合 baseline，目标=超过近两年方法，最好超过 ODSR(KDD26) 本身。

## 当前执行服务器、项目路径与 Git commit
- CCNU1 `/mnt/data/zzx/Code/ODSR`，上游 commit 075733b（nicoco96/odsr-douban-recall-0189761）
- 环境 `/mnt/data/zzx/conda_envs/odsr`：Python 3.9 / torch 2.8.0+cu128 / numpy 2.0.2 / scipy 1.13.1
- 参考仓库：`../ODSR_official`（KDD26 官方，含 LFS checkpoint）、`../ODSR_ref_GBSR`、`../ODSR_ref_SELFRec`
- **GPU 阻塞**：CCNU1 GPU1(4090, PCI 05:00.0) 掉卡，导致 CUDA 驱动初始化整体失败（4080 也不可用），需管理员重启；CCNU2 两卡被他人占满且内存 0 空闲。目前只能 CPU。

## 数据
| 数据集 | 来源 | 校验 |
|---|---|---|
| douban-book | SELFRec 仓库 dataset/douban-book | 与上游 dataset_checksums.json 逐字节一致 |
| epinions | GBSR 公开 npy 原样导出 | 逐字节一致 |
| yelp-gbsr | GBSR 公开 npy 导出（probes/conv_gbsr_npy.py） | 上游 Yelp2018 不是这份（校验不匹配）；官方 ODSR 仓库的 yelp2018 待 LFS 下载 |

## 评估协议（注意）
上游与 GBSR/SGIL 相同：80/20，无验证集，每轮在测试集评估并按测试集 Recall@20 选最佳轮次。所有对比方法必须同协议；论文中如实写明。

## 近两年对手（同 GBSR 划分，论文报告值 R@20/N@20）
| 方法 | Douban | Epinions | Yelp(GBSR) |
|---|---|---|---|
| GBSR KDD24 | .1694/.1523 | .0793/.0464 | .1243/.0724 |
| SGIL SIGIR25 | .1809/.1627 | .0844/.0489 | .1323/.0787 |
| ODSR KDD26（论文） | .1867/.1669 | .0885/.0509 | Yelp2018 另一版本 .0817/.0536 |
| 上游本仓库 | .1898/.1721 | .0863/.0496 | — |
来源：Codex 检索（scratchpad lit_social_rec.md），ODSR 官方 ckpt 已在本机复现 Douban .1867/.1669。

## 诊断实验（Step-0）
| 探针 | 数据 | R@20 | N@20 | 备注 |
|---|---|---:|---:|---|
| EASE λ=100（免训练） | Douban | **.2136** | **.2033** | 已用上游自己的 evaluate_full_ranking 逐位交叉验证 |
| GF-CF α=0 | Douban | .1810 | .1616 | |
| EASE+朴素好友交互 β=0.1/0.3/1.0 | Douban | .2113/.1994/.1545 | | 社交朴素注入单调变差 |
| ODSR 官方 ckpt | Douban | .1867 | .1669 | 精确复现论文 |
| z(ODSR)+w·z(EASE) 最好 | Douban | .2163 | .2045 | 只比 EASE 单独高 ~1% |
| EASE λ=50 | Epinions | .0754 | .0479 | 低于 ODSR；λ 扫描进行中 |
| GF-CF α=1 | Epinions | .0457 | .0251 | 稀疏数据上很弱 |

## 下一步
1. Epinions EASE 扫描完 + ODSR×EASE 融合探针 → 判断互补性
2. 官方 Yelp2018 数据/ckpt 下载完成后同样探针
3. 根据探针设计重构模型（候选：社交图骨干 + 闭式物品-物品分支 + 门控融合）
