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
| yelp2018 | 官方 ODSR 仓库 LFS（train/test/trust） | 与上游逐字节一致；官方 val.txt 与 test.txt 是同一文件 |
| yelp-gbsr | GBSR 公开 npy 导出（probes/conv_gbsr_npy.py） | 备用：SGIL/GBSR 使用的 Yelp 版本 |

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

| 每用户交互量分桶最优融合权重（oracle 上界） | Epinions / Douban | +1.19% / +0.65% | | 两折诚实估计 +0.52% / +0.20% → **用户级门控机制 Step-0 毙掉** |
| ODSR 官方 ckpt / 融合最好 | Epinions | .0885→.0919 | .0509→.0543 | 稀疏数据上互补明显（w=0.25–0.5） |
| 闭式物品模型（验证集选参 λ50 p0.1 β0.25）| Douban | **.2172** | **.2043** | 选参只看验证集 |
| 闭式岭回归社交去噪 | Douban | .2167 | .2036 | 无增益 |

## 重构（分支 refactor-odsrx）
- `odsrx/`：data / prior / backbone / model / evaluate / train；`tests/test_equiv.py` 三项等价检查通过（图矩阵 diff=0；官方 ckpt 复现 .186746；α→∞ 退化为 EASE .21365）
- commit 2a7f4da（重构）、185eb7f（难负样本选项）

## MODE C pilot 记录
| run | 设置 | 结果 | 结论 |
|---|---|---|---|
| 225541 douban odsrx-l50p01b025 | 联合残差训练，随机负样本 | 4 轮 .2162→.2175 | 停：图分量饿死（rec loss≈0.01） |
| 225541 epinions odsrx-l200 | 同上 | 5 轮恒 ≈.0782（=先验单独） | 停：同上 |
| 231443 epinions odsrx-l200-hard05 | + 先验引导难负样本 ρ=0.5 K=100 | 4 轮恒 ≈.0783，rec loss 升至 .09 但无效 | 停：联合残差第 2 个实现无信号（§13）→ 子方向关闭 |
| 231443 epinions odsr-baseline | 无先验 = ODSR 同管线复现，测试集选轮 | 最佳 ep5 .0829/.0480，之后单调降 | 低于官方 ckpt .0885（疑种子方差，待多种子） |
| 232530 epinions odsr-valsplit | fit 80% 训练，验证集选轮 | 最佳 ep4 val .0740 | E*=4 |
| 234625 epinions odsr-full-E4 | 全量训练 4 轮 | test .0822/.0477 | 与 baseline ep4 逐位相同（CPU 确定性） |

## 主路线：验证集标定的晚融合（odsrx/fuse.py）
选参（只看验证集，results/fuse/select_epinions.json）：λ=200 p=0.25 β=0.25 **w=0.25**，val .0740→.0797
| 数据集 | 图视图 | 图单独 R@20/N@20 | ODSR-X R@20/N@20 | 提升 |
|---|---|---|---|---|
| Epinions | 自训 E4 | .0822/.0477 | **.0895/.0522** | +8.9%/+9.5% |
| Epinions | 官方 ckpt | .0885/.0509 | .0928/.0540 | +4.9%/+6.0% |
| Douban | 自训 E17（val 选） | .1804/.1620 | **.2182/.2048** | +20.9%/+26.4% |
| Douban | 官方 ckpt | .1867/.1669 | .2182/.2046 | +16.9%/+22.6% |

Douban 选参（val）：λ=50 p=0.25 β=0.25 **w=2**，val .1669→.1986。
Douban 消融：物品视图单独（同配置）test .2172/.2043 → 融合只再 +0.5%/+0.2%。**Douban 的增益主要来自闭式物品视图，必须如实写。**
Yelp2018：官方 ckpt 复现 .0817/.0536；EASE λ400 .0729/.0487；GF-CF α0.3 .0753/.0493；官方 ckpt+EASE 融合（测试集探针）最好 w=0.25 .0827/.0545（+1.2%，互补性弱）。

## 运行中（无人值守链）
- scripts/chain_yelp2018.sh（log: logs/chain_yelp2018.log）：Yelp val-run → 选参 → 全量 E* → test 一次，结束打印 CHAIN_YELP_DONE
- scripts/chain_seeds_epinions.sh（log: logs/chain_seeds_epinions.log）：Epinions seed 2027/2028

## 事故记录
- 2026-10-11 01:4x /mnt/data 写满（我方先验缓存+探针 B 矩阵共 51GB）→ Douban 选参进程 OSError 退出；已删全部可再生缓存、缓存改为 opt-in（ODSRX_CACHE），重跑。

## 下一步
1. Epinions EASE 扫描完 + ODSR×EASE 融合探针 → 判断互补性
2. 官方 Yelp2018 数据/ckpt 下载完成后同样探针
3. 根据探针设计重构模型（候选：社交图骨干 + 闭式物品-物品分支 + 门控融合）
