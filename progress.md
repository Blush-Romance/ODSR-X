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

## Epinions 多种子（配置在 val 上冻结：E*=4, λ200 p.25 β.25 w.25）
| seed | 同管线 ODSR | ODSR-X |
|---|---|---|
| 2026 | .0822/.0477 | .0895/.0522 |
| 2027 | .0836/.0482 | .0882/.0520 |
| 2028 | .0813/.0469 | .0906/.0522 |
| 均值 | .0824/.0476 | **.0894/.0521** |
vs ODSR 论文 .0885/.0509：+1.0%/+2.4%（R@20 seed2027 略低，**R@20 不能写稳定领先**）；vs SGIL +5.9%/+6.5%；vs 同管线 ODSR 3/3 种子全赢。

## 运行中（无人值守链）
- scripts/chain_yelp2018.sh（log: logs/chain_yelp2018.log）：Yelp val-run → 选参 → 全量 E* → test 一次，结束打印 CHAIN_YELP_DONE
- ~~scripts/chain_seeds_epinions.sh~~ 已完成

## 事故记录
- 2026-10-11 01:4x /mnt/data 写满（我方先验缓存+探针 B 矩阵共 51GB）→ Douban 选参进程 OSError 退出；已删全部可再生缓存、缓存改为 opt-in（ODSRX_CACHE），重跑。

## 下一步
1. Epinions EASE 扫描完 + ODSR×EASE 融合探针 → 判断互补性
2. 官方 Yelp2018 数据/ckpt 下载完成后同样探针
3. 根据探针设计重构模型（候选：社交图骨干 + 闭式物品-物品分支 + 门控融合）

## 2026-10-11 Yelp2018 结果（chain_yelp2018.sh 完成）
- val 选参：E*=5；λ=400, p=.25, β=0, w=.25（val R@20 图 .1133 → 融合 .1191）
- test（只评一次，seed 2026）：
  | 图来源 | 图单独 R@20/N@20 | ODSR-X R@20/N@20 |
  |---|---|---|
  | 自训（全 train，E*=5） | .0800 / .0529 | **.0822 / .0543** |
  | 官方 ckpt | .0817 / .0536 | .0828 / .0546 |
- 对照：ODSR 论文 .0817/.0536，上游仓库 .0808/.0533 ⇒ ODSR-X(自训) +0.6% / +1.3%，单种子，幅度小，不能说稳定领先；融合对同管线图 +2.7%/+2.7%。
- Yelp 上物品视图互补性弱（与 step-0 一致）。

## 2026-10-11 诊断：自训图模型为何弱于官方 ckpt（进行中）
- 代码层：odsrx/backbone.py+model.py 与官方 ODSR.py/main.py 逐行对照，loss、三次前向、SSL、均匀负采样、丢尾 batch 全一致；图矩阵此前已验证差为 0。
- 同协议（全 train、每轮测 test、按 test 选轮）下自训 Epinions 峰值 .0829@ep5（run 231443），官方 ckpt .0885。
- 上游仓库自训（GPU、test 选轮）：Epinions .0863@ep4、Yelp .0808@ep3，同样低于官方 ckpt（.0885/.0817）⇒ 官方 ckpt 难以从 released config 复现不是我们独有。
- 官方 ckpt 提交于 cac11ca (2026-04-09)，main.py（含 released config）晚至 5acd95a (2026-06-13) 才加入 ⇒ ckpt 可能出自更早/不同的配置。
- ckpt 统计：官方 item 嵌入范数更小（Yelp .305 vs 自训 .438；Epinions .316 vs .327~.358），无法单凭此定论。
- 决定性实验：原样运行官方 main.py --train（epinions, cpu, seed 2026, 8 线程）→ runs/hs-Ubuntu_20261011_071744_epinions_official-main-s2026/
- **结论（2026-10-11 08:00）**：官方 main.py 原样训练 Epinions（cpu, seed 2026）test R@20 = ep2 .0772 / ep4 .0822 / ep6 **.0830** / ep8 .0801；
  我方 odsrx 同协议 run 231443 = ep2 .0772 / ep4 .0822 / ep6 .0828 ⇒ **两份代码逐轮吻合，我方实现无缺陷**。
  官方 ckpt 的 .0885 无法由公开代码+公开配置复现（差 6.6%）；上游仓库 GPU 自训 .0863 亦未达到。
  ⇒ 论文主表：同管线对比（ODSR vs ODSR-X，均自训）为主；"官方 ckpt + 物品视图"作为补充行，并注明 ckpt 不可复现。

## 2026-10-11 方向1：多视图融合 Step-0 探针（Epinions val，probes/probe_multiview.py）
- 单视图 val R@20：G .0739 / E .0711 / SOC(好友交互均值) .0375 / SOCE(好友交互经EASE) .0400 / RP3β .0104(全稠密未截断，偏弱) / POP .0361
- 贪心前向加权（val）：G+E(.25) .0796 → +SOC(.1) **.0820 (+3.0%)** → +RP3(.1) .0824 → +SOCE(.1) .0829
- 结论：社交共现视图是唯一有分量的新信息源（+3%）；RP3/SOCE 合计 +0.1pp，舍弃以保持方法简单。
- 落地：odsrx/prior.py 新增 social_scores；fuse.py 新增 --soc_ws（默认 0，与旧行为逐位一致）；scripts/chain_social_epinions.sh 在 val 上选 (w, w_soc) 后 test 只评一次（3 种子 + 官方 ckpt）。
- 已知小坑：首次运行因 ssh 单引号剥离导致 f-string 报错，未产生任何结果，已修复重跑。

## 2026-10-11 方向2：图模型正则调参（CCNU2 luokekingdom，val 选轮）
- CCNU2 环境：/mnt/data/zzx/conda_envs/odsr（torch 2.8.0+cpu, numpy 2.0.2, scipy 1.13.1）；代码 git bundle 同步 ab28cee；数据 sha256 与 CCNU1 一致。
- 网格：reg0(对照) / reg1e-3 / lr5e-4（队列A）；reg3e-3 / reg1e-2（队列B）；patience 4、每轮 val。
- 方向2 结果（Epinions val R@20 峰值@轮）：reg0 对照 .0740@4（与 CCNU1 .0740@4 一致）｜reg1e-3 .0699@2｜reg3e-3 .0468@1｜reg1e-2 .0246@1 ⇒ 范数正则全有害；
  **lr5e-4 .0771@7（+4.3%）**。继续下探：lr2.5e-4、lr3.5e-4（max_epoch 40）。调好的超参将同时用于 ODSR 对照行和 ODSR-X，保证公平。
- **方向1 test 结果（Epinions，val 选出 w=.25, w_soc=.2，test 只评一次）**：
  | 图 | 图单独 R@20/N@20 | ODSR-X(G+E) | **ODSR-X+SOC** |
  |---|---|---|---|
  | 自训 s2026 | .0822/.0477 | .0895 | **.0945/.0549** |
  | 自训 s2027 | .0836/.0482 | .0882 | **.0935/.0546** |
  | 自训 s2028 | .0813/.0469 | .0906 | **.0939/.0544** |
  | 均值 | .0824/.0476 | .0894/.0521 | **.0940/.0546** |
  | 官方 ckpt | .0885/.0509 | .0928/.0540 | .0955/.0558 |
  ⇒ 对 ODSR 论文 .0885/.0509：+6.2%/+7.3%，3 种子全部超过；对同管线 ODSR +14.1%/+14.7%；社交视图在 G+E 之上再 +5.1%。
- **方向1 Douban**：val 上 w_soc 任何正值都不升（w=2 时 0→.1986/.1985/.1985/.1980…），选出 w_soc=0 ⇒ 社交视图在 Douban 无增益（与 step-0「好友行注入在 Douban 有害」一致）。
  test：自训 s2026 .2182/.2048、**s2027 .2181/.2041**（图单独 .1843/.1654）、官方 ckpt .2182/.2046。论文须写明：社交视图收益集中在社交信号强/交互稀疏的 Epinions，Douban 上权重被验证集自动选为 0。
- 方向2 续（CCNU2，Epinions val 峰值）：lr5e-4 .0771@7｜lr3.5e-4 .0770@13｜**lr2.5e-4 .0776@16**（三者在 ±.0005 内基本持平，按严格 val 最大选 lr2.5e-4, E*=16）。
  已启动：CCNU2 全量 train lr2.5e-4 E16 × seeds 2026/2027/2028（luokekingdom_20261011_0938xx_*）；CCNU1 用 val 图重选 (w, w_soc)（select_epinions_lr25_soc）；完成后 ckpt 拷回 CCNU1 在 test 只评一次。
  公平性：调参后的 ODSR（图单独）将作为同管线对照行一起报告。

## 2026-10-11 Douban / Yelp 补种子完成（配置冻结于 val，G+E 融合）
| 数据集 | seed | 图单独 R@20/N@20 | ODSR-X R@20/N@20 |
|---|---|---|---|
| Douban | 2026/2027/2028 | .1804/.1843/.1818 → 均值 .1822/.1634 | .2182/.2181/.2177 → **均值 .2180/.2043** |
| Yelp2018 | 2026/2027/2028 | .0800/.0800/.0792 → 均值 .0797/.0528 | .0822/.0822/.0826 → **均值 .0823/.0545** |
- Yelp：3 种子全部高于 ODSR 论文 .0817（+0.8%/+1.6%），幅度小但稳定；Douban 3 种子极稳（std≈.0003）。
- Yelp 社交视图链（after_seeds_yelp_social.sh）此时自动开始。
