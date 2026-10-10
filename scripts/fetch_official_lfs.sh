#!/usr/bin/env bash
cd /mnt/data/zzx/Code/ODSR_official
for f in checkpoints/Yelp2018.pth dataset/douban-book/train.txt dataset/douban-book/test.txt dataset/douban-book/trust.txt dataset/epinions/train.txt dataset/epinions/test.txt dataset/epinions/trust.txt dataset/yelp2018/train.txt dataset/yelp2018/test.txt dataset/yelp2018/trust.txt dataset/yelp2018/val.txt; do
  grep -q "^version https://git-lfs" $f 2>/dev/null || { echo "SKIP $f"; continue; }
  oid=$(grep -o "sha256:[0-9a-f]*" $f | cut -d: -f2)
  for try in 1 2 3; do
    curl -sL --max-time 1800 -o $f.tmp "https://media.githubusercontent.com/media/Rongfeng-Guo/ODSR-KDD-2026/main/$f"
    [ "$(sha256sum $f.tmp | cut -d" " -f1)" = "$oid" ] && { mv $f.tmp $f; echo "OK $f"; break; } || echo "retry $try $f"
  done
done
echo FETCH_DONE
