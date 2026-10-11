"""Convert exported .json splits to the official GBSR/SGIL .npy format (run with the TF1 env's numpy 1.x).
usage: json_to_npy.py <dir>  -> writes <name>.npy next to every <name>.json"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np
for f in glob.glob(os.path.join(sys.argv[1], "**", "*.json"), recursive=True):
    if os.path.basename(f) == "meta.json":
        continue
    d = defaultdict(list, {int(k): [int(x) for x in v] for k, v in json.load(open(f)).items()})
    np.save(f[:-5] + ".npy", d)
    print(f, len(d))
