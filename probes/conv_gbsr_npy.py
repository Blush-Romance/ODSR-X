import numpy as np, sys, os
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
for name, f in [("train", "traindata"), ("test", "testdata"), ("trust", "user_users_d")]:
    d = np.load(f"{src}/{f}.npy", allow_pickle=True).tolist()
    with open(f"{dst}/{name}.txt", "w") as fo:
        fo.write("".join(f"{u} {i}\n" for u in d for i in d[u]))
