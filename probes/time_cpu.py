import sys, os, time, torch, numpy as np
off_src, data_root, ds = sys.argv[1:4]
sys.path.insert(0, off_src); os.environ["SELFREC_DATA_ROOT"] = os.path.abspath(data_root)
from common import GraphData
from ODSR import ODSR
torch.set_num_threads(int(os.environ.get("THREADS", "16")))
dev = torch.device("cpu"); data = GraphData(ds, dev, social_norm="row", verbose=False)
m = ODSR(data, dict(emb_size=64, n_layer=3, dropout=0.01, reg=1e-5, eps=0.8, ssl_reg=0.5, tau=0.2), dev)
opt = torch.optim.Adam(m.parameters(), lr=0.002)
uu = np.array(data.train_data)
for it in range(6):
    b = uu[np.random.randint(0, len(uu), 2048)]
    t = time.time()
    loss = m.compute_loss(torch.from_numpy(b[:, 0]), torch.from_numpy(b[:, 1]), torch.randint(0, data.item_num, (2048,)))
    opt.zero_grad(); loss.backward(); opt.step()
    print(f"step {it}: {time.time()-t:.2f}s  batches/epoch={len(uu)//2048}", flush=True)
