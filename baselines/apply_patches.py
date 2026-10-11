"""Fetch the official GBSR (KDD24) / SGIL (SIGIR25) TensorFlow code and apply protocol-only patches.

The official repositories carry no license, so their code is not redistributed here. This script clones them
into baselines/_official/ and writes patched copies to baselines/{gbsr,sgil}/. Every change is marked `# PATCH`;
model, loss and hyper-parameters are untouched (see README, section Baselines).
"""
import subprocess, sys
REPOS = {"gbsr": ("https://github.com/yimutianyang/KDD24-GBSR.git", "881d712"),
         "sgil": ("https://github.com/yimutianyang/SIGIR2025-SGIL.git", "bee3a39")}  # commits used in our runs
import os, shutil, re
B = os.path.dirname(os.path.abspath(__file__))
SRC = {name: os.path.join(B, "_official", name) for name in ("gbsr", "sgil")}
for name, src in SRC.items():
    if not os.path.exists(src):
        subprocess.check_call(["git", "clone", "-q", REPOS[name][0], src])
        subprocess.check_call(["git", "-C", src, "checkout", "-q", REPOS[name][1]])
    dst = os.path.join(B, name)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("datasets", "torch_version", ".git", "saved", "*.jpg"))
    ds_file = os.path.join(dst, "rec_dataset.py" if name == "gbsr" else "utils/rec_dataset.py")
    run_file = os.path.join(dst, "run_GBSR.py" if name == "gbsr" else "run_SGIL.py")
    s = open(ds_file).read()
    old = "self.valdata = np.load(self.data_path + 'testdata.npy', allow_pickle=True).tolist()"
    assert s.count(old) == 1, (name, "valdata")
    s = s.replace(old, "self.valdata = np.load(self.data_path + ('valdata.npy' if os.path.exists(self.data_path + 'valdata.npy') else 'testdata.npy'), allow_pickle=True).tolist()  # PATCH: real validation split")
    open(ds_file, "w").write(s)

    s = open(run_file).read()
    def rep(old, new, count=1):
        global s
        assert s.count(old) == count, (name, old, s.count(old))
        s = s.replace(old, new)
    # seeds
    s = re.sub(r"np\.random\.seed\((\d+)\)\ntf\.set_random_seed\((\d+)\)",
               lambda m: f"SEED = int(os.environ.get('SEED', {m.group(1)}))  # PATCH\nnp.random.seed(SEED)\ntf.set_random_seed(SEED)\nimport random; random.seed(SEED)", s)
    assert "SEED = int" in s, name
    # data / run dir / sizes from env (dataset-specific hparam block above still applies)
    rep("    args.data_path = './datasets/' + args.dataset + '/'",
        "    args.data_path = os.environ.get('DATA_PATH', './datasets/' + args.dataset + '/')  # PATCH\n"
        "    if 'NUM_USER' in os.environ: args.num_user = int(os.environ['NUM_USER']); args.num_item = int(os.environ['NUM_ITEM'])  # PATCH\n"
        "    FIXED = int(os.environ.get('FIXED_EPOCHS', 0))  # PATCH: >0 -> train exactly FIXED epochs on full train, no per-epoch eval")
    rec = "    record_path = './saved/' + args.dataset + '/GBSR/' + args.runid + '/'" if name == "gbsr" else "    record_path = './saved/' + args.dataset + '/' + args.runid + '/'"
    rep(rec, "    record_path = os.environ['RUN_DIR'].rstrip('/') + '/'  # PATCH")
    rep("    config = tf.ConfigProto()",
        "    config = tf.ConfigProto(intra_op_parallelism_threads=int(os.environ.get('TF_THREADS', 8)), inter_op_parallelism_threads=2)  # PATCH")
    rep("        if (epoch % 1) == 0:",
        "        if FIXED and epoch + 1 >= FIXED:  # PATCH\n            break\n        if (epoch % 1) == 0 and not FIXED:  # PATCH")
    rep("                saver.save(sess, model_save_path + best_ckpt)",
        "                saver.save(sess, model_save_path + best_ckpt)\n                open(record_path + 'best_epoch.txt', 'w').write(str(epoch + 1))  # PATCH: number of epochs trained")
    rep("    saver.restore(sess, model_save_path + best_ckpt)",
        "    if not FIXED:  # PATCH\n        saver.restore(sess, model_save_path + best_ckpt)")
    hdr = "    log.write('=================Evaluation results==================\\n')\n    user_matrix, item_matrix = sess.run([rec_model.user_emb, rec_model.item_emb])\n"
    rep(hdr, hdr + "    np.save(record_path + 'user_emb.npy', user_matrix); np.save(record_path + 'item_emb.npy', item_matrix)  # PATCH: scored by odsrx.evaluate\n")
    open(run_file, "w").write(s)
    print(name, "patched")
