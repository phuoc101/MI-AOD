import mmcv
import numpy as np


def get_X_L_0(cfg):
    # load dataset anns
    anns = load_ann_list(cfg.data.train.dataset.ann_file)
    unanns = load_ann_list(cfg.data.unlabeled.ann_file)
    # get all indexes
    X_all = np.arange(len(anns) + len(unanns), dtype=int)
    X_L = X_all[: len(anns)].copy()
    X_U = X_all[len(anns) :].copy()
    anns = np.hstack((anns, unanns))
    return X_L, X_U, X_all, anns


def create_X_L_file(cfg, X_L, anns, cycle):
    # create labeled ann files
    save_folder = cfg.work_directory + "/cycle" + str(cycle)
    mmcv.mkdir_or_exist(save_folder)
    save_path = save_folder + "/X_L.txt"
    np.savetxt(save_path, anns[X_L], fmt="%s")
    X_L_path = save_path
    # update cfg
    cfg.data.train.dataset.ann_file = X_L_path
    cfg.data.train.times = cfg.X_L_repeat
    return cfg


def create_X_U_file(cfg, X_U, anns, cycle):
    # create unlabeled ann files
    X_U_path = []
    save_folder = cfg.work_directory + "/cycle" + str(cycle)
    mmcv.mkdir_or_exist(save_folder)
    save_path = save_folder + "/X_U.txt"
    np.savetxt(save_path, anns[X_U], fmt="%s")
    X_U_path = save_path
    # update cfg
    cfg.data.train.dataset.ann_file = X_U_path
    cfg.data.train.times = cfg.X_U_repeat
    return cfg


def load_ann_list(path):
    # anns = []
    # for path in paths:
    #     anns.append(np.loadtxt(path, dtype="str"))
    return np.loadtxt(path, dtype="str")


def update_X_L(uncertainty, X_all, X_L, X_S_size):
    uncertainty = uncertainty.cpu().numpy()
    all_X_U = np.array(list(set(X_all) - set(X_L)))
    arg = uncertainty.argsort()
    X_S = all_X_U[arg[-X_S_size:]]
    X_L_next = np.concatenate((X_L, X_S))
    all_X_U_next = np.array(list(set(X_all) - set(X_L_next)))
    np.random.shuffle(all_X_U_next)
    X_U_next = all_X_U_next[: X_L_next.shape[0]]
    if X_L_next.shape[0] > X_U_next.shape[0]:
        np.random.shuffle(X_L_next)
        X_U_next = np.concatenate((X_U_next, X_L_next[: X_L_next.shape[0] - X_U_next.shape[0]]))
    X_L_next.sort()
    X_U_next.sort()
    return X_L_next, X_U_next
