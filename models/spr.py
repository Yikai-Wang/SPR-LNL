import multiprocessing
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import enet_path
from utils import get_one_hot


def linear(X, Y, num_classes, num_inlier, label):
    H = np.dot(np.dot(X, np.linalg.inv(np.dot(X.T, X))), X.T)
    X_hat = np.eye(H.shape[0]) - H
    y_hat = np.dot(X_hat, Y)

    if not isinstance(num_inlier, list):
        num_inlier = [num_inlier for _ in range(num_classes)]

    _, coefs, _ = enet_path(X_hat, y_hat, l1_ratio=1.0)
    coefs = np.sum(np.abs(coefs.transpose(2, 1, 0)[::-1, :, :]), axis=2)
    selected = np.zeros(num_classes)
    clean_set = []
    for gamma in coefs:
        for i, g in enumerate(gamma):
            if g == 0.0 and i not in clean_set and (selected[label[i]] < num_inlier[label[i]]):
                clean_set.append(i)
                selected[label[i]] += 1
        if np.sum(selected >= num_inlier) == num_classes:
            break

    return clean_set

def label2onehot(label, num_class):
    result = np.zeros((label.shape[0], num_class))
    for ind, num in enumerate(label):
        if num != -1:
            result[ind, num] = 1.0
        else:
            result[ind] = np.ones(num_class) / num_class
    return result

def spr(args, ep_stats, clean_set):
    idx = np.array(ep_stats['idx'], dtype=int)
    X, label = ep_stats['feature'][idx], ep_stats['label'][idx]
    Y = get_one_hot(label, args.num_classes)
    idx = np.arange(len(idx))

    res_list = []

    super2sub = []
    num_classes_sub = args.num_classes_sub

    if num_classes_sub == args.num_classes:
        super2sub = [np.arange(args.num_classes).tolist()]
    else:
        if clean_set is None:
            proto = []
            for i in range(args.num_classes):
                proto.append(np.mean(X[label==i], axis=0, keepdims=True))
            proto = np.concatenate(proto)
        else:
            proto = np.zeros((args.num_classes, X.shape[1]))
            count = np.zeros(args.num_classes)
            for i in range(len(X)):
                if int(idx[i]) in clean_set:
                    count[label[i]] += 1
                    proto[label[i]] += X[i]
            proto = proto / count.reshape(-1,1)
        similarity = np.dot(proto, proto.T) 
        candidate = np.arange(args.num_classes).tolist()
    
        while len(candidate) > 0:
            sub = []
            x = candidate[0]
            sub.append(x)
            candidate.remove(x)
            for i in range(num_classes_sub-1):
                # select the new class by the smallest sum of similarities.
                sim2sub = similarity[sub][:, candidate].sum(0)
                x = candidate[sim2sub.argmin()]
                sub.append(x)
                candidate.remove(x)
            super2sub.append(sub)
        
    num_per_task = 10
    num_inlier = int(0.5*num_per_task)

    X_list = []
    Y_list = []
    num_classes_sub = len(super2sub[0])
    label_list = []
    label2sub_list = []
    indexes2full_list = []
    for sub in super2sub:
        sub_stats = defaultdict(dict)
        min_num = 100000000000
        max_num = 0
        for l in sub:
            sub_stats[l]['idx'] = idx[label==l]
            sub_stats[l]['X'] = X[label==l]
            sub_stats[l]['Y'] = Y[label==l]
            sub_stats[l]['label'] = label[label==l]
            min_num = min(len(sub_stats[l]['idx']), min_num)
            max_num = max(len(sub_stats[l]['idx']), max_num)
        total_num = max_num
        for i in range(0, total_num-num_per_task, num_per_task):   
        # drop last
            indexes2full = {}
            X_sub = []
            Y_sub = []
            label_sub = []
            for ind, l in enumerate(sub):
                num_sub = len(sub_stats[l]['X'])
                if i+num_per_task < num_sub or i >= num_sub: 
                    X_sub.append(sub_stats[l]['X'][i%num_sub:(i+num_per_task)%num_sub])
                    Y_sub.append(sub_stats[l]['Y'][i%num_sub:(i+num_per_task)%num_sub])
                    label_sub.append(sub_stats[l]['label'][i%num_sub:(i+num_per_task)%num_sub])
                else:
                    X_sub.append(sub_stats[l]['X'][i:])
                    X_sub.append(sub_stats[l]['X'][:(i+num_per_task)%num_sub])
                    Y_sub.append(sub_stats[l]['Y'][i:])
                    Y_sub.append(sub_stats[l]['Y'][:(i+num_per_task)%num_sub])
                    label_sub.append(sub_stats[l]['label'][i:])
                    label_sub.append(sub_stats[l]['label'][:(i+num_per_task)%num_sub])
                for j in range(num_per_task):
                    indexes2full[num_per_task*ind+j]=sub_stats[l]['idx'][(i+j)%num_sub]
            X_sub = np.concatenate(X_sub)
            Y_sub = np.concatenate(Y_sub)[:,sub]
            label_sub = np.concatenate(label_sub)

            if args.pca:
                X_sub = PCA(n_components=len(sub)).fit_transform(X_sub)
            else:
                X_sub = X_sub
            sub2label = {c:i for i, c in enumerate(sub)}
            label2sub = {i:c for i, c in enumerate(sub)}
            label_sub = [sub2label[int(c)] for c in label_sub]
            X_list.append(X_sub)
            Y_list.append(Y_sub)
            label_list.append(label_sub)
            label2sub_list.append(label2sub)
            indexes2full_list.append(indexes2full)
    pool = Pool(processes=int(multiprocessing.cpu_count()*args.ratio_cpu))
    for i in range(len(X_list)):
        res = pool.apply_async(func=linear, args=(X_list[i], Y_list[i], num_classes_sub, num_inlier, label_list[i]))
        res_list.append(res)
    pool.close()
    pool.join()
    clean_set = []
    for i, res in enumerate(res_list):
        sub_set = res.get()
        for j in sub_set:
            clean_set.append(indexes2full_list[i][j])
    clean_set = set(clean_set)
    return clean_set
