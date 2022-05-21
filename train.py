import os
import os.path as osp
import sys
from collections import defaultdict
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import config
from dataset import DatasetGenerator
from models.spr import spr
from utils import Logger, calculate_loss, evaluate, evaluate_top5, pNorm, rand_bbox, save_checkpoint, set_seed

args = config()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
device = 'cuda:0' if torch.cuda.is_available()  else 'cpu'

if args.seed is not None:
    set_seed(args.seed)

if os.path.exists(args.save_dir) and args.overwrite:
    os.system('rsync -a {0} logs/trashes/ && rm -r {0}'.format(args.save_dir))
    print('Existing log folder, move it to trashes!')
writter = SummaryWriter(args.save_dir)

data_loader = DatasetGenerator(data_path=os.path.join(args.root, args.dataset),
                               num_of_workers=args.num_workers,
                               seed=args.seed,
                               train_batch_size=args.batch_size,
                               noise_type=args.noise_type,
                               dataset=args.dataset,
                               noise_rate=args.noise_rate,
                               cutmix=args.cutmix,
                               ).getDataLoader()
train_loader, test_loader = data_loader['train_dataset'], data_loader['test_dataset']
if args.dataset == 'WebVision':
    test_loader_imagenet = data_loader['test_imagenet']

if args.backbone == 'conv2':
    from models.models import CNN
    model = CNN(type=args.backbone, num_classes=args.num_classes, show=True)
    nFeat = 128
elif args.backbone == 'res18' and 'CIFAR' in args.dataset:
    from models.resnet_cifar import resnet18
    model = resnet18(num_classes=args.num_classes, show=True)
    nFeat = 512
elif args.backbone == 'vgg':
    from models.vgg import vgg19_bn
    model = vgg19_bn(num_classes=args.num_classes, pretrained=False, show=True)
    nFeat = 4096
elif args.backbone == 'inception':
    from models.inception import InceptionResNetV2
    model = InceptionResNetV2(num_classes=args.num_classes, show=True)
    nFeat = 1536
else:
    raise NameError

if args.resume is not None:
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    if args.start_epoch is None:
        args.start_epoch = epoch + 1
else:
    args.start_epoch = 0

if len(args.gpus) > 1:
    model = nn.DataParallel(model)
model = model.to(device)

if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay= args.weight_decay)

if args.scheduler == 'cos':
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0, last_epoch=-1)
elif args.scheduler == 'step':
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

criterion = nn.CrossEntropyLoss(reduction='none')

norm = pNorm(args.norm)

sys.stdout = Logger(args.save_dir+'.txt', 'a')

print(args)
print('Epoch LR Loss Best Acc Time')
best_acc = 0
if args.spr:
    clean_set = None
    ep_stats = {}
    ep_stats['label'] = np.array(train_loader.dataset.mislabeled_targets).astype(int)
    num_train = len(ep_stats['label'])
    ep_stats['idx'] = np.empty(num_train)
    ep_stats['feature'] = np.zeros((num_train, nFeat))
    ep_stats['pred'] = np.zeros_like(ep_stats['label']) - 1

for ep in range(args.start_epoch, args.epochs):
    start = time()
    model.train()
    count_info = defaultdict(float)
    visited = 0
    if args.tqdm:
        train_loader = tqdm(train_loader, ncols=0)
    for batch in train_loader:
        if args.cutmix:
            x, x1, y, idx = batch
            x1 = x1.to(device)
        else:
            x, y, idx = batch
        x, y = x.to(device), y.to(device)

        model.zero_grad()
        optimizer.zero_grad()

        if not ep or not args.cutmix:
            logit, feature = model(x)
            if args.cutmix and args.cutmix_prob == 1.0:
                loss = calculate_loss(criterion, logit, y)
            else:
                loss = calculate_loss(criterion, logit, y, norm, args.lamb)
        else:
            r = np.random.rand(1)
            if r >= args.cutmix_prob:
                logit, feature = model(x)
                if args.spr:
                    weight = torch.zeros_like(y)
                    for i in range(len(weight)):
                        if int(idx[i]) in clean_set:
                            weight[i] = 1.0
                else:
                     weight = torch.ones_like(y)
                loss = calculate_loss(criterion, logit, y, norm, args.lamb, weight)
            else:
                onehot = F.one_hot(y, num_classes=args.num_classes)
                with torch.no_grad():
                    logit, feature = model(x)
                    logit1, _ = model(x1)
                    p = (logit.softmax(1)+logit1.softmax(1)) / 2
                    p = p**(1/0.5)
                    y_u = p / p.sum(1, keepdim=True)
                    y_u = y_u.detach()
                labeled = []
                unlabeled = []
                if not args.spr:
                    bs = len(idx)
                    _permutation = np.random.permutation(bs)
                    labeled = _permutation[:int(bs//2)].tolist()
                    unlabeled = _permutation[int(bs//2):].tolist()
                else:
                    for i, id in enumerate(idx):
                        if int(id) in clean_set:
                            labeled.append(i)
                        else:
                            unlabeled.append(i)
                n_labeled = len(labeled)
                orded_idx = labeled + unlabeled
                l = np.random.beta(1, 1)
                l = max(l, 1-l)

                rand_idx = torch.randperm(x1.shape[0])
                bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), l)
                mixed_x = x[orded_idx]
                mixed_y = y[orded_idx]
                mixed_y[n_labeled:] = y_u.argmax(1)[orded_idx][n_labeled:]
                mixed_x[:, :, bbx1:bbx2, bby1:bby2] = mixed_x[rand_idx, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (mixed_x.size()[-1] * mixed_x.size()[-2]))
                target_a = mixed_y
                target_b = mixed_y[rand_idx]
                # compute output
                mixed_logit, _ = model(mixed_x)
                loss = calculate_loss(criterion, mixed_logit, target_a) * lam + calculate_loss(criterion, mixed_logit, target_b) * (1. - lam)
            
        
        if args.spr:
            logit = logit.argmax(-1).detach().cpu().numpy()
            feature = feature.detach().cpu().numpy()
            for i in range(len(logit)):
                ep_stats['idx'][visited] = idx[i]
                visited += 1
                ep_stats['pred'][idx[i]] = logit[i]
                ep_stats['feature'][idx[i]] = feature[i]

        loss.backward()
        if args.grad_bound:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_bound)
        optimizer.step()

        count_info['loss'] += loss.item()
        count_info['num_batches'] += 1

    lr = scheduler.get_last_lr()[0]
    scheduler.step()
    
    if args.spr:
        ep_stats['idx'] = ep_stats['idx'][:visited]
        clean_set = spr(args, ep_stats, clean_set) 


    torch.cuda.empty_cache()

    if args.dataset == 'WebVision':
        top1, top5 = evaluate_top5(test_loader, model, device)
        top1_imagenet, top5_imagenet = evaluate_top5(test_loader_imagenet, model, device)
        test_acc = top1
    else:
        test_acc = evaluate(test_loader, model, device)

    torch.cuda.empty_cache()

    end = time()  

    if (ep + 1) % args.freq == 0:
        args.lamb = args.lamb * args.rho

    if test_acc > best_acc:
        best_acc = test_acc
        state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        save_checkpoint({
            'epoch': ep,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
        }, osp.join(args.save_dir, 'best_model.pth.tar'))
    

    print_info = '{}/{} {:.4f} {:.3f} '.format(ep, args.epochs, lr, count_info['loss'] / count_info['num_batches'])
    writter.add_scalar('Loss/Toal', count_info['loss'] / count_info['num_batches'], ep)
    
    if args.dataset == 'WebVision':
        print_info += '{:.2f} {:.2f}({:.2f}) {:.2f}({:.2f})'.format(100 * best_acc, 100 * test_acc, 100 * top5, 100 * top1_imagenet, 100 * top5_imagenet)
        writter.add_scalar('Test Accuracy/top1', test_acc, ep)
        writter.add_scalar('Test Accuracy/top5', top5, ep)
        writter.add_scalar('Test Accuracy/top1_imagenet', top1_imagenet, ep)
        writter.add_scalar('Test Accuracy/top5_imagenet', top5_imagenet, ep)
    else:
        print_info += '{:.2f} {:.2f} '.format(100 * best_acc, 100 * test_acc)
        writter.add_scalar('Test Accuracy', test_acc, ep)

    mins = int((end-start)//60)
    if mins:
        print_info += '{}m{}s'.format(mins, int((end-start)%60))
    else:
        print_info += '{}s'.format(int((end-start)%60))
    print(print_info)
