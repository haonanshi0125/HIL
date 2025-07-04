# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
from sklearn import metrics
import argparse
import ot
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os.path as osp
import random
import heapq
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys
import collections
import time
from datetime import timedelta
from solver import make_optimizer, WarmupMultiStepLR
from sklearn.cluster import DBSCAN
from PIL import Image
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from config import cfg
from clustercontrast import datasets
# from clustercontrast import models
from clustercontrast.model_vit_cmrefine import make_model
from torch import einsum
from clustercontrast.models.cm import ClusterMemory,Memory_wise_v3,ClusterMemory_shared
from clustercontrast.trainers import ClusterContrastTrainer_HIL
from clustercontrast.evaluators import Evaluator, extract_features
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.preprocessor import Preprocessor,Preprocessor_color
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint,save_checkpoint10
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance,compute_ranked_list,compute_ranked_list_cm
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam,MoreCameraSampler
import os
import torch.utils.data as data
from torch.autograd import Variable
import math
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing,ChannelExchange,Gray
from collections import Counter
from solver.scheduler_factory import create_scheduler
from typing import Tuple, List, Optional
from torch import Tensor
import numbers
from typing import Any, BinaryIO, List, Optional, Tuple, Union
import cv2

import copy
import os.path as osp
import errno
import shutil
start_epoch = best_mAP = 0
def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
part=1
torch.backends.cudnn.enable =True,
torch.backends.cudnn.benchmark = True




# l2norm = Normalize(2)



def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset




def get_train_loader_ir(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False,train_transformer=None):



    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            # sampler = MoreCameraSampler(train_set, num_instances)
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def get_train_loader_color(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False,train_transformer=None,train_transformer1=None):




    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            # sampler = MoreCameraSampler(train_set, num_instances)
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    if train_transformer1 is None:
        train_loader = IterLoader(
            DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    else:
        train_loader = IterLoader(
            DataLoader(Preprocessor_color(train_set, root=dataset.images_dir, transform=train_transformer,transform1=train_transformer1),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None,test_transformer=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    if test_transformer is None:
        test_transformer = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.ToTensor(),
            normalizer
        ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)#,output_device=1)
    return model




class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)

def process_query_sysu(data_path, mode = 'all', relabel=False):
    if mode== 'all':
        ir_cameras = ['cam3','cam6']
    elif mode =='indoor':
        ir_cameras = ['cam3','cam6']
    
    file_path = os.path.join(data_path,'exp/test_id.txt')
    files_rgb = []
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in ir_cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)
    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    return query_img, np.array(query_id), np.array(query_cam)

def process_gallery_sysu(data_path, mode = 'all', trial = 0, relabel=False):
    
    random.seed(trial)
    
    if mode== 'all':
        rgb_cameras = ['cam1','cam2','cam4','cam5']
    elif mode =='indoor':
        rgb_cameras = ['cam1','cam2']
        
    file_path = os.path.join(data_path,'exp/test_id.txt')
    files_rgb = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_rgb.append(random.choice(new_files))
    gall_img = []
    gall_id = []
    gall_cam = []
    for img_path in files_rgb:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)
    return gall_img, np.array(gall_id), np.array(gall_cam)
    

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_gall_feat(model,gall_loader,ngall):
    pool_dim=768*2
    net = model
    net.eval()
    print ('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_pool = np.zeros((ngall, pool_dim))
    gall_feat_fc = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(gall_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            input = Variable(input.cuda())
            feat_fc,feat_fc_s = net( input,input, 1)
            feat_fc = torch.cat((feat_fc,feat_fc_s),dim=1)
            flip_input = Variable(flip_input.cuda())
            feat_fc_1,feat_fc_1_s = net( flip_input,flip_input, 1)
            feat_fc_1 = torch.cat((feat_fc_1,feat_fc_1_s),dim=1)
            feature_fc = (feat_fc.detach() + feat_fc_1.detach())/2
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            gall_feat_fc[ptr:ptr+batch_num,: ]   = feature_fc.cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return gall_feat_fc

def extract_query_feat(model,query_loader,nquery):
    pool_dim=768*2
    net = model
    net.eval()
    print ('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_pool = np.zeros((nquery, pool_dim))
    query_feat_fc = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(query_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            input = Variable(input.cuda())
            feat_fc,feat_fc_s = net( input, input,2)
            feat_fc = torch.cat((feat_fc,feat_fc_s),dim=1)
            flip_input = Variable(flip_input.cuda())
            feat_fc_1,feat_fc_1_s= net( flip_input,flip_input, 2)
            feat_fc_1 = torch.cat((feat_fc_1,feat_fc_1_s),dim=1)
            feature_fc = (feat_fc.detach() + feat_fc_1.detach())/2
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            query_feat_fc[ptr:ptr+batch_num,: ]   = feature_fc.cpu().numpy()
            
            ptr = ptr + batch_num         
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return query_feat_fc



def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank = 20):
    """Evaluation with sysu metric
    Key: for each query identity, its gallery images from the same camera view are discarded. "Following the original setting in ite dataset"
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    pred_label = g_pids[indices]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    # compute cmc curve for each query
    new_all_cmc = []
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (q_camid == 3) & (g_camids[order] == 2)
        keep = np.invert(remove)
        
        # compute cmc curve
        # the cmc calculation is different from standard protocol
        # we follow the protocol of the author's released code
        new_cmc = pred_label[q_idx][keep]
        new_index = np.unique(new_cmc, return_index=True)[1]
        new_cmc = [new_cmc[index] for index in sorted(new_index)]
        
        new_match = (new_cmc == q_pid).astype(np.int32)
        new_cmc = new_match.cumsum()
        new_all_cmc.append(new_cmc[:max_rank])
        
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(orig_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx]/ (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q   # standard CMC
    
    new_all_cmc = np.asarray(new_all_cmc).astype(np.float32)
    new_all_cmc = new_all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return new_all_cmc, mAP, mINP

def pairwise_distance(features_q, features_g):
    x = torch.from_numpy(features_q)
    y = torch.from_numpy(features_g)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m.numpy()





class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / float(self.warmup_iters)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out



def main():
    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.freeze()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    log_name = 'sysu_train'
    main_worker_stage2(args,log_name)

def main_worker_stage2(args,log_name):

    l2norm = Normalize(2)
    ir_batch=128
    rgb_batch=128

    global start_epoch, best_mAP 

    args.logs_dir = osp.join('./logs',log_name)
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))
    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, 'r') as cf:
        config_str = "\n" + cf.read()
    print(config_str)
    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset_ir = get_data('sysu_ir', args.data_dir)
    dataset_rgb = get_data('sysu_rgb', args.data_dir)

    test_loader_ir = get_test_loader(dataset_ir, args.height, args.width, args.batch_size, args.workers)
    test_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width, args.batch_size, args.workers)

    model = make_model(cfg, num_class=0, camera_num=0, view_num = 0)
    model.cuda()
    device_ids = [0,1]
    print("GPU数量:", torch.cuda.device_count())
    model = nn.DataParallel(model, device_ids=device_ids)
    trainer = ClusterContrastTrainer_HIL(model)
    trainer.cmlabel = 0#30
    trainer.hm = 0#20
    trainer.ht = 0#10

    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]

    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    evaluator = Evaluator(model)

    @torch.no_grad()
    def generate_cluster_features_kmeans(labels, features, n_clusters=9, max_small_centers=9, weight=0.9):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if label == -1:
                continue
            centers[labels[i]].append(features[i])

        cluster_features = []

        unique_labels_count = []  # 用于统计每次 KMeans 的 unique_labels 数量

        for idx in sorted(centers.keys()):
            cluster_samples = centers[idx]

            if len(cluster_samples) == 0:
                continue

            cluster_samples_array = np.array([sample.numpy() for sample in cluster_samples])
            cluster_samples_tensor = torch.tensor(cluster_samples_array, dtype=torch.float32)

            large_center = cluster_samples_tensor.mean(0)

            small_centers = []
            if len(cluster_samples) > n_clusters:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    try:
                        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
                        labels_kmeans = kmeans.fit_predict(cluster_samples_array)

                        # 统计 KMeans 的 unique_labels 数量分布
                        unique_labels, counts = np.unique(labels_kmeans, return_counts=True)
                        unique_labels_count.append({label: count for label, count in zip(unique_labels, counts)})

                        # 检查 KMeans 的输出是否符合预期
                        if len(unique_labels) < n_clusters:
                            raise ValueError(
                                f"KMeans found fewer clusters ({len(unique_labels)}) than expected ({n_clusters}).")

                        for label in range(n_clusters):
                            small_cluster_samples = cluster_samples_tensor[labels_kmeans == label]
                            small_center = small_cluster_samples.mean(0)
                            # 计算加权平均值，使小中心点更接近大中心点
                            weighted_small_center = weight * large_center + (1 - weight) * small_center
                            small_centers.append(weighted_small_center)

                        if len(small_centers) > max_small_centers:
                            small_centers = small_centers[:max_small_centers]
                        elif len(small_centers) < max_small_centers:
                            small_centers += [large_center] * (max_small_centers - len(small_centers))
                    except Exception as e:
                        # 如果 KMeans 失败，直接使用整个簇的均值作为中心点
                        # print(f"KMeans failed with error: {e}")
                        small_centers = [large_center] * max_small_centers
            else:
                small_centers = [large_center] * max_small_centers

            cluster_features.append(
                torch.stack([large_center] + small_centers[:max_small_centers], dim=0))

        cluster_features = torch.stack(cluster_features, dim=0)

        return cluster_features

    @torch.no_grad()
    def generate_cluster_features(labels, features):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if label == -1:
                continue
            centers[labels[i]].append(features[i])

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0)
        return centers

    @torch.no_grad()
    def correct_label_transfer(features, pseudo_labels, prototypes):

        pseudo_labels_hat = copy.deepcopy(pseudo_labels)
        max_similarity = []

        for i in range(len(prototypes)):
            similarities = np.dot(prototypes[i], features.T) / (np.linalg.norm(prototypes[i]) * np.linalg.norm(features, axis=1))
            max_similarity.append(np.max(similarities))

        for i in range(len(features)):
            labels = pseudo_labels[i]
            if labels == -1:
                continue

            similarities = np.dot(features[i], prototypes.T) / (np.linalg.norm(features[i]) * np.linalg.norm(prototypes, axis=1))
            best_ir_center_idx = np.argmax(similarities)
            best_ir_center_similarity = similarities[best_ir_center_idx]

            if best_ir_center_similarity >= max_similarity[best_ir_center_idx] * 0.5:
                pseudo_labels_hat[i] = best_ir_center_idx
            else:
                pseudo_labels_hat[i] = -1

        return pseudo_labels_hat

    color_aug = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    height=args.height
    width=args.width
    train_transformer_rgb = T.Compose([
    color_aug,
    T.Resize((height, width)),#, interpolation=3
    T.Pad(10),
    T.RandomCrop((height, width)),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    normalizer,
    ChannelRandomErasing(probability = 0.5)
    ])

    train_transformer_rgb1 = T.Compose([
    color_aug,
    T.Resize((height, width)),
    T.Pad(10),
    T.RandomCrop((height, width)),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    normalizer,
    ChannelRandomErasing(probability = 0.5),
    ChannelExchange(gray = 2)
    ])

    transform_thermal = T.Compose( [
        color_aug,
        T.Resize((height, width)),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        ChannelRandomErasing(probability = 0.5),
        ChannelAdapGray(probability =0.5)
        ])
    transform_thermal1 = T.Compose( [
        color_aug,
        T.Resize((height, width)),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        ChannelRandomErasing(probability = 0.5),
        ChannelAdapGray(probability =0.5)])

    for epoch in range(args.epochs):

        if (epoch == 0):
            checkpoint = load_checkpoint('/root/autodl-tmp/HIL-main/logs/sysu_train_baseline/model_best.pth.tar')
            model.load_state_dict(checkpoint['state_dict'])

        if (epoch == 20):
            checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
            model.load_state_dict(checkpoint['state_dict'])

        with torch.no_grad():
            ir_eps = 0.6
            rgb_eps = 0.6
            print('IR Clustering criterion: eps: {:.3f}'.format(ir_eps))
            cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
            print('RGB Clustering criterion: eps: {:.3f}'.format(rgb_eps))
            cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)

            print('==> Create pseudo labels for unlabeled RGB data')

            cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width,
                                             256, args.workers, 
                                             testset=sorted(dataset_rgb.train))
            features_rgb, features_rgb_s = extract_features(model, cluster_loader_rgb, print_freq=50,mode=1)
            features_rgb_s = torch.cat([features_rgb_s[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)
            del cluster_loader_rgb,
            features_rgb = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)
            features_rgb_ori=features_rgb
            
            features_rgb_s_=F.normalize(features_rgb_s, dim=1)
            features_rgb_ori_=F.normalize(features_rgb_ori, dim=1)

            features_rgb = torch.cat((features_rgb,features_rgb_s), 1)
            features_rgb_=F.normalize(features_rgb, dim=1)

            print('==> Create pseudo labels for unlabeled IR data')
            cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
                                             256, args.workers, 
                                             testset=sorted(dataset_ir.train))
            features_ir, features_ir_s = extract_features(model, cluster_loader_ir, print_freq=50,mode=2)
            del cluster_loader_ir
            features_ir = torch.cat([features_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)
            features_ir_ori=features_ir
            
            features_ir_s = torch.cat([features_ir_s[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)
            features_ir_s_=F.normalize(features_ir_s, dim=1)

            features_ir = torch.cat((features_ir,features_ir_s), 1)
            features_ir_=F.normalize(features_ir, dim=1)
            features_ir_ori_=F.normalize(features_ir_ori, dim=1)

            rerank_dist_ir = compute_jaccard_distance(features_ir_, k1=30, k2=args.k2,search_option=3)
            pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)

            rerank_dist_rgb = compute_jaccard_distance(features_rgb_, k1=30, k2=args.k2,search_option=3)
            pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)

            del rerank_dist_rgb
            del rerank_dist_ir

        rgb_id = [data[1] for data in sorted(dataset_rgb.train)]
        ir_id = [data[1] for data in sorted(dataset_ir.train)]

        # 分别计算 RGB 模态的指标
        rgb_true_labels = rgb_id  # RGB 的真实标签
        rgb_pred_labels = pseudo_labels_rgb  # RGB 的伪标签

        rgb_ari = metrics.adjusted_rand_score(rgb_true_labels, rgb_pred_labels)
        rgb_fmi = metrics.fowlkes_mallows_score(rgb_true_labels, rgb_pred_labels)
        rgb_ami = metrics.adjusted_mutual_info_score(rgb_true_labels, rgb_pred_labels)
        rgb_v_measure = metrics.v_measure_score(rgb_true_labels, rgb_pred_labels)

        print(f"RGB 模态 - Adjusted Rand Index (ARI): {rgb_ari:.4f}")
        print(f"RGB 模态 - Fowlkes-Mallows Index (FMI): {rgb_fmi:.4f}")
        print(f"RGB 模态 - Adjusted Mutual Information (AMI): {rgb_ami:.4f}")
        print(f"RGB 模态 - V-measure: {rgb_v_measure:.4f}")

        # 分别计算 IR 模态的指标
        ir_true_labels = ir_id  # IR 的真实标签
        ir_pred_labels = pseudo_labels_ir  # IR 的伪标签

        ir_ari = metrics.adjusted_rand_score(ir_true_labels, ir_pred_labels)
        ir_fmi = metrics.fowlkes_mallows_score(ir_true_labels, ir_pred_labels)
        ir_ami = metrics.adjusted_mutual_info_score(ir_true_labels, ir_pred_labels)
        ir_v_measure = metrics.v_measure_score(ir_true_labels, ir_pred_labels)

        print(f"IR 模态 - Adjusted Rand Index (ARI): {ir_ari:.4f}")
        print(f"IR 模态 - Fowlkes-Mallows Index (FMI): {ir_fmi:.4f}")
        print(f"IR 模态 - Adjusted Mutual Information (AMI): {ir_ami:.4f}")
        print(f"IR 模态 - V-measure: {ir_v_measure:.4f}")

        '''''''''
        cluster_features_ir_c = generate_cluster_features(pseudo_labels_ir, features_ir)
        cluster_features_rgb_c = generate_cluster_features(pseudo_labels_rgb, features_rgb)
        pseudo_labels_ir = correct_label(features_ir, pseudo_labels_ir, cluster_features_ir_c)
        pseudo_labels_rgb = correct_label(features_rgb, pseudo_labels_rgb, cluster_features_rgb_c)
        '''''''''

        num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
        num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)

        cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir_ori)
        cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb_ori)

        cluster_features_ir_d = generate_cluster_features_kmeans(pseudo_labels_ir, features_ir_ori)
        cluster_features_rgb_d = generate_cluster_features_kmeans(pseudo_labels_rgb, features_rgb_ori)

        memory_ir = ClusterMemory(768, num_cluster_ir, temp=args.temp,
                                  momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory_rgb = ClusterMemory(768, num_cluster_rgb, temp=args.temp,
                                   momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
        memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()

        memory_ir.features_d = F.normalize(cluster_features_ir_d, dim=2).cuda()
        memory_rgb.features_d = F.normalize(cluster_features_rgb_d, dim=2).cuda()

        trainer.memory_ir = memory_ir
        trainer.memory_rgb = memory_rgb
        wise_momentum = 0.9
        print('wise_momentum', wise_momentum)
        wise_memory_rgb = Memory_wise_v3(768, len(dataset_rgb.train), num_cluster_rgb, temp=args.temp,
                                         momentum=wise_momentum).cuda()  # args.momentum
        wise_memory_ir = Memory_wise_v3(768, len(dataset_ir.train), num_cluster_ir, temp=args.temp,
                                        momentum=wise_momentum).cuda()
        wise_memory_ir.features = F.normalize(features_ir_ori, dim=1).cuda()
        wise_memory_rgb.features = F.normalize(features_rgb_ori, dim=1).cuda()
        wise_memory_ir.centers_d = F.normalize(cluster_features_ir_d, dim=2).cuda()
        wise_memory_rgb.centers_d = F.normalize(cluster_features_rgb_d, dim=2).cuda()


        nameMap_ir = {val[0]: idx for (idx, val) in enumerate(sorted(dataset_ir.train))}

        nameMap_rgb = {val[0]: idx for (idx, val) in enumerate(sorted(dataset_rgb.train))}

        wise_memory_rgb.labels = torch.from_numpy(pseudo_labels_rgb)
        wise_memory_ir.labels = torch.from_numpy(pseudo_labels_ir)

        trainer.wise_memory_ir = wise_memory_ir
        trainer.wise_memory_rgb = wise_memory_rgb
        trainer.nameMap_ir = nameMap_ir
        trainer.nameMap_rgb = nameMap_rgb

        ######################
        cluster_features_ir_s = generate_cluster_features(pseudo_labels_ir, features_ir_s)
        cluster_features_rgb_s = generate_cluster_features(pseudo_labels_rgb, features_rgb_s)

        cluster_features_ir_s_d = generate_cluster_features_kmeans(pseudo_labels_ir, features_ir_s)
        cluster_features_rgb_s_d = generate_cluster_features_kmeans(pseudo_labels_rgb, features_rgb_s)

        memory_ir_s = ClusterMemory(768, num_cluster_ir, temp=args.temp,
                                    momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory_rgb_s = ClusterMemory(768, num_cluster_rgb, temp=args.temp,
                                     momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory_ir_s.features = F.normalize(cluster_features_ir_s, dim=1).cuda()
        memory_rgb_s.features = F.normalize(cluster_features_rgb_s, dim=1).cuda()
        memory_ir_s.features_d = F.normalize(cluster_features_ir_s_d, dim=2).cuda()
        memory_rgb_s.features_d = F.normalize(cluster_features_rgb_s_d, dim=2).cuda()

        trainer.memory_ir_s = memory_ir_s
        trainer.memory_rgb_s = memory_rgb_s

        wise_memory_rgb_s = Memory_wise_v3(768, len(dataset_rgb.train), num_cluster_rgb, temp=args.temp,
                                           momentum=wise_momentum).cuda()  # 0.9
        wise_memory_ir_s = Memory_wise_v3(768, len(dataset_ir.train), num_cluster_ir, temp=args.temp,
                                          momentum=wise_momentum).cuda()  # args.momentum
        wise_memory_ir_s.features = F.normalize(features_ir_s, dim=1).cuda()
        wise_memory_rgb_s.features = F.normalize(features_rgb_s, dim=1).cuda()
        wise_memory_ir_s.centers_d = F.normalize(cluster_features_ir_s_d, dim=2).cuda()
        wise_memory_rgb_s.centers_d = F.normalize(cluster_features_rgb_s_d, dim=2).cuda()


        trainer.wise_memory_ir_s = wise_memory_ir_s
        trainer.wise_memory_rgb_s = wise_memory_rgb_s

        pseudo_labeled_dataset_ir = []
        ir_label = []
        pseudo_real_ir = {}
        cams_ir = []
        modality_ir = []
        outlier = 0
        cross_cam = []
        idxs_ir = []
        ir_cluster = collections.defaultdict(list)

        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir)):
            cams_ir.append(cid)
            modality_ir.append(1)
            cross_cam.append(int(cid + 4))
            ir_label.append(label.item())
            ir_cluster[cid].append(label.item())
            if label != -1:
                pseudo_labeled_dataset_ir.append((fname, label.item(), cid))

                pseudo_real_ir[label.item()] = pseudo_real_ir.get(label.item(), []) + [_]
                pseudo_real_ir[label.item()] = list(set(pseudo_real_ir[label.item()]))

            else:
                outlier = outlier + 1

        print('==> Statistics for IR epoch {}: {} clusters outlier {}'.format(epoch, num_cluster_ir, outlier))

        pseudo_labeled_dataset_rgb = []
        rgb_label = []
        pseudo_real_rgb = {}
        cams_rgb = []
        modality_rgb = []
        outlier = 0
        idxs_rgb = []
        rgb_cluster = collections.defaultdict(list)

        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb)):
            cams_rgb.append(cid)
            modality_rgb.append(0)
            cross_cam.append(int(cid))
            rgb_label.append(label.item())
            rgb_cluster[cid].append(label.item())
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))

                pseudo_real_rgb[label.item()] = pseudo_real_rgb.get(label.item(), []) + [_]
                pseudo_real_rgb[label.item()] = list(set(pseudo_real_rgb[label.item()]))
            else:
                outlier = outlier + 1

        print('==> Statistics for RGB epoch {}: {} clusters outlier {} '.format(epoch, num_cluster_rgb, outlier))

        if epoch >= trainer.cmlabel:

            # 标签传递
            if epoch % 1 == 0:
                cluster_features_ir= generate_cluster_features(pseudo_labels_ir, features_ir)
                pseudo_labels_rgb= correct_label_transfer(features_rgb, pseudo_labels_rgb, cluster_features_ir)
                num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)
                num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
                features_all = torch.cat((features_rgb_ori, features_ir_ori), dim=0)
                pseudo_labels_all = torch.cat((torch.from_numpy(pseudo_labels_rgb), torch.from_numpy(pseudo_labels_ir)),dim=-1).view(-1).cpu().numpy()
                cluster_features = generate_cluster_features(pseudo_labels_all, features_all)
                cluster_features_d = generate_cluster_features_kmeans(pseudo_labels_all, features_all)
                shared_memory = ClusterMemory_shared(768, num_cluster_ir, temp=args.temp, momentum=args.momentum, use_hard=args.use_hard)
                shared_memory.features = F.normalize(cluster_features, dim=1).cuda()
                shared_memory.features_d = F.normalize(cluster_features_d, dim=2).cuda()
                trainer.shared_memory_ir = shared_memory
                trainer.shared_memory_rgb = shared_memory
                features_all_s = torch.cat((features_rgb_s, features_ir_s), dim=0)
                cluster_features_ir_s = generate_cluster_features(pseudo_labels_all, features_all_s)
                cluster_features_ir_s_d = generate_cluster_features_kmeans(pseudo_labels_all, features_all_s)
                shared_memory_s = ClusterMemory_shared(768, num_cluster_ir, temp=args.temp,
                                                momentum=0.1, use_hard=args.use_hard)
                shared_memory_s.features = F.normalize(cluster_features_ir_s, dim=1).cuda()
                shared_memory_s.features_d = F.normalize(cluster_features_ir_s_d, dim=2).cuda()
                trainer.shared_memory_ir_s = shared_memory_s
                trainer.shared_memory_rgb_s = shared_memory_s

                # 计算 ALL 模态的聚类指标
                true_labels_all = torch.cat((torch.tensor(rgb_true_labels), torch.tensor(ir_true_labels)), dim=0).cpu().numpy()  # ALL 模态的真实标签
                pred_labels_all = pseudo_labels_all  # ALL 模态的伪标签

                # 调用指标计算函数
                all_ari = metrics.adjusted_rand_score(true_labels_all, pred_labels_all)
                all_fmi = metrics.fowlkes_mallows_score(true_labels_all, pred_labels_all)
                all_ami = metrics.adjusted_mutual_info_score(true_labels_all, pred_labels_all)
                all_v_measure = metrics.v_measure_score(true_labels_all, pred_labels_all)

                # 输出 ALL 模态的结果
                print(f"ALL 模态 - Adjusted Rand Index (ARI): {all_ari:.4f}")
                print(f"ALL 模态 - Fowlkes-Mallows Index (FMI): {all_fmi:.4f}")
                print(f"ALL 模态 - Adjusted Mutual Information (AMI): {all_ami:.4f}")
                print(f"ALL 模态 - V-measure: {all_v_measure:.4f}")



            else:
                cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb)
                pseudo_labels_ir = correct_label_transfer(features_ir, pseudo_labels_ir, cluster_features_rgb)
                num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)
                num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
                features_all = torch.cat((features_rgb_ori, features_ir_ori), dim=0)
                pseudo_labels_all = torch.cat((torch.from_numpy(pseudo_labels_rgb), torch.from_numpy(pseudo_labels_ir)),dim=-1).view(-1).cpu().numpy()
                cluster_features = generate_cluster_features(pseudo_labels_all, features_all)
                cluster_features_d = generate_cluster_features_kmeans(pseudo_labels_all, features_all)
                shared_memory = ClusterMemory_shared(768, num_cluster_rgb, temp=args.temp, momentum=args.momentum, use_hard=args.use_hard)
                shared_memory.features = F.normalize(cluster_features, dim=1).cuda()
                shared_memory.features_d = F.normalize(cluster_features_d, dim=2).cuda()
                trainer.shared_memory_ir = shared_memory
                trainer.shared_memory_rgb = shared_memory
                features_all_s = torch.cat((features_rgb_s, features_ir_s), dim=0)
                cluster_features_ir_s = generate_cluster_features(pseudo_labels_all, features_all_s)
                cluster_features_ir_s_d = generate_cluster_features_kmeans(pseudo_labels_all, features_all_s)
                shared_memory_s = ClusterMemory_shared(768, num_cluster_rgb, temp=args.temp,
                                                momentum=0.1, use_hard=args.use_hard)
                shared_memory_s.features = F.normalize(cluster_features_ir_s, dim=1).cuda()
                shared_memory_s.features_d = F.normalize(cluster_features_ir_s_d, dim=2).cuda()
                trainer.shared_memory_ir_s = shared_memory_s
                trainer.shared_memory_rgb_s = shared_memory_s

            ##############################################################################################################
            pseudo_labeled_dataset_ir = []
            cams_ir = []
            modality_ir = []
            outlier = 0
            cross_cam = []
            for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir)):
                cams_ir.append(int(cid))
                modality_ir.append(int(1))
                cross_cam.append(int(cid))
                indexes = torch.tensor([trainer.nameMap_ir[fname]])
                ir_label_ms = trainer.wise_memory_ir.labels[indexes]
                if (label != -1) and (ir_label_ms != -1):
                    pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
                else:
                    outlier = outlier + 1
            print('==> Statistics for IR epoch {}: {} clusters outlier {}'.format(epoch, num_cluster_ir, outlier))
            pseudo_labeled_dataset_rgb = []
            cams_rgb = []
            modality_rgb = []
            outlier = 0
            cross_cam = []
            for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb)):
                cams_rgb.append(int(cid))
                modality_rgb.append(int(0))
                cross_cam.append(int(cid))
                indexes = torch.tensor([trainer.nameMap_rgb[fname]])
                rgb_label_ms = trainer.wise_memory_rgb.labels[indexes]
                if (label != -1) and (rgb_label_ms != -1):
                    pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
                else:
                    outlier = outlier + 1
            print(
                '==> Statistics for RGB epoch {}: {} clusters outlier {} '.format(epoch, num_cluster_rgb, outlier))
            ##############################################################################################################

        train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                    ir_batch, args.workers, args.num_instances, iters,
                                    trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,train_transformer=transform_thermal)
        train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                rgb_batch, args.workers, args.num_instances, iters,
                                trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,train_transformer=train_transformer_rgb,train_transformer1=train_transformer_rgb1)

        train_loader_ir.new_epoch()
        train_loader_rgb.new_epoch()
        trainer.train(epoch, train_loader_ir,train_loader_rgb, optimizer, print_freq=args.print_freq, train_iters=len(train_loader_ir))

        if epoch>=0 and ( (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1)):
            #_,mAP_homo = evaluator.evaluate(test_loader_ir, dataset_ir.query, dataset_ir.gallery, cmc_flag=True,modal=2)
            #_,mAP_homo = evaluator.evaluate(test_loader_rgb, dataset_rgb.query, dataset_rgb.gallery, cmc_flag=True,modal=1)
##############################
            args.test_batch=64
            args.img_w=args.width
            args.img_h=args.height
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            transform_test = T.Compose([
                T.ToPILImage(),
                T.Resize((args.img_h,args.img_w)),
                T.ToTensor(),
                normalize,
            ])
            mode='all'
            data_path='/root/autodl-tmp/HIL-main/data/sysu'
            query_img, query_label, query_cam = process_query_sysu(data_path, mode=mode)
            nquery = len(query_label)
            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
            query_feat_fc = extract_query_feat(model,query_loader,nquery)
            for trial in range(10):
                gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=mode, trial=trial)
                ngall = len(gall_label)
                trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

                gall_feat_fc = extract_gall_feat(model,trial_gall_loader,ngall)

                distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
                cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)

                if trial == 0:
                    all_cmc = cmc
                    all_mAP = mAP
                    all_mINP = mINP

                else:
                    all_cmc = all_cmc + cmc
                    all_mAP = all_mAP + mAP
                    all_mINP = all_mINP + mINP

                print('Test Trial: {}'.format(trial))
                print(
                    'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                        cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

            cmc = all_cmc / 10
            mAP = all_mAP / 10
            mINP = all_mINP / 10
            print('All Average:')
            print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

#################################

            if epoch >= 0:
                is_best = (cmc[0]+mAP > best_mAP)
                best_mAP = max(cmc[0]+mAP, best_mAP)
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'best_mAP': cmc[0]+mAP,
                }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            mode='indoor'
            data_path='/root/autodl-tmp/HIL-main/data/sysu'
            query_img, query_label, query_cam = process_query_sysu(data_path, mode=mode)
            nquery = len(query_label)
            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
            query_feat_fc = extract_query_feat(model,query_loader,nquery)
            for trial in range(10):
                gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=mode, trial=trial)
                ngall = len(gall_label)
                trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

                gall_feat_fc = extract_gall_feat(model,trial_gall_loader,ngall)

                distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))

                cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
                if trial == 0:
                    all_cmc = cmc
                    all_mAP = mAP
                    all_mINP = mINP

                else:
                    all_cmc = all_cmc + cmc
                    all_mAP = all_mAP + mAP
                    all_mINP = all_mINP + mINP


                print('Test Trial: {}'.format(trial))
                print(
                    'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                        cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
            cmc1 = all_cmc / 10
            mAP1 = all_mAP / 10
            mINP1 = all_mINP / 10
            print('indoor All Average:')
            print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc1[0], cmc1[4], cmc1[9], cmc1[19], mAP1, mINP1))

############################
        lr_scheduler.step()
    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    _,mAP_homo = evaluator.evaluate(test_loader_ir, dataset_ir.query, dataset_ir.gallery, cmc_flag=True,modal=2)
    _,mAP_homo = evaluator.evaluate(test_loader_rgb, dataset_rgb.query, dataset_rgb.gallery, cmc_flag=True,modal=1)
    mode='all'
    data_path='/root/autodl-tmp/HIL-main/data/sysu'
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=mode)
    nquery = len(query_label)
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    query_feat_fc = extract_query_feat(model,query_loader,nquery)
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=mode, trial=trial)
        ngall = len(gall_label)
        trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

        gall_feat_fc = extract_gall_feat(model,trial_gall_loader,ngall)

        distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))

        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP

        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP


        print('Test Trial: {}'.format(trial))
        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    cmc = all_cmc / 10
    mAP = all_mAP / 10
    mINP = all_mINP / 10
    print('all search All Average:')
    print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))


    mode='indoor'
    data_path='/root/autodl-tmp/HIL-main/data/sysu'
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=mode)
    nquery = len(query_label)
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    query_feat_fc = extract_query_feat(model,query_loader,nquery)
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=mode, trial=trial)
        ngall = len(gall_label)
        trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

        gall_feat_fc = extract_gall_feat(model,trial_gall_loader,ngall)

        distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))

        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP

        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP


        print('Test Trial: {}'.format(trial))
        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    cmc = all_cmc / 10
    mAP = all_mAP / 10
    mINP = all_mINP / 10
    print('indoor All Average:')
    print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

#################################
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    parser.add_argument(
        "--config_file", default="vit_base_ics_288.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    # data
    parser.add_argument('-d', '--dataset', type=str, default='dukemtmcreid',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=2)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=288, help="input height")#288 384
    parser.add_argument('--width', type=int, default=144, help="input width")#144 128
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,#30
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        )
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.000035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=30)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action="store_true")
    parser.add_argument('--no-cam',  action="store_true")
    parser.add_argument('--warmup-step', type=int, default=0)
    parser.add_argument('--milestones', nargs='+', type=int, default=[20,40],
                        help='milestones for the learning rate decay')
    main()
