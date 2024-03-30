from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import open3d as o3d
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._jit_internal import Optional
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import collections
import matplotlib.pyplot as plt
import glob
import h5py
from torch.utils.data import Dataset


import argparse
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.utils.data import DataLoader
import sklearn.metrics as metrics
import tqdm

from DOConv import DOConv2d

from sklearn.model_selection import KFold

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
#     print(x.size())
    x = x.view(batch_size, -1, num_points)
    
#     print("After View: ",x.size())
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class DGCNN(nn.Module):
    def __init__(self, k=20, emb_dims=1024, dropout=0.5, output_channels=40,depth_mul=None):
        super(DGCNN, self).__init__()
        
        self.k = k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(DOConv2d(6, 64, kernel_size=1, bias=False, D_mul=depth_mul),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(DOConv2d(64*2, 64, kernel_size=1, bias=False, D_mul=depth_mul),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(DOConv2d(64*2, 128, kernel_size=1, bias=False, D_mul=depth_mul),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(DOConv2d(128*2, 256, kernel_size=1, bias=False, D_mul=depth_mul),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        
        x = get_graph_feature(x, k=self.k)
        
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x



def load_data(partition):
    
    DATA_DIR = "/kaggle/input/modelnet40-h5/modelnet40_ply_hdf5_2048"
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def load_data_our_human_pose():
    # download_modelnet40()
    all_data = []
    all_label = []
#     DATA_DIR = "/kaggle/input/modelnet40-h5"
    data_path = "/kaggle/input/modelnet40-h5/human_pose_no_ground_point.h5"
    print(data_path)
    for h5_name in glob.glob(data_path):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        
        # data = data.reshape(1, -1)
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

class OurHumanPose(Dataset):
    def __init__(self, num_points):
        self.data, self.label = load_data_our_human_pose()
        self.num_points = num_points
        
    def __getitem__(self, item):
#         pointcloud = self.data[item]
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
    
def load_data_scanobjectnn(partition):
    
    DATA_DIR = "dataset"
    all_data = []
    all_label = []
    h5_file_path = ''
    if partition == 'train':
        h5_file_path = 'training_objectdataset_augmentedrot_scale75.h5'
    elif partition == 'test':
        h5_file_path = 'test_objectdataset_augmentedrot_scale75.h5'
    
    for h5_name in glob.glob(os.path.join(DATA_DIR, h5_file_path)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label
    
class ScanObjectNN_hardest(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data_scanobjectnn(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def train(
    batch_size=32, test_batch_size=32, cuda='gpu',
    num_points=1024,
    lr = 0.001,
    momentum=0.9,
    epochs=50,
    exp_name='exp',
    dataset_name = 'human',
    model_name = 'pointnet'
):
    
    num_channels = 6
    if dataset_name == 'modelnet40':
        train_loader = DataLoader(ModelNet40(partition='train', num_points=num_points), num_workers=2,
                                  batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ModelNet40(partition='test', num_points=num_points), num_workers=2,
                                 batch_size=test_batch_size, shuffle=True, drop_last=False)
        num_channels = 40
    elif dataset_name == 'scanobject':
        train_loader = DataLoader(ScanObjectNN_hardest(partition='train', num_points=num_points), num_workers=2,
                                  batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(ScanObjectNN_hardest(partition='test', num_points=num_points), num_workers=2,
                                 batch_size=test_batch_size, shuffle=True, drop_last=False)
        num_channels = 15
    elif dataset_name == 'human':
        dataset = OurHumanPose(num_points=2048)
        train_size = int(0.8  * len(dataset))
        test_size = len(dataset) - train_size
        train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_set, num_workers=2,
                                  batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_set, num_workers=2,
                                 batch_size=test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if cuda else "cpu")

    #Try to load models
    
    if model_name == 'pointnet':
        model = PointNet(emb_dims=num_points,output_channels=num_channels).to(device)
    elif model_name == 'dgcnn':
        model = DGCNN(emb_dims=num_points,output_channels=num_channels,depth_mul=30).to(device)
    elif model_name == 'gdanet':
        model = GDANET(output_channels=num_channels).to(device)
    
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

#     if use_sgd:
#         print("Use SGD")
    opt = optim.SGD(model.parameters(), lr=lr*100, momentum=momentum, weight_decay=1e-4)
#     else:
#         print("Use Adam")
#     opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, epochs, eta_min=lr)
    
    criterion = cal_loss

    best_test_acc = 0
    list_train_loss  = []
    list_test_loss  = []
    list_train_acc  = []
    list_test_acc  = []
    for epoch in tqdm.tqdm(range(epochs)):
        
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
#             print("Data shape before permutation",data.size())
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
#             print("Data shape after permutation",data.size())
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        avg_train_acc = metrics.balanced_accuracy_score(train_true, train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,train_acc,avg_train_acc
                                                                                 )
        print(outstr)
        
        list_train_loss.append(train_loss*1.0/count)
        list_train_acc.append(train_acc)

        ###################
        ## Test
        ###################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
#             print(data.size()[0])
            if batch_size > 2:
                logits = model(data)
                loss = criterion(logits, label)
                preds = logits.max(dim=1)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
            
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        print(outstr)
        list_test_loss.append(test_loss*1.0/count)
        list_test_acc.append(test_acc)
        scheduler.step()
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), model_name+'_'+dataset_name+'.t7')
            torch.save(model,model_name+'_'+dataset_name+'.pt')
            
    return list_train_loss, list_test_loss, list_train_acc, list_test_acc
            

def test(dataset_name='human', model_name='pointnet'):
    
    device = torch.device("cuda")
    test_batch_size = 8
    if(dataset_name == 'human'):
        dataset = OurHumanPose(num_points=2048)
        train_size = int(0.8  * len(dataset))
        test_size = len(dataset) - train_size
        
        train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
        test_loader = DataLoader(test_set, num_workers=4, batch_size=test_batch_size, shuffle=False, drop_last=False)
    elif dataset_name == 'scanobject':
        test_loader = DataLoader(ScanObjectNN_hardest(partition='test', num_points=1024), num_workers=2,
                                 batch_size=test_batch_size, shuffle=True, drop_last=False)
#         num_channels = 15
    else:
        test_loader = DataLoader(ModelNet40(partition='test', num_points=1024), num_workers=2, batch_size=test_batch_size, shuffle=True, drop_last=False)
    
    model_path = 'dgcnn_scanobject.pt'
#     model_path = model_name+'_'+dataset_name+'.pt'
#     model_path = '/kaggle/input/dgcnn-doconv18/pytorch/dgcnn-scanobject-pt/1/dgcnn_scanobject.pt'
    model = torch.load(model_path, map_location=device)
    model.eval()
    criterion = cal_loss
    test_loss = 0.0
    count = 0.0
#     model.eval()
    test_pred = []
    test_true = []
    
    all_predictions = []
    all_labels = []
    all_points = []  # Assuming your data includes point clouds
    predictions = []
    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        
        if batch_size > 2:
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            
            all_points.append(data.cpu().numpy())
            all_predictions.append(preds.cpu().numpy())
            all_labels.append(label.cpu().numpy())

            predictions.extend(preds.cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    # test_acc = metrics.accuracy_score(test_true, test_pred)
    #         count += batch_size
    #         test_loss += loss.item() * batch_size
    #         test_true.append(label.cpu().numpy())
    #         test_pred.append(preds.detach().cpu().numpy())


    # test_true = np.concatenate(test_true)
    # test_pred = np.concatenate(test_pred)
    #         print(np.array(test_true).shape)
    #         print(np.array(test_pred).shape)
    yt = np.asarray(test_true).reshape(-1,1)
    yp = np.asarray(test_pred).reshape(-1,1)

    #         print(np.array(yt).shape)
    #         print(np.array(yp).shape)
    #         print(test_true[:5], test_pred[:5])
    test_acc = metrics.accuracy_score(yt, yp)
    avg_per_class_acc = metrics.balanced_accuracy_score(yt, yp)
    outstr = 'Test loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (test_loss*1.0/count,
                                                                          test_acc,
                                                                          avg_per_class_acc)

    print(outstr)
    
    return yt, yp, all_points, all_predictions, all_labels, predictions


list_labels = ['bag','bin','box','cabinet','chair','desk','display','door','shelf','table','bed','pillow','sink','sofa','toilet']

dataset_name = 'scanobject'
model_name = 'dgcnn'

NUM_EPOCH = 1

import gc
# !nvidia-smi --gpu-reset
gc.collect()
torch.cuda.empty_cache()
# !nvidia-smi
# torch.manual_seed(1)

# list_train_loss, list_test_loss, list_train_acc, list_test_acc = train(
#     num_points=1024,
#     batch_size=8, 
#     test_batch_size=8, 
#     epochs=NUM_EPOCH, 
#     model_name=model_name,
#     dataset_name=dataset_name
# )
# yt, yp, all_points, all_predictions, all_labels, predictions_extended = test(dataset_name, model_name)

predictions_extended = []
with h5py.File('predictions_extended.h5', 'r') as hf:
    predictions_extended = hf['predictions_extended'][:]

def visualize_point_cloud(point_cloud, actual_label, predicted_label):

    

    # Convert the point cloud to Open3D format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
    # Optionally, you can also color the point cloud
    # pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(point_cloud.shape[0], 3)))

    # Visualize the point cloud
    # o3d.visualization.draw_geometries([pcd], window_name=f"Label: {label}")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Actual: {actual_label}, Predicted: {predicted_label}")

    # Add the point cloud to the visualizer
    vis.add_geometry(pcd)

    # Access the render option and adjust the point size
    render_option = vis.get_render_option()
    render_option.point_size = 10  # Set your desired point size here
    render_option.show_coordinate_frame = True
    # Run the visualizer
    vis.run()
    vis.destroy_window()

test_loader = DataLoader(ScanObjectNN_hardest(partition='test', num_points=1024), num_workers=2,
                                 batch_size=8, shuffle=True, drop_last=False)

all_labels = []
with h5py.File('all_labels.h5', 'r') as hf:
    all_labels = hf['all_labels'][:]

all_points = []
with h5py.File('all_points.h5', 'r') as hf:
    all_points = hf['all_points'][:]

all_predictions = []
with h5py.File('all_predictions.h5', 'r') as hf:
    all_predictions = hf['all_predictions'][:]

for i, data in enumerate(all_points):
    # print(data.shape)
    # print(all_labels[i], all_predictions[i])
    batch_index = 0
    actual_label = all_labels[i][batch_index]
    predicted_label = all_predictions[i][batch_index]
    print(list_labels[actual_label],list_labels[predicted_label])
    
    # print(data[0].shape)
    
    pts = data[batch_index].T
    # print(pts.shape)

    visualize_point_cloud(pts, list_labels[actual_label], list_labels[predicted_label])
    if i == 7:  # Adjust this value to display more or fewer point clouds
        break



# from sklearn.metrics import classification_report

# print(classification_report(yt, yp))

# with h5py.File('all_predictions.h5', 'w') as hf:
#     hf.create_dataset("all_predictions",  data=all_predictions)

# with h5py.File('predictions_extended.h5', 'w') as hf:
#     hf.create_dataset("predictions_extended",  data=predictions_extended)

# with h5py.File('all_labels.h5', 'w') as hf:
#     hf.create_dataset("all_labels",  data=all_labels)

# def visualize_predictions(points, labels, predictions, index=0):
#     fig = plt.figure(figsize=(10, 5))
    
#     ax = fig.add_subplot(121, projection='3d')
#     ax.scatter(points[index][:, 0], points[index][:, 1], points[index][:, 2], c='skyblue', s=15)
#     ax.set_title(f'Actual: {labels[index]}')
    
#     ax = fig.add_subplot(122, projection='3d')
#     ax.scatter(points[index][:, 0], points[index][:, 1], points[index][:, 2], c='salmon', s=15)
#     ax.set_title(f'Predicted: {predictions[index]}')

#     plt.show()

# # Visualize the first point cloud and its prediction
# visualize_predictions(all_points, all_labels, all_predictions, index=0)

# index = 5
# pts = all_points[index][0].T
# # print(np.array(pts).shape)

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(pts)

# vis = o3d.visualization.Visualizer()
# vis.create_window()

# # Add the point cloud to the visualizer
# vis.add_geometry(pcd)

# # Access the render option and adjust the point size
# render_option = vis.get_render_option()
# render_option.point_size = 10  # Set your desired point size here

# # Run the visualizer
# vis.run()
# vis.destroy_window()
