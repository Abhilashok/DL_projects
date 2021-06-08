import os
import csv
import cv2 as cv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import Adam

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

#
# print(yolo)
# print(architecture)
######################################################


# ########################

# Images = []
# Labels = []
# with open(file_path, 'r') as csvfile:
#     reader = csv.reader(csvfile)
#     for image_file,label_file in reader:
#         image_path = os.path.join(Images_dir,image_file)
#         label_path = os.path.join(Labels_dir,label_file)


#         image = cv.imread(image_path)
#         image = cv.resize()
#         with open(label_path, 'r') as l_file:
#             label = []
#             for row in l_file:
#                 print(row)
#                 class_label, x, y, width, height = [
#                     float(x) if float(x) != int(float(x)) else int(float(x))
#                     for x in row.replace("\n","").split()
#                     ]
#                 label.append([class_label, x, y, width, height])

#         Images.append(image)
#         Labels.append(label)

#         break

######################get data#########################


class VOC_Dataset(Dataset):
    def __init__(self, voc_dir, Images_dir, Labels_dir, S=3, B=1, C=20, transforms=None):
        super().__init__()
        self.voc_dir = voc_dir
        self.Images_dir = Images_dir
        self.Labels_dir = Labels_dir
        self.df = pd.read_csv(self.voc_dir, header=None)

        self.S = S
        self.B = B
        self.C = C

        # self.Images = None
        # self.Labels = None
        self.transform = transforms

    def get_all(self, voc_dir, Images_dir, Labels_dir):
        file_path = os.path.join(dir, 'train.csv')
        Images = []
        Labels = []
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for image_file, label_file in reader:
                image_path = os.path.join(Images_dir, image_file)
                label_path = os.path.join(Labels_dir, label_file)

                image = cv.imread(image_path)
                with open(label_path, 'r') as l_file:
                    labels = []
                    for row in l_file:
                        print(row)
                        class_label, x, y, width, height = [
                            float(x) if float(x) != int(float(x)) else int(float(x))
                            for x in row.replace("\n", "").split()
                        ]
                        labels.append([class_label, x, y, width, height])

                Images.append(image)

                break

    def __getitem__(self, index):

        image_file, label_file = self.df.iloc[index]
        image_path = os.path.join(self.Images_dir, image_file)
        label_path = os.path.join(self.Labels_dir, label_file)

        image = cv.imread(image_path)
        # print(image.shape)
        # image = cv.resize(image, (256,256))

        with open(label_path, 'r') as l_file:
            labels = []
            for row in l_file:
                # print(row)
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(float(x))
                    for x in row.replace("\n", "").split()
                ]
                labels.append([class_label, x, y, width, height])
        #print(labels)
        if self.transform:
            image = self.transform(image)
        t_plots = []
        new_image_targ = image.copy()
        for t_box in labels:
            print(t_box[1:])
            x,y,w,h = t_box[1:]
            H = image.shape[0]
            W = image.shape[1]
            print((int(x*256),int(y*256)),(int(256*(x+w)),int(256*(y+h))))
            new_image_targ = cv.rectangle(new_image_targ, (int(x*W),int(y*H)),(int(W*(x+w)),int(H*(y+h))),color = (255, 0, 0),thickness = 2)
            
        
            plt.imshow(new_image_targ)
            plt.show()




        label_matrix = torch.zeros((self.S, self.S, self.C + self.B * 5))
        for box in labels:

            class_label, x, y, width, height = box

            i, j = int(self.S * x), int(self.S * y)
            # print(x,y)
            # print(i,j)
            # print(width,height)
            x_cell = (self.S * x - i )
            y_cell = (self.S * y - j )
            # print(x_cell,y_cell)
            
            width_cell = self.S * width
            height_cell = self.S * height
            # print(width_cell,height_cell)
            if label_matrix[i, j, 20] == 0:
                label_matrix[i,j,class_label] = 1
                label_matrix[i, j, 20] = 1
                label_matrix[i, j, 21:25] = torch.tensor([x_cell, y_cell, width_cell, height_cell])

            # if label_matrix[i, j, 25] == 0:
            #     label_matrix[i, j, 25] = 1
            #     label_matrix[i, j, 26:30] = torch.tensor([x_cell, y_cell, width_cell, height_cell])
        image = torch.Tensor(image).view(3,256,256) 
        # print(image.shape)
        # plt.imshow(image)
        # plt.show()
        return [image, label_matrix]

    def __len__(self):
        return len(self.df)


#####################################


dir = 'dataset'
Images_dir = 'images'
Labels_dir = 'labels'
file_path = os.path.join(dir, 'train.csv')

dataset = VOC_Dataset(file_path, Images_dir, Labels_dir)

##########################################


architecture = [(3, 192, 7, 2, 6),
                (192, 256, 3, 1, 2),
                (256, 512, 5, 1, 4),
                (512, 1024, 5, 1, 4),
                (1024, 1024, 3, 1, 2),
                ]


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        # print(kwargs.items())

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = F.max_pool2d(self.leakyrelu(self.batchnorm(self.conv(x))), (2, 2))
        # print(x.shape)
        return x




class Yolo(nn.Module):
    def __init__(self, in_channels, architecture, split_size):
        super().__init__()

        self.architecture = architecture
        self.in_channels = in_channels
        self.split_size = split_size
        self.conv_net = self.create_conv_net(self.architecture)
        # self.fc_net = self.create_fc_net(**kwargs)

        x = torch.randn((3, 256, 256)).view(-1, 3, 256, 256)
        print(x.shape)

        self._to_linear = None
        if self._to_linear == None:
            y = self.conv_net(x)
            (_, b, c, d) = y.shape
            self._to_linear = b * c * d

        self.fc1 = nn.Linear(self._to_linear, 1024)
        self.fc2 = nn.Linear(1024, 25 * self.split_size * self.split_size)

    def forward(self, x):
        x = self.conv_net(x).view(-1,self._to_linear)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x.view(-1,  self.split_size, self.split_size,25)

    def create_conv_net(self, architecture):
        layers = []

        for step in architecture:
            layers += [CNN(in_channels=step[0],
                           out_channels=step[1],
                           kernel_size=step[2],
                           stride=step[3],
                           padding=step[4]
                           )]
        # print (*layers)
        return nn.Sequential(*layers)


yolo = Yolo(in_channels=3, split_size=3, architecture=architecture)


def IoU(pred_boxs, target_boxs):


    # b1_x1 = pred_box[..., 0].view(-1, 3,3,1)
    # b1_y1 = pred_box[..., 1].view(-1, 3,3,1)
    # b1_w = pred_box[..., 2].view(-1,3,3, 1)
    # b1_h = pred_box[..., 3].view(-1, 3,3,1)
    # b1_x2 = b1_x1 + b1_w
    # b1_y2 = b1_y1 + b1_h
    #
    # b2_x1 = target_box[..., 0].view(-1, 3,3,1)
    # b2_y1 = target_box[..., 1].view(-1,3,3, 1)
    # b2_w = target_box[..., 2].view(-1,3,3, 1)
    # b2_h = target_box[..., 3].view(-1,3,3, 1)
    # b2_x2 = b2_x1 + b2_w
    # b2_y2 = b2_y1 + b2_h
    #
    # x1_max = torch.max(b1_x1, b2_x1)
    # y1_max = torch.max(b1_y1, b2_y1)
    #
    # x2_min = torch.max(b1_x2, b2_x2)
    # y2_min = torch.max(b1_y2, b2_y2)
    #
    # width = (x2_min - x1_max).clamp(0)
    # height = (y2_min - y1_max).clamp(0)
    #
    # iou = (width * height) / (b1_w * b1_h + b2_w * b2_h - width * height)

    for index in range(pred.shape[0]):
        pred_box = pred_boxs[index]
        target_box = target_boxs[index]
        print(target_box.shape)
        break

    return iou


loss_function = nn.MSELoss()
optimizer = Adam(yolo.parameters())

# class Yolo_Loss(nn.Module):
#     def __init__(self, S=3, B=2, C=20):
#         super().__init__()

#         self.S = S
#         self.C = C
#         self.B = B

#         self.mse = nn.MSELoss()

#     def forward(self, pred, target):
#         IoU_b1 = IoU(pred[..., 21:25], target[..., 21:25])
#         IoU_b2 = IoU(pred[..., 26:30], target[..., 21:25])

#         print(IoU_b2.shape)
#         print(IoU_b1.shape)

# Loss_fn = Yolo_Loss()




Train_Loader = DataLoader(dataset,batch_size=16, shuffle = True)


def print_step(Images,pred,target):
    conf_threshold = 0.25
    p_plots = []
    t_plots = []
    Images_ = Images.view(-1,448,448,3)
    # plt.imshow(Images_[15])
    # plt.show()
    print(pred[...,20:])
    print(targ[..., 20:])
    for Image,prediction,target in zip(Images_,pred, target):
        #print(Image.shape)
        pred_boxes = []
        targ_boxes = []
        # plt.imshow(Image)
        # plt.show()
        #print(target)
        for (i,j) in np.ndindex(3,3):
            if prediction[i, j, 20] > conf_threshold:
                #print(prediction[i,j,21:].shape)
                (x_cell, y_cell, w_cell, h_cell) = prediction[i,j,21:]
                
                x = (x_cell + i)/3
                y = (y_cell + j)/3
                h = h_cell /3
                w = w_cell /3
                pred_boxes.append([x,y,w,h])
            if target[i, j, 20] > conf_threshold:
                #print(prediction[i,j,21:].shape)
                #print(target[i,j,21:])
                (x_cell, y_cell, w_cell, h_cell) = target[i,j,21:]
                
                x = (x_cell + i)/3
                y = (y_cell + j)/3
                h = h_cell /3
                w = w_cell/3
                
                targ_boxes.append([x,y,w,h])
        print(f'targ_boxes{targ_boxes}')
        print(f'pred_boxes{pred_boxes}')
        new_image_pred = Image.numpy().copy()
        new_image_targ = Image.numpy().copy()
        W = Image.shape[0]
        H = Image.shape[1]
        #print(W,H)
        #print(len(boxes))
        print(len(pred_boxes))
        # print()
        print(len(targ_boxes))
        
        for p_box in pred_boxes:
        
            x,y,w,h = p_box
            new_image_pred = cv.rectangle(new_image_pred, (int(x*W),int(y*H)), (int(H*(y+h)),int(W*(x+w))),color = (255, 0, 0),thickness = 2)
            
        p_plots.append(new_image_pred)
        
        for t_box in targ_boxes:
        
            x,y,w,h = t_box
            new_image_targ = cv.rectangle(new_image_targ, (int(y*H),int(x*W)),(int(H*(y+h)),int(W*(x+w))),color = (255, 0, 0),thickness = 2)
            
        t_plots.append(new_image_targ)


        
    plt.figure(figsize=(10, 10))
    for i, image in enumerate(Images_):
        if i < 4:
            plt.subplot(2, 2, i+1)
            plt.imshow(image)
    plt.show()
        
    plt.figure(figsize=(10, 10))
    for i, image in enumerate(p_plots):
        if i < 4:
            plt.subplot(2, 2, i+1)
            plt.imshow(image)
    plt.show()
    
    plt.figure(figsize=(10, 10))
    for i, image in enumerate(t_plots):
        if i < 4:
            plt.subplot(2, 2, i+1)
            plt.imshow(image)
    plt.show()
    plt.show()





def Train_fn():
    for epoch in range(5):
        loop = tqdm(Train_Loader, leave = False)
        for index,(X,Y) in enumerate(loop):
            loop.set_description(f"Epoch {epoch}")
            #print(index)
            X = X.view(-1, 3, 256, 256)/255.0
            Y = Y.view(-1,3,3,25)
            if  index % 25 == 0:
                loss,pred = feed_forward(X, Y, train = True, verbose = True)
            else:  
                loss,pred = feed_forward(X, Y, train=True)
                
            loop.set_postfix(loss=loss)
            
            
#


Train_fn()

def get_batch():
    X, Y = [], []
    for i in range(4):
        Image, Label = dataset.__getitem__(index=i)

        X.append(Image)
        Y.append(Label)
        

        # X = torch.tensor([item for item in X])
        # Y = torch.Tensor([item for item in Y])
    print(X)
    print(Y)
    return torch.stack(X), torch.stack(Y)


X, Y = get_batch()


print(Y.shape)

# Y = torch.stack(Y)

loss, pred = feed_forward(X/255.0, Y, train=False, verbose = True)



# X = torch.FloatTensor(X)
#
#




def print_boxes(Image, label):
    











