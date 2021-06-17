# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 14:09:54 2021

@author: abhil
"""
####################################################
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import optim

import os
from bs4 import BeautifulSoup

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# torch.autograd.set_detect_anomaly(True)

###################Utils###############


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

File_dir = 'VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007'
Images_dir = os.path.join(File_dir,'JPEGImages')
Labels_dir = os.path.join(File_dir,'Annotations')



classes = ['aeroplane','bicycle','bird','boat','bottle',
                        'bus','car','cat','chair','cow',
                        'diningtable','dog','horse','motorbike','person',
                        'pottedplant','sheep','sofa','train','tvmonitor'
                        ]
 
torch.autograd.set_detect_anomaly(True)

S = 3
B = 2
C = 20

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0


Batch_size = 8
confidence_thres = 0.3




def to_encodings(boxes):
    label_matrix = torch.zeros((S, S, C + B * 5))
    
    for box in boxes:
        class_label, x, y, width, height = box
        i, j = int(S * x), int(S * y)
        
        x_cell = (S * x - i )
        y_cell = (S * y - j )
        width_cell = S * width
        height_cell = S * height
        
        if label_matrix[j, i, 20] == 0:
            label_matrix[j, i, class_label] = 1
            label_matrix[j, i, 20] = 1
            label_matrix[j, i, 21:25] = torch.tensor([x_cell, y_cell, width_cell, height_cell])
        else:
            break
        
    return label_matrix




def to_boxes(Label_matrix):
    
    boxes = []
    for (i,j) in np.ndindex(3,3):
            #print(Label_matrix[i,j,20])
                #print(prediction[i,j,21:].shape)
            confidence = Label_matrix[i,j,20]    
            if (confidence > confidence_thres):
                Class = torch.argmax(Label_matrix[i,j,:20])
                (x_cell, y_cell, w_cell, h_cell) = Label_matrix[i,j,21:]
                    
                x = (x_cell + j)/3
                y = (y_cell + i)/3
                h = h_cell /3
                w = w_cell /3
                boxes.append([Class, x ,y ,w, h])

    return boxes

def detected_image(image, boxes):
    new_image = image.copy()
    for box in boxes:
        
        #print(box[1:])
        x,y,w,h = box[1:]
        H = image.shape[0]
        W = image.shape[1]
        #print((int(x*W),int(y*H)),(int(W*(x+w)),int(H*(y+h))))
        new_image = cv.rectangle(new_image, (int(x*W),int(y*H)),(int(W*(x+w)),int(H*(y+h))),color = (1.0, 0, 0),thickness = 2)
        new_image = cv.putText(new_image,classes[box[0]],(int(x*W),int(y*H)),
                                   fontFace = cv.FONT_HERSHEY_SIMPLEX,fontScale = 1,
                                   color = (255, 0, 0),thickness = 2)
    return new_image


def print_step(Images,pred,target):
    
    p_plots = []
    t_plots = []
    Images_ = Images.view(-1,256,256,3)
    Pred = get_bestbox(pred)
    
    for Image,prediction,targ in zip(Images_,Pred, target):
        
            # print(prediction.shape)
            # print(target.shape)
            # print(prediction)
            # print('target')
            # print(target)
            pred_boxes = to_boxes(prediction)
            targ_boxes = to_boxes(targ)
           
            # print(f'targ_boxes{targ_boxes}')
            # print(f'pred_boxes{pred_boxes}')
            
            image = Image.numpy()
            
            image_targ = detected_image(image, targ_boxes)
            image_pred = detected_image(image, pred_boxes)
            
            t_plots.append(image_targ)
            p_plots.append(image_pred)


        
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
    







################CLASS_DATA####################





class Yolo_Dataset(Dataset):
    def __init__(self,Images_dir, Labels_dir, classes, S = S, B = B, C = C):
        super().__init__()
        
        self.Images_dir = Images_dir
        self.Labels_dir = Labels_dir
        self.S = S
        self.B = B
        self.C = C
        self.Files = os.listdir(Labels_dir)
        self.classes = classes
        
    def __getitem__(self, index):
        
        file = self.Files[index]
       # Xml_file = os.path.splitext(file)[0]
        Xml_file = os.path.join(self.Labels_dir,file)
        #print(Xml_file)
        with open(Xml_file, 'r') as Xml:
            data = Xml.read()
            Bs_data = BeautifulSoup(data,features="html.parser")
            
            img_file = Bs_data.find('filename').get_text()
            image = cv.imread(os.path.join(Images_dir,img_file))
            
            Y,X,Z = image.shape
            
            objects = Bs_data.find_all('object')
            boxes = []
            for Object in objects:
                
                Class = Object.find('name').get_text()
                xmin,ymin,xmax,ymax = [int(x) for x in Object.find('bndbox').get_text().split()]
                x , y, w, h = xmin/X, ymin/Y ,(xmax-xmin)/X ,(ymax - ymin)/Y 
                boxes.append([self.classes.index(Class), x, y, w, h])
                
        image = cv.resize(image, (256,256))
        
        
        # new_image = detected_image(image, boxes)
        # plt.imshow(new_image)
        # plt.show()
        
        label_matrix = to_encodings(boxes)
        # BBoxes = to_boxes(label_matrix) 
        
        # New_image = detected_image(image, BBoxes)
        # plt.imshow(New_image)
        # plt.show()

        image = torch.Tensor(image)
        return [image, label_matrix]
                
    def __len__(self):
        return len(self.Files)
    
    
    
yolo_dataset = Yolo_Dataset(Images_dir, Labels_dir, classes)

Train_Loader = DataLoader(yolo_dataset, batch_size = Batch_size, shuffle = True)

###################################################################

architecture = [(3, 64, 7, 2, 6 , 2),
                
                (64, 192, 3, 1, 2, 2),
                
                (192, 256, 1, 1, 0, 1),
                (256, 256, 3, 1, 2, 1),
                (256, 256, 1, 1, 0, 1),
                (256, 512, 1, 1, 0, 2),
                
                (512, 256, 1, 1, 0, 1),
                (256, 512, 3, 1, 2, 1),
                (512, 256, 1, 1, 0, 1),
                (256, 512, 3, 1, 2, 1),
                (512, 256, 1, 1, 0, 1),
                (256, 512, 3, 1, 2, 1),
                (512, 256, 1, 1, 0, 1),
                (256, 512, 1, 1, 0, 1),
                (512, 256, 1, 1, 0, 1),
                (256, 512, 1, 1, 0, 1),
                (512, 1024, 3, 1, 2, 2),
                
                (1024, 512, 1, 1, 0, 1),
                (512, 1024, 3, 1, 2, 1),
                (1024, 512, 1, 1, 0, 1),
                (512, 1024, 3, 1, 2, 1),
                (1024, 1024, 3, 1, 2, 2),
                
                (1024, 1024, 3, 1, 2, 1),
                (1024, 1024, 3, 1, 2, 1),
              
                ]


class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, maxpool_dim, **kwargs):
        super().__init__()
        # print(kwargs.items())

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.maxpool_dim = maxpool_dim

    def forward(self, x):
        #print(x.shape)
        x = F.max_pool2d(self.leakyrelu(self.batchnorm(self.conv(x))), (self.maxpool_dim, self.maxpool_dim))
        # print(x.shape)
        return x




class Yolo(nn.Module):
    def __init__(self, in_channels, architecture, split_size, C, B):
        super().__init__()

        self.architecture = architecture
        self.in_channels = in_channels
        self.split_size = split_size
        self.B = B
        self.C = C
        self.conv_net = self.create_conv_net(self.architecture)
        # self.fc_net = self.create_fc_net(**kwargs)

        x = torch.randn((3, 256, 256)).view(-1, 3, 256, 256)
        #print(x.shape)

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

        return x.view(-1,  self.split_size, self.split_size,self.C + 5*self.B)

    def create_conv_net(self, architecture):
        layers = []

        for step in architecture:
            #print(step)
            layers += [CNN(in_channels=step[0],
                           out_channels=step[1],
                           maxpool_dim = step[5],
                           kernel_size=step[2],
                           stride=step[3],
                           padding=step[4]
                           )]
            
        # print (*layers)
        return nn.Sequential(*layers)


yolo = Yolo(in_channels=3, split_size=3, architecture=architecture).to(DEVICE)

#print(yolo.parameters())

#####################################################


#loss_function = nn.MSELoss()
optimizer = optim.Adam(yolo.parameters(), lr = LEARNING_RATE, weight_decay= WEIGHT_DECAY)


def get_bestbox(pred):
    
    

        
    ious1 = get_ious(pred[..., 21:25], Y)
    ious2 = get_ious(pred[..., 26:], Y)
        
    get_1 = (ious1>ious2).unsqueeze(3)
    get_2 = ~get_1
        

        
    Best_pred = get_1*pred[..., :25]+get_2*torch.cat((pred[..., :20], pred[..., 25:]), dim = 3)
    
    return Best_pred

def get_ious(pred,Y):
    
    
    
    
    x1_p, y1_p, w_p, h_p = pred[..., 1],pred[..., 2],pred[..., 3],pred[..., 4]
    
    x1_t, y1_t, w_t, h_t = Y[..., 1], Y[..., 2], Y[..., 3], Y[..., 4]
    
    x2_p, y2_p = x1_p + pred[...,3], x1_p + pred[...,4]
    x2_t, y2_t = x1_t + Y[...,3], x1_t + Y[...,4]
    

    max_x1 = torch.max(x1_p,x1_t)
    max_y1 = torch.max(y1_p,y1_t)
    min_x2 = torch.min(x2_p,x2_t)
    min_y2 = torch.min(y2_p,y2_t)
    # print(max_x1)
    # print(max_x1.shape)
    # print(min_x2.shape)
    
    intersection = (min_x2 - max_x1).clamp(0)*(min_y2 - max_y1).clamp(0)
    # print(intersection)
    union = w_p*h_p + w_t*h_t - intersection
    # print(union)
    ious = intersection/(union + 1e-5)
    
    # print(ious.shape)
    return ious

# def loss_fn(pred, Y):
    
     
#          # print(pred.shape)
#          # print(Y.shape)
         
#     class_t = Y[..., :20]
#     class_p = Y[..., :20]
#     # print(class_t.shape,class_p.shape)
    
#     c_t = Y[..., 20]
#     c_p = pred[..., 20]
#          #print( pred[:,:, 21].shape)
#     x_p, y_p, w_p, h_p = pred[..., 21],pred[..., 22],pred[..., 23],pred[..., 24]
#     x_t, y_t, w_t, h_t = Y[..., 21], Y[..., 22], Y[..., 23], Y[..., 24]
    
#     # print(c_t.shape)
#     # print(x_p.shape)
         
#          #########boundingbosError#############
         
#     bb_loc_Error = c_t*((x_t-x_p)**2 + (y_t-y_p)**2)
#     bb_size_Error =  c_t*((w_t**1/2 - w_p**1/2)**2 + (w_t**1/2 - w_p**1/2)*2)
    
#     box_loss =self.L_coord*(bb_loc_Error + bb_size_Error) + 0.5*(1 - c_t)*(bb_loc_Error + bb_size_Error)
    
#     Box_loss = torch.sum(box_loss, (2, 1))
#     # print()
#     # print(box_loss.shape,Box_loss.shape)
         
#     #      #########ConfidenceError#########
#     ious = get_ious(pred, Y)
         
#     object_Error = ious*c_t*(1 - c_p)**2
#     no_object_Error = (1 - ious*c_t)*(c_p)**2    
    
#     confidence_loss = 5*object_Error + 0.5*no_object_Error
#     Confidence_loss = torch.sum(confidence_loss, (2, 1))
#     # print()
#     # print(object_Error.shape, no_object_Error.shape, Confidence_loss.shape)
#     #      ########ClassError#############
 
#     class_error = c_t*torch.sum((class_t-class_p)**2, dim = 3)
#     #print()
#     Class_loss = torch.sum(class_error, (2, 1))
#     #print(class_error.shape,Class_loss.shape)
#     #      # print(y_p.shape)
#     #      # print(w_p.shape)
#     #      # print(h_p.shape)
         
#     total_error = torch.mean(Box_loss + Confidence_loss + Class_loss)
    
    
#     # print(total_error)
    
#     return total_error
       
    


class Yolo_Loss(nn.Module):
    def __init__(self,S = 3,B = 2,C = 20):
        super().__init__()
        
        self.S = S
        self.C = C
        self.B = B
        
        
        self.L_noobj = 0.5
        self.L_coord = 5
        
        
        self.mse = nn.MSELoss()
        
        
    def forward(self, pred, Y):
        class_t = Y[..., :20]
        class_p = pred[..., :20]
        # print(class_t.shape,class_p.shape)
        
        c_t = Y[..., 20]
        c_p1 = pred[..., 20]
        c_p2 = pred[..., 25]
             #print( pred[:,:, 21].shape)

        
        ious1 = get_ious(pred[..., 21:25], Y)
        ious2 = get_ious(pred[..., 26:], Y)
        
        get_1 = ious1>ious2
        get_2 = ~get_1
        
        c_p = get_1*c_p1 + get_2*c_p2
        
        
        pred[..., 23:25] = torch.sign(pred[...,23:25])*torch.sqrt(torch.abs(pred[...,23:25] + 1e-6))
        pred[..., 28:] = torch.sign(pred[...,28:])*torch.sqrt(torch.abs(pred[...,28:] + 1e-6))
        Y[...,23:] = torch.sqrt(Y[...,23:])
        
        x_p1, y_p1, w_p1, h_p1 = get_1*(pred[..., 21],pred[..., 22],pred[..., 23],pred[..., 24])
        x_p2, y_p2, w_p2, h_p2 = get_2*(pred[..., 26],pred[..., 27],pred[..., 28],pred[..., 29])
        
        x_p, y_p, w_p, h_p = x_p1+x_p2 , y_p1+y_p2, w_p1+w_p2, h_p1+h_p2
        
        x_t, y_t, w_t, h_t = Y[..., 21], Y[..., 22], Y[..., 23], Y[..., 24]
        
             #########boundingboxError#############
             
             
        
        bb_loc_Error = self.mse(c_t*x_t,c_t*x_p) + self.mse(c_t*y_t,c_t*y_p)
        
        no_bb_loc_Error = self.mse((1-c_t)*x_t,(1-c_t)*x_p) + self.mse((1-c_t)*y_t,(1-c_t)*y_p)
        
        
        bb_size_Error =  self.mse(c_t*w_t,c_t*w_p) + self.mse(c_t*h_t,c_t*h_p)
        
        no_bb_size_Error =  self.mse((1-c_t)*w_t,(1-c_t)*w_p) + self.mse((1-c_t)*h_t,(1-c_t)*h_p)
        
        
        # print(pred_box.shape,Y_box.shape)
        # box_loss = self.mse(pred_box,Y_box)
        
        
        box_loss = (bb_loc_Error + bb_size_Error)
        no_box_loss = (no_bb_loc_Error + no_bb_size_Error)
        
        
        # print(box_loss.shape)
        # print()
         #################object Loss###############
         
         
        # print(c_t.shape, (c_t*c_p).shape )
        object_Error = self.mse(c_t, c_t*c_p)
        # print(object_Error.shape)
        no_object_Error = self.mse((1-c_t)*c_p,c_t)
        # print(no_object_Error.shape)     
        # print()
         #      #########Class Error#########
        c_t1 = c_t.unsqueeze(3) 
        class_loss = self.mse(c_t1*class_t,c_t1*class_p)
        
        # print(f'box_loss = {self.L_coord*box_loss},objectlose = {self.L_coord*object_Error},np_object_loss = {self.L_noobj*no_object_Error},class_loss = {class_loss}')

        total_loss = self.L_coord*box_loss + self.L_coord*object_Error + self.L_noobj*no_object_Error + class_loss
             
        return [total_loss,self.L_coord*box_loss,self.L_noobj*no_box_loss,self.L_coord*object_Error,self.L_noobj*no_object_Error,class_loss]
        
            
loss_fn = Yolo_Loss()

def feed_forward(X, Y, train=True, verbose = False):
    if train:
        yolo.zero_grad()

    pred = yolo(X)
    # print()
    # print(X.shape)
    # print(Y.shape)
    # print(pred.shape)

    # Loss_fn(pred, Y)
    if verbose:
        print_step(X,pred,Y)
    
    


    loss_List = loss_fn(pred, Y)
    loss = loss_List[0]
    # print(pred)
    # print(loss)
    # print(loss.shape)

    if train:
        loss.backward()
        optimizer.step()
        
    return loss_List,pred


def Train_fn():
    for epoch in range(5):
        loop = tqdm(Train_Loader, leave = False)
        for index,(X,Y) in enumerate(loop):
            loop.set_description(f"Epoch {epoch}")
            #print(index)
            X = X.view(-1, 3, 256,256)/255.0
            Y = Y.view(-1,3,3,30)
            if  index % 25 == 0:
                loss,pred = feed_forward(X, Y, train = True, verbose = True)
            else:  
                loss,pred = feed_forward(X, Y, train=True)
                
            loop.set_postfix(loss=loss)
            
            
            


                    
                
                
        
        