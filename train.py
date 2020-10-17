import torch
from torch_model import network, w_b_init
import torchvision.transforms as transforms
import pickle
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from dataset import LaneDataset
import sys
import time
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
from dice_loss import BinaryDiceLoss


#print('Epoch: %d || Time passed: %4f || Step: (%0d, %d) || Avg Val Loss: %4f '%(epoch + 1, time.time() - t1 ,len(testset) / batch_size, i + 1, running_loss1 / (i + 1)))
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def check_training(batches, freq):
    for i in batches:
        img = i.detach().cpu()
        img = img.reshape(img.shape[1], img.shape[2], img.shape[0])
        img = img.numpy()
        cv2.imshow('Out', img)
        cv2.waitKey(freq)
       

def train(epoch_num, batch_size, lr):
    
    transform = transforms.Compose([
        #transforms.CenterCrop((120, 208)), # 240, 416
        transforms.ToTensor(),
        #transforms.Normalize(mean = [0.5], std = [0.5])
    ])
   
    train_image_path = 'data/train_images/'
    train_mask_path = 'data/train_masks/'

    #test_image_path = 'data/test_images/'
    #test_mask_path = 'data/test_masks/'
  

    trainset = LaneDataset(train_image_path, train_mask_path, transform = transform)
    trainloader = DataLoader(trainset, batch_size = batch_size, shuffle = 0, num_workers = 0)
    '''
    # visualize 
    dataiter = iter(trainloader)
    images, masks = dataiter.next()
    imshow(torchvision.utils.make_grid(masks))
    '''
    #testset = LaneDataset(test_image_path, test_mask_path, transform = transform)
    #testloader = DataLoader(testset, batch_size = batch_size, shuffle = 0, num_workers = 0)
    
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = network()
    net.to(device)
    net.apply(w_b_init)

    loss_foo = BinaryDiceLoss()
    optimizer = optim.Adam(net.parameters(), lr = lr)
    
    print('Total train image number: %d || Total test image number: %d'%(len(trainset), 0))
    div_loss = 1
    cp_count = 1
    all_losses = []
    for epoch in range(epoch_num):
        print('Epoch {} started.'.format(epoch + 1))
        running_loss = 0.0
        running_loss1 = 0.0
        tt = 0
        tt1 = 0 
        tpas = 0
        step_tpas = 0 
        t0 = time.time()
        for i, data in enumerate(trainloader, 0):
            st0 = time.time()
            images, masks = data
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = net(images)   
            
            #check_training(images, 1)

            loss = loss_foo(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            step_tpas = time.time() - st0
            print('Epoch: (%d/%d) || Step: (%d/%d) || Time passed: %4f || Avg Training Loss: %4f '%(epoch_num, epoch + 1, len(trainset) // batch_size, i, step_tpas, running_loss / (i + 1)))

        '''  
        net.eval()
        with torch.no_grad():
            for ii, data in enumerate(testloader, 0):
                t1 = time.time()
                images, masks = data
                images = images.to(device)
                masks = masks.to(device)
                outputs = net(images)
                loss1 = loss_foo(outputs, masks)
                running_loss1 += loss.item()
                tt1 += time.time() - t1
        '''
        tpas = time.time() - t0
        all_losses.append(running_loss / (i + 1))
        #print('Epoch: %d || Time passed: %4f || Avg Training Loss: %4f '%(epoch + 1, tpas, running_loss / (i + 1)))
        #print('Epoch: %d || Time passed: %4f || Avg Validation Loss: %4f '%(epoch + 1, tt1, running_loss1 / (ii + 1)))
        div_loss += 1
         
        checkpoint_num = int(epoch_num * 0.1)
        if checkpoint_num < 1:
            checkpoint_num = 1
        if (epoch + 1) % checkpoint_num == 0:
            print('Saving checkpoint %d...'%cp_count)
            torch.save(net, 'saved_models/checkpoints/lane_model_checkpoint{}.pth'.format(cp_count))
            print('Checkpoint %d saved.'%cp_count)
            cp_count += 1
        print(all_losses)

    print('Training finished. Saving final model model...') 
    torch.save(net, 'saved_models/lane_model_final.pth')
    print('Model saved.')


if __name__ == "__main__":
    train_time0 = time.time()

    train(
        epoch_num = 5,   # 5
        batch_size = 16, # 16
        lr = 3e-4    # 3e-4
    )
    print('Train took {} seconds.'.format(time.time() - train_time0))
