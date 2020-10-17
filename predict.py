import torch
import numpy as np
from torchvision import transforms
from torch.autograd import Variable
import cv2
from torch_model import network, w_b_init
import time
from displayers import show_weighted
import imageio
import os
import torchvision
from detect_pts import detect

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
res = (416, 240)

def create_gif(image_path, model):
    images = []
    result = cv2.VideoWriter('filename.mp4',  
                        cv2.VideoWriter_fourcc(*'MP4V'), 
                        10, (res)) 
    color_filter = cv2.imread('green.png')
    for filename in os.listdir(image_path):
        if filename.endswith('.jpg'):
            image = imageio.imread(image_path + filename)
            image = cv2.resize(image, (res))
            
            out = predict_image(image, model) * 255
            out = out.reshape(720, 1280, 1)
            out = np.array(out).astype(np.uint8)
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

            out = cv2.bitwise_and(out, color_filter)
            out = show_weighted(image, 0.7, out, 0.5)
            '''
            cv2.imshow('image', out)
            cv2.waitKey(0)
            '''
            images.append(out)
    for i in range(len(images)):
        result.write(images[i])
    result.release()

def predict_image(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (res))
    w, h = image.shape[1], image.shape[0]
    image = image.reshape(1, h, w)

    batch = torch.tensor(image / 255).unsqueeze(0).float()
    with torch.no_grad():
        batch = batch.to(device)
        output = model(batch) 
    output = output.cpu().numpy()
    return output

def predict_video(video, model):
    color_filter = cv2.imread('green.png')
    while True:
        check, frame = video.read()
        if check:
            t0 = time.time()
            frame = cv2.resize(frame, (res))
            w, h = frame.shape[1], frame.shape[0]
            out = predict_image(frame, model) * 255
            out = out.reshape(h, w, 1) 
            out = np.array(out, dtype = np.uint8)
            #_, out = cv2.threshold(out, 250, 255, cv2.THRESH_BINARY)
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
            '''
            pts = detect(out)
            for i in pts:
                u, v = i.ravel()
                out = cv2.circle(out, (u, v), 3, (0, 0, 255))
            
    
            black_out = out
            #black_out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
            black_out = cv2.bitwise_and(black_out, color_filter)
            out = show_weighted(frame, 0.7, black_out, 0.5)
            '''
            predict_time = time.time() - t0
            FPS = 1 / predict_time

            cv2.putText(out, 'FPS: {}'.format(str(FPS)), (w - int(w * 0.9), h - int(h * 0.9)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
            cv2.putText(out, 'Quality: {}p'.format(str(frame.shape[0])), (w - int(w * 0.9) + 30, h - int(h * 0.9) + 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
            print(FPS)
            cv2.imshow('Out', out)
            cv2.waitKey(1)

if __name__ == "__main__":
    model = torch.load('saved_models/lane_model_final.pth')
    #model = torch.load('saved_models/checkpoints/lane_model_checkpoint2.pth')

    model.eval()
    transform = transforms.ToTensor()
    video = cv2.VideoCapture('test_videos/7.mp4')
    predict_video(video, model)
    #create_gif('gif_images/', model)
    
