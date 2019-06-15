# -*- coding: utf-8 -*-
import torch #has pytorch which contains dynamic graph. Helps to build the gradients to update weight for back propagation.
from torch.autograd import Variable #stores a tensor and a gradient in a single variable.
import cv2 #to only draw rectancles around the image, not used for any detection.
from data import BaseTransform, VOC_CLASSES as labelmap #to transform as per the neural network and for mapping.
from ssd import build_ssd # constructor which will build the architecture of ssd
import imageio #to process the images in the video

#detect function to detect a single image. imageio will be used for a video
def detect(frame, net, transform): #4 tranformation are to be done before feeding it in the neural network.
    height, width = frame.shape[:2]
    t_frame = transform(frame)[0] #1st tranform for right dimentions and colour.
    x = torch.from_numpy(t_frame).permute(2,0,1) #2nd transform to convert numpy array to tensor state.
    x = Variable(x.unsqueeze(0)) #3rd and 4th transform to make it into batches and assign it to a variable.
    y = net(x) #apply x to the neural network and store it in y.\
    detections = y.data #Tensor detections contain data attribute of y.
    scale = torch.Tensor([width, height, width, height]) #to normalize the position of image between 0 and 1
    # detections = [batch, no of classes, no of occurences, [score, x0, y0, x1, y1]]
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= 0.6: #if no of occurences crosses the thrushold
            pt = (detections[0,i,j,1:] * scale).numpy() #assigns the coordinates
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255,0,0), 2)
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
            j += 1
    return frame
         
#Creating SSD neural network
net = build_ssd('test') #we use test phase cause we have a trained model
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc:storage)) #the weights are attributed to net
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0)) #tranform the image according to the trained neural network.

#doing object detection on a video
reader = imageio.get_reader('input.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('output.mp4', fps = fps)
for i,frame in enumerate(reader):
     frame = detect(frame, net.eval(), transform)
     writer.append_data(frame)
     print(i)
writer.close()
    