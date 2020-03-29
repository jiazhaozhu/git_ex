#!/usr/bin/env python

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
import time
from openvino.inference_engine import IENetwork, IECore, IEPlugin


class Face_detect():
    def __init__(self,model_xml,cpu_extension,device):
        self.model_xml = model_xml
        self.cpu_extension = cpu_extension
        self.device = device

        self.load_model()

    def load_model(self):

        model_bin = os.path.splitext(self.model_xml)[0] + ".bin"

        log.info("Creating Inference Engine")
        ie = IECore()

        if self.cpu_extension and 'CPU' in self.device:
            ie.add_extension(self.cpu_extension, "CPU")
        # Read IR
        log.info("Loading network files:\n\t{}\n\t{}".format(self.model_xml, model_bin))
        net = IENetwork(model=self.model_xml, weights=model_bin)


        if "CPU" in self.device:
            supported_layers = ie.query_network(net, "CPU")
            not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                          format(self.device, ', '.join(not_supported_layers)))
                log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                          "or --cpu_extension command line argument")
                sys.exit(1)
        assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
        assert len(net.outputs) == 1, "Sample supports only single output topologies"

        log.info("Preparing input blobs")
        self.input_blob = next(iter(net.inputs))
        net.batch_size = 1


        net.reshape({self.input_blob: (1, 3, 120, 150)})
        # Read and pre-process input images
        self.n, self.c, self.h, self.w = net.inputs[self.input_blob].shape
        print(self.n, self.c, self.h, self.w)
        self.images = np.ndarray(shape=(self.n, self.c, self.h, self.w))

        # Loading model to the plugin
        log.info("Loading model to the plugin")
        self.exec_net = ie.load_network(network=net, device_name=self.device)


    def detect(self,img):
        label_ = 1
        # img = cv2.imread(img_path)
        x1_ = int(0.2 * img.shape[1])
        y1_ = int(0.05 * img.shape[0])
        x2_ = int(0.8 * img.shape[1])
        y2_ = int(0.95 * img.shape[0])

        # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # videoWriter = cv2.VideoWriter('wanwan.mp4',fourcc,25,(img.shape[1],img.shape[0]))
        # cv2.rectangle(img, (x1_, y1_), (x2_, y2_), (0, 0, 0), 2)

        # img = cv2.imread(r"C:\Users\Administrator\source\repos\1.png")
        start = time.time()
        frame = cv2.resize(img, dsize=(self.w, self.h))

        self.images[0] = frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW

        res = self.exec_net.infer(inputs={self.input_blob: self.images})
        res0 = res["527"]
        #id_max = np.argmax(res0[0, 0, :, 2])
        id_max = np.where(res0[0, 0, :, 2] > 0.5)[0]
        if len(id_max) == 0:
          return []
        narry0 = res0[0, 0, id_max, :]
        id_max0 = np.argmax(narry0[:, 6] - narry0[:, 4])
        narry1 = narry0[id_max0, :]
        x1 = int(narry1[3]*img.shape[1])
        y1 = int(narry1[4]*img.shape[0])
        x2 = int(narry1[5]*img.shape[1])
        y2 = int(narry1[6]*img.shape[0])
        if x1 < 0:
            x1 = 0
        if y1< 0:
            y1 = 0
        if x2 >= img.shape[1]:
            x2 = img.shape[1]-1
        if y2 >= img.shape[0]:
            y2 = img.shape[0]-1

        # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        # print((y2-y1)/(x2-x1))
        if ((center_x < x1_) or (center_y < y1_) or (center_x > x2_) or (center_y > y2_)):
            label_ = 0
        elif (y2-y1)/(x2-x1)<0.9:
            label_ = 0
        elif (y2-y1)/(x2-x1)>1.75:
            label_ = 0
        else:
            label_ = 1
        return [x1,y1,x2,y2,label_]

    def detect0(self,img):
        label_ = 1
        # img = cv2.imread(img_path)
        pro = 0
        x1_ = int(0.15 * img.shape[1])
        y1_ = int(0.1 * img.shape[0])
        x2_ = int(0.85 * img.shape[1])
        y2_ = int(0.9 * img.shape[0])
        x1_sco = int(0.4 * img.shape[1])
        y1_sco = int(0.4 * img.shape[0])
        x2_sco = int(0.6 * img.shape[1])
        y2_sco = int(0.6 * img.shape[0])

        # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # videoWriter = cv2.VideoWriter('wanwan.mp4',fourcc,25,(img.shape[1],img.shape[0]))
        # cv2.rectangle(img, (x1_, y1_), (x2_, y2_), (0, 0, 0), 2)

        # img = cv2.imread(r"C:\Users\Administrator\source\repos\1.png")
        start = time.time()
        frame = cv2.resize(img, dsize=(self.w, self.h))

        self.images[0] = frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW

        res = self.exec_net.infer(inputs={self.input_blob: self.images})
        res0 = res["527"]
        #id_max = np.argmax(res0[0, 0, :, 2])
        id_max = np.where(res0[0, 0, :, 2] > 0.5)[0]
        if len(id_max) == 0:
          return []
        narry0 = res0[0, 0, id_max, :]
        id_max0 = np.argmax(narry0[:, 6] - narry0[:, 4])
        narry1 = narry0[id_max0, :]
        
        x1 = int(narry1[3]*img.shape[1])
        y1 = int(narry1[4]*img.shape[0])
        x2 = int(narry1[5]*img.shape[1])
        y2 = int(narry1[6]*img.shape[0])
        if x1 < 0:
            x1 = 0
        if y1< 0:
            y1 = 0
        if x2 >= img.shape[1]:
            x2 = img.shape[1]-1
        if y2 >= img.shape[0]:
            y2 = img.shape[0]-1

        # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        # print((y2-y1)/(x2-x1))
        if ((center_x < x1_) or (center_y < y1_) or (center_x > x2_) or (center_y > y2_)):
            label_ = 0
        elif (y2-y1)/(x2-x1)<0.9:
            label_ = 0
        elif (y2-y1)/(x2-x1)>1.75:
            label_ = 0
        else:
            label_ = 1
            if ((center_x > x1_sco) and (center_y > y1_sco) and (center_x < x2_sco) and (center_y < y2_sco)):
                pro = 100            
            else:
                min_x = min([(center_x-x1_)/img.shape[1],(x2_-center_x)/img.shape[1]])
                min_y = min([(center_y - y1_) / img.shape[0], (y2_ - center_y) / img.shape[0]])
                pro_x = int(400 * min_x)
                pro_y = int(2000 * min_y / 3 - 100)
                # print(center_x,center_y,x1_,y1_,x2_,y2_,min_)
                pro = min([pro_x,pro_y])
                if pro<0:
                    pro = 0
                elif pro>100:
                    pro = 100

        return [x1,y1,x2,y2,label_,pro]




if __name__ == '__main__':
    # sys.exit(main() or 0)

    face_ = Face_detect("face-detection-retail-0005.xml","libcpu_extension_avx2.so","CPU")
    cap = cv2.VideoCapture("3.mp4")
    # 读视频帧
    while True:

        ret, frame = cap.read()
        
        if ret==True:
          st_time = time.time()
          list_ = face_.detect0(frame)
          print(list_,time.time()-st_time)
        else:
          break
