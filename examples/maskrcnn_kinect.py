import sys
import os
import argparse
from PIL import Image
from matplotlib import pyplot as plt
import pycocotools.mask as maskUtils
import time
import numpy as np
import cv2
import threading
from scipy.interpolate import splprep, splev
from ctypes import cdll
import ctypes

from numpy.ctypeslib import ndpointer
import shutil
import mmcv
from mmdet.apis import init_detector, inference_detector
from mmcv.visualization.color import color_val


i=0
depth_mem=None
first_flag=True

config_file = 'configs/cascade_mask_rcnn_x101_64x4d_fpn_1x.py'
checkpoint_file = 'checkpoints/cascade_mask_rcnn_x101_64x4d_fpn_20e_20181218-630773a7.pth'
model = init_detector(config_file, checkpoint_file)
class_names = model.CLASSES
score_thr=0.3


lib = cdll.LoadLibrary('./viewer_opengl.so')
st = lib.Foo_start
t0 = threading.Thread(target=st)
t0.start()
end = lib.Foo_end
dataread =lib.Foo_dataread
dataread_color =lib.Foo_dataread_color
dataread_depth =lib.Foo_dataread_depth
dataread_color_to_depth =lib.Foo_dataread_color_to_depth
dataread.restype = ndpointer(dtype=ctypes.c_uint8, shape=(720,1280,2))
dataread_color.restype = ndpointer(dtype=ctypes.c_uint8, shape=(720,1280,4))
dataread_depth.restype = ndpointer(dtype=ctypes.c_uint16, shape=(512,512))#ctypes.POINTE
dataread_color_to_depth.restype = ndpointer(dtype=ctypes.c_uint8, shape=(512,512,4))

classname = "test"
classname1 = classname
smooth_rate = 200
classnumber = 4
scale_factor = 1.1
home_path=os.getcwd() 

classname = classname+"_"
rgb_segmentation =1
darker = 0;
sensitivity = 245;
x= 0
y = 0
w = 0
h = 0
def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)
color_img = np.zeros((1280,720),dtype = np.uint8)
result_mask_img =np.zeros((1280,720),dtype = np.uint8)
result_bbox_img =np.zeros((1280,720),dtype = np.uint8)
result_mask =np.zeros((1280,720),dtype = np.uint8)
mask_rcnn_flag = 0
def run_maskrcnn():
    global color_img
    global result_mask_img
    global result_bbox_img
    global result_mask
    global mask_rcnn_flag
    while 1:
        start = time.time()
        mask_rcnn_flag=1
        result = inference_detector(model, color_img)
        result_mask_img,result_bbox_img,result_mask = show_result(color_img, result, model.CLASSES)
        print("time : ",time.time()-start)
        #print(result)

def show_result(img,result,class_names):
       global mask_rcnn_flag
       img_mask = img.copy()
       mask_temp = img.copy()
       bbox_result, segm_result = result
       bboxes = np.vstack(bbox_result)
       labels = [
           np.full(bbox.shape[0], i, dtype=np.int32)
           for i, bbox in enumerate(bbox_result)
       ]

       labels = np.concatenate(labels)
       bbox_color = 'green'
       text_color = 'green'
       thickness=1
       font_scale = 3
       show=True
       win_name=''
       wait_time = 0
       out_file = None
       assert bboxes.ndim == 2
       assert labels.ndim == 1
       assert bboxes.shape[0] == labels.shape[0]
       assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5

       bbox_color = color_val(bbox_color)
       text_color = color_val(text_color)

       prev_point = [0,0]

       for i in range(0,len(bboxes)):
          label_text = class_names[labels[i]] if class_names is not None else 'cls {}'.format(labels[i])
          if label_text =="banana":
            if len(bboxes[i]) > 4:
                    label_text += '|{:.02f}'.format(bboxes[i][-1])
            left_top = (int(bboxes[i][0]),int(bboxes[i][1]) )
            right_bottom = (int(bboxes[i][2]), int(bboxes[i][3]))
            if bboxes[i][-1]<0.5 :
                pass
            else :
                img = cv2.putText(img , str(label_text) , (int(bboxes[i][0]),int(bboxes[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0) ,1)
                img = cv2.rectangle(img , (int(bboxes[i][0]),int(bboxes[i][1])),(int(bboxes[i][2]),int(bboxes[i][3])),(255,0,0) ,1)
       mask_temp=np.zeros((720,1280),dtype = np.uint8)
       if segm_result is not None:
           
           segms = mmcv.concat_list(segm_result)
           inds = np.where(bboxes[:, -1] > score_thr)[0]
           mask_temp = mask_temp*0
           for i in inds:

               label_text = class_names[labels[i]] if class_names is not None else 'cls {}'.format(labels[i])
               if label_text =="banana":
                 color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)

                 mask = maskUtils.decode(segms[i]).astype(np.bool)

                 mask_temp[mask] = 255
                 #print(color_mask.shape)
                 #print(color_mask)
                 img_mask[mask] = img_mask[mask] * 0.5 + color_mask * 0.5 
                 img_mask = cv2.putText(img_mask, str(label_text) , (int(bboxes[i][0]),int(bboxes[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1,(int(color_mask[0,0]),int(color_mask[0,1]),int(color_mask[0,2])) ,2)
       mask_rcnn_flag=0
       return (img_mask,img,mask_temp)



def detect_img():
    global color_img
    global result_mask_img
    global result_bbox_img
    global result_mask
    cv2.namedWindow("rgb", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("rgb", 1280,720)
    cv2.namedWindow("depth", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("depth", 1280,720)
    while 1:
      color_data = np.array(dataread_color(),dtype=np.uint8)
      color_img = color_data[:,:,0:3]
      depth_to_color_data = np.array(dataread(),dtype=np.uint8)
      depth_to_color_img = depth_to_color_data[:,:,0]
      depth_img = depth_to_color_img.copy()

      cv2.imshow("rgb",color_img)
      cv2.imshow("depth",result_mask_img)
      k = cv2.waitKey(5) & 0xFF
      if k == ord('s'):
         cv2.destroyWindow("rgb")
    end()
if __name__ == '__main__':
    t1 = threading.Thread(target=detect_img)
    t1.start()
    t2 = threading.Thread(target=run_maskrcnn)
    t2.start()
