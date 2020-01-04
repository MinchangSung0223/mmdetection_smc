## Install
```bash

export MMDET_HOME=$PWD
pip3 install mmcv
pip3 install opencv-contrib-python
apt-get -y install libjpeg-dev
python3 setup.py develop 

wget https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_mask_rcnn_x101_64x4d_fpn_20e_20181218-630773a7.pth

mkdir checkpoints
mv cascade_mask_rcnn_x101_64x4d_fpn_20e_20181218-630773a7.pth checkpoints/cascade_mask_rcnn_x101_64x4d_fpn_20e_20181218-630773a7.pth

```

https://drive.google.com/drive/folders/1YbPUQVTYw_slAvk_DchvRY-7B6rnSXP9

```bash
mv /path/to/model.pth $MMDET_HOME/model.pth 
```
## RUN
python3 examples/maskrcnn_kinect_siammask.py 
