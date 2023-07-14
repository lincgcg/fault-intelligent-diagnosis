
#### 目录结构

* ./
  * train/
    * 1/
      * 1.5929.jpg
      * label.ID.jpg
      * ...
    * ...
    * 9/
  * valid/
    * 1/
      * 1.5929.jpg
      * label.ID.jpg
      * ...
    * ...
    * 9/


#### 训练命令

CUDA_VISIBLE_DEVICES=2,3 /data0/mxy/linchungang/fault-intelligent-diagnosis/image/CNN.py --lr 0.01 --name CNN_0.01
