
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

CUDA_VISIBLE_DEVICES=0,1 python3 /root/DL-project/1.py --lr 0.001 --name CNN_0.00005
