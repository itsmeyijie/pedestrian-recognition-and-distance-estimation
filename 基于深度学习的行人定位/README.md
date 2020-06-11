### 基于深度学习的测距

首先将下载的KITTI数据集有标注图片和标注放入original_data

执行

```
python generate-csv.py --input=original_data/train_annots --output=annotations.csv
```

将train_annots .txt文件转换为.csv格式

执行

```
python generate-depth-annotations.py
```

将文件按照9：1分为训练集和测试集

执行

```
python generate-depth-annotations.py
```

将数据集和测试集的数据变成.csv格式

执行

```
python testnewdis.py 
```

训练网络

执行

```
prediction-visualizer.py
```

可视化预测结果

其中网络结构为：

![1591843827883](C:\Users\blair\AppData\Roaming\Typora\typora-user-images\1591843827883.png)

数据预处理参考项目：[harshilpatel312/KITTI-distance-estimation](https://github.com/harshilpatel312/KITTI-distance-estimation)

yolov3的Pytorch实现参考项目：[packyan](https://github.com/packyan)/**PyTorch-YOLOv3-kitti**


  