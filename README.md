# yolov5训练自己的数据集

## 1.下载`yolo v5`源码

在GitHub中搜索`yolov5`，找到[ultralytics](https://github.com/ultralytics)/**[yolov5](https://github.com/ultralytics/yolov5)**项目，下载`.zip`文件，这里我下载的是`5.0`版本，或者使用git bash 输入https://github.com/ultralytics/yolov5.git下载代码。若下载的是`yolov5-5.0.zip`文件，下载完成后解压至代码编辑的地方。

## 2.使用Anaconda创建虚拟环境

若无anaconda环境，也可直接使用python环境

在Anaconda Prompt中输入`conda create --name yolov5 python=3.8`

输入y回车，然后输入命令`conda activate yolov5`进入虚拟环境。

yoloV5[要求](https://github.com/ultralytics/yolov5/blob/master/requirements.txt)[**在Python>= 3.7.0**](https://www.python.org/)环境中，包括[**PyTorch> = 1.7。**](https://pytorch.org/get-started/locally/)

然后我们进入解压后的YOLO V5项目文件夹，使用`pip install -r requirements.txt`命令下载项目所需依赖包（无anaconda可直接使用本命令安装依赖库，默认你安装好了python）

安装完成后,我们进入[PyTorch](https://pytorch.org/)官网,这里我选择以下配置:

`PyTorch Build`选择`Stable (1.10.2)`

`Your OS`选择`Windows` 系统

`Package`选择`Pip` 注意这里最好选用pip,conda会一直出现报错

`Language`选择`Python`

`Compute Platform`选择`CUDA 10.2`有显卡建议选这个,没有显卡选择`CPU`

`Run this Command:`显示

```python
pip3 install torch==1.10.2+cu102 torchvision==0.11.3+cu102 torchaudio===0.10.2+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
```

将上面命令复制到控制台,安装pytorch,显示Successful即可

## 3.建立VOC格式标准文件夹

在`yolov5-5.0\`下创建`make_voc_dir.py`

```python
import os
os.makedirs('VOCdevkit/VOC2007/Annotations')
os.makedirs('VOCdevkit/VOC2007/JPEGImages')
```

运行`make_voc_dir.py`

在`\yolov5-5.0\VOCdevkit\VOC2007\Annotations`中存放`xml`格式文件

在`\yolov5-5.0\VOCdevkit\VOC2007\JPEGImages`中存放`JPG`格式文件

## 4.将xml格式转换成yolo格式

在`yolov5-5.0\`下创建`voc_to_yolo.py`

```python
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import random
from shutil import copyfile

classes = ["crack","helmet"]  # 这个列表里存放的是你的类别

TRAIN_RATIO = 90  # 训练的比例


# 遍历文件夹
def clear_hidden_files(path):
    dir_list = os.listdir(path)
    for i in dir_list:
        abspath = os.path.join(os.path.abspath(path), i)
        if os.path.isfile(abspath):
            if i.startswith("._"):
                os.remove(abspath)
        else:
            clear_hidden_files(abspath)


# 对宽高进行归一化操作 size:原图的宽和高
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)



# 解析XML
def convert_annotation(image_id):
    in_file = open('VOCdevkit/VOC2007/Annotations/%s.xml' % image_id,'rb')
    out_file = open('VOCdevkit/VOC2007/YOLOLabels/%s.txt' % image_id, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    in_file.close()
    out_file.close()


wd = os.getcwd()
wd = os.getcwd()
data_base_dir = os.path.join(wd, "VOCdevkit/")
if not os.path.isdir(data_base_dir):
    os.mkdir(data_base_dir)
work_sapce_dir = os.path.join(data_base_dir, "VOC2007/")
if not os.path.isdir(work_sapce_dir):
    os.mkdir(work_sapce_dir)
annotation_dir = os.path.join(work_sapce_dir, "Annotations/")
if not os.path.isdir(annotation_dir):
    os.mkdir(annotation_dir)
clear_hidden_files(annotation_dir)
image_dir = os.path.join(work_sapce_dir, "JPEGImages/")
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)
clear_hidden_files(image_dir)
yolo_labels_dir = os.path.join(work_sapce_dir, "YOLOLabels/")
if not os.path.isdir(yolo_labels_dir):
    os.mkdir(yolo_labels_dir)
clear_hidden_files(yolo_labels_dir)
yolov5_images_dir = os.path.join(data_base_dir, "images/")
if not os.path.isdir(yolov5_images_dir):
    os.mkdir(yolov5_images_dir)
clear_hidden_files(yolov5_images_dir)
yolov5_labels_dir = os.path.join(data_base_dir, "labels/")
if not os.path.isdir(yolov5_labels_dir):
    os.mkdir(yolov5_labels_dir)
clear_hidden_files(yolov5_labels_dir)
yolov5_images_train_dir = os.path.join(yolov5_images_dir, "train/")
if not os.path.isdir(yolov5_images_train_dir):
    os.mkdir(yolov5_images_train_dir)
clear_hidden_files(yolov5_images_train_dir)
yolov5_images_test_dir = os.path.join(yolov5_images_dir, "val/")
if not os.path.isdir(yolov5_images_test_dir):
    os.mkdir(yolov5_images_test_dir)
clear_hidden_files(yolov5_images_test_dir)
yolov5_labels_train_dir = os.path.join(yolov5_labels_dir, "train/")
if not os.path.isdir(yolov5_labels_train_dir):
    os.mkdir(yolov5_labels_train_dir)
clear_hidden_files(yolov5_labels_train_dir)
yolov5_labels_test_dir = os.path.join(yolov5_labels_dir, "val/")
if not os.path.isdir(yolov5_labels_test_dir):
    os.mkdir(yolov5_labels_test_dir)
clear_hidden_files(yolov5_labels_test_dir)

train_file = open(os.path.join(wd, "yolov5_train.txt"), 'w')
test_file = open(os.path.join(wd, "yolov5_val.txt"), 'w')
train_file.close()
test_file.close()
train_file = open(os.path.join(wd, "yolov5_train.txt"), 'a')
test_file = open(os.path.join(wd, "yolov5_val.txt"), 'a')
list_imgs = os.listdir(image_dir)  # list image_one files
prob = random.randint(1, 100)
print("Probability: %d" % prob)
for i in range(0, len(list_imgs)):
    path = os.path.join(image_dir, list_imgs[i])
    if os.path.isfile(path):
        image_path = image_dir + list_imgs[i]
        voc_path = list_imgs[i]
        (nameWithoutExtention, extention) = os.path.splitext(os.path.basename(image_path))
        (voc_nameWithoutExtention, voc_extention) = os.path.splitext(os.path.basename(voc_path))
        annotation_name = nameWithoutExtention + '.xml'
        annotation_path = os.path.join(annotation_dir, annotation_name)
        label_name = nameWithoutExtention + '.txt'
        label_path = os.path.join(yolo_labels_dir, label_name)
    prob = random.randint(1, 100)
    print("Probability: %d" % prob)
    if (prob < TRAIN_RATIO):  # train dataset
        if os.path.exists(annotation_path):
            train_file.write(image_path + '\n')
            convert_annotation(nameWithoutExtention)  # convert label
            copyfile(image_path, yolov5_images_train_dir + voc_path)
            copyfile(label_path, yolov5_labels_train_dir + label_name)
    else:  # test dataset
        if os.path.exists(annotation_path):
            test_file.write(image_path + '\n')
            convert_annotation(nameWithoutExtention)  # convert label
            copyfile(image_path, yolov5_images_test_dir + voc_path)
            copyfile(label_path, yolov5_labels_test_dir + label_name)
train_file.close()
test_file.close()
```

执行一下`voc_to_yolo.py`

```shell
(yoloV5) E:\PythonCode\yoloV5_toukui\yolov5-5.0>python voc_to_yolo.py
Probability: 6
Probability: 79
Probability: 26
Probability: 19
Probability: 64
Probability: 5
Probability: 80
Probability: 40
Probability: 46
Probability: 23
Probability: 87
Probability: 19
Probability: 71
Probability: 62
Probability: 53
Probability: 74
Probability: 10
Probability: 19
Probability: 90
Probability: 35
Probability: 100
Probability: 27
Probability: 77
Probability: 65
Probability: 34
Probability: 95
Probability: 43
```

可以看到`\yolov5-5.0\VOCdevkit`文件下内生成了`images`和`labels`文件夹,文件夹内有`train`(训练样本)和`val`(验证样本)文件夹,文件夹内yolov5已经对标签进行了归一化处理,里面的参数分别为|类别数|中心点坐标归一化后的结果|宽和高的一个结果|；在`\yolov5-5.0`文件夹内生成了`yolov5_train.txt`和`yolov5_val.txt`两个文件

以及`\yolov5-5.0\VOCdevkit\VOC2007\YOLOLabels\`内的文件对项目没有影响,可以直接删除

## 5.修改yaml配置文件

进入`\yolov5-5.0\data\`文件夹内，打开`voc.yaml`文件,



原`voc.yaml`文件

```yaml
# PASCAL VOC dataset http://host.robots.ox.ac.uk/pascal/VOC/
# Train command: python train.py --data voc.yaml
# Default dataset location is next to /yolov5:
#   /parent_folder
#     /VOC
#     /yolov5


# download command/URL (optional)
download: bash data/scripts/get_voc.sh

# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
train: ../VOC/images/train/  # 16551 images
val: ../VOC/images/val/  # 4952 images

# number of classes
nc: 20

# class names
names: [ 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
         'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor' ]

```

修改后的`voc.yaml`文件

```yaml
# PASCAL VOC dataset http://host.robots.ox.ac.uk/pascal/VOC/
# Train command: python train.py --data voc.yaml
# Default dataset location is next to /yolov5:
#   /parent_folder
#     /VOC
#     /yolov5


# download command/URL (optional)
download: bash data/scripts/get_voc.sh

# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
train: VOCdevkit/images/train/ # train: ../VOC/images/train/  # 16551 images
val: VOCdevkit/images/val/ # val: ../VOC/images/val/  # 4952 images

# number of classes
nc: 2 # nc: 20

# class names
names: [ 'crack', 'helmet' ]
# names: [ 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
#          'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor' ]

```

将`train` `val` 修改为目的路径

`nc`为类别数量

`names`为类别名

## 6.开始训练

### 6.1权重文件下载

在`\yolov5-5.0\weights\`里放置`.pt`文件,可以执行`download_weights.sh`下载官方`.pt`文件,也可直接去GitHub上[下载](https://github.com/ultralytics/yolov5/releases)

这里我使用`yolov5m.pt`标准版模型

### 6.2 参数修改

再点开`train.py`,找到`if __name__ == '__main__':`开始修改参数

---

修改权重文件地址为`default='weights/yolov5m.pt'`

`parser.add_argument('--weights', type=str, default='weights/yolov5m.pt'`

---

config可改可不改,这里修改为`default='models/yolov5m.yaml'`

`parser.add_argument('--cfg', type=str, default='models/yolov5m.yaml', `

这里的`yolov5m.yaml`里的`anchors:`需要通过`kmeans`进行聚类

可参考这篇博客进行配置[YOLOv5训练自己的数据集](https://blog.csdn.net/qq_36756866/article/details/109111065)

---

修改data文件地址为`default='data/voc.yaml', `

`parser.add_argument('--data', type=str, default='data/voc.yaml', `

---

`hyp`为随机数相关参数,不用修改

---

`epochs`为训练的轮次,这里是300次`default=300`

---

`batch-size`每次给的批次,这里由于我的内存限制我每次只给1次`default=1`
`parser.add_argument('--batch-size', type=int, default=1,`

---

`img-size`输入图像大小,这里是640x640`default=[640, 640]`

---

其余参数不用修改,***接下来我们就可以开始训练了!!!***

---

这里出现报错`AttributeError: Can't get attribute 'SPPF' on <module 'models.common' from 'E:\\PythonCode\\yoloV5_toukui\\yolov5-5.0\\models\\common.py'>`

解决方法:

去Tags6里面的[model/common.py](https://github.com/ultralytics/yolov5/blob/v6.0/models/common.py)里面去找到这个SPPF的类,把它拷过来到你这个Tags5的model/common.py里面,这样你的代码就也有这个类了,还要引入一个warnings包就行了！

增加SPPF这个类内容如下，将下面的代码复制到你们的`common.py`里面即可，记得把`import warnings`放在上面去：

```python
import warnings

class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))
```

完美解决

---

运行`train.py`

```shell
(yoloV5_toukui) E:\PythonCode\yoloV5_toukui\yolov5-5.0>python train.py
github: skipping check (not a git repository)
YOLOv5  2021-4-11 torch 1.10.1+cpu CPU

Namespace(adam=False, artifact_alias='latest', batch_size=16, bbox_interval=-1, bucket='', cache_images=False, cfg='models/yolov5m.yaml', data='data/voc.yaml', device='', entity=None, epochs=300, evolve=False, exist_ok=False, global_rank=-1, hyp='data/hyp.scratch.yaml', image_weights=False, img_size=[640, 640], label_smoothing=0.0, linear_lr=False, local_rank=-1, multi_scale=False, name='exp', noautoanchor=False, nosave=False, notest=False, project='runs/train', quad=False, rect=False, resume=False, save_dir='runs\\train\\exp', save_period=-1, single_cls=False, sync_bn=False, total_batch_size=16, upload_dataset=False, weights='weights/yolov5m.pt', workers=8, world_size=1)
tensorboard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/
hyperparameters: lr0=0.01, lrf=0.2, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0
wandb: Install Weights & Biases for YOLOv5 logging with 'pip install wandb' (recommended)
Overriding model.yaml nc=80 with nc=2

                 from  n    params  module                                  arguments
  0                -1  1      5280  models.common.Focus                     [3, 48, 3]
  1                -1  1     41664  models.common.Conv                      [48, 96, 3, 2]
  2                -1  1     65280  models.common.C3                        [96, 96, 2]
  3                -1  1    166272  models.common.Conv                      [96, 192, 3, 2]
  4                -1  1    629760  models.common.C3                        [192, 192, 6]
  5                -1  1    664320  models.common.Conv                      [192, 384, 3, 2]
  6                -1  1   2512896  models.common.C3                        [384, 384, 6]
  7                -1  1   2655744  models.common.Conv                      [384, 768, 3, 2]
  8                -1  1   1476864  models.common.SPP                       [768, 768, [5, 9, 13]]
  9                -1  1   4134912  models.common.C3                        [768, 768, 2, False]
 10                -1  1    295680  models.common.Conv                      [768, 384, 1, 1]
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']
 12           [-1, 6]  1         0  models.common.Concat                    [1]
 13                -1  1   1182720  models.common.C3                        [768, 384, 2, False]
 14                -1  1     74112  models.common.Conv                      [384, 192, 1, 1]
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']
 16           [-1, 4]  1         0  models.common.Concat                    [1]
 17                -1  1    296448  models.common.C3                        [384, 192, 2, False]
 18                -1  1    332160  models.common.Conv                      [192, 192, 3, 2]
 19          [-1, 14]  1         0  models.common.Concat                    [1]
 20                -1  1   1035264  models.common.C3                        [384, 384, 2, False]
 21                -1  1   1327872  models.common.Conv                      [384, 384, 3, 2]
 22          [-1, 10]  1         0  models.common.Concat                    [1]
 23                -1  1   4134912  models.common.C3                        [768, 768, 2, False]
 24      [17, 20, 23]  1     28287  models.yolo.Detect                      [2, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [192, 384, 768]]
D:\software\Anaconda3\envs\yoloV5_toukui\lib\site-packages\torch\functional.py:445: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  ..\aten\src\ATen\native\TensorShape.cpp:2157.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
Model Summary: 391 layers, 21060447 parameters, 21060447 gradients, 50.4 GFLOPS

Transferred 402/506 items from weights/yolov5m.pt
Scaled weight_decay = 0.0005
Optimizer groups: 86 .bias, 86 conv.weight, 83 other
train: Scanning 'VOCdevkit\labels\train' images and labels... 23 found, 0 missing, 0 empty, 0 corrupted: 100%|█| 23/23
train: New cache created: VOCdevkit\labels\train.cache
val: Scanning 'VOCdevkit\labels\val' images and labels... 3 found, 0 missing, 0 empty, 0 corrupted: 100%|█| 3/3 [00:00<
val: New cache created: VOCdevkit\labels\val.cache
Plotting labels...
```

最终结果

```shell
   Epoch   gpu_mem       box       obj       cls     total    labels  img_size   299/299     1.21G   0.04496   0.02003   0.02956   0.09454         1       640:   4%|███                                                                   | 1/23 [00:00<00:03,  5.90it/   299/299     1.21G   0.06122   0.02847    0.0249    0.1146         6       640:   4%|███                                                                   | 1/23 [00:00<00:03,  5.90it/   299/299     1.21G   0.06122   0.02847    0.0249    0.1146         6       640:   9%|██████                                                                | 2/23 [00:00<00:03,  5.88   299/299     1.21G   0.06042   0.02963   0.02202    0.1121         2       640:   9%|██████                                                                | 2/23 [00:00<00:03,  5.88   299/299     1.21G   0.06042   0.02963   0.02202    0.1121         2       640:  13%|█████████▏                                                            | 3/23 [00:00<00:03,     299/299     1.21G   0.05973   0.02952   0.02186    0.1111         3       640:  13%|█████████▏                                                            | 3/23 [00:00<00:03,     299/299     1.21G   0.05973   0.02952   0.02186    0.1111         3       640:  17%|████████████▏                                                         | 4/23 [00:00<00:03   299/299     1.21G   0.05656   0.02924   0.02161    0.1074         2       640:  17%|████████████▏                                                         | 4/23 [00:00<00:03   299/299     1.21G   0.05656   0.02924   0.02161    0.1074         2       640:  22%|███████████████▏                                                      | 5/23 [00:00<00   299/299     1.21G   0.05358   0.02645    0.0194   0.09942         1       640:  22%|███████████████▏                                                      | 5/23 [00:01<00   299/299     1.21G   0.05358   0.02645    0.0194   0.09942         1       640:  26%|██████████████████▎                                                   | 6/23 [00:01   299/299     1.21G   0.05662   0.02863   0.01956    0.1048         5       640:  26%|██████████████████▎                                                   | 6/23 [00:01   299/299     1.21G   0.05662   0.02863   0.01956    0.1048         5       640:  30%|█████████████████████▎                                                | 7/23 [00   299/299     1.21G   0.05714   0.02914   0.02038    0.1067         2       640:  30%|█████████████████████▎                                                | 7/23 [00   299/299     1.21G   0.05714   0.02914   0.02038    0.1067         2       640:  35%|████████████████████████▎                                             | 8/23    299/299     1.21G   0.05828   0.03214    0.0208    0.1112         5       640:  35%|████████████████████████▎                                             | 8/23    299/299     1.21G   0.05828   0.03214    0.0208    0.1112         5       640:  39%|███████████████████████████▍                                          | 9/   299/299     1.21G   0.06111   0.03179   0.02086    0.1138         3       640:  39%|███████████████████████████▍                                          | 9/   299/299     1.21G   0.06111   0.03179   0.02086    0.1138         3       640:  43%|██████████████████████████████                                       | 1   299/299     1.21G   0.05965   0.03091   0.01985    0.1104         1       640:  43%|██████████████████████████████                                       | 1   299/299     1.21G   0.05965   0.03091   0.01985    0.1104         1       640:  48%|█████████████████████████████████                                       299/299     1.21G   0.06099   0.03505   0.01993     0.116        10       640:  48%|█████████████████████████████████                                       299/299     1.21G   0.06099   0.03505   0.01993     0.116        10       640:  52%|████████████████████████████████████                                 299/299     1.21G   0.06144   0.03449   0.01998    0.1159         3       640:  52%|████████████████████████████████████                                 299/299     1.21G   0.06144   0.03449   0.01998    0.1159         3       640:  57%|███████████████████████████████████████                           299/299     1.21G   0.06225   0.03479   0.02013    0.1172         3       640:  57%|███████████████████████████████████████                           299/299     1.21G   0.06225   0.03479   0.02013    0.1172         3       640:  61%|██████████████████████████████████████████                     299/299     1.21G   0.06229   0.03422   0.02033    0.1168         3       640:  61%|██████████████████████████████████████████                     299/299     1.21G   0.06229   0.03422   0.02033    0.1168         3       640:  65%|█████████████████████████████████████████████               299/299     1.21G   0.06267   0.03524   0.02006     0.118        10       640:  65%|█████████████████████████████████████████████               299/299     1.21G   0.06267   0.03524   0.02006     0.118        10       640:  70%|████████████████████████████████████████████████         299/299     1.21G   0.06301   0.03419   0.01984     0.117         1       640:  70%|████████████████████████████████████████████████         299/299     1.21G   0.06301   0.03419   0.01984     0.117         1       640:  74%|███████████████████████████████████████████████████   299/299     1.21G    0.0642   0.03599   0.02012    0.1203        11       640:  74%|███████████████████████████████████████████████████   299/299     1.21G    0.0642   0.03599   0.02012    0.1203        11       640:  78%|███████████████████████████████████████████████████   299/299     1.21G   0.06377   0.03509   0.02048    0.1193         1       640:  78%|███████████████████████████████████████████████████   299/299     1.21G   0.06377   0.03509   0.02048    0.1193         1       640:  83%|███████████████████████████████████████████████████   299/299     1.21G   0.06399   0.03754   0.02059    0.1221         8       640:  83%|███████████████████████████████████████████████████   299/299     1.21G   0.06399   0.03754   0.02059    0.1221         8       640:  87%|███████████████████████████████████████████████████   299/299     1.21G   0.06392   0.03851   0.02078    0.1232         6       640:  87%|███████████████████████████████████████████████████   299/299     1.21G   0.06392   0.03851   0.02078    0.1232         6       640:  91%|███████████████████████████████████████████████████   299/299     1.21G   0.06446   0.03858   0.02099     0.124         4       640:  91%|███████████████████████████████████████████████████   299/299     1.21G   0.06446   0.03858   0.02099     0.124         4       640:  96%|███████████████████████████████████████████████████   299/299     1.21G   0.06421   0.03818   0.02112    0.1235         2       640:  96%|███████████████████████████████████████████████████   299/299     1.21G   0.06421   0.03818   0.02112    0.1235         2       640: 100%|███████████████████████████████████████████████████   299/299     1.21G   0.06421   0.03818   0.02112    0.1235         2       640: 100%|█████████████████████████████████████████████████████████████████████| 23/23 [00:04<00:00,  5.65it/s]               Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95:  50%|█████████████████████████████▌                             |               Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100%|█████████████████████████████████████████████               Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100%|███████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  5.45it/s]                 all           3          12       0.182        0.55       0.177      0.0306               crack           3          10       0.166         0.1      0.0881      0.0248              helmet           3           2       0.199           1       0.265      0.0364300 epochs completed in 0.638 hours.Optimizer stripped from runs\train\exp4\weights\last.pt, 42.5MBOptimizer stripped from runs\train\exp4\weights\best.pt, 42.5MB
```

各参数意义: 

box    回归框的损失   

obj       置信度的损失

cls     类别的损失

total    总的损失

mAP@.5  mAP@.5:.95 置信度为0.5~0.95的一个置信度的map值



## 7.使用训练好的权重文件进行识别

打开`detect.py`,找到`if __name__ == '__main__':`

载入预训练权重`default='weights/yolov5s.pt'`修改为`default='runs/train/exp/weights/last.pt'`

将要测试的图片放入`data/images`

运行`python detect.py`,运行完成后,会在`runs`下生成`detect`文件夹,里面也有一个`exp`文件夹里面存放着预测后的结果.

若要进行视频流的检测,只需修改`source`里文件的路径为视频所在路径即可.

```python
parser.add_argument('--source', type=str, default='data/images', help='source') 
```

修改为

```python
parser.add_argument('--source', type=str, default='/Desktop/a.mp4', help='source') 
```

## 8 使用USB摄像头进行识别

打开`detect.py`,找到`if __name__ == '__main__':`

将路径设置为0

`parser.add_argument('--source', type=str, default='0`

再进入`yolov5-5.0\utils\datasets.py`的279行~282行,将这四行注释掉,

```python
if 'youtube.com/' in url or 'youtu.be/' in url:  # if source is YouTube video                check_requirements(('pafy', 'youtube_dl'))                import pafy                url = pafy.new(url).getbest(preftype="mp4").url
```

再运行`python detect.py`即可
