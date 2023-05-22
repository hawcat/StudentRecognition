## 学生行为识别

闲暇之余把毕设开源了，真的很水，各位不要对这个项目要求太高~

https://github.com/hawcat/StudentRecognition

大家可以看看俞刚老师的文章，这是指导老师推荐给我的一篇文章，也推荐给大家，算是我的姿态估计启蒙吧，你们也可以当作综述来看。https://zhuanlan.zhihu.com/p/85506259

参考了很多知网上的优化方向，针对OpenPose方面的话主要是苏超的优化方式，在我的项目里面应该很能体现他的论文内容。

时间太紧代码写的有点乱，特别是s0的main函数，当初为了实现功能省了太多步骤了，有时间重构一下这一块的代码。大多都是开源大牛的杰作，我的工作仅仅只是把他们升级优化然后整合到一起了而已。

项目框架Yolov5 5.0 + tf-OpenPose

~~因为项目同时用到了两种框架（虽然说没必要）~~，所以请严格按照requirement.txt安装环境，才能利用CUDA进行显卡加速。

### 依赖

你可能需要按照这个环境版本进行配置

- cudatoolkit==11.0.3
- cudnn==8.0.5.39
- torch==1.9.1+cu111
- tensorflow-gpu==2.4.0

### 架构

![项目架构](https://cdn.hawcat.cn/%E5%9B%BE%E7%89%871.png)

OpenPose的网络在src的Pose文件夹里面，Github上面上传不了，默认是VGG19，各位可以自行替换优化好的文件，我这里可以提供Mobile-net的网络文件。

### Yolo 模块

Yolo的模型是我训练好的检测手机模型，可以自行替换为你自己的模型。

Yolov5s的训练矩阵：

*其实可以发现在前50轮的时候就已经有了很好的精准度了*

![图片2.png (684×676) (hawcat.cn)](https://cdn.hawcat.cn/articlePicture/图片2.png)

Yolo模块演示：

![图片3.png (588×369) (hawcat.cn)](https://cdn.hawcat.cn/articlePicture/图片3.png)

数据集来源：[违规使用手机 - 飞桨AI Studio (baidu.com)](https://aistudio.baidu.com/aistudio/datasetdetail/105315)

### OpenPose 模块

具体训练数据就不给出了，大家可以自己录视频或者拍照然后按照格式放在data文件夹中自行分类训练。

![OpenPose模块](https://cdn.hawcat.cn/articlePicture/图片1.png)

### 引用

https://github.com/Zumbalamambo/tf-openpose

https://github.com/mpj1234/yolov5-5.0-simpleUI





## StudentRecognition

The time is too tight and the code is messy, most of it is the masterpiece of open source predecessors, and my job is just to put them together.

Project based on Yolov5 5.0 + OpenPose

Please strictly follow the requirements .txt installation environment to use CUDA for graphics card acceleration due to the project using two frameworks.

### Install

You need dependencies below.

- cudatoolkit==11.0.3
- cudnn==8.0.5.39
- torch==1.9.1+cu111
- tensorflow-gpu==2.4.0

### Architecture

![项目架构](https://cdn.hawcat.cn/%E5%9B%BE%E7%89%871.png)

OpenPose's network is in the Pose folder of src, Github can not uploaded, the default is VGG19, you can replace the optimized file by yourself, I can provide Mobile-net network files.

### Yolov5 module

Yolo's model is a phone detection model that I trained, ofc you can replace it with your own model.

Training matrix of Yolov5s:

*In fact, it can be found that in the first 50 rounds, there is already a good accuracy*

![图片2.png (684×676) (hawcat.cn)](https://cdn.hawcat.cn/articlePicture/图片2.png)

Yolo Module Demo:

![图片3.png (588×369) (hawcat.cn)](https://cdn.hawcat.cn/articlePicture/图片3.png)

Dataset Source: [Illegal Use of Mobile Phone - Paddle AI Studio (baidu.com)] (https://aistudio.baidu.com/aistudio/datasetdetail/105315)

### OpenPose module

You can record your own video or take photos, and then put it in the data folder according to the format to classify the training by yourself.

![OpenPose模块](https://cdn.hawcat.cn/articlePicture/图片1.png)

### References

https://github.com/Zumbalamambo/tf-openpose

https://github.com/mpj1234/yolov5-5.0-simpleUI
