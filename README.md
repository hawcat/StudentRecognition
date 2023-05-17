## StudentRecognition 学生行为识别
我的博客:https://hawcat.cn
闲暇之余把毕设开源了，真的很水，各位不要对这个项目要求太高~

https://github.com/hawcat/StudentRecognition
推荐大家看看俞刚老师的文章，有很多有趣的工作。
https://zhuanlan.zhihu.com/p/85506259
参考了很多知网上的优化方向，针对OpenPose方面的话主要是苏超的优化方式，在我的项目里面应该很能体现他的论文内容。
时间太紧代码写的有点乱，特别是s0的main函数，当初为了实现功能省了太多步骤了，有时间重构一下这一块的代码。大多都是开源大牛的杰作，我的工作仅仅只是把他们升级优化然后整合到一起了而已。

项目框架Yolov5 5.0 + tf-OpenPose

Yolov5需要Pytorch，OpenPose需要tensorflow2.x，请严格按照requirement.txt安装环境，才能利用CUDA进行显卡加速。

cudatoolkit==11.0.3

cudnn==8.0.5.39

![项目架构](https://cdn.hawcat.cn/%E5%9B%BE%E7%89%871.png)

OpenPose的网络在src的Pose文件夹里面，Github上面上传不了，默认是VGG19，各位可以自行替换优化好的文件，我这里可以提供Mobile-net的网络文件。

Yolo的模型是我训练好的检测手机模型，可以自行替换为你自己的模型。

感谢

https://github.com/Zumbalamambo/tf-openpose

https://github.com/mpj1234/yolov5-5.0-simpleUI





## StudentRecognition

The time is too tight and the code is messy, most of it is the masterpiece of open source predecessors, and my job is just to put them together.

Project framework Yolov5 5.0 + OpenPose

Yolov5 based on Pytorch, OpenPose based on tensorflow 2.x, please strictly follow the requirements .txt installation environment to use CUDA for graphics card acceleration .

cudatoolkit==11.0.3

cudnn==8.0.5.39

![项目架构](https://cdn.hawcat.cn/%E5%9B%BE%E7%89%871.png)

OpenPose's network is in the Pose folder of src, Github can not uploaded, the default is VGG19, you can replace the optimized file by yourself, I can provide Mobile-net network files here.

Yolo's model is a phone detection model that I trained, ofc you can replace it with your own model.

Gratitude for

https://github.com/Zumbalamambo/tf-openpose

https://github.com/mpj1234/yolov5-5.0-simpleUI
