## StudentRecognition 学生行为识别

时间太紧代码写的有点乱，大多都是开源大牛的杰作，我的工作仅仅只是把他们整合到一起了而已

项目框架Yolov5 5.0 + OpenPose

Yolov5需要Pytorch，OpenPose需要tensorflow2.x，请严格按照requirement.txt安装环境，才能利用CUDA进行显卡加速。

cudatoolkit==11.0.3

cudnn==8.0.5.39

![项目架构](https://cdn.hawcat.cn/%E5%9B%BE%E7%89%871.png)

OpenPose的网络在src的Pose文件夹里面，Github上面上传不了，默认是VGG19，各位可以自行替换优化好的文件，我这里可以提供Mobile-net的网络文件。

Yolo的模型是我训练好的检测手机模型，可以自行替换为你自己的模型。

感谢

https://github.com/Zumbalamambo/tf-openpose

https://github.com/mpj1234/yolov5-5.0-simpleUI
