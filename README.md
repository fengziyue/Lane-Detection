# Lane-Detection

### Video Demo:
[![image](https://user-images.githubusercontent.com/21237230/146306411-19f2017f-407c-4217-ad86-6d9227de9cf6.png)](https://youtu.be/Ryo1Iv46Bc8)

我的第一个深度学习项目。大概是2016年秋冬季，使用CNN检测车道线。

说是Detection，事实上应该是Localization。

一个8层的网络（5层卷积，3层FC），输入一张图片，回归出4个连续值，代表车道线位置的坐标。

这个方法太naive，用真正的Detection的方法效果应该会更好。可惜没有继续做下去。

涵盖了晴天，雨天，雪天，夜间，高速公路，城区道路等多种场景。

这个网络只能检测直线的车道线，后面还有适应了曲线的版本，代码还没有上传。

检测效果见https://www.youtube.com/watch?v=Ryo1Iv46Bc8
