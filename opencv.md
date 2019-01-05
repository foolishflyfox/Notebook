# OpenCV3 笔记

OpenCV 的绘图函数可以在容易的图像上进行，但在大多数情况下，它们针对图像的前3个通道有影响，如果是单通道图像，则默认只影响第一个通道。

大多数绘图函数都支持操作对象的颜色、宽度、线型和亚像素对齐等参数；

在 OpenCV 中颜色通道的顺序为：BGR，可以通过`cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` 进行颜色转换后显示。注：所有的操作过程都应该保持 BGR 的顺序，只有在显示的时候才将颜色通道进行转换；   

- `cv2.line(img, pt1, pt2, color, [, thickness, lineType, shift])`: e.g. `cv2.line(img, (0,0), (20,30), (255,0,0), 10)`，注意：pt 的坐标为左上角为(0,0),向右为x轴正向，向下为y轴正向；
- `cv2.circle(img, center, radius, color, [, thickness, lineType, shift])`: e.g. `cv2.circle(img, (100,100), 50, (255,0,0), -1)`, `thickness=-1`表示完全填充；

- `cv2.clipLine(imgRect, pt1, pt2)->retval, pt1, pt2`：例如 `cv2.clipLine((3,5,4,4), (4,5), (9,9))` 输出为 `True, (4,5), (6,7)`, imgRect=(x,y,w,h)
