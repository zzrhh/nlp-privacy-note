import cv2
import numpy as np
# 以灰度图形式读入图像
img1 = cv2.imread('data/license+plate.jpg')

# 像素拉伸
def stretch(img):
    max_ = float(img.max())
    min_ = float(img.min())

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = (255 / (max_ - min_)) * img[i, j] - (255 * min_) / (max_ - min_)
    return img

#查找图中矩形
def find_retangle(contour):
    y, x = [], []

    for p in contour:
        y.append(p[0][0])
        x.append(p[0][1])

    return [min(y), min(x), max(y), max(x)]

img = stretch(img1)

kernel = np.ones((5, 15), np.uint8)
#开运算是指图像先进行腐蚀再膨胀的运算，所以对图像进行开运算可以去除图像中的一些噪声
openingimg = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


#然后，获取两个图像之间的差分图，这个函数可以把两幅图的差的绝对值输出到另一幅图上面来，利用这种办法可以去除图片中的大面积噪声。
img = cv2.absdiff(img, openingimg)

#边缘检测的目的是标识数字图像中亮度变化明显的点，所以，利用边缘检测可提高对图像有效信息的感知能力
#img = cv2.GaussianBlur(img, (3, 3), 0)

#先二值化
ret, binary_img = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)

#再做边缘检测
img = cv2.Canny(binary_img, binary_img.shape[0],binary_img.shape[1])


# 先闭运算将车牌数字部分连接，再开运算将不是块状的或是较小的部分去掉
close_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
open_img = cv2.morphologyEx(close_img, cv2.MORPH_OPEN, kernel)

#再次开运算
kernel = np.ones((11, 5), np.uint8)
open_img = cv2.morphologyEx(open_img, cv2.MORPH_OPEN, kernel)

#得到图形不规整，进行多次膨胀操作
element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilation_img = cv2.dilate(open_img, element, iterations=5)

cv2.imshow("dilation_img", dilation_img)

# 获取轮廓
contours, hierarchy = cv2.findContours(dilation_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

blocks = []
for c in contours:
    # 找出轮廓的左上点和右下点，由此计算它的面积和长宽比
    r = find_retangle(c)
    a = (r[2] - r[0]) * (r[3] - r[1])
    s = (r[2] - r[0]) / (r[3] - r[1])
    if(s > 2 and s<5):
        blocks.append([r, a, s])

# 选出面积较小的3个区域
blocks = sorted(blocks, key=lambda b: b[2])

for i in range(3):
    rect = blocks[i][0]
    cv2.rectangle(img1, (rect[0], rect[1]), (rect[2], rect[3]), (0,255,0),2)


cv2.imshow("rectangle", img1)
cv2.imwrite('data/license.jpg',img1)
cv2.drawContours(img1, contours, -1, (0, 0, 255), 3)
cv2.imshow("lpr", img1)

cv2.waitKey(0)
