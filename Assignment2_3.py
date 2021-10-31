import cv2
import numpy as np

img1 = cv2.imread('data/cards.jpeg')
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


def cut_license(afterimg, rect):
    '''
    图像分割函数
    '''
    # 转换为宽度和高度
    rect[2] = rect[2] - rect[0]
    rect[3] = rect[3] - rect[1]
    rect_copy = tuple(rect.copy())
    rect = [0, 0, 0, 0]
    # 创建掩膜
    mask = np.zeros(afterimg.shape[:2], np.uint8)
    # 创建背景模型  大小只能为13*5，行数只能为1，单通道浮点型
    bgdModel = np.zeros((1, 65), np.float64)
    # 创建前景模型
    fgdModel = np.zeros((1, 65), np.float64)
    # 分割图像
    cv2.grabCut(afterimg, mask, rect_copy, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img_show = afterimg * mask2[:, :, np.newaxis]

    return img_show

img = stretch(img1)

kernel = np.ones((5, 5), np.uint8)
#开运算是指图像先进行腐蚀再膨胀的运算，所以对图像进行开运算可以去除图像中的一些噪声
openingimg = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

#边缘检测的目的是标识数字图像中亮度变化明显的点，所以，利用边缘检测可提高对图像有效信息的感知能力
#先二值化
ret, binary_img = cv2.threshold(openingimg, 110, 255, cv2.THRESH_BINARY)

#再做边缘检测
img = cv2.Canny(binary_img, binary_img.shape[0],binary_img.shape[1])


# 先闭运算将数字部分连接
close_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

#得到图形不规整，进行多次膨胀操作
element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
dilation_img = cv2.dilate(close_img, element, iterations=1)

cv2.imshow('binary_cut_img',dilation_img)

# 获取轮廓
contours, hierarchy = cv2.findContours(dilation_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

blocks = []
for c in contours:
    # 找出轮廓的左上点和右下点，由此计算它的面积和长宽比
    r = find_retangle(c)
    blocks.append(r)

cutimg = []
image = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
for i in range(3):
    rect = blocks[i]
    cv2.rectangle(img1, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)

cv2.imshow('img1',img1)

cv2.waitKey(0)
cv2.destroyAllWindows()