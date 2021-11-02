import cv2
import numpy as np

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

def find_end(start,arg,black,white,width,black_max,white_max):
    end=start+1
    for m in range(start+1,width-1):
        print(black[m],black_max)
        if (black[m] if arg else white[m])>(0.99*black_max if arg else 0.98*white_max):
            end=m
            break
    return end

img = stretch(img1)

kernel = np.ones((5, 15), np.uint8)
#开运算是指图像先进行腐蚀再膨胀的运算，所以对图像进行开运算可以去除图像中的一些噪声
openingimg = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


#然后，获取两个图像之间的差分图，这个函数可以把两幅图的差的绝对值输出到另一幅图上面来，利用这种办法可以去除图片中的大面积噪声。
img = cv2.absdiff(img, openingimg)

#边缘检测的目的是标识数字图像中亮度变化明显的点，所以，利用边缘检测可提高对图像有效信息的感知能力
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

cutimg = []
image = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
for i in range(3):
    rect = blocks[i][0]
    cv2.rectangle(img1, (rect[0]+5, rect[1]+5), (rect[2]-5, rect[3]-5), (0, 0, 0), 2)
    cutimg.append(cut_license(img1, rect))
    cutimg[i] = cv2.cvtColor(cutimg[i], cv2.COLOR_BGR2GRAY)
    image = cutimg[i] + image

ret, thresh = cv2.threshold(cutimg[0],100, 255, cv2.THRESH_BINARY)

# 分割字符
'''
判断底色和字色
'''
# 记录黑白像素总和
white = []
black = []
height = thresh.shape[0]
width = thresh.shape[1]

white_max = 0
black_max = 0
# 计算每一列的黑白像素总和
for i in range(width):        #width
    line_white = 0
    line_black = 0
    for j in range(height):         #height
        if thresh[j][i] == 255:
            line_white += 1
        if thresh[j][i] == 0:
            line_black += 1
    white_max = max(white_max, line_white)
    black_max = max(black_max, line_black)
    white.append(line_white)
    black.append(line_black)
    #print('white', white)
    #print('black', black)
# arg为true表示黑底白字，False为白底黑字
arg = True
if black_max < white_max:
    arg = False

n = 1
start = 1
end = 2
while n < width - 2:
    n += 1
    # 判断是白底黑字还是黑底白字  0.05参数对应上面的0.95 可作调整
    if (white[n] if arg else black[n]) > (0.01 * white_max if arg else 0.01 * black_max):
        start = n
        end = find_end(start, arg, black, white, width, black_max, white_max)
        n = end
        print(end,start)
        if end - start > 5:
            cj = thresh[1:height, start:end]
            cv2.imshow('cutlicense', cj)


cv2.imshow('image',image)
cv2.imshow('binary_cut_img',thresh)
cv2.imshow("rectangle", img1)
cv2.imwrite('data/license.jpg',img1)

cv2.waitKey(0)
cv2.destroyAllWindows()