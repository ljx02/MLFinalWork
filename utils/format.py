import numpy as np
import cv2


def convert(size, box, class_id):
    """归一化box坐标，符合yolo3格式"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0  # x的中心点
    y = (box[1] + box[3]) / 2.0  # 中心点y
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return class_id, x, y, w, h


def formatAnno(label_path, img_path):
    img = cv2.imread(img_path)
    height, width, channels = img.shape
    classes = ["带电芯充电宝", "不带电芯充电宝"]
    with open(label_path, "r", encoding='utf-8') as f1:
        dataread = f1.readlines()
        ndarray_box = []
        for annotation in dataread:
            temp = annotation.split()
            name = temp[1]
            if name != '带电芯充电宝' and name != '不带电芯充电宝':
                continue

            cls_id = classes.index(name)

            xmin = int(temp[2])
            if int(xmin) > width:
                continue
            if xmin < 0:
                xmin = 1
            ymin = int(temp[3])
            if ymin < 0:
                ymin = 1
            xmax = int(temp[4])
            if xmax > width:
                xmax = width - 1
            ymax = int(temp[5])
            if ymax > height:
                ymax = height - 1

            bb = [float(xmin) - 1, float(ymin) - 1, float(xmax) - 1, float(ymax) - 1]
            bb = convert((width, height), bb, cls_id)
            ndarray_box.append(bb)
        return np.array(ndarray_box, dtype=float)


if __name__ == '__main__':
    img_path = "data/coreless_battery00003474.jpg"
    label_path = "data/coreless_battery00003474.txt"
    print(formatAnno(label_path, img_path))
