import cv2
from ultralytics import YOLO
from toolkit import random_color
from distanceConfig import *
def run(weight, source, conf):
    model = YOLO(weight)
    cap = cv2.VideoCapture(source)
    while True:
        # 读取图像
        ret, img = cap.read()
        if not ret:
            break
        # 推理
        results = model(img, conf=conf)[0]
        # 获取模型包含标签和预测结果
        names = results.names
        boxes = results.boxes.data.tolist()
        # 遍历预测结果
        for obj in boxes:
            # x1, y1, x2, y2
            left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
            # 置信度
            confidence = obj[4]
            # 标签
            label = int(obj[5])
            # 颜色
            color = random_color(label)
            # 目标像素宽度
            height = abs(bottom - top)
            # 根据类型选择样本库, 0为行人, 1为车辆,按照你自己模型的标签去判断
            # 这里只简单列举两类
            # 用if判断语句对cls(类别)进行判断
            # 如果是行人,则调用box_distance函数计算距离,如果为车辆,则调用car_distance函数计算距离
            # 其他物体操作方法一样
            if label == 'car':
                dis = car_distance(height)
            elif label == 'person':
                dis = person_distance(height)
            else:
                dis = chair_distance(height)
            dis = str(dis) + 'm'
            # 绘制目标框
            cv2.rectangle(img, (left, top), (right, bottom), color=color ,thickness=2, lineType=cv2.LINE_AA)
            caption = f"{names[label]} {confidence:.2f} {dis}"
            w, h = cv2.getTextSize(caption, 0, 1, 2)[0]
            cv2.rectangle(img, (left - 3, top - 33), (left + w + 10, top), color, -1)
            cv2.putText(img, caption, (left, top - 5), 0, 1, (0, 0, 0), 2, 16)
        # 保存结果 result.jpg
        cv2.imwrite("result.jpg", img)
        print(f"结果已保存")
        cv2.imshow("test", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    run("yolov8n.pt", 'test.jpg', 0.3)