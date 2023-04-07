# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 17:11:06 2022

@author: liuzh
"""
import gradio as gr
import time
import cv2

#############这里需要添加绝对路径###################
pathf = './haarcascades/haarcascade_frontalface_alt.xml'
pathe = './haarcascades/haarcascade_eye.xml'


###########################################

# 人脸检测函数
def face_rec(img):
    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 创建人脸识别分类器
    face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    face_cascade.load('./haarcascades/haarcascade_frontalface_default.xml')
    # 创建人眼识别分类器
    eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
    face_cascade.load('./haarcascades/haarcascade_eye.xml')
    # 检测人脸
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.15,
                                          minNeighbors=3,
                                          flags=cv2.IMREAD_GRAYSCALE,
                                          minSize=(40, 40))

    # 在人脸周围绘制方框
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # 进行眼部检测
    eyes = eye_cascade.detectMultiScale(gray,
                                        scaleFactor=1.1,
                                        minNeighbors=3,
                                        flags=cv2.IMREAD_GRAYSCALE,
                                        minSize=(3, 3))
    for (ex, ey, ew, eh) in eyes:
        # 绘制眼部方框
        img = cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    
    cv2.imwrite(f"result/{time.time_ns()}.jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img


demo = gr.Interface(
    face_rec,
    gr.Image(),
    "image",    
    examples=["images/1.jpg", "images/2.jpg", "images/3.jpg", "images/4.jpg"],
)

if __name__ == "__main__":
    demo.launch()
