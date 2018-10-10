#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import os
import dlib
import numpy as np
from skimage import io

AlignmentModel='model/face_alignment.dat'
FaceModel='model/face_model.dat'
FaceDetection=dlib.get_frontal_face_detector()
FaceAlignment=dlib.shape_predictor(AlignmentModel)
Description=dlib.face_recognition_model_v1(FaceModel)

def read_names_lib(filename):
    f = open(filename,'r')
    r = f.readlines()
    f.close()
    return r

feature_lib = np.loadtxt("faceFeatureLib/features.txt")
names = read_names_lib("faceFeatureLib/names.txt")
numbers = len(names)
print("***********Model load ok and Matching faces numbers is {} ***********".format(numbers))
print("Everthing is Ready ! Please enter image :")

def Matching_top3(target_feature,feature_lib):
    comparison=[]
    for row in range(numbers):
        distance = np.sqrt(np.sum(np.square(target_feature - feature_lib[row])))
        comparison.append(distance)
    max_1=max_2=max_3=10
    max_1_id=max_2_id=max_3_id=0
    for nu in range(numbers):
        if comparison[nu]<max_1:
            max_2=max_1
            max_2_id=max_1_id
            max_1=comparison[nu]
            max_1_id=nu
        elif comparison[nu]>max_1 and comparison[nu]<max_2:
            max_3=max_2
            max_3_id=max_2_id
            max_2=comparison[nu]
            max_2_id=nu
        elif comparison[nu]>max_2 and comparison[nu]<max_3:
            max_3=comparison[nu]
            max_3_id=nu
    print("Top1 : {}".format(names[max_1_id]))
    print("Top2 : {}".format(names[max_2_id]))
    print("Top3 : {}".format(names[max_3_id]))

while True:
    line=sys.stdin.readline().strip()
    if len(line)==0:
        break
    try:
        img = io.imread(line)
        dets = FaceDetection(img, 1)
    except IOError:
        print("!~Error Error loading image") 
        continue
    except:
        print("!~Error ERROR_ANALIZING")
        continue
    
    if len(dets)==0:
         print("Error {} FACE NOT FOUND".format(line))
         continue
    
    face_descriptor=[]
    face = max(dets, key=lambda rect: rect.width() * rect.height())
    shape = FaceAlignment(img, face)
    face_descriptor=Description.compute_face_descriptor(img, shape, 100)
    
    print("The Top3 result of Matching {} is as below :\n".format(line)) 
    Matching_top3(face_descriptor,feature_lib)   
    print("==============Please continue to enter the picture=====================")


















