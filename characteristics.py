#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import sys
import dlib
import glob
from skimage import io 
from numpy import *

AlignmentModel='model/face_alignment.dat'
FaceModel='model/face_model.dat'
FaceDetection=dlib.get_frontal_face_detector()
FaceAlignment=dlib.shape_predictor(AlignmentModel)
Description=dlib.face_recognition_model_v1(FaceModel)

def check_parameter(param, param_type, create_new_if_missing=False):
    assert param_type == 'file' or param_type == 'directory'
    if param_type == 'file':
        assert os.path.exists(param)
        assert os.path.isfile(param)
    else:
        if create_new_if_missing is True:
            if not os.path.exists(param):
                os.makedirs(param)
            else:
                assert os.path.isdir(param)
        else:
           assert os.path.exists(param)
           assert os.path.isdir(param)

def listdir(top_dir, type='image'):
    tmp_file_lists = os.listdir(top_dir)
    file_lists = []
    if type == 'image':
        for e in tmp_file_lists:
           if e.endswith('.jpg') or e.endswith('.png') or e.endswith('.bmp') or e.endswith('.JPG'):
               file_lists.append(e)
    elif type == 'dir':
        for e in tmp_file_lists:
           if os.path.isdir(top_dir + e):
              file_lists.append(e)
    else:
        raise Exception('Unknown type in listdir')
    return file_lists

def extract_feature(src_dir, tgt_dir,type_name):
    check_parameter(src_dir, 'directory')
    check_parameter(tgt_dir, 'directory', True)
    if src_dir[-1] != '/':
        src_dir += '/'
    if tgt_dir[-1] != '/':
        tgt_dir += '/'    
    image_lists = listdir(src_dir, type_name)
    names_lists=open(tgt_dir+"names.txt",'w')
    feature_lists=[]
    for n,e in enumerate(image_lists):
        img=io.imread(''.join([src_dir,e]))
        dets=FaceDetection(img,1) #upsampling
        if len(dets)==0:
            print("The faces of {}  >>>>>>>>>>  detecting defeat".format(e))
        else:
            params=e.split(".")
            name_id=params[0]
            print("ID : {} , Name : {}  ".format(n,name_id)) 
            Best_face=max(dets, key=lambda rect: rect.width() * rect.height())
            shape = FaceAlignment(img, Best_face)
            face_descriptor=Description.compute_face_descriptor(img, shape, 100)
            feature_lists.append(face_descriptor)
            names_lists.write("{}\n".format(name_id))
    savetxt(tgt_dir+"features.txt",feature_lists)
    names_lists.close()

if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print("Please enter the command : python FeatureLib.py $image_path\n"
              "$image_path is the location of images")
        exit()
    src_dir=sys.argv[1]
    extract_feature(src_dir,'faceFeatureLib','image')












