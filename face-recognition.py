#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import sys
import dlib
import glob
from skimage import io 
from numpy import *

face_detection = dlib.get_frontal_face_detector()
face_alignment = dlib.shape_predictor('model/face_alignment.dat')
description = dlib.face_recognition_model_v1('model/face_model.dat')

def read_names_lib(filename):
	f = open(filename,'r')
	r = f.readlines()
	f.close()
	return r

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

def extract_feature(src_images, des_names, des_feature):
	feature_lists=[]
	image_lists = listdir(src_images)
	name_lists = open(des_names,'w')
	
	for n, e in enumerate(image_lists):
		img=io.imread(''.join([src_images, e]))
		dets = face_detection(img, 1) #upsampling
		if len(dets) == 0:
			print("The faces of {} >>>>>>>>>> detecting defeat".format(e))
		else:
			params = e.split(".")
			name_id=params[0]
			print("ID : {} , Name : {} ".format(n,name_id))
			best_face = max(dets, key = lambda rect: rect.width() * rect.height())
			shape = face_alignment(img, best_face)
			face_descriptor = description.compute_face_descriptor(img, shape, 100)
			feature_lists.append(face_descriptor)
			name_lists.write("{}\n".format(name_id))
	savetxt(des_feature, feature_lists)
	name_lists.close()

def match(image_path, features, names_lib, numbers):
	comparison = []
	face_descriptor = []
	optimal = 10
	optimal_id = 0
	
	img = io.imread(image_path)
	dets = face_detection(img, 1)
	
	if len(dets) != 0:
		face = max(dets, key = lambda rect: rect.width() * rect.height())
		shape = face_alignment(img, face)
		face_descriptor = description.compute_face_descriptor(img, shape, 100)
		
		for row in range(numbers):
			distance = sqrt(sum(square(face_descriptor - features[row])))
			comparison.append(distance)
		
		for nu in range(numbers):
			if comparison[nu] < optimal:
				optimal = comparison[nu]
				optimal_id = nu
		
		return names_lib[optimal_id]
	else:
		print("Face not found !")

#extract_feature('picture/', "names.txt", 'features.txt')

feature_lib = loadtxt("features.txt")
names_lib = read_names_lib("names.txt")
numbers = len(names_lib)
print(match('zqc.jpg', feature_lib, names_lib, numbers))