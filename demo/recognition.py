#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import sys
import cv2
import dlib
import glob
from skimage import io 
from numpy import loadtxt, sqrt, sum, square
from datetime import datetime

face_detection = dlib.get_frontal_face_detector()
face_alignment = dlib.shape_predictor('model/face_alignment.dat')
description = dlib.face_recognition_model_v1('model/face_model.dat')

def read_names_lib(filename):
	f = open(filename,'r')
	r = f.readlines()
	f.close()
	return r
feature_lib = loadtxt("features.txt")
names_lib = read_names_lib("names.txt")
numbers = len(names_lib)

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

def face_id_by_image(image_path, features, names_lib, numbers):
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
		distance = sqrt(sum(square(face_descriptor - feature_lib[row])))
		comparison.append(distance)
		#Take the smallest gap.
		if distance < optimal:
			optimal = distance
			optimal_id = row
		
		return names_lib[optimal_id]
	else:
		print("Face not found !")

def face_id_by_frame(frame, last_names = []):
	optimal = 5
	det_names = []
	last_names_index = []
	face_descriptors = []
	
	img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	#Get face numbers in frame
	dets = face_detection(img_gray, 0)
	# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
	rgb_frame = frame[:, :, ::-1]
	face_num = len(dets)
	
	for i in range(len(dets)):
		face_descriptors.append(description.compute_face_descriptor(rgb_frame, face_alignment(rgb_frame, dets[i])))
		optimal_id = -1
		for name in last_names:
			last_names_index.append(names_lib.index(name + '\n'))
			distance = sqrt(sum(square(face_descriptors[i] - feature_lib[last_names_index[-1]])))
			#Take the smallest gap.
			if distance <= optimal:
				det_names.append(name)
				face_num -= 1
	
	for i in range(face_num):
		for row in range(numbers):
			if row in last_names_index:
				continue
			for face_descriptor in face_descriptors:
				distance = sqrt(sum(square(face_descriptor - feature_lib[row])))
				#Take the smallest gap.
				if distance < optimal:
					optimal = distance
					optimal_id = row
		if optimal_id != -1:
			det_names.append(names_lib[optimal_id].replace('\n', ''))
			optimal_id = -1
	
	return rgb_frame, det_names

def face_id_by_frame2(frame, last_names = []):
	optimal = 10
	det_names = []
	
	img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)	
	#Get face numbers in frame
	dets = face_detection(img_gray, 0)
	rgb_frame = frame[:, :, ::-1]
	small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
	rgb_small_frame = small_frame[:, :, ::-1]

	for i in range(len(dets)):
		shape = face_alignment(rgb_small_frame, dets[i])
		start = datetime.now()
		face_descriptor = description.compute_face_descriptor(rgb_small_frame, shape)
		end = datetime.now()
		print (end - start).seconds
		optimal_id = -1
		for row in range(numbers):
			distance = sqrt(sum(square(face_descriptor - feature_lib[row])))
			#Take the smallest gap.
			if distance < optimal:
				optimal = distance
				optimal_id = row
		det_names.append(names_lib[optimal_id].replace('\n', ''))
	
	return rgb_frame, det_names

def face_id_by_fram3(frame, last_names = []):
	optimal = 5
	det_names = []
	last_names_index = []
	face_descriptors = []
	
	img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	#Get face numbers in frame
	dets = face_detection(img_gray, 0)
	# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
	rgb_frame = frame[:, :, ::-1]
	face_num = len(dets)
	small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
	rgb_small_frame = small_frame[:, :, ::-1]
	for i in range(len(dets)):
		face_descriptors.append(description.compute_face_descriptor(rgb_frame, face_alignment(rgb_frame, dets[i])))
		optimal_id = -1
		for name in last_names:
			last_names_index.append(names_lib.index(name + '\n'))
			distance = sqrt(sum(square(face_descriptors[i] - feature_lib[last_names_index[-1]])))
			#Take the smallest gap.
			if distance <= optimal:
				det_names.append(name)
				face_num -= 1
	
	for i in range(face_num):
		for row in range(numbers):
			if row in last_names_index:
				continue
			for face_descriptor in face_descriptors:
				distance = sqrt(sum(square(face_descriptor - feature_lib[row])))
				#Take the smallest gap.
				if distance < optimal:
					optimal = distance
					optimal_id = row
		if optimal_id != -1:
			det_names.append(names_lib[optimal_id].replace('\n', ''))
			optimal_id = -1
	
	return rgb_frame, det_names
