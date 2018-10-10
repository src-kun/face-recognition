## face-recognition

云创大数据人脸识别开源系统

## Installation

### Requirements
	* Python 2.7
	* CentOS 7

## Installation Options

### Installing

```bash
git clone https://github.com/src-kun/dlib
cd dlib
python setup.py install
yum install python2-scikit-image.x86_64
```

## Example

### Face feature library generated by comparison

```bash
python characteristics.py
```

	The faceFeatureLib folder is generated, which includes: names.txt and features.txt, which refer to the label of the face and the corresponding face feature library respectively.

### Matching face

```bash
python matching.py
```

## Model test description

##### The number of N: N is less than 100 thousand, Top1 can almost hit directly.
##### Face test picture requirements: face angle is less than 20 degrees, face resolution is greater than 110*110
