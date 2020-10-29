# Generate Synthetic Image for Deep Learning

## 1. Install Python Libraries
```
$ git clone https://github.com/kawashiii/generate_synimg
$ cd generate_synimg
$ pip install -r requirements.txt
```

## 2. Install V-HACD
```
# Latest version
$ git clone https://github.com/kmammou/v-hacd
$ cd v-hacd/install
$ python run.py --cmake
$ cd ../build/linux
$ make
$ sudo cp test/testVHACD /usr/bin

or

# My version
$ cd {THIS_REPOSITORY}
$ sudo cp v-hacd/testVHACD /usr/bin
```
### Problems
1. Offscreen Rendering by OSMesa  
https://pyrender.readthedocs.io/en/latest/install/index.html

2. Export plane urdf  
Because there is no convex decomposition for plane, the export will be failed.  
https://github.com/BerkeleyAutomation/sd-maskrcnn/issues/15

3. Load obj model made by blender  
It'll be error. Load the model in meshlab, then export same file type.


### MEMO
!git clone https://Funmatu@github.com/Funmatu/generate_synimg.git -b master  
!cp /content/generate_synimg/v-hacd/testVHACD /usr/bin  
!chmod 777 /usr/bin/testVHACD
!pip install -r requirements.txt  
!pip install -U imgaug  
%cd generate_synimg/scripts/  
!python generate_datasets.py object bin  
