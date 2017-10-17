# synthetic-target-line-of-bearing

Create function to provide angle of bearing for person in an image. The following code was created using both the Nvidia and Tensorflow docker. The code should work if only using CPU but will still need to install tensorflow


## Dependencies

#### Python Packages

* tensorflow
* numpy 
* os
* six.moves.urllib
* sys
* tarfile
* zipfile
* pandas
* math
* collections
* io
* matplotlib
* pil

#### Docker

* Tensorflow docker container https://github.com/tensorflow/tensorflow.git
* Nvidia docker https://github.com/NVIDIA/nvidia-docker
        - This will be needed if you want to use the gpu for this script
* need to update python package protobuf to 3.4.1 - I used code "pip install protobuf --upgrade"  

