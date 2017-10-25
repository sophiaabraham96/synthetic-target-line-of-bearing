# synthetic-target-line-of-bearing

Create function to provide angle of bearing for person in an image. The following code was created using both the Nvidia and Tensorflow docker. The code should work if only using CPU but will still need to install tensorflow

## Discussion
The method selected to perform object recognition was the Single Shot MultiBox Detector (SSD) https://research.google.com/pubs/pub44872.html. This model was selected due to its low computational overhead and low resolution input to improve processing speed, providing greater accuracy in object detection.The COCO dataset was selected for training due to its advantages in project recognition of small objects within an image. The current method of identifying the object angle utilizes the line of bearing and field view of the camera as input parameters. Simple mathematical operations were performed to derive the object angle from the center of the image.

## Contribution
This is the first iteration of this project. Please feel free to contribute to this project by providing issue notificaitons, suggestions of methodology improvements, or suggestions of any kind. Please fork this project first before you make updates to this code. 

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
* need to update python package protobuf to 3.4.0 - I used code "pip install protobuf --upgrade"  

