## VideoFaceSample

Simple Python scripts using OpenCV to detect faces in video frames and Tensorflow with Keras to train a neural network recognizing the person the face belongs to.

`faces_from_video.py`: Detect faces on video frames, cutting them out and saving as JPG files.

`distribute_images.py`: Shuffle and distribute images to directory structure later used by Keras' `flow_from_directory` method.

`train_network.py`: Train simple neural network with the prepared image data.

`use_network.py`: Check and use the network in a live video stream from webcam.
