## MICC Dataset

The MICC dataset that was used for evaluation of the different losses can be download from this [site](https://www.micc.unifi.it/resources/datasets/florence-3d-faces/).

It is composed by the facial scans of 53 subjects together with 3 videos, each of them recorded in three different situations:
- Controlled indoor setting (HD video)
- Semi-controlled indoor using PTZ survillance camera.
- Unconstrained and outdoor under challeging conditions.

## Procedure for evaluation

Our models are evaluated **only** using the frames from the controlled indoor videos.

First of all, we extract the frames from the controlled indoor videos for each subject. We make use of ffmpeg:

```
brew install ffmpeg
ffmpeg -i video.mov frame%04d.jpg
```

We manually select the most frontal frame from each subject, which we list in annotations.json

Then, the face in each selected frontal frame is cropped using the face detector from [DLIB](http://dlib.net/) and then, the crop is passed through the network to predict the 3D face.

In order to evaluate the errors of the predictions, we manually annotate four 3D landmarks in the scans from MICC, which are used to perform an initial alignment and that we provide in the file annotations.json. After this, Iterative Closest Point (ICP) is applied to refine the alignment between the prediction and the groundtruth.
