## FaceWarehouse Dataset

The [FaceWarehouse dataset]](http://kunzhou.net/zjugaps/facewarehouse/) was used for the evaluation of the different losses.

This dataset is composed by scans of 150 individuals. For each subject, they extract:
- Facial scan with neutral expression
- 19 facial scans with different expressions.

The 3D data is given in two different formats:
- Raw data from Kinect RGBD camera.
- Registered 3D mesh template to the RGBD.

## Procedure for evaluation

In order to evaluate our models, we select only those images corresponding with the neutral expression as well as the corresponding registered 3D template. We do not use the raw RGBD data.

For each selected frontal frame, the face region is cropped using the face detector from [DLIB](http://dlib.net/) and then, the crop is passed through the network to predict the 3D face.

In order to evaluate the errors of the predictions, we select 4 vertex identifiers from each registered 3D template to perform an initial alignment. Since the registered 3D templates have fixed topology, we select these indices only once and we provide the id's in vertex_identifiers.txt. After the alignment, Iterative Closest Point (ICP) is applied to refine the alignment between the prediction and the groundtruth.