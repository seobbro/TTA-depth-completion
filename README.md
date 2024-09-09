# Test-Time Adaptation for Depth Completion

PyTorch implementation of *Test-Time Adaptation for Depth Completion*

[[publication]]() [[arxiv]](https://arxiv.org/pdf/2402.03312.pdf) [[poster]]() [[talk]]()

Model have been tested on Ubuntu 20.04 using Python 3.7, 3.8,  PyTorch 1.10.1 and 1.11.0 (CUDA 11.1)


Authors: [Hyoungseob Park](https://www.linkedin.com/in/hyoungseob-park-00692a188/), [Anjali Gupta](https://www.anjaliwgupta.com/), [Alex Wong](https://www.cs.yale.edu/homes/wong-alex/)

If this work is useful to you, please cite our paper:

**News**
```
2024.09.09: I got back to the school, and the full repository including bash script and data setup will be available by Sep 15th.
```
**Table of Contents**
[About ProxyTTA](#about-ProxyTTA)
[Setting up your virtual environment](#set-up-virtual-environment)
[Setting up your datasets](#set-up-datasets)
[Training your models](#training-ProxyTTA)
[Related projects](#related-projects)
[License and disclaimer](#license-disclaimer)

## About ProxyTTA <a name="about-ProxyTTA"></a>

### Motivation:

It is common to observe performance degradation when transferring models trained on some (source) datasets to target testing data due to a domain gap between them.
Existing methods for bridging this gap, such as domain adaptation (DA), may require the source data on which the model was trained (often not available), while others, i.e., source-free DA, require many passes through the testing data. As we can only assume that a single pair of image and sparse depth map is available in the target domain for the depth completion, models belonging to either learning paradigms cannot easily be trained or adapted to the new domain even when given the testing data.


### Our Solution:

#### Use of sparse depth modality as proxy:

we investigate a test-time adaptation approach that learns an embedding for guiding the model parameter update by exploiting the data modality (sparse depth) that is less sensitive to the domain shift. The embedding module maps the latent features encoding sparse depth to the latent features encoding both image and sparse depth. The mapping is trained in the source domain and frozen when deployed to the target domain for adaptation. During test time, sparse depth is first fed through the encoder and mapped, through the embedding module, to yield a proxy for image and sparse depth embeddings from the source domain -- we refer to the embedded sparse depth features as proxy embeddings. Note: As the mapping is learned in the source domain, the proxy embeddings will also follow the distribution of source image and sparse depth embeddings. Next, both image and sparse depth from the target test domain are fed as input to the encoder. By maximizing the similarity between test-time input embeddings and the proxy embeddings, we align the target distribution to that of the source to reduce the domain gap. In other words, our method exploits a proxy modality for guiding test-time adaptation and we call the approach, ProxyTTA. When used in conjunction with typical loss functions to penalize discrepancies between predictions and input sparse depth, and abrupt depth transitions, i.e., Total Variation, the embeddings serve as regularization to guide the model parameter update and prevent excessive drift from those trained on the source data.


<center>
<img src="figures/demo.gif">
</center>


## Setting up your virtual environment <a name="set-up-virtual-environment"></a>
We will create a virtual environment using virtualenv with dependencies for running our results.
```
virtualenv -p /usr/bin/python3.8 ~/venvs/proxytta
source ~/venvs/proxytta/bin/activate

export TMPDIR=./
```

Nvidia RTX achitectures i.e. 20 and 30 series (CUDA 11.1)
```
pip install torch==1.10.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements-rtx.txt
```

## Setting up your datasets <a name="set-up-datasets"></a>

In setup folder, we uploaded the python scripts for setting up test datasets.

For datasets, we will use [KITTI][kitti_dataset] for outdoors and [VOID][void_github] for indoors. Below are instructions to run our setup script for each dataset. The setup script will (1) store images as sequential temporal triplets and (2) produce paths for training, validation and testing splits.
```
mkdir -p data
ln -s /path/to/kitti_raw_data data/
ln -s /path/to/kitti_depth_completion data/
ln -s /path/to/void_release data/
```

If you already have KITTI and VOID datasets, you can set them up using
```
python setup/setup_dataset_kitti.py
python setup/setup_dataset_void.py
```

In case you do not already have KITTI dataset downloaded, we provide download a scripts:
```
bash bash/setup_dataset_kitti.sh
```
For the KITTI dataset, the `bash/setup_dataset_kitti.sh` script will download and set up `kitti_raw_data` and `kitti_depth_completion` for you in your data folder.

For the VOID dataset, you may download them via:
```
https://drive.google.com/open?id=1kZ6ALxCzhQP8Tq1enMyNhjclVNzG8ODA
https://drive.google.com/open?id=1ys5EwYK6i8yvLcln6Av6GwxOhMGb068m
https://drive.google.com/open?id=1bTM5eh9wQ4U8p2ANOGbhZqTvDOddFnlI
```
which will give you three files `void_150.zip`, `void_500.zip`, `void_1500.zip`.

Assuming you are in the root of the repository, to construct the same dataset structure as the setup script above:
```
mkdir void_release
unzip -o void_150.zip -d void_release/
unzip -o void_500.zip -d void_release/
unzip -o void_1500.zip -d void_release/
bash bash/setup_dataset_void.sh unpack-only
```
If you encounter `error: invalid zip file with overlapped components (possible zip bomb)`. Please do the following
```
export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE
```
and run the above again.

For more detailed instructions on downloading and using VOID and obtaining the raw rosbags, you may visit the [VOID][void_github] dataset webpage.

and store the paths to the training, validation and testing data as `.txt` files in
```
training/kitti
validation/kitti
testing/kitti
training/void
testing/void
```

## Training ProxyTTA <a name="training-ProxyTTA"></a>
Bash file will be uploaded soon! (~Jul 12th)
```
bash
```

## Related projects <a name="related-projects"></a>
You may also find the following projects useful:

- [KBNet][kbnet_github]: *Unsupervised Depth Completion with Calibrated Backprojection Layers*. A fast (15 ms/frame) and accurate unsupervised sparse-to-dense depth completion method that introduces a calibrated backprojection layer that improves generalization across sensor platforms. This work is published as an oral paper in the International Conference on Computer Vision (ICCV) 2021.
- [ScaffNet][scaffnet_github]: *Learning Topology from Synthetic Data for Unsupervised Depth Completion*. An unsupervised sparse-to-dense depth completion method that first learns a map from sparse geometry to an initial dense topology from synthetic data (where ground truth comes for free) and amends the initial estimation by validating against the image. This work is published in the Robotics and Automation Letters (RA-L) 2021 and the International Conference on Robotics and Automation (ICRA) 2021.
- [AdaFrame][adaframe_github]: *An Adaptive Framework for Learning Unsupervised Depth Completion*. An adaptive framework for learning unsupervised sparse-to-dense depth completion that balances data fidelity and regularization objectives based on model performance on the data. This work is published in the Robotics and Automation Letters (RA-L) 2021 and the International Conference on Robotics and Automation (ICRA) 2021.
- [VOICED][voiced_github]: *Unsupervised Depth Completion from Visual Inertial Odometry*. An unsupervised sparse-to-dense depth completion method, developed by the authors. The paper introduces Scaffolding for depth completion and a light-weight network to refine it. This work is published in the Robotics and Automation Letters (RA-L) 2020 and the International Conference on Robotics and Automation (ICRA) 2020.
- [VOID][void_github]: from *Unsupervised Depth Completion from Visual Inertial Odometry*. A dataset, developed by the authors, containing indoor and outdoor scenes with non-trivial 6 degrees of freedom. The dataset is published along with this work in the Robotics and Automation Letters (RA-L) 2020 and the International Conference on Robotics and Automation (ICRA) 2020.
- [XIVO][xivo_github]: The Visual-Inertial Odometry system developed at UCLA Vision Lab. This work is built on top of XIVO. The VOID dataset used by this work also leverages XIVO to obtain sparse points and camera poses.
- [GeoSup][geosup_github]: *Geo-Supervised Visual Depth Prediction*. A single image depth prediction method developed by the authors, published in the Robotics and Automation Letters (RA-L) 2019 and the International Conference on Robotics and Automation (ICRA) 2019. This work was awarded **Best Paper in Robot Vision** at ICRA 2019.
- [AdaReg][adareg_github]: *Bilateral Cyclic Constraint and Adaptive Regularization for Unsupervised Monocular Depth Prediction.* A single image depth prediction method that introduces adaptive regularization. This work was published in the proceedings of Conference on Computer Vision and Pattern Recognition (CVPR) 2019.

## License and disclaimer <a name="license-disclaimer"></a>
This software is property of Yale University, and is provided free of charge for research purposes only. 
