# HGI-SLAM

**Authors:** Shuhul Mujoo and Jerry Ng

HGI-SLAM (Human and Geometric Importance SLAM) is a ...

This is the accompanying source code for the paper **[HGI-SLAM: Loop Closure With Human and Geometric Importance Features](...)** 

## 1. License

HGI-SLAM is released under a GPLv3 [license](https://github.com/shuhul/HGI-SLAM/blob/main/LICENSE).

## 2. Video

[![HGI-SLAM: Loop Closure With Human and Geometric Importance Features](assets/video.png)](...)

## 3. Citation
If you use HGI-SLAM in an academic work, please cite:

```
    @article{mujng2022,
      title={{HGI-SLAM}: Loop Closure With Human and Geometric Importance Features},
      author={Shuhul Mujoo, Jerry Ng},
      journal={...},
      volume={..},
      year={2022},
      url={...},
      doi={...},
      eprinttype={arXiv},
      eprint={1909.05214}
     }
```

## 4. Setup

We have tested the code on **Ubuntu 16.04** and on a **i7 processor**. It should work on other platforms but we recommend this configuration.

### 4.1. Prerequisites

You need to have the following packages on your machine.

* [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2)
* [SalinetDSO](https://github.com/prgumd/SalientDSO)
* [Superpoint](https://github.com/rpautrat/SuperPoint)

### 4.2. Cloning

```
git clone https://github.com/shuhul/HGI-SLAM.git
```

### 4.3 Required Dependencies

The required dependencies are 

```
...
```

### 4.4 Building

In order to build the project run

```
...
```

## 5. Usage

### 5.1 Datasets

#### TUM Dataset

1. Download and uncompress a [sequence](http://vision.in.tum.de/data/datasets/rgbd-dataset/download)
2. Download the corresponding ground truth [trajectory](https://vision.in.tum.de/data/datasets/rgbd-dataset/download#freiburg1_xyz)
3. Move it into a folder called ORB_FRx where x is the sequence number (ex: 1, 2 ...)
4. Move the ground truth file into the folder and rename it to groundtruth.txt (Used for validation)

#### KITTI Dataset  

1. Download and uncompress a [sequence](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) (Note you will have to make an account for this)
2. Download the corresponding ground truth [trajectory](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip) (These are all the trajectorys for the whole dataset, select the one you need)
3. Move it into a folder called KITTI_x where x is the sequence number (ex: 00, 01, ...)
4. Move the ground truth file into the folder and rename it to groundtruth.txt

The rest of the instructions will use TUM sequence 1 but the steps are essentially the same for other datasets.


### 5.2 Example

Run the following code to start the example

```
...
```

## 6. Acknowledgments

We would like to thank the authors of [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2), [SalinetDSO](https://github.com/prgumd/SalientDSO), and [Superpoint](https://github.com/rpautrat/SuperPoint) on which this code is based on.
