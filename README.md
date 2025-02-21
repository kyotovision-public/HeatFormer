# HeatFormer: A Neural Optimizer for Multiview Human Mesh Recovery

# installation

# Data preparation
1. [Human3.6M](http://vision.imar.ro/human3.6m/description.php)
2. [MPI-INF-3DHP](http://gvv.mpi-inf.mpg.de/3dhp-dataset/)
3. [BEHAVE](https://virtualhumans.mpi-inf.mpg.de/behave/)
4. [RICH](https://rich.is.tue.mpg.de/)

More specifically:
1. **Human3.6M**: You regidter from this [link](http://vision.imar.ro/human3.6m/description.php) and download data. Then, you preprocess Human3.6M dataset by [H36M-Toolbox](https://github.com/CHUNYUWANG/H36M-Toolbox). We provide the preprocessed data => [Google Drive or One Drive](). After preprocessing data or download preprocedded data, you place data look like this:

```
${HeatFormer root}
|-- data
    |-- preprocessed_data
        |--h36m_train_25fps_new_db.pt
        |--h36m_test_25fps_new_db.pt
        |--h36m_train_25fps_ex_db.pt
        |--h36m_test_25fps_ex_db.pt
        |extra_data
            |-- Human36M_subject*_camera.json
            |-- Human36M_subject*_joint_3d.json
            |-- Human36M_subject*_SMPL_NeuralAnnot.json
    |-- dataset
        |-- images_h36m
            |-- s_01_act_02_subact_01_ca_01
            ..
```

2. **MPI-INF-3DHP**: You visit the [website](http://gvv.mpi-inf.mpg.de/3dhp-dataset/) of dataset, download zip file and, run the scripts. After running the scripts, you place the data look like this:

${HeatFormer root}
|-- mpi_inf_3dhp

# Run demo

# Training

# Evaluation