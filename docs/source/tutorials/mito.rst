Mitochondria Segmentation
=========================

Introduction
------------

`Mitochondria <https://en.wikipedia.org/wiki/Mitochondrion>`__ are the primary energy providers for cell activities, thus essential for metabolism. Quantification of the size and geometry of mitochondria is not only crucial to basic neuroscience research, but also informative to the clinical studies of several diseases including bipolar disorder and diabetes.

This tutorial has two parts. In the first part, you will learn how to make **pixel-wise class prediction** on the widely used benchmark dataset released by `Lucchi et al. <https://ieeexplore.ieee.org/document/6619103>`__ in 2012. In the second part, you will learn how to predict the **instance masks** of individual mitochondrion from the large-scale MitoEM dataset released by `Wei et al. <https://donglaiw.github.io/paper/2020_miccai_mitoEM.pdf>`__ in 2020.

Semantic Segmentation
---------------------

This section provides step-by-step guidance for mitochondria segmentation with the EM benchmark datasets released by `Lucchi et al. (2012) <https://cvlab.epfl.ch/research/page-90578-en-html/research-medical-em-mitochondria-index-php/>`__. We approach the task as a **semantic segmentation** task and predict the mitochondria pixels with encoder-decoder ConvNets similar to the models used for affinity prediction in `neuron segmentation <neuron.html>`_. The evaluation of the mitochondria segmentation results is based on the F1 score and Intersection over Union (IoU).

    .. note:: Unlike other EM connectomics datasets used in these tutorials, the dataset released by Lucchi et al. is an isotropic dataset, which means the spatial resolution along all three axes is the same. Therefore a completely 3D U-Net and data augmentation along x-z and y-z planes (alongside the standard practice of applying augmentation along the x-y plane) is applied.

The scripts needed for this tutorial can be found at ``pytorch_connectomics/scripts/main.py``. The corresponding configuraion file is found at ``configs/Lucchi-Mitochondria.yaml``.

.. figure:: ../_static/img/lucchi_qual.png
    :align: center
    :width: 800px

A benchmark model's qualitative results on the Lucchi dataset, presented without any post-processing

1 - Get the data
^^^^^^^^^^^^^^^^

Download the dataset from our server:

.. code-block:: bash

    mkdir -p datasets/Lucchi
    wget -q --show-progress -O lucchi.zip http://rhoana.rc.fas.harvard.edu/dataset/lucchi.zip
    unzip -d datasets/Lucchi datasets/lucchi.zip
    rm lucchi.zip

For description of the data please check `the author page <https://www.epfl.ch/labs/cvlab/data/data-em/>`_.

2 - Run training
^^^^^^^^^^^^^^^^
.. code-block:: bash

    source activate pytc
    python scripts/main.py --config-file configs/Lucchi-Mitochondria.yaml

3 (*optional*) - Visualize the training progress
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    tensorboard --logdir outputs/Lucchi_UNet/

4 - Inference on test data
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    source activate pytc
    python scripts/main.py \
    --config-file configs/Lucchi-Mitochondria.yaml --inference \
    --checkpoint outputs/Lucchi_UNet/checkpoint_100000.pth.tar

5 - Run evaluation
^^^^^^^^^^^^^^^^^^

Since the ground-truth label of the test set is public, we can run the evaluation locally. A script is provided:

.. code-block:: bash

    python scripts/tutorials/lucchi_eval.py

..

    .. note:: Our pretained model achieves a foreground IoU and IoU of **0.892** and **0.943** on the test set, respectively. The results are better or on par with state-of-the-art approaches. Please check `BENCHMARK.md <https://github.com/zudi-lin/pytorch_connectomics/blob/master/BENCHMARK.md>`_  for detailed performance comparison and the pre-trained models.

Instance Segmentation
---------------------

This section provides step-by-step guidance for mitochondria segmentation with the `MitoEM <https://donglaiw.github.io/page/mitoEM/index.html>`_ dataset. We approach the task as a 3D **instance segmentation** task and provide three different confiurations of the model output. We utilize the ``UNet3D`` model similar to the one used in `neuron segmentation <neuron.html>`_. The evaluation of the segmentation results is based on the AP-75 (average precision with an IoU threshold of 0.75).

.. figure:: ../_static/img/mito_complex.png
    :align: center
    :width: 800px

Complex mitochondria in the MitoEM dataset:(**a**) mitochondria-on-a-string (MOAS), and (**b**) dense tangle of touching instances. Those challenging cases are prevalent but not covered in previous datasets.

   .. note:: Since the dataset is very large and can not be directly loaded into memory, we designed the :class:`connectomics.data.dataset.TileDataset` class that only loads part of the whole volume each time by opening involved ``PNG`` or ``TIFF`` images.

..

    .. note:: A benchmark evaluation with validation data and pretrained weights is provided for users at `this Colab notebook <https://colab.research.google.com/drive/1ll3a0F2VbmmKBTQ_RBqSrEsU3gpTUdam>`_.

1 - Dataset introduction
^^^^^^^^^^^^^^^^^^^^^^^^

The dataset is publicly available at the `MitoEM Challenge <https://mitoem.grand-challenge.org/>`_ page. To provide a brief description of the dataset:

- ``im``: includes 1,000 single-channel ``*.png`` files (**4096x4096**) of raw EM images (with a spatial resolution of **30x8x8** nm).
  The 1,000 images are splited into 400, 100 and 500 slices for training, validation and inference, respectively.

- ``mito_train/``: includes 400 single-channel ``*.png`` files (**4096x4096**) of instance labels for training. Similarly, the ``mito_val/`` folder contains 100 slices for validation. The ground-truth annotation of the test set (rest 500 slices) is not publicly provided but can be evaluated online at the `MitoEM challenge page <https://mitoem.grand-challenge.org>`_.

2 - Get the data
^^^^^^^^^^^^^^^^

.. code-block:: bash
  
  mkdir -p datasets/MitoEM
  wget -q --show-progress -O datasets/MitoEM/EM30-R-im.zip https://huggingface.co/datasets/pytc/EM30/resolve/main/EM30-R-im.zip?download=true
  unzip -q datasets/MitoEM/EM30-R-im.zip -d datasets/MitoEM/EM30-R-im
  rm -r datasets/MitoEM/EM30-R-im/__MACOSX
  rm datasets/MitoEM/EM30-R-im.zip
  wget -q --show-progress -O datasets/MitoEM/mito_val.zip https://huggingface.co/datasets/pytc/MitoEM/resolve/main/EM30-R-mito-train-val-v2.zip?download=true
  unzip -q datasets/MitoEM/mito_val.zip -d datasets/MitoEM/EM30-R-val
  rm datasets/MitoEM/mito_val.zip

3 - Model configuration
^^^^^^^^^^^^^^^^^^^^^^^

Multiple ``*.yaml`` configuration files are provided at ``configs/MitoEM`` for different learning targets:

- ``MitoEM-R-A.yaml``: output 3 channels for predicting the affinty between voxels.

- ``MitoEM-R-AC.yaml``: output 4 channels for predicting both affinity and instance contour.

- ``MitoEM-R-BC.yaml``: output 2 channels for predicting both the binary foreground mask and instance contour.

The lattermost configuration achieves the best overall performance according to our `experiments <https://donglaiw.github.io/paper/2020_miccai_mitoEM.pdf>`_. This tutorial will move forward using this configuration file.

4 - Run training
^^^^^^^^^^^^^^^^

.. code-block:: bash

    python -u scripts/main.py \
    --config-base configs/MitoEM/MitoEM-R-Base.yaml \
    --config-file configs/MitoEM/MitoEM-R-BC.yaml

..

5 (*optional*) - Visualize the training progress
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    tensorboard --logdir outputs/MitoEM_R_BC/

6 - Run inference
^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python -u scripts/main.py \
    --config-base configs/MitoEM/MitoEM-R-Base.yaml \
    --config-file configs/MitoEM/MitoEM-R-BC.yaml --inference \
    --checkpoint outputs/MitoEM_R_BC/checkpoint_100000.pth.tar

6 - Post-process
^^^^^^^^^^^^^^^^

The post-processing step requires merging output volumes and applying watershed segmentation. As mentioned before, the dataset is very large and cannot be directly loaded into memory for processing. Therefore our code run prediction on smaller chunks sequentially, which produces multiple ``*.h5`` files with the coordinate information. To merge the chunks into a single volume and apply the segmentation algorithm:

.. code-block:: python

    import glob
    import numpy as np
    from connectomics.data.utils import readvol
    from connectomics.utils.process import bc_watershed

    output_files = 'outputs/MitoEM_R_BC/test/*.h5' # output folder with chunks
    chunks = glob.glob(output_files)
    
    # Mitochondria Segmentation 
    vol_shape = (2, 500, 4096, 4096) # MitoEM test set
    pred = np.ones(vol_shape, dtype=np.uint8)
    for x in chunks:
        pos = x.strip().split("/")[-1]
        print("process chunk: ", pos)
        pos = pos.split("_")[1].split("-")
        pos = list(map(int, pos))
        chunk = readvol(x)
        pred[:, pos[0]:pos[1], pos[2]:pos[3], pos[4]:pos[5]] = chunk

    # This function process the array in numpy.float64 format.
    # Please allocate enough memory for processing.
    segm = bc_watershed(pred, thres1=0.85, thres2=0.6, thres3=0.8, thres_small=1024)

The generated segmentation map should be ready for submission to the `MitoEM <https://mitoem.grand-challenge.org/>`_ challenge website for evaluation. Please note that this tutorial only outlines training on **MitoEM-Rat** subset. Results on the **MitoEM-Human** subset, which can be generated using a similar pipeline as above, also need to be provided for online evaluation.
