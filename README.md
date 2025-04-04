<h2>Tensorflow-Image-Segmentation-Left-Atrial (2025/04/04)</h2>

Sarah T. Arai<br>
Software Laboratory antillia.com<br><br>

This is the first experiment of Image Segmentation for Left-Atrial 
 based on 
the latest <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, 
and  <a href="https://drive.google.com/file/d/1VAySrFfcHS9LwqfdId-nL7pt03eLV9tm/view?usp=sharing">
Left-Atrial-ImageMask-Dataset.zip</a>, which was derived by us from 
<a href="https://www.kaggle.com/datasets/adarshsng/heart-mri-image-dataset-left-atrial-segmentation">
<b>
Heart MRI Image DataSet : Left Atrial Segmentation
</b>
</a>
<br>
<br>
<hr>
<b>Actual Image Segmentation for Images of 512x512 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>

<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/mini_test/images/la_003_1049.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/mini_test/masks/la_003_1049.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/mini_test_output/la_003_1049.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/mini_test/images/la_003_1065.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/mini_test/masks/la_003_1065.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/mini_test_output/la_003_1065.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/mini_test/images/la_007_1103.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/mini_test/masks/la_007_1103.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/mini_test_output/la_007_1103.jpg" width="320" height="auto"></td>
</tr>
</table>

<hr>
<br>
In this experiment, we used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this Left-AtrialSegmentation Model.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>

<br>

<h3>1. Dataset Citation</h3>
The dataset used here has been derived from the following web site:<br>

<a href="https://www.kaggle.com/datasets/adarshsng/heart-mri-image-dataset-left-atrial-segmentation">
<b>
Heart MRI Image DataSet : Left Atrial Segmentation
</b>
</a>
<br><br>
<b>About Dataset</b><br>
<b>Left Atrial Segmentation Challenge</b><br>
Authors: Catalina Tobon-Gomez (catactg@gmail.com) and Arjan Geers (ajgeers@gmail.com)<br>
<br>
<b>About</b><br>

This repository is associated with the Left Atrial Segmentation Challenge 2013 (LASC'13). LASC'13 was part of 
the STACOM'13 workshop, held in conjunction with MICCAI'13. Seven international research groups, 
comprising 11 algorithms, participated in the challenge.
<br>
For a detailed report, please refer to:

Tobon-Gomez C, Geers AJ, Peters, J, Weese J, Pinto K, Karim R, Ammar M, Daoudi A, Margeta J, Sandoval Z, <br>
Stender B, Zheng Y, Zuluaga, MA, Betancur J, Ayache N, Chikh MA, Dillenseger J-L, Kelm BM, Mahmoudi S, <br>
Ourselin S, Schlaefer A, Schaeffter T, Razavi R, Rhode KS.<br>
 Benchmark for Algorithms Segmenting the Left Atrium From 3D CT and MRI Datasets.<br>
  IEEE Transactions on Medical Imaging, 34(7):1460–1473, 2015.
<br>
<br>
<b>
License</b><br>
<a href="https://commission.europa.eu/legal-notice_en">
EU ODP Legal Notice
</a>
<br>
<h3>
<a id="2">
2 Left-Atrial ImageMask Dataset
</a>
</h3>
 If you would like to train this Left-Atrial Segmentation model by yourself,
 please download our 512x512 pixels dataset from the google drive  
<a href="https://drive.google.com/file/d/1VAySrFfcHS9LwqfdId-nL7pt03eLV9tm/view?usp=sharing">
Left-Atrial-ImageMask-Dataset.zip</a>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─Left-Atrial
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
On the derivation of this dataset, please refer to the following Python scripts:<br>
<li><a href="./generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a></li>
<li><a href="./generator/split_master.py">split_master.py.</a></li>
<br>
<br>
<b>Left-Atrial Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/Left-Atrial_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not enough to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorflowUNet Model
</h3>
 We trained Left-Atrial TensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorflowSlightlyFlexibleUNet/Left-Atrialand run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters</b> and large <b>base_kernels</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dilation       = (3,3)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Online augmentation</b><br>
Enabled our online augmentation tool. 
<pre>
[model]
model         = "TensorflowUNet"
generator     = True
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>


<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>

By using this epoch_change_infer callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (1,2,3)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/asset/epoch_change_infer_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (93,94,95)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/asset/epoch_change_infer_end.png" width="1024" height="auto"><br>
<br>

In this experiment, the training process was stopped at epoch 95  by EarlyStopping Callback.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/asset/train_console_output_at_epoch_95.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/eval/train_losses.png" width="520" height="auto"><br>
<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for Left-Atrial.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/asset/evaluate_console_output_at_epoch_95.png" width="720" height="auto">
<br><br>Image-Segmentation-Left-Atrial

<a href="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/evaluation.csv">evaluation.csv</a><br>
The loss (bce_dice_loss) to this Left-Atrial/test was low, but dice_coef not so high as shown below.
<br>
<pre>
loss,0.0631
dice_coef,0.8787
</pre>
<br>
<h3>5 Inference</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Left-Atrial.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/asset/mini_test_output.png" width="1024" height="auto"><br>

<br>
<hr>
<b>Enlarged images and masks </b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/mini_test/images/la_003_1049.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/mini_test/masks/la_003_1049.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/mini_test_output/la_003_1049.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/mini_test/images/la_003_1064.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/mini_test/masks/la_003_1064.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/mini_test_output/la_003_1064.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/mini_test/images/la_003_1097.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/mini_test/masks/la_003_1097.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/mini_test_output/la_003_1097.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/mini_test/images/la_004_1044.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/mini_test/masks/la_004_1044.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/mini_test_output/la_004_1044.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/mini_test/images/la_005_1052.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/mini_test/masks/la_005_1052.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/mini_test_output/la_005_1052.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/mini_test/images/la_007_1071.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/mini_test/masks/la_007_1071.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Left-Atrial/mini_test_output/la_007_1071.jpg" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>

<h3>
References
</h3>
<b>1. Deep Learning for Cardiac Image Segmentation: A Review</b><br>
Chen Chen, Chen Qin, Huaqi Qiu, Giacomo Tarroni,Jinming Duan,Wenjia Bai, Daniel Rueckert<br>

<a href="https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2020.00025/full">
https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2020.00025/full</a>
<br>
<br>
<b>2. Medical Image Analysis on Left Atrial LGE MRI for Atrial Fibrillation Studies: A Review</b>
<br>
Lei Li, Veronika A Zimmer, Julia A Schnabel, Xiahai Zhuang<br>
<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC7614005/">https://pmc.ncbi.nlm.nih.gov/articles/PMC7614005/</a>

<br>
<br>
<b>3. A Two-stage Method with a Shared 3D U-Net for Left Atrial Segmentation of Late Gadolinium-Enhanced MRI Images</b>
<br>
Jieyun Bai, Ruiyu Qiu, Jianyu Chen, Liyuan Wang, Lulu Li, Yanfeng Tian, Huijin Wang, Yaosheng Lu, Jichao Zhao
<br>
<a href="https://www.scienceopen.com/hosted-document?doi=10.15212/CVIA.2023.0039">
https://www.scienceopen.com/hosted-document?doi=10.15212/CVIA.2023.0039</a>
<br>
<br>
<b>4. Heart Segmentation — Neural Networks for Medical Imaging</b><br>
Matheus Ramos Parracho<br>
<a href="https://medium.com/@mathparracho/heart-segmentation-neural-networks-for-medical-imaging-a75138906ed1">
https://medium.com/@mathparracho/heart-segmentation-neural-networks-for-medical-imaging-a75138906ed1</a>

<br>
<br>
<b>5. Fully Automated 3D Cardiac MRI Localisation and Segmentation Using Deep Neural Networks </b><br>
Sulaiman Vesa, Andreas Maier, Nishant Ravikumar<br>
<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC8321054/">https://pmc.ncbi.nlm.nih.gov/articles/PMC8321054/</a>

<br>


