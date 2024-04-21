# YoloV3
________
YoloV3 Simplified for training on Colab with custom dataset. 

_A Collage of Training images_
![image](https://github.com/shylane/YoloV3/blob/master/output/train.png)


We have added a very 'smal' Coco sample imageset in the folder called smalcoco. This is to make sure you can run it without issues on Colab.

Full credit goes to [this](https://github.com/ultralytics/yolov3), and if you are looking for much more detailed explainiation and features, please refer to the original [source](https://github.com/ultralytics/yolov3). 

You'll need to download the weights from the original source. 
1. Create a folder called weights in the root (YoloV3) folder
2. Download from: https://drive.google.com/file/d/1vRDkpAiNdqHORTUImkrpD7kK_DkCcMus/view?usp=share_link
3. Place 'yolov3-spp-ultralytics.pt' file in the weights folder:
  * to save time, move the file from the above link to your GDrive
  * then drag and drop from your GDrive opened in Colab to weights folder
4. run this command
`python train.py --data data/smalcoco/smalcoco.data --batch 10 --cache --epochs 25 --nosave`

For custom dataset:
1. Clone this repo: https://github.com/miki998/YoloV3_Annotation_Tool
2. Follow the installation steps as mentioned in the repo. 
3. For the assignment, download 500 images of your unique object. 
4. Annotate the images using the Annotation tool. 
```
data
  --customdata
    --images/
      --img001.jpg
      --img002.jpg
      --...
    --labels/
      --img001.txt
      --img002.txt
      --...
    custom.data #data file
    custom.names #your class names
    custom.txt #list of name of the images you want your network to be trained on. Currently we are using same file for test/train
```
5. As you can see above you need to create **custom.data** file. For 1 class example, your file will look like this:
```
  classes=1
  train=data/customdata/custom.txt
  test=data/customdata/custom.txt 
  names=data/customdata/custom.names
```
6. As you it a poor idea to keep test and train data same, but the point of this repo is to get you up and running with YoloV3 asap. You'll probably do a mistake in writing to custom.txt file. This is how our file looks like (please note the .s and /s):
```
./data/customdata/images/img001.jpg
./data/customdata/images/img002.jpg
./data/customdata/images/img003.jpg
...
```
7. You need to add custom.names file as you can see above. For our example, we downloaded images of Walle. Our custom.names file look like this:
```
walle
```
8. Walle above will have a class index of 0. 
9. For COCO's 80 classes, VOLOv3's output vector has 255 dimensions ( (4+1+80)*3). Now we have 1 class, so we would need to change it's architecture.
10. Copy the contents of 'yolov3-spp.cfg' file to a new file called 'yolov3-custom.cfg' file in the data/cfg folder. 
11. Search for 'filters=255' (you should get entries entries). Change 255 to 18 = (4+1+1)*3
12. Search for 'classes=80' and change all three entries to 'classes=1'
13. Since you are lazy (probably), you'll be working with very few samples. In such a case it is a good idea to change:
  * burn_in to 100
  * max_batches to 5000
  * steps to 4000,4500
14. Don't forget to perform the weight file steps mentioned in the sectio above. 
15. Run this command `python train.py --data data/customdata/custom.data --batch 10 --cache --cfg cfg/yolov3-custom.cfg --epochs 3 --nosave`

As you can see in the collage image above, a lot is going on, and if you are creating a set of say 500 images, you'd get a bonanza of images via default augmentations being performed. 

*As per instruction above, I have added the class 'mirror' as I am trying to detect mirrors in the images. The concerned folder in data is marked 'shmirror'.*

**Results**
After training for 100 Epochs, results are better than my expectation!
Below are some test images on which training was not conducted.

![image](https://github.com/shylane/YoloV3/blob/master/output/76520898_medium.jpg)
1 correct detection!

![image](https://github.com/shylane/YoloV3/blob/master/output/73493454_medium.jpg)
1 missed detection!

![image](https://github.com/shylane/YoloV3/blob/master/output/main-qimg-4a2e6fa6db96b5f95e964d146a74f59d-lq.jpg)
1 partial detection!

![image](https://github.com/shylane/YoloV3/blob/master/output/a178327161977bed19c26881b8753d5a9949c8d0.jpg)
1 correct detection!

![image](https://github.com/shylane/YoloV3/blob/master/output/tips-for-a-mirror-facing-the-front-door-1274597-hero-9d27ddf9e70a4bcab825fb9521efff61.jpg)
1 correct detection and 2 false positives!

![image](https://github.com/shylane/YoloV3/blob/master/output/16843991316465e41b32c382_58791889.jpg)
1 correct detection (albeit the score does not appear) and 1 missed detection!

![image](https://github.com/shylane/YoloV3/blob/master/output/Arch2O-decorating-with-mirrors-25-creative-ways-to-reflect-light-and-space-1.jpg)
2 correct detections!! (completely overlapping for same object)

![image](https://github.com/shylane/YoloV3/blob/master/output/16843990626465e3d64c5d71_99978625.jpg)
1 correct detection!

![image](https://github.com/shylane/YoloV3/blob/master/output/3-AIS-1.jpg)
1 correct detection and 1 false positive!

![image](https://github.com/shylane/YoloV3/blob/master/output/How-is-the-Troxler-effect-applicable-to-mirrors-1.jpg)
1 correct detection!

![image](https://github.com/shylane/YoloV3/blob/master/output/funny-pics-people-selling-mirrors-12-6613f76c47457__700.jpg)
1 correct detection! (score does not appear)


**Training Log**
(.venv) ubuntu@ip-172-31-14-70:/mnt/files/teShting/yoloa/YoloV3-masterTSAI$ python train.py --data data/shmirror/shmirror.data --batch 16 --cache --cfg cfg/yolov3-custom.cfg --epochs 100 --nosave
Namespace(epochs=100, batch_size=16, accumulate=4, cfg='cfg/yolov3-custom.cfg', data='data/shmirror/shmirror.data', multi_scale=False, img_size=[512], rect=False, resume=False, nosave=True, notest=False, evolve=False, bucket='', cache_images=True, weights='weights/yolov3-spp-ultralytics.pt', name='', device='', adam=False, single_cls=False)
Using CUDA device0 _CudaDeviceProperties(name='NVIDIA A10G', total_memory=22723MB)
           device1 _CudaDeviceProperties(name='NVIDIA A10G', total_memory=22723MB)
           device2 _CudaDeviceProperties(name='NVIDIA A10G', total_memory=22723MB)
           device3 _CudaDeviceProperties(name='NVIDIA A10G', total_memory=22723MB)
           device4 _CudaDeviceProperties(name='NVIDIA A10G', total_memory=22723MB)
           device5 _CudaDeviceProperties(name='NVIDIA A10G', total_memory=22723MB)
           device6 _CudaDeviceProperties(name='NVIDIA A10G', total_memory=22723MB)
           device7 _CudaDeviceProperties(name='NVIDIA A10G', total_memory=22723MB)

Run 'tensorboard --logdir=runs' to view tensorboard at http://localhost:6006/
WARNING: smart bias initialization failure.
WARNING: smart bias initialization failure.
WARNING: smart bias initialization failure.
Model Summary: 225 layers, 6.25733e+07 parameters, 6.25733e+07 gradients
/mnt/files/teShting/.venv/lib/python3.11/site-packages/torch/optim/lr_scheduler.py:143: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
Caching labels:   0%|                                                                                                                 | 0/106 [00:00<?, ?it/s]missing labels for image ./data/shmirror/images/gettyimages-526295555-2048x2048.jpg
missing labels for image ./data/shmirror/images/gettyimages-1489392969-2048x2048.jpg
Caching labels (104 found, 2 missing, 0 empty, 0 duplicate, for 106 images): 100%|████████████████████████████████████████| 106/106 [00:00<00:00, 9846.66it/s]
Caching images (0.1GB): 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 106/106 [00:00<00:00, 162.87it/s]
Caching labels:   0%|                                                                                                                 | 0/106 [00:00<?, ?it/s]missing labels for image ./data/shmirror/images/gettyimages-526295555-2048x2048.jpg
missing labels for image ./data/shmirror/images/gettyimages-1489392969-2048x2048.jpg
Caching labels (104 found, 2 missing, 0 empty, 0 duplicate, for 106 images): 100%|███████████████████████████████████████| 106/106 [00:00<00:00, 10199.27it/s]
Caching images (0.1GB): 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 106/106 [00:00<00:00, 115.87it/s]
Image sizes 512 - 512 train, 512 test
Using 8 dataloader workers
Starting training for 100 epochs...

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
  0%|                                                                                                                                   | 0/7 [00:00<?, ?it/s][rank0]:[W reducer.cpp:1360] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
/mnt/files/teShting/yoloa/YoloV3-masterTSAI/utils/utils.py:375: UserWarning: The torch.cuda.*DtypeTensor constructors are no longer recommended. It's best to use methods such as torch.tensor(data, dtype=*, device='cuda') to create tensors. (Triggered internally at ../torch/csrc/tensor/python_tensor.cpp:83.)
  lcls, lbox, lobj = ft([0]), ft([0]), ft([0])
/mnt/files/teShting/.venv/lib/python3.11/site-packages/torch/cuda/memory.py:440: FutureWarning: torch.cuda.memory_cached has been renamed to torch.cuda.memory_reserved
  warnings.warn(
      0/99     13.2G      6.24        88         0      94.2        26       512: 100%|█████████████████████████████████████████| 7/7 [00:06<00:00,  1.00it/s]
/mnt/files/teShting/.venv/lib/python3.11/site-packages/torch/functional.py:507: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3549.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:03<00:00,  1.79it/s]
                 all       106       132   0.00129    0.0985  0.000938   0.00256

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      1/99     13.2G      5.82      7.78         0      13.6        20       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  3.65it/s]
                 all       106       132         0         0  0.000876         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      2/99     13.2G      6.69      2.86         0      9.55        21       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.38it/s]
                 all       106       132         0         0   0.00135         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      3/99     13.2G      6.23      3.39         0      9.62        21       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.86it/s]
                 all       106       132         0         0   0.00171         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      4/99     13.2G      5.35      3.91         0      9.26        25       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  6.26it/s]
                 all       106       132         0         0   0.00512         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      5/99     13.2G      4.87      4.32         0      9.19        26       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  6.92it/s]
                 all       106       132         0         0    0.0225         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      6/99     13.2G      5.12       4.5         0      9.62        22       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  6.93it/s]
                 all       106       132         0         0    0.0334         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      7/99     13.2G      4.29      4.72         0      9.02        20       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  6.78it/s]
                 all       106       132         0         0    0.0957         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      8/99     13.2G      4.87      4.17         0      9.04        18       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  6.12it/s]
                 all       106       132         0         0     0.313         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      9/99     13.2G      4.33      3.37         0       7.7        19       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.89it/s]
                 all       106       132         1   0.00758     0.384     0.015

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     10/99     13.2G      4.42      4.13         0      8.55        22       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.55it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.41it/s]
                 all       106       132     0.842    0.0809     0.437     0.148

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     11/99     13.2G      4.85      3.31         0      8.15        25       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.88it/s]
                 all       106       132     0.592     0.352     0.438     0.442

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     12/99     13.2G      4.55      2.92         0      7.47        25       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.89it/s]
                 all       106       132     0.209     0.674     0.355     0.319

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     13/99     13.2G      4.39      2.59         0      6.98        25       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.84it/s]
                 all       106       132     0.151     0.742     0.493     0.251

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     14/99     13.2G      5.05      2.56         0      7.61        37       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.83it/s]
                 all       106       132     0.102      0.78     0.298      0.18

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     15/99     13.2G      4.88       1.9         0      6.79        19       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.86it/s]
                 all       106       132     0.167     0.788     0.499     0.275

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     16/99     13.2G      5.54      1.52         0      7.05        25       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.82it/s]
                 all       106       132      0.29     0.682     0.417     0.407

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     17/99     13.2G      4.06      1.76         0      5.82        24       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.84it/s]
                 all       106       132     0.198     0.629     0.234     0.301

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     18/99     13.2G       4.3      1.59         0      5.89        38       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.81it/s]
                 all       106       132     0.177     0.447     0.139     0.254

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     19/99     13.2G      3.77      1.33         0       5.1        22       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.80it/s]
                 all       106       132     0.054     0.318    0.0435    0.0923

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     20/99     13.2G       3.7      1.33         0      5.03        32       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.80it/s]
                 all       106       132    0.0948     0.311    0.0773     0.145

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     21/99     13.2G      3.66      1.25         0      4.91        25       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.82it/s]
                 all       106       132     0.143       0.5     0.137     0.223

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     22/99     13.2G      3.68      1.14         0      4.82        19       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.82it/s]
                 all       106       132     0.555     0.606     0.601     0.579

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     23/99     13.2G      3.32      1.16         0      4.48        26       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.85it/s]
                 all       106       132      0.54     0.561     0.529      0.55

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     24/99     13.2G      3.04      1.01         0      4.05        18       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.71it/s]
                 all       106       132     0.302     0.735     0.285     0.428

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     25/99     13.2G      3.47     0.854         0      4.33        15       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.79it/s]
                 all       106       132      0.31     0.689     0.309     0.428

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     26/99     13.2G      3.49     0.819         0      4.31        15       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.83it/s]
                 all       106       132     0.318     0.811      0.43     0.457

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     27/99     13.2G      3.09      1.06         0      4.15        25       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.84it/s]
                 all       106       132      0.52     0.848     0.639     0.645

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     28/99     13.2G      3.99     0.802         0      4.79        15       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.85it/s]
                 all       106       132     0.313     0.826     0.356     0.454

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     29/99     13.2G       3.7      0.96         0      4.66        33       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.84it/s]
                 all       106       132      0.38     0.813     0.361     0.518

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     30/99     13.2G      3.17     0.887         0      4.06        19       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.79it/s]
                 all       106       132     0.346     0.836     0.352     0.489

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     31/99     13.2G      3.62     0.862         0      4.49        19       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.85it/s]
                 all       106       132     0.391     0.833      0.38     0.532

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     32/99     13.2G      3.34     0.841         0      4.19        23       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.84it/s]
                 all       106       132     0.657     0.879     0.807     0.752

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     33/99     13.2G      3.63     0.881         0      4.51        28       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.85it/s]
                 all       106       132     0.721     0.856     0.847     0.783

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     34/99     13.2G      3.09     0.906         0         4        27       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.85it/s]
                 all       106       132     0.423     0.939     0.582     0.584

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     35/99     13.2G      3.08     0.819         0       3.9        20       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.84it/s]
                 all       106       132     0.663     0.886     0.785     0.759

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     36/99     13.2G      3.71     0.791         0       4.5        24       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.83it/s]
                 all       106       132     0.518     0.904     0.686     0.658

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     37/99     13.2G      3.62     0.755         0      4.37        21       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.87it/s]
                 all       106       132     0.567     0.932     0.732     0.705

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     38/99     13.2G      3.26     0.957         0      4.22        22       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.86it/s]
                 all       106       132     0.658     0.939     0.759     0.774

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     39/99     13.2G      2.63     0.858         0      3.48        21       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.84it/s]
                 all       106       132     0.617     0.947      0.78     0.747

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     40/99     13.2G      2.89     0.749         0      3.64        18       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.86it/s]
                 all       106       132      0.55     0.924     0.738      0.69

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     41/99     13.2G      2.85     0.701         0      3.55        26       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.86it/s]
                 all       106       132     0.739     0.835     0.863     0.784

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     42/99     13.2G      2.88     0.788         0      3.66        32       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.84it/s]
                 all       106       132     0.496     0.902     0.517      0.64

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     43/99     13.2G      2.83     0.811         0      3.64        19       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.26it/s]
                 all       106       132     0.522     0.947     0.645     0.673

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     44/99     13.2G       3.5     0.713         0      4.22        21       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.78it/s]
                 all       106       132     0.657     0.909     0.714     0.763

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     45/99     13.2G      2.99     0.734         0      3.72        14       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.79it/s]
                 all       106       132     0.655     0.892     0.724     0.756

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     46/99     13.2G       4.3     0.735         0      5.04        22       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.84it/s]
                 all       106       132     0.787     0.924     0.893      0.85

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     47/99     13.2G      3.57     0.644         0      4.22        25       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.85it/s]
                 all       106       132     0.703     0.939     0.843     0.804

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     48/99     13.2G         3      0.67         0      3.67        27       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.87it/s]
                 all       106       132     0.718     0.917     0.787     0.805

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     49/99     13.2G      3.59     0.771         0      4.36        20       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.82it/s]
                 all       106       132     0.834     0.932     0.928      0.88

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     50/99     13.2G      2.95     0.756         0       3.7        17       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.85it/s]
                 all       106       132     0.791      0.92      0.85     0.851

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     51/99     13.2G      2.52     0.695         0      3.22        21       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.59it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.88it/s]
                 all       106       132     0.715     0.929     0.756     0.808

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     52/99     13.2G      3.09     0.762         0      3.85        28       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.87it/s]
                 all       106       132      0.69     0.928     0.721     0.792

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     53/99     13.2G      2.62     0.763         0      3.38        29       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.88it/s]
                 all       106       132     0.782     0.947     0.894     0.857

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     54/99     13.2G      2.76     0.606         0      3.37        21       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.55it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.86it/s]
                 all       106       132     0.726     0.909     0.862     0.807

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     55/99     13.2G      2.93     0.641         0      3.57        26       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.83it/s]
                 all       106       132     0.681     0.886     0.784      0.77

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     56/99     13.2G      2.23       0.8         0      3.03        26       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.86it/s]
                 all       106       132     0.532      0.93     0.683     0.677

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     57/99     13.2G      2.84      0.67         0      3.51        17       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.88it/s]
                 all       106       132     0.609     0.939     0.843     0.739

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     58/99     13.2G      2.62     0.698         0      3.32        20       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.32it/s]
                 all       106       132     0.687     0.947     0.824     0.796

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     59/99     13.2G      2.78     0.795         0      3.57        29       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.65it/s]
                 all       106       132     0.756     0.955     0.902     0.844

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     60/99     13.2G      2.45     0.684         0      3.13        30       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.79it/s]
                 all       106       132     0.841      0.96     0.946     0.897

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     61/99     13.2G      2.96     0.586         0      3.55        21       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.84it/s]
                 all       106       132     0.769     0.962     0.888     0.855

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     62/99     13.2G      2.58     0.595         0      3.17        23       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.85it/s]
                 all       106       132     0.666     0.955      0.69     0.785

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     63/99     13.2G      2.99     0.516         0       3.5        19       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.84it/s]
                 all       106       132     0.726     0.945     0.738     0.822

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     64/99     13.2G      2.48     0.688         0      3.17        17       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.87it/s]
                 all       106       132     0.904     0.932     0.908     0.918

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     65/99     13.2G      2.88     0.621         0       3.5        21       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.87it/s]
                 all       106       132     0.964     0.947     0.975     0.955

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     66/99     13.2G      2.28     0.639         0      2.92        26       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.55it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.84it/s]
                 all       106       132     0.962     0.961     0.972     0.962

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     67/99     13.2G         2     0.604         0      2.61        18       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.86it/s]
                 all       106       132     0.939     0.955      0.97     0.947

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     68/99     13.2G       2.4     0.551         0      2.95        25       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.84it/s]
                 all       106       132     0.907     0.939     0.912     0.923

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     69/99     13.2G       1.8     0.501         0       2.3        23       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.88it/s]
                 all       106       132     0.862     0.962     0.858     0.909

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     70/99     13.2G      2.47     0.515         0      2.99        24       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.86it/s]
                 all       106       132     0.875     0.962     0.906     0.917

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     71/99     13.2G      1.91     0.567         0      2.48        32       512:  43%|█████████████████▌                       | 3/7 [00:01<00:01,  2.43it/s]
Model Bias Summary:    layer        regression        objectness    classification
                          89      -0.11+/-0.26     -10.96+/-2.57       0.02+/-0.01 
                         101       0.13+/-0.18     -12.23+/-1.27       0.04+/-0.00 
                         113       0.20+/-0.24     -11.17+/-2.07      -0.00+/-0.01 
     71/99     13.2G      1.98     0.535         0      2.52        27       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.87it/s]
                 all       106       132     0.936     0.962     0.959     0.949

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     72/99     13.2G      2.09     0.553         0      2.65        30       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.88it/s]
                 all       106       132     0.937     0.977      0.96     0.957

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     73/99     13.2G      1.77      0.48         0      2.25        22       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.87it/s]
                 all       106       132     0.913     0.977     0.979     0.944

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     74/99     13.2G      1.78     0.474         0      2.25        17       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.85it/s]
                 all       106       132     0.916     0.977     0.969     0.946

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     75/99     13.2G      1.68     0.502         0      2.18        25       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.59it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.84it/s]
                 all       106       132     0.927     0.977     0.978     0.951

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     76/99     13.2G      2.12     0.446         0      2.57        14       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.86it/s]
                 all       106       132     0.921     0.978     0.984     0.949

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     77/99     13.2G      2.12     0.463         0      2.58        22       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.78it/s]
                 all       106       132      0.91     0.977     0.983     0.942

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     78/99     13.2G      2.03     0.566         0       2.6        26       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.55it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.84it/s]
                 all       106       132     0.923     0.985     0.986     0.953

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     79/99     13.2G      1.73     0.477         0       2.2        24       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.84it/s]
                 all       106       132     0.922     0.985     0.986     0.952

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     80/99     13.2G      1.63     0.484         0      2.12        21       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.79it/s]
                 all       106       132     0.916     0.985     0.986     0.949

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     81/99     13.2G      1.75     0.426         0      2.18        20       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.86it/s]
                 all       106       132     0.917     0.985     0.985      0.95

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     82/99     13.2G      1.59     0.496         0      2.08        25       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.80it/s]
                 all       106       132     0.942     0.985     0.987     0.963

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     83/99     13.2G         2     0.429         0      2.43        23       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.83it/s]
                 all       106       132     0.949     0.982     0.987     0.965

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     84/99     13.2G      1.43     0.503         0      1.93        25       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.55it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.42it/s]
                 all       106       132     0.956     0.977     0.988     0.966

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     85/99     13.2G      1.92     0.529         0      2.45        19       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.55it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.68it/s]
                 all       106       132     0.955     0.985     0.988      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     86/99     13.2G      1.51     0.451         0      1.96        21       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.74it/s]
                 all       106       132      0.96     0.985     0.988     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     87/99     13.2G      1.79     0.491         0      2.28        25       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.85it/s]
                 all       106       132      0.96     0.985     0.988     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     88/99     13.2G      1.53     0.514         0      2.05        19       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.84it/s]
                 all       106       132     0.959     0.985     0.989     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     89/99     13.2G      1.33     0.422         0      1.76        18       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.54it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.38it/s]
                 all       106       132     0.958     0.985     0.989     0.971

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     90/99     13.2G      1.75     0.433         0      2.19        23       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.82it/s]
                 all       106       132     0.958     0.985     0.989     0.971

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     91/99     13.2G      2.07     0.397         0      2.47        21       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.87it/s]
                 all       106       132     0.958     0.985     0.989     0.971

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     92/99     13.2G      1.69     0.438         0      2.13        26       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.55it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.85it/s]
                 all       106       132      0.96     0.985     0.989     0.972

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     93/99     13.2G      2.03       0.4         0      2.42        22       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.87it/s]
                 all       106       132     0.961     0.985     0.989     0.973

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     94/99     13.2G      1.54     0.455         0      1.99        23       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.55it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.85it/s]
                 all       106       132     0.961     0.985     0.989     0.973

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     95/99     13.2G      1.53     0.372         0       1.9        17       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.86it/s]
                 all       106       132     0.973     0.985     0.989     0.979

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     96/99     13.2G      1.36     0.456         0      1.81        27       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.86it/s]
                 all       106       132     0.986     0.985     0.989     0.986

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     97/99     13.2G      1.54     0.467         0         2        29       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.86it/s]
                 all       106       132     0.988     0.985     0.989     0.987

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     98/99     13.2G      1.27     0.384         0      1.65        21       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.85it/s]
                 all       106       132     0.992     0.975     0.989     0.983

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     99/99     13.2G      1.54     0.362         0       1.9        23       512: 100%|█████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|█████████████████████████████████████████| 7/7 [00:01<00:00,  5.86it/s]
                 all       106       132     0.987      0.97     0.988     0.978
100 epochs completed in 0.118 hours.

![image](https://github.com/shylane/YoloV3/blob/master/results.png)