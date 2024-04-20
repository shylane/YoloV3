# YoloV3
________
YoloV3 Simplified for training on Colab with custom dataset. 

_A Collage of Training images_
![image](https://github.com/shylane/YoloV3/blob/master/output/train_batch0.png)


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

*As per instruction above, I have added the class 'mirror' as I am trying to detect mirrors in the images. the concerned folder in data is marked 'shmirror'.*

**Results**
After training for 100 Epochs, results are better than my expectation!
Below are some test images on which training was not conducted.

![image](https://github.com/shylane/YoloV3/blob/master/output/76520898_medium.jpg)
Correctly detected!
![image](https://github.com/shylane/YoloV3/blob/master/output/73493454_medium.jpg)
Missed detection!
![image](https://github.com/shylane/YoloV3/blob/master/output/main-qimg-4a2e6fa6db96b5f95e964d146a74f59d-lq.jpg)
Missed detection!
![image](https://github.com/shylane/YoloV3/blob/master/output/a178327161977bed19c26881b8753d5a9949c8d0.jpg)
Missed detection!
![image](https://github.com/shylane/YoloV3/blob/master/output/tips-for-a-mirror-facing-the-front-door-1274597-hero-9d27ddf9e70a4bcab825fb9521efff61.jpg)
One correct detection and one false detection!
![image](https://github.com/shylane/YoloV3/blob/master/output/16843991316465e41b32c382.58791889.jpg)
One correctly detected (albeit the score does not appear) and missed detecting one!
![image](https://github.com/shylane/YoloV3/blob/master/output/Arch2O-decorating-with-mirrors-25-creative-ways-to-reflect-light-and-space-1.jpg)
Correctly detected!
![image](https://github.com/shylane/YoloV3/blob/master/output/16843990626465e3d64c5d71.99978625.jpg)
Correctly detected!
![image](https://github.com/shylane/YoloV3/blob/master/output/3-AIS-1.jpg)
Correctly detected!
![image](https://github.com/shylane/YoloV3/blob/master/output/How-is-the-Troxler-effect-applicable-to-mirrors-1.jpg)
Correctly detected!
![image](https://github.com/shylane/YoloV3/blob/master/output/funny-pics-people-selling-mirrors-12-6613f76c47457__700.jpg)
Correctly detected!


**Training Log**
      (.venv) ubuntu@ip-172-31-14-70:/mnt/files/teShting/yoloa/YoloV3-masterTSAI$ python train.py --data data/shmirror/shmirror.data --batch 16 --cache --cfg cfg/yolov3-shmirror.cfg --epochs 100 --nosave
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
Caching labels:   0%|                                                                                                                           | 0/106 [00:00<?, ?it/s]missing labels for image ./data/shmirror/images/TwU7Gk9nKEkd3Ytzb3oYYF-768-80.jpg.jpg
missing labels for image ./data/shmirror/images/gettyimages-526295555-2048x2048.jpg
missing labels for image ./data/shmirror/images/gettyimages-1489392969-2048x2048.jpg
missing labels for image ./data/shmirror/images/CcoCPCkPrr8hZKQ3pqTbCF-768-80.jpg.jpg
missing labels for image ./data/shmirror/images/TSvCX5CmvMxzBt8JqMyU2F-768-80.jpg.jpg
missing labels for image ./data/shmirror/images/PA5cUo5DG47dLKeRfoL78G-768-80.jpg.jpg
Caching labels (100 found, 6 missing, 0 empty, 0 duplicate, for 106 images): 100%|█████████████████████████████████████████████████| 106/106 [00:00<00:00, 10051.69it/s]
Caching images (0.1GB): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 106/106 [00:00<00:00, 162.62it/s]
Caching labels:   0%|                                                                                                                           | 0/106 [00:00<?, ?it/s]missing labels for image ./data/shmirror/images/gettyimages-526295555-2048x2048.jpg
missing labels for image ./data/shmirror/images/gettyimages-1489392969-2048x2048.jpg
missing labels for image ./data/shmirror/images/TSvCX5CmvMxzBt8JqMyU2F-768-80.jpg.jpg
missing labels for image ./data/shmirror/images/PA5cUo5DG47dLKeRfoL78G-768-80.jpg.jpg
missing labels for image ./data/shmirror/images/CcoCPCkPrr8hZKQ3pqTbCF-768-80.jpg.jpg
missing labels for image ./data/shmirror/images/TwU7Gk9nKEkd3Ytzb3oYYF-768-80.jpg.jpg
Caching labels (100 found, 6 missing, 0 empty, 0 duplicate, for 106 images): 100%|█████████████████████████████████████████████████| 106/106 [00:00<00:00, 10054.42it/s]
Caching images (0.1GB): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 106/106 [00:00<00:00, 115.29it/s]
Image sizes 512 - 512 train, 512 test
Using 8 dataloader workers
Starting training for 100 epochs...

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
  0%|                                                                                                                                             | 0/7 [00:00<?, ?it/s][rank0]:[W reducer.cpp:1360] Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass. This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
/mnt/files/teShting/yoloa/YoloV3-masterTSAI/utils/utils.py:375: UserWarning: The torch.cuda.*DtypeTensor constructors are no longer recommended. It's best to use methods such as torch.tensor(data, dtype=*, device='cuda') to create tensors. (Triggered internally at ../torch/csrc/tensor/python_tensor.cpp:83.)
  lcls, lbox, lobj = ft([0]), ft([0]), ft([0])
/mnt/files/teShting/.venv/lib/python3.11/site-packages/torch/cuda/memory.py:440: FutureWarning: torch.cuda.memory_cached has been renamed to torch.cuda.memory_reserved
  warnings.warn(
      0/99     13.2G      6.24        88         0      94.2        26       512: 100%|███████████████████████████████████████████████████| 7/7 [00:06<00:00,  1.00it/s]
/mnt/files/teShting/.venv/lib/python3.11/site-packages/torch/functional.py:507: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3549.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:04<00:00,  1.74it/s]
                 all       106       128    0.0013     0.102  0.000893   0.00256

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      1/99     13.2G      5.83      7.74         0      13.6        20       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  3.59it/s]
                 all       106       128         0         0   0.00082         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      2/99     13.2G      6.67       2.8         0      9.48        20       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.33it/s]
                 all       106       128         0         0   0.00138         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      3/99     13.2G      6.26      3.35         0      9.61        21       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.55it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.76it/s]
                 all       106       128         0         0   0.00187         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      4/99     13.2G      5.37      3.83         0       9.2        25       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  6.03it/s]
                 all       106       128         0         0   0.00523         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      5/99     13.2G       5.1       4.2         0       9.3        25       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.55it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  6.62it/s]
                 all       106       128         0         0      0.02         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      6/99     13.2G      5.33      4.47         0      9.79        22       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  6.76it/s]
                 all       106       128         0         0    0.0349         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      7/99     13.2G      4.33      4.56         0      8.88        18       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  6.54it/s]
                 all       106       128         0         0    0.0548         0

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      8/99     13.2G      5.16         4         0      9.16        17       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.92it/s]
                 all       106       128         1   0.00781     0.295    0.0155

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      9/99     13.2G      3.75      3.14         0      6.89        18       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.79it/s]
                 all       106       128         1   0.00781     0.425    0.0155

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     10/99     13.2G      4.16      4.12         0      8.28        22       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.77it/s]
                 all       106       128     0.666    0.0778     0.328     0.139

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     11/99     13.2G      5.34      3.22         0      8.56        25       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.78it/s]
                 all       106       128     0.631      0.32     0.447     0.425

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     12/99     13.2G      4.77      2.86         0      7.63        24       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.75it/s]
                 all       106       128     0.326     0.664     0.439     0.437

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     13/99     13.2G      4.25      2.57         0      6.82        24       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.72it/s]
                 all       106       128     0.174     0.758      0.46     0.283

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     14/99     13.2G      3.88      2.51         0      6.39        35       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.70it/s]
                 all       106       128     0.126     0.789      0.36     0.217

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     15/99     13.2G      3.71       1.9         0      5.61        17       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.76it/s]
                 all       106       128     0.171      0.68     0.472     0.274

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     16/99     13.2G      4.31      1.57         0      5.88        23       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.55it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.75it/s]
                 all       106       128      0.33     0.773     0.512     0.463

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     17/99     13.2G      3.35      1.62         0      4.96        24       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.75it/s]
                 all       106       128     0.411     0.633      0.38     0.499

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     18/99     13.2G      3.98      1.49         0      5.47        38       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.74it/s]
                 all       106       128     0.294     0.336     0.317     0.313

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     19/99     13.2G       3.9      1.34         0      5.24        22       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.75it/s]
                 all       106       128     0.216     0.306      0.28     0.254

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     20/99     13.2G      4.12      1.32         0      5.44        31       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.71it/s]
                 all       106       128    0.0469     0.391    0.0428    0.0837

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     21/99     13.2G      4.25      1.24         0      5.49        24       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.70it/s]
                 all       106       128     0.228     0.586     0.206     0.328

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     22/99     13.2G      3.77      1.24         0      5.02        19       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.69it/s]
                 all       106       128     0.439      0.52      0.37     0.476

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     23/99     13.2G      3.73      1.23         0      4.97        26       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.76it/s]
                 all       106       128     0.419     0.398     0.364     0.408

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     24/99     13.2G      3.31      1.14         0      4.45        17       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.74it/s]
                 all       106       128     0.423     0.474     0.382     0.447

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     25/99     13.2G      3.06      0.94         0         4        15       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.73it/s]
                 all       106       128     0.333     0.492     0.309     0.397

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     26/99     13.2G      3.96     0.911         0      4.87        15       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.74it/s]
                 all       106       128     0.506     0.657     0.478     0.572

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     27/99     13.2G      3.25       1.1         0      4.35        25       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.74it/s]
                 all       106       128     0.475     0.758     0.461     0.584

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     28/99     13.2G      4.01      0.88         0      4.89        14       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.75it/s]
                 all       106       128      0.49     0.849      0.63     0.622

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     29/99     13.2G      3.66      1.03         0       4.7        32       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.75it/s]
                 all       106       128     0.582     0.675     0.569     0.625

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     30/99     13.2G      3.31     0.994         0      4.31        19       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.74it/s]
                 all       106       128     0.479     0.639     0.466     0.548

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     31/99     13.2G      3.53     0.982         0      4.51        19       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.74it/s]
                 all       106       128     0.653     0.625      0.68     0.639

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     32/99     13.2G      3.14      1.01         0      4.15        21       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.74it/s]
                 all       106       128     0.633     0.579     0.631     0.605

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     33/99     13.2G       3.1     0.954         0      4.06        26       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.79it/s]
                 all       106       128     0.537     0.375      0.44     0.442

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     34/99     13.2G      3.37         1         0      4.37        27       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.74it/s]
                 all       106       128     0.556     0.492     0.453     0.522

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     35/99     13.2G      3.13     0.842         0      3.97        20       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.75it/s]
                 all       106       128       0.5     0.664     0.519     0.571

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     36/99     13.2G      3.42     0.891         0      4.31        24       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.70it/s]
                 all       106       128      0.43     0.656     0.502      0.52

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     37/99     13.2G      3.33     0.858         0      4.19        21       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.72it/s]
                 all       106       128     0.377     0.715     0.373     0.494

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     38/99     13.2G      3.19     0.953         0      4.14        22       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.72it/s]
                 all       106       128     0.587     0.655     0.555     0.619

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     39/99     13.2G      2.88     0.957         0      3.84        21       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.77it/s]
                 all       106       128     0.676     0.719      0.68     0.697

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     40/99     13.2G      2.77     0.844         0      3.62        18       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.55it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.75it/s]
                 all       106       128     0.714     0.727     0.722      0.72

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     41/99     13.2G       2.8     0.851         0      3.65        25       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.73it/s]
                 all       106       128     0.634     0.797     0.684     0.706

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     42/99     13.2G      2.76     0.874         0      3.63        32       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.73it/s]
                 all       106       128     0.838     0.734     0.815     0.783

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     43/99     13.2G      3.18     0.965         0      4.14        18       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.74it/s]
                 all       106       128     0.652     0.633     0.676     0.642

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     44/99     13.2G      3.35     0.772         0      4.12        21       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.75it/s]
                 all       106       128     0.776     0.594     0.704     0.673

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     45/99     13.2G      2.98     0.807         0      3.78        13       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.74it/s]
                 all       106       128     0.933     0.648     0.771     0.764

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     46/99     13.2G      3.73     0.798         0      4.53        22       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.76it/s]
                 all       106       128      0.82     0.773     0.852     0.796

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     47/99     13.2G      3.81     0.734         0      4.55        23       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.76it/s]
                 all       106       128     0.819     0.816     0.877     0.818

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     48/99     13.2G      2.69     0.699         0      3.39        27       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.75it/s]
                 all       106       128     0.706     0.827     0.834     0.762

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     49/99     13.2G      3.47     0.815         0      4.28        20       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.77it/s]
                 all       106       128      0.51     0.836      0.71     0.633

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     50/99     13.2G      2.65     0.909         0      3.56        51       512:  57%|████████████████████████████                                      
     50/99     13.2G      2.65     0.909         0      3.56        51       512:  71%|████████████████████████████████████                          
     50/99     13.2G       2.7      0.88         0      3.58        31       512:  71%|████████████████████████████████████                          
     50/99     13.2G       2.7      0.88         0      3.58        31       512:  86%|████████████████████████████████████████████              
     50/99     13.2G      2.77     0.851         0      3.62        17       512:  86%|████████████████████████████████████████████             
     50/99     13.2G      2.77     0.851         0      3.62        17       512: 100%|███████████████████████████████████████████████████     
     50/99     13.2G      2.77     0.851         0      3.62        17       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.74it/s]
                 all       106       128     0.284     0.812     0.425     0.421

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     51/99     13.2G       2.8     0.776         0      3.58        21       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.74it/s]
                 all       106       128     0.318     0.773     0.448     0.451

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     52/99     13.2G      2.98     0.863         0      3.84        28       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.70it/s]
                 all       106       128     0.346     0.819       0.5     0.487

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     53/99     13.2G       2.8     0.788         0      3.59        28       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.69it/s]
                 all       106       128     0.389     0.906     0.652     0.545

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     54/99     13.2G      3.23     0.687         0      3.92        19       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.71it/s]
                 all       106       128     0.554     0.891     0.693     0.683

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     55/99     13.2G      3.06     0.691         0      3.75        25       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.73it/s]
                 all       106       128     0.658     0.916     0.794     0.766

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     56/99     13.2G      2.35     0.846         0       3.2        24       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.73it/s]
                 all       106       128     0.736     0.958     0.936     0.833

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     57/99     13.2G      2.81     0.685         0      3.49        16       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.53it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.72it/s]
                 all       106       128     0.846     0.945     0.941     0.893

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     58/99     13.2G      2.49     0.722         0      3.21        20       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.71it/s]
                 all       106       128     0.798     0.938     0.908     0.862

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     59/99     13.2G      2.82     0.871         0      3.69        29       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.74it/s]
                 all       106       128     0.866     0.945     0.946     0.904

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     60/99     13.2G       2.4     0.712         0      3.11        30       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.67it/s]
                 all       106       128     0.872     0.945     0.951     0.907

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     61/99     13.2G      2.74     0.634         0      3.37        21       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.55it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.72it/s]
                 all       106       128     0.882     0.945     0.957     0.912

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     62/99     13.2G      2.57     0.659         0      3.23        23       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.74it/s]
                 all       106       128      0.84     0.942     0.936     0.888

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     63/99     13.2G      2.57     0.542         0      3.11        19       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.74it/s]
                 all       106       128     0.857     0.933     0.928     0.893

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     64/99     13.2G      2.53     0.679         0      3.21        16       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.74it/s]
                 all       106       128     0.937     0.932     0.956     0.935

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     65/99     13.2G      2.85     0.629         0      3.48        21       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.72it/s]
                 all       106       128     0.916     0.941     0.953     0.929

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     66/99     13.2G      2.42      0.63         0      3.05        25       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.74it/s]
                 all       106       128     0.924     0.956     0.949      0.94

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     67/99     13.2G      2.12      0.62         0      2.74        17       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.76it/s]
                 all       106       128     0.897     0.969     0.921     0.931

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     68/99     13.2G       2.4     0.596         0         3        23       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.73it/s]
                 all       106       128     0.934     0.969     0.958     0.951

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     69/99     13.2G      1.79     0.524         0      2.31        23       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.71it/s]
                 all       106       128     0.895     0.969       0.9      0.93

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     70/99     13.2G      2.34     0.594         0      2.94        23       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.55it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.73it/s]
                 all       106       128     0.924     0.961      0.95     0.942

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     71/99     13.2G      1.81     0.585         0      2.39        32       512:  43%|█████████████████████▊                             | 3/7 [00:01<00:01,  2.40it/s]
Model Bias Summary:    layer        regression        objectness    classification
                          89      -0.11+/-0.25     -11.01+/-2.56       0.02+/-0.01 
                         101       0.12+/-0.20     -12.23+/-1.25       0.04+/-0.00 
                         113       0.29+/-0.31     -11.16+/-2.07      -0.00+/-0.01 
     71/99     13.2G      1.88     0.566         0      2.44        25       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.73it/s]
                 all       106       128     0.958     0.961     0.968      0.96

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     72/99     13.2G      1.81      0.63         0      2.44        30       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.62it/s]
                 all       106       128     0.978     0.961     0.971     0.969

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     73/99     13.2G      1.82     0.528         0      2.35        21       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.75it/s]
                 all       106       128     0.971     0.953     0.972     0.962

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     74/99     13.2G      1.66     0.494         0      2.15        17       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.74it/s]
                 all       106       128     0.982     0.953     0.974     0.967

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     75/99     13.2G      1.74     0.506         0      2.24        23       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.65it/s]
                 all       106       128     0.974     0.953     0.974     0.963

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     76/99     13.2G      2.07     0.479         0      2.55        14       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.64it/s]
                 all       106       128      0.97     0.953     0.977     0.962

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     77/99     13.2G      1.73     0.493         0      2.22        21       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.55it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.73it/s]
                 all       106       128     0.977     0.953     0.979     0.965

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     78/99     13.2G      2.06     0.601         0      2.66        26       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.76it/s]
                 all       106       128     0.979     0.961      0.98      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     79/99     13.2G      1.69     0.515         0       2.2        22       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.75it/s]
                 all       106       128     0.979     0.969     0.982     0.974

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     80/99     13.2G      1.61     0.478         0      2.09        21       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.71it/s]
                 all       106       128     0.973     0.969     0.983     0.971

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     81/99     13.2G      1.69     0.449         0      2.14        20       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.55it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.75it/s]
                 all       106       128     0.939     0.984     0.983     0.961

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     82/99     13.2G      1.57     0.517         0      2.09        24       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.55it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.70it/s]
                 all       106       128     0.924     0.984     0.985     0.953

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     83/99     13.2G      1.79     0.467         0      2.25        23       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.77it/s]
                 all       106       128     0.913     0.984     0.984     0.947

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     84/99     13.2G      1.43     0.479         0      1.91        25       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.73it/s]
                 all       106       128     0.905     0.984     0.983     0.943

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     85/99     13.2G      1.72     0.546         0      2.27        19       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.76it/s]
                 all       106       128     0.902     0.984     0.983     0.941

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     86/99     13.2G       1.4     0.475         0      1.87        21       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.74it/s]
                 all       106       128     0.909     0.984     0.983     0.945

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     87/99     13.2G      1.63      0.43         0      2.06        23       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.74it/s]
                 all       106       128     0.893     0.984     0.981     0.937

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     88/99     13.2G       1.4     0.506         0      1.91        18       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.73it/s]
                 all       106       128      0.91     0.984     0.982     0.946

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     89/99     13.2G      1.43     0.431         0      1.86        17       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.75it/s]
                 all       106       128     0.911     0.984     0.981     0.946

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     90/99     13.2G      1.71     0.453         0      2.16        22       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.55it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.75it/s]
                 all       106       128     0.915     0.984     0.983     0.948

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     91/99     13.2G      1.99     0.427         0      2.41        20       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.75it/s]
                 all       106       128     0.925     0.984     0.984     0.954

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     92/99     13.2G      1.62     0.492         0      2.11        25       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.55it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.73it/s]
                 all       106       128     0.928     0.984     0.984     0.955

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     93/99     13.2G      2.12     0.441         0      2.56        21       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.55it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.75it/s]
                 all       106       128     0.927     0.984     0.985     0.955

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     94/99     13.2G      1.62     0.437         0      2.06        23       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.75it/s]
                 all       106       128     0.912     0.984     0.984     0.947

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     95/99     13.2G      1.49     0.387         0      1.88        17       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.77it/s]
                 all       106       128     0.944     0.984     0.984     0.964

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     96/99     13.2G      1.52     0.465         0      1.99        26       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.78it/s]
                 all       106       128     0.956     0.984     0.985      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     97/99     13.2G      1.38     0.459         0      1.84        29       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.57it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.73it/s]
                 all       106       128     0.956     0.984     0.985      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     98/99     13.2G      1.17     0.397         0      1.57        20       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.56it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.77it/s]
                 all       106       128     0.957     0.984     0.984      0.97

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     99/99     13.2G      1.46     0.388         0      1.85        22       512: 100%|███████████████████████████████████████████████████| 7/7 [00:02<00:00,  2.58it/s]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100%|███████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.77it/s]
                 all       106       128     0.957     0.984     0.985      0.97
100 epochs completed in 0.119 hours.