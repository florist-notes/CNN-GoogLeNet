# CNN-GoogLeNet
🕵🏻 Vision : Model 4: GoogLeNet : Image Classification


GoogleNet V-1 ( Inception_V1 ) : 
-
<b>Paper : </b> " [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) "  <b>Talk :</b> [ILSVRC 2014](https://youtu.be/ySrj_G5gHWI) <b> CVPR Presentation : </b> [CVPR 2015](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html)

2014 [ILSVRC](http://www.image-net.org/challenges/LSVRC/) ([ImageNet](http://www.image-net.org/) Large-Scale Visual Recognition Challenge) <b>Winner</b>.

It is a benchmark competition where teams across the world compete to classify, localize, detect ... images of 1000 categories, taken from the imagenet dataset.The imagenet dataset holds 15M images of 22K categories but for this contest: 1.2M images in 1K categories were chosen.The ILSVRC 2014 classification challenge involves the task of classifying the image into one of 1000 leaf-node categories in the Imagenet hierarchy. There are about 1.2 million images for training, 50,000 for validation and 100,000 images for testing. Each image is associated with one ground truth category, and performance is measured based on the highest scoring classifier predictions. Two numbers are usually reported: the top-1 accuracy rate, which compares the ground truth against the first predicted class, and the top-5 error rate, which compares the ground truth against the first 5 predicted classes: an image is deemed correctly classified if the ground truth is among the top-5, regardless of its rank in them. The challenge uses the top-5 error rate for ranking purposes. Team " GoogLeNet ", achieved top 5 test error rate of 6.7%, It was  better than any other model's performance at that time. Check ILSVRC 2014 [results](http://www.image-net.org/challenges/LSVRC/2014/results).

This paper is important as it is one of the first CNN architectures that really strayed from the general approach of simply stacking conv and pooling layers on top of each other in a sequential structure.The authors of the paper also emphasized that this new model places notable consideration on memory and power usage.

<img src="https://github.com/SKKSaikia/CNN-GoogLeNet/blob/master/img/GoogleNet.gif">

Overview
-
When we first take a look at the structure of GoogLeNet ( above gif ), we notice immediately that not everything is happening sequentially, as seen in previous architectures. We have pieces of the network that are happening in parallel as well. 'GoogLeNet' is a 22 layer deep Convolutional Neural Network architecture with considerable computational efficiency, introduced in 2014 by [Christian Szegedy](https://research.google.com/pubs/ChristianSzegedy.html), [Wei Liu](https://scholar.google.com/citations?user=yFMX138AAAAJ), [Yangqing Jia](https://scholar.google.com/citations?user=mu5Y2rYAAAAJ), [Pierre Sermanet](https://scholar.google.com/citations?user=0nPi5YYAAAAJ), [Scott Reed](https://scholar.google.com/citations?user=jEANvfgAAAAJ&hl=en), [Dragomir Anguelov](http://dblp.uni-trier.de/pers/hd/a/Anguelov:Dragomir), [Dumitru Erhan](https://scholar.google.co.in/citations?user=wfGiqXEAAAAJ&hl=en), [Vincent Vanhoucke](https://research.google.com/pubs/VincentVanhoucke.html) and [Andrew Rabinovich](https://scholar.google.co.in/citations?user=qn1ejaQAAAAJ&hl=en)

Architecture
-

 Let's look at the structure of GoogLeNet. 

<img src = "https://github.com/SKKSaikia/CNN-GoogLeNet/blob/master/img/gg.png">

model.summary()
-

Important Points
-

    ❅ 22 Layers ( stacking of Inception modules )
    ❅ Efficient Inception module
    ❅ No Fully Connected Layer
    ❅ Only 5 Million parameters ( 12x less than AlexNet )
    ❅ Computational efficiency ( 2x times less than AlexNet )
  
Practical
-

<img src="https://github.com/SKKSaikia/CNN-GoogLeNet/blob/master/img/g1.jpg">


<img src="https://github.com/SKKSaikia/CNN-GoogLeNet/blob/master/img/g2.jpg">

GoogleNet V-2 ( Inception_V2 )
-
Paper : " [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) "

GoogleNet V-3 ( Inception_V3 )
-
Paper : " [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) "

GoogleNet V-4 ( InceptionResNet_V2 )
-
Paper : " [Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261) "

References
-
    
      CS231n : Lecture 9 | CNN Architectures - GoogLeNet
      adeshpande3.github.io for /gif 
