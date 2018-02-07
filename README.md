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

Let's look at the structure of GoogLeNet. It is a <b>22 layer</b> network, which starts with general CONV-POOL-NORM layers and ends with POOL-FC-SOFTMAX-> but is not sequential (linear) in between. We have chunks of parallel blocks in between, called the 'inception' block, which forms the network architecture. I will explain the network in detail as we go through the post, so lets check out <b>'GoogLeNet'</b> :

<img src = "https://github.com/SKKSaikia/CNN-GoogLeNet/blob/master/img/gg.png">

So, we know that the parallel portion in the network is called the <b>inception module</b>, right? As we can see, stacking of these inception modules make up the network.

<img src="https://github.com/SKKSaikia/CNN-GoogLeNet/blob/master/img/gin.jpg">

But what is this inception module?

The authors wanted to design a good local network topology, a network within a network and then stack a lot of these on top of each other. These local topologies are being called the <b>“inception module”</b>. As we look into the module below, we can see, what’s inside an inception module.

<img src="https://github.com/SKKSaikia/CNN-GoogLeNet/blob/master/img/in.jpg">

In the inception module (Naïve inception module) they are applying different kinds of filter operations in parallel. We have our input coming from the previous layer, and apply conv ( 1x1, 3x3 & 5x5 ) & a pooling layer in parallel. We get different outputs from these filter operations and then we concatenate all these filter outputs together ‘depth’ wise. Then it creates a single tensor output, that is going to pass down to the next layer. This was the initial idea of the authors, but <b>THE PROBLEM?</b> = Computational Complexity.

Let's take an example, suppose we have an input image ( 28x28x256 ) to an inception module & we have convolutions :

        1 x 1 of 128 filters
        3 x 3 of 192 filters 
        5 x 5 of 96  filters
        3 x 3 pool
        
Each one of these convolves and outputs images of 28x28x128, 28x28x192, 28x28x96 & 28x28x256. All these operations are concatenated at end, 'depth' wise. So, finally we get an output of 28x28x(128+192+96+256) = 28x28x672. This is huge, we kept the same spatial dimension, but we blew out the depth.

<img src="https://github.com/SKKSaikia/CNN-GoogLeNet/blob/master/img/opinN.jpg">

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
