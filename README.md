# CNN-GoogLeNet
🕵🏻 Vision : Model 4: GoogLeNet : Image Classification


GoogleNet ( Inception-v1 ) : 
-
<b>Paper : </b> " [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) "  <b>Talk :</b> [ILSVRC 2014](https://youtu.be/ySrj_G5gHWI) <b> CVPR Presentation : </b> [CVPR 2015](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html)

2014 [ILSVRC](http://www.image-net.org/challenges/LSVRC/) ([ImageNet](http://www.image-net.org/) Large-Scale Visual Recognition Challenge) <b>Winner</b>.

It is a benchmark competition where teams across the world compete to classify, localize, detect ... images of 1000 categories, taken from the imagenet dataset.The imagenet dataset holds 15M images of 22K categories but for this contest: 1.2M images in 1K categories were chosen.The ILSVRC 2014 classification challenge involves the task of classifying the image into one of 1000 leaf-node categories in the Imagenet hierarchy. There are about 1.2 million images for training, 50,000 for validation and 100,000 images for testing. Each image is associated with one ground truth category, and performance is measured based on the highest scoring classifier predictions. Two numbers are usually reported: the top-1 accuracy rate, which compares the ground truth against the first predicted class, and the top-5 error rate, which compares the ground truth against the first 5 predicted classes: an image is deemed correctly classified if the ground truth is among the top-5, regardless of its rank in them. The challenge uses the top-5 error rate for ranking purposes. Team " GoogLeNet ", achieved top 5 test error rate of 6.7%, It was  better than any other model's performance at that time. Check ILSVRC 2014 [results](http://www.image-net.org/challenges/LSVRC/2014/results).

This paper is important as it is one of the first CNN architectures that really strayed from the general approach of simply stacking conv and pooling layers on top of each other in a sequential structure.The authors of the paper also emphasized that this new model places notable consideration on memory and power usage.

<img src="https://github.com/SKKSaikia/CNN-GoogLeNet/blob/master/img/GoogleNet.gif">

Overview
-
When we first take a look at the structure of GoogLeNet ( above gif ), we notice immediately that not everything is happening sequentially, as seen in previous architectures. We have pieces of the network that are happening in parallel as well. 'GoogLeNet' is a 22 layer deep Convolutional Neural Network architecture with considerable computational efficiency, introduced in 2014 by [Christian Szegedy](https://research.google.com/pubs/ChristianSzegedy.html), [Wei Liu](https://scholar.google.com/citations?user=yFMX138AAAAJ), [Yangqing Jia](https://scholar.google.com/citations?user=mu5Y2rYAAAAJ), [Pierre Sermanet](https://scholar.google.com/citations?user=0nPi5YYAAAAJ), [Scott Reed](https://scholar.google.com/citations?user=jEANvfgAAAAJ&hl=en), [Dragomir Anguelov](http://dblp.uni-trier.de/pers/hd/a/Anguelov:Dragomir), [Dumitru Erhan](https://scholar.google.co.in/citations?user=wfGiqXEAAAAJ&hl=en), [Vincent Vanhoucke](https://research.google.com/pubs/VincentVanhoucke.html) and [Andrew Rabinovich](https://scholar.google.co.in/citations?user=qn1ejaQAAAAJ&hl=en). The GoogLeNet is spelled so and not GoogleNet, in order to pay homage to [LeNet](https://github.com/SKKSaikia/CNN-LeNet). This community is supportive and homely :)

Architecture
-

Let's look at the structure of GoogLeNet. It is a <b>22 layer</b> network, which starts with general CONV-POOL-NORM layers and ends with POOL-FC-SOFTMAX-> but is not sequential (linear) in between. We have chunks of parallel blocks in between, called the 'inception' block, which forms the network architecture. I will explain the network in detail as we go through the post, so lets check out <b>'GoogLeNet'</b> :

<img src = "https://github.com/SKKSaikia/CNN-GoogLeNet/blob/master/img/gg.png">

So, we know that the parallel portion in the network is called the <b>inception module</b>, right? As we can see, stacking of these inception modules make up the network.

<img src="https://github.com/SKKSaikia/CNN-GoogLeNet/blob/master/img/gin.jpg">

But what is this inception module?

The authors wanted to design a good local network topology, a network within a network and then stack a lot of these on top of each other. These local topologies are being called the <b>“inception module”</b>. The name inception comes from the movie "[Inception](http://www.imdb.com/title/tt1375666/)", where the model is inspired from the meme " [We need to go Deeper](http://knowyourmeme.com/memes/we-need-to-go-deeper) ". The authors even referenced this meme in the paper ^ - ^ .As we look into the module below, we can see, what’s inside an inception module.

<img src="https://github.com/SKKSaikia/CNN-GoogLeNet/blob/master/img/in.jpg">

In the inception module (Naïve inception module) they are applying different kinds of filter operations in parallel. We have our input coming from the previous layer, and apply conv ( 1x1, 3x3 & 5x5 filters ) & a pooling layer in parallel. We get different outputs from these filter operations and then we concatenate all these filter outputs together ‘depth’ wise. Then it creates a single tensor output, that is going to pass down to the next layer. This was the initial idea of the authors, but <b>THE PROBLEM?</b> = Computational Complexity.

Let's take an example, suppose we have an input image ( 28x28x256 ) to an inception module & we have convolutions :

        1 x 1 of 128 filters
        3 x 3 of 192 filters 
        5 x 5 of 96  filters
        3 x 3 pool
        
Each one of these convolves and outputs images of 28x28x128, 28x28x192, 28x28x96 & 28x28x256 respectively. All these operations are concatenated at end, 'depth' wise. So, finally we get an output of 28x28x(128+192+96+256) = 28x28x672. This is huge!!!!!!! we kept the same spatial dimension, but we blew out the depth.

<img src="https://github.com/SKKSaikia/CNN-GoogLeNet/blob/master/img/opinN.jpg">

In terms of operations, we can note that the total operations ( Conv Ops ) reaches 854 Million! this is hugely expensive. Also, the fact that Pooling layer preserves feature depth, which means total depth after concatenation can only grow at every layer! We need an alternative, which can reduce dimensions at every layer?

<b>Solution:</b> We can project the "Bottleneck" layers into lower dimensions, i.e, we can use 1x1 convolutions to reduce feature depth. Let's review what a 1x1 conv does, it preserves spatial dimension, but reduces the depth.

<img src="https://github.com/SKKSaikia/CNN-GoogLeNet/blob/master/img/1x1.JPG">

How does it do that? Andre Ng expalins that pretty well [here](https://youtu.be/c1RBQzKsDCk). So, it's main idea is that its projecting your depth down, right ? So, to eleviate these expensive compute, the authors added 1x1 conv prior to the other filters and after the pooling layer to reduce its depth. Thus in Figure 2 (b) we get the desired inception module by adding the 1x1 bottleneck layers. Now we can check the computation requirements for the same input image.

<img src="https://github.com/SKKSaikia/CNN-GoogLeNet/blob/master/img/ch.JPG">

So, Inception module all done! Now we can say that, GoogLeNet = stack the inception modules. But wait ! What are these ?

<img src="https://github.com/SKKSaikia/CNN-GoogLeNet/blob/master/img/auxI.jpg">

These are extra stems coming out from the main network, called the auxiliary layers. Apart from the final softmax, there are two 1000 way imagenet training classification loss in these seperate places, it is used to control parameters when we train the networks in case of the [gradient vanishing](https://en.wikipedia.org/wiki/Vanishing_gradient_problem). This is done so that more gradient injected in these intermediate layers, which directs more helpful signal flows through the network.Given the relatively large depth of the network, the ability to propagate gradients back through all the layers in an effective manner was a concern. One interesting insight is that the strong performance of relatively shallower networks on this task suggests that the features produced by the layers in the middle of the network should be very discriminative. By adding auxiliary classifiers connected to these intermediate layers, we would expect to encourage discrimination in the lower stages in the classifier, increase the gradient signal that gets propagated back, and provide additional regularization. These classifiers take the form of smaller convolutional networks put on top of the output of the Inception (4a) and (4d) modules. During training, their loss gets added to the total loss of the network with a discount weight (the losses of the auxiliary classifiers were weighted by 0.3). At inference time, these auxiliary networks are discarded.

The network in network conv is able to extract information about the very fine grain details in the volume, while the 5x5 filter is able to cover a large receptive field of the input, and thus able to extract its information as well. You also have a pooling operation that helps to reduce spatial sizes and combat overfitting. On top of all of that, you have ReLUs after each conv layer, which help improve the nonlinearity of the network. Basically, the network is able to perform the functions of these different operations while still remaining computationally considerate.

So, we are now left with the final GoogLeNet architecture, with 22 weight layers.
<img src="https://github.com/SKKSaikia/CNN-GoogLeNet/blob/master/img/fg.jpg">

You can also follow Andrew NG's lectures [Inception Network Motivation](https://youtu.be/C86ZXvgpejM) & [Inception Network](https://youtu.be/KfV8CJh7hE0) for visual understanding. 

model.summary()
-

                                                                  |-auxiliary layer
      Input -> CONV-POOL-NORM-CONV-CONV-NORM-POOL-{ 9 inception modules }-POOL-FC-Softmax->{ 1000 class classification
                                                           |_auxiliary layer
        

Important Points
-

    ❅ 22 Layers ( stacking of Inception modules )
    ❅ Efficient Inception module
    ❅ No Fully Connected Layer
    ❅ Only 5 Million parameters ( 12x less than AlexNet )
    ❅ Computational efficiency ( 2x times less than AlexNet )
    ❅ Used 9 Inception modules in the whole architecture, with over 100 layers in total!
    ❅ No use of fully connected layers! They use an average pool instead, to go from a 7x7x1024
       volume to a 1x1x1024 volume. This saves a huge number of parameters.
    ❅ Utilized concepts from R-CNN for their detection model.
    ❅ During testing, multiple crops of the same image were created, fed into the network, and 
       the softmax probabilities were averaged to give us the final solution.
    ❅ resize the image to 4 scales where the shorter dimension (height or width) is 256, 288, 320 and 352
       respectively, take the left, center and right square of these resized images (in the case of portrait
       images, take the top, center and bottom squares).For each square, then take the 4 corners and the 
       center 224x224 crop as well as the square resized to 224x224, and their mirrored versions. This results
       in 4x3x6x2 = 144 crops per image.
    ❅ dropout layer with 70% ratio of dropped outputs.
    ❅ trained using the DistBelief, distributed machine learning system.
    ❅ asynchronous stochastic gradient descent with 0.9 momentum.
    ❅ fixed learning rate schedule (decreasing the learning rate by 4% every 8 epochs).
  
Practical
-

Will update : Training ...

GoogleNet V-2 ( Inception-v2 )
-

Paper 1 : [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

Paper 2 : [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)

[Sergey Ioffe](https://research.google.com/pubs/SergeyIoffe.html) & [Christian Szegedy](https://research.google.com/pubs/ChristianSzegedy.html) published this paper called "[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)" where they introduced <b>Batch Normalization (BN)</b> , which provides a new method for normalization of layer, giving us higher learning rates. Local Response Normalization was used earlier, but with the introduction of Batch Normalization similar accuracy was achieved with 14 times fewer training steps. This paper, " [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) " stands as a motivation the modern Deep Networks.This inception module (v1) is also called the <b>Inception-BN module</b>.

Motivation: Deep networks are ill-posed (internal covariate shift) – Carefully parameters Initialization & – Small learning rate.

The difference between Inception & Inception V1 (Inception-BN) is :

     1. Batch Normalization
     2. Replace a 5x5 convolutional kernels with two 3x3 kernels.

<img src="https://github.com/SKKSaikia/CNN-GoogLeNet/blob/master/img/inV2.jpg">

We can check the performance graph for Inception & its BN variants, here:

<img src="https://github.com/SKKSaikia/CNN-GoogLeNet/blob/master/img/inBN.jpg">

By replacing 5x5 kernels with two 3x3 kernel, It can save many calculation operations and memories, also 3x3 can get more information than 5x5 kernels, and the overfit also can be avoided, which we have talked in Inception-v1.

GoogleNet V-3 ( Inception-v3 )
-
Paper : " [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) "

In the Inception-v3, they introduced Factorization (factorize convolutions into smaller convolutions) and some minor change into Inception-v2. Factorization is a very important trick in Inception-v3, it factorization big kernels into small kernels, here {one 7x7= two 5x5 with strides 2 = three 3x3 with stride 1).However, the networks doesn’t factorization thoroughly, so they continue to factorization, 3x3=(3x1 + 1x3), a picture shows as follows:

<img src="https://github.com/SKKSaikia/CNN-GoogLeNet/blob/master/img/inV2a.png">

In the picture, the module is the component of the networks. There are several inception grid in networks, for example 17x17 grid can be shows in following picture:

<img src="https://github.com/SKKSaikia/CNN-GoogLeNet/blob/master/img/inv2b.png">

Important Points:
-

    ★ Smaller kernels
    ★ Several inception grid
    
Practical:
-

Will update : Training ...

GoogleNet V-3 ( Inception_V3 )
-
Paper : " [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) "

As for Inception-v3, it is a variant of Inception-v2 which adds BN-auxiliary. BN auxiliary refers to the version in which the fully connected layer of the auxiliary classifier is also-normalized, not just convolutions. We are refering to the model [Inception-v2 + BN auxiliary] as Inception-v3.

Important Points:
-

Practical:
-

GoogleNet V-4 ( InceptionResNet )
-
Paper : " [Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261) "

Inception-v4 is proposed in this fourth paper. It integrates Residual Connection into the network. I felt that first covering ResNet will help me to write this section better. Will update this section soon :)


References
-
    
      CS231n : Lecture 9 | CNN Architectures - GoogLeNet
