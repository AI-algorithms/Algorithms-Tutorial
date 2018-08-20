### Machine Learning
机器学习是人工智能的一个分支。人工智能的研究历史有着一条从以“推理”为重点，到以“知识”为重点，再到以“学习”为重点的自然、清晰的脉络。显然，机器学习是实现人工智能的一个途径，即以机器学习为手段解决人工智能中的问题。机器学习在近30多年已发展为一门多领域交叉学科，涉及概率论、统计学、逼近论、凸分析、计算复杂性理论等多门学科。机器学习理论主要是设计和分析一些让计算机可以自动“学习”的算法。机器学习算法是一类从数据中自动分析获得规律，并利用规律对未知数据进行预测的算法。因为学习算法中涉及了大量的统计学理论，机器学习与推断统计学联系尤为密切，也被称为统计学习理论。算法设计方面，机器学习理论关注可以实现的，行之有效的学习算法。很多推论问题属于无程序可循难度，所以部分的机器学习研究是开发容易处理的近似算法。

机器学习已广泛应用于数据挖掘、计算机视觉、自然语言处理、生物特征识别、搜索引擎、医学诊断、检测信用卡欺诈、证券市场分析、DNA序列测序、语音和手写识别、战略游戏和机器人等领域。

#### 机器学习的框架

 * [Theano](#Theano) 
 * [Lasagne](#Lasagne) 
 * [Blocks](#Blocks) 
 * [TensorFlow](#TensorFlow) 
 * [Keras](#Keras) 
 * [MXNet](#MXNet) 
 * [PyTorch](#PyTorch)    
 * [Caffe](#Caffe) 
 * [CNTK](#CNTK)    
 * [Neon](#Neon)
 * [NumPy](#NumPy)
 
#### Theano

Theano 是一个Python 库，允许你定义、优化并且有效地评估涉及到多维数组的数学表达式。Theano 是数值计算的主力，它支持了许多其他的深度学习框架。Theano由 Frédéric Bastien创建，这是蒙特利尔大学机器学习研究所（MILA）背后的一个非常优秀的研究团队。它的API 水平较低，并且为了写出效率高的Theano，需要对隐藏在其他框架幕后的算法相当的熟悉。如果有着丰富的学术机器学习知识，正在寻找模型的精细控制方法，或者想要实现一个新奇的或不同寻常的模型，Theano是首选库。为了灵活性，Theano牺牲了易用性。

优点：灵活，正确使用时的高性能。

缺点：较高的学习难度，低水平的 API，编译复杂的符号图可能很慢。


#### Lasagne

通常在Theano上建立和训练神经网络的轻量级库。因为 Theano致力于成为符号数学中最先且最好的库，Lasagne提供了在 Theano顶部的抽象，这使得它更适合于深度学习。它主要由当前 DeepMind研究科学家 Sander Dieleman编写并维护。Lasagne并非是根据符号变量之间的函数关系来指定网络模型，而是允许用户在层级思考，为用户提供了例如「Conv2DLayer」和「DropoutLayer」的构建块。Lasagne在牺牲了很少的灵活性的同时，提供了丰富的公共组件来帮助图层定义、图层初始化、模型正则化、模型监控和模型训练。Theano+Lasagne是很多一线开发者的最爱。

优点：仍旧非常灵活，比 Theano更高级的抽象，文档和代码中包含了各种 Pasta Puns。

缺点：社区小。

#### Blocks

Blocks用于构建和训练神经网络的 Theano框架。与 Lasagne类似，Blocks 是在 Theano 顶部添加一个抽象层使深度学习模型比编写原始的 Theano更清晰、更简单、定义更加标准化。它是由蒙特利尔大学机器学习研究所（MILA）编写，其中一些人为搭建Theano 和第一个神经网络定义的高级接口（已经淘汰的PyLearn2）贡献了自己的一份力量。比起Lasagne，Blocks灵活一点，代价是入门台阶较高，想要高效的使用它有不小的难度。除此之外，Blocks对递归神经网络架构（recurrent neural network architectures）有很好的支持，所以如果有兴趣探索这种类型的模型，它值得一看。

优点：仍旧非常灵活，比 Theano更高级的抽象，易于测试。

缺点：学习难度比较高，而且社区比较小众。


#### TensorFlow

TensorFlow用于数值计算的使用数据流图的开源软件库。TensorFlow 是较低级别的符号库（比如 Theano）和较高级别的网络规范库（比如Blocks 和Lasagne）的混合。虽然它是Python 深度学习库集合的最新成员，不过在Google Brain 团队支持下，它已经是最大的活跃社区了。它支持在多GPUs 上运行深度学习模型，为高效的数据流水线提供使用程序，并具有用于模型的检查，可视化和序列化的内置模块。且TensorFlow支持 Keras（一个很优秀的深度学习库）。

优点：由软件巨头 Google支持，非常大的社区，低级和高级接口网络训练，比基于 Theano配置更快的模型编译，完全地多 GPU支持。

缺点：虽然 Tensorflow正在追赶，但是在许多基准上比基于 Theano的慢，RNN支持仍不如 Theano。


#### Keras

Python 的深度学习库。支持Convnets（基于GPU实现的卷积神经网络）、递归神经网络等。在Theano 或者TensorFlow 上运行。Keras 也许是水平最高，对用户最友好的库了。由 Francis Chollet（Google Brain团队中的另一个成员）编写和维护。它允许用户选择其所构建的模型是在 Theano上或是在 TensorFlow上的符号图上执行。Keras的用户界面受启发于 Torch。由于部分非常优秀的文档和其相对易用性，Keras的社区非常大并且非常活跃。TensorFlow已经与 Keras一起支持内置，所以很快 Keras将是 TensorFlow项目的一个分组。

优点：可供选择的 Theano或者 TensorFlow后端，直观、高级别的端口，更易学习。

缺点：Keras缺点是不太灵活。

#### MXNet

MXNet是一个旨在提高效率和灵活性的深度学习框架。MXNet是亚马逊（Amazon）选择的深度学习库，也许是最优秀的库。它拥有类似于Theano 和TensorFlow 的数据流图，架构设计得可以利用更多内存复用机会和为多GPU 配置提供了良好的配置，有着类似于Lasagne 和Blocks 更高级别的模型构建块，并且可以在你可以想象的任何硬件上运行（包括手机）。对Python 的支持只是其冰山一角——MXNet同样提供了对 R、Julia、C++、Scala、Matlab和Javascript 的接口。


#### PyTorch

Python 中的张量（Tensors）和动态神经网络，有着强大的GPU 加速。PyTorch 也是Python 深度学习框架列表中的一个新成员。它是从Lua 的Torch 库到Python 的松散端口，由Facebook 的人工智能研究团队（Artificial Intelligence Research team (FAIR)）支持且因为它较早支持用于处理动态计算图，也是非常优秀的一款深度学习框架。

优点：来自 Facebook 组织的支持，完全地对动态图的支持，高级和低级API 的混合。

缺点：目前PyTorch还不太成熟，除了官方文档以外，只有有限的参考文献资料。

#### Caffe

Caffe 起初并不是一个通用框架，而仅仅关注计算机视觉，但它具有非常好的通用性。Caffe 具有很好的 CNN建模能力，但是 RNN资源就少很多，所以它更多的是面向图像识别、推荐引擎和自然语言识别等方向的应用，不面向其他深度学习应用诸如语音识别、时间序列预测、图像字幕和文本等其他需要处理顺序信息的任务。

优点：良好的CNN建模能力。

缺点：不够灵活，有限的参考文献/资源。

#### CNTK

CNTK 是微软的开源深度学习框架，是「Computational Network Toolkit（计算网络工具包）」的缩写。或是另一种称呼认知工具包（Cognitive Toolkit）。虽然也是强大的工具，但是社区小，文献资源和相关评论也是很少。

优点：丰富的 RNN教程和预构建模型。

缺点：社区小，新手级资料少。


#### Neon

Neon是Intel收购的一个深度学习框架，因此在Intel处理器架构平台性能相当好，且具有良好的CNN 建模能力。至于其他方面就逊色许多。

优点：较好的 CNN建模能力和Intel架构出色的性能。

缺点：社区小，新手级资料少。

#### NumPy（Numeric Python）

 一个用Python实现的科学计算包，是Python的一种开源数值计算扩展，包括：1、一个强大的N维数组对象Array；2、比较成熟的（广播）函数库；3、用于整合C/C++和Fortran代码的工具包；4、实用的线性代数、傅里叶变换和随机数生成函数。提供了许多高级的数值编程工具，可用来存储和处理大型矩阵，如矩阵数据类型、矢量处理、以及精密的运算库，是专为进行严格的数字处理而产生。基本可以认为NumPy将Python变成了一种免费的更强大的MatLab系统。
 
优点：速度的标杆，非常灵活。

缺点：最小的社区，比 Theano更高的学习难度。

####  卷积神经网络（Convolutional Neural Network，CNN）

CNN是一种前馈神经网络，它的人工神经元可以响应一部分覆盖范围内的周围单元，且该网络避免了对图像的复杂前期预处理，可以直接输入原始图像，对于大型图像处理有出色表现。主要用来识别位移、缩放及其他形式扭曲不变性的二维图形，由于CNN的特征检测层是通过训练数据进行学习的，所以在使用CNN时，避免了显式的特征抽取，而是隐式地从训练数据中进行学习；再者由于同一特征映射面上的神经元权值相同，所以网络可以并行学习，这也是卷积网络相对于神经元彼此相连网络的一大优势。卷积神经网络以其局部权值共享的特殊结构在语音识别和图像处理方面有着独特的优越性，其布局更接近于实际的生物神经网络，权值共享降低了网络的复杂性，特别是多维输入向量的图像可以直接输入网络这一特点避免了特征提取和分类过程中数据重建的复杂度。

 
#### 递归神经网络（Recurrent neural Network，RNN）

RNN神经网络是一种节点定向连接成环的人工神经网络。网络的内部状态可以展示动态时序行为。不同于前馈神经网络的是，RNN可以利用它内部的记忆来处理任意时序的输入序列，这让它可以更容易处理如不分段的手写识别、语音识别等。具有更强的动态行为和计算能力。

#### 比较分析
 
通过对上面的机器学习框架的介绍,我们从以下几个方面对各个机器学习框架做个对比:

*框架教程

我们可以看出各类深度学习框架的Api和可利用的资源在质量和数量上有着显著的不同。
TensorFlow，MXNet，PyTorch和Theano有着很详尽的文档教程，与此相比，虽然微软的CNTK 和英特尔的Nervana Neon 也是强大的工具，却很少能见到有关它们的新手级资料。下面我们从一下几个方面

* 语言和平台

毫无疑问，python是目前机器学习语言的趋势。MXNet（Python、R、Scala、Julia、Matlab、Javascript、C++）与TensorFlow（python、C/C++）具有丰富的多语言支持。

多平台的支持首属MXNet（跨平台Linux、OS X、Windows、Android、iOS），Theano跨平台，TensorFlow暂时不支持Windows。

*架构和速度

训练深度网络非常耗时，所以为在特定框架中构建和训练新模型，易于使用和模块化的前端是至关重要的。TensorFlow，PyTorch和 MXNet都有直观而模块化的架构，让开发相对变得简单。另外，因为有 TensorBoard web GUI等应用的存在，TensorFlow极易在训练中和训练后进行 debug和监控。

Caffe 也发布了一些预训练模型/权重（model zoo），能够作为初始权重被用于特殊领域或自定义图像的迁移学习或微调深度网络。能够转换基于caffe 的预训练模型权重，使其可以适应MXNET。

至于速度则需要考虑每个深度学习的框架所针对的领域了，RNN性能比较出色的有CNTK和 PyTorch，CNN性能比较出色的有Theano，Caffe和MXNet。另外Theano和MXNet在综合性能中是比较出色的，Tensorflow的性能在大多数测试中也是很有竞争力的。

* GPU和Keras兼容性

大多数深度学习应用都需要用到巨量的浮点运算（FLOP），了减少构建模型所需的时间，需要使用多 GPU并联的方式组建自己的学习平台。根据公开发表的文献，MXNet有着最好的多 GPU优化引擎，这也是其综合性能中比较出色的原因之一。另外TensorFlow和PyTorch对多GPU也有很好的支持。

基于Theano的深度学习框架（Lasagne）和 Keras是深度学习模型中较早且较广泛使用的框架，很容易用 Lasagne/Keras 实现新网络或者编辑现存网络。且Keras是一个用于快速构建深度学习原型的高级库，是数据科学家应用深度学习的好帮手，目前兼容Keras的只有Theano和TensorFlow。

* CNN 建模能力

卷积神经网络（CNN）经常被用于图像识别、推荐引擎和自然语言识别等方向的应用。CNN由一组多层的神经网络组成，在运行时会将输入的数据进行预定义分类的评分。CNN也可用于回归分析，例如构成自动驾驶汽车中有关转向角的模型。评价一种框架的 CNN建模能力考虑以下几个特性：定义模型的机会空间、预构建层的可用性、以及可用于连接这些层的工具和功能。其中Theano，Caffe和MXNet都有很好的 CNN建模能力。另外，TensorFlow因为易于建立的 Inception V3模型，PyTorch 因为其丰富的 CNN 资源（易于使用的时间卷积集）使得这两种框架在 CNN 建模能力上脱颖而出。

* RNN 建模能力

循环神经网络（RNN）常用于语音识别，时间序列预测，图像字幕和其他需要处理顺序信息的任务。目前，Microsoft的 CNTK和 PyTorch 有着丰富的 RNN教程和预构建模型。另外，目前很流行的 TensorFlow中也有一些 RNN 资源，且Keras中更是有很多使用 TensorFlow的 RNN 示例。

* 高级支持和扩展

主要考虑低级的张量（Tensor）运算符和控制流运算符的支持程度，高效的低级运算符实现能充当新模型的原料，控制流运算符增加符号引擎的表达性和通用性，这方面表现良好的主要是Theano和TensorFlow。

基于Theano的架构库可以说是最多的，Keras、Lasagne、blocks就是成功的案例。而TensorFlow是较低级别的符号库（比如Theano）和较高级别的网络规范库（比如Blocks 和Lasagne）的混合，且兼容Keras，良好的设计使得图像、队列、图像增加器等能成为更高级包装的有用构造块。

 

 通过对比其他机器学习框架我们之后，我们在看看各个机器学习框架对语言和平台的支持情况：

MXNet：真正的跨平台，语言支持丰富。

Theano：跨平台Win、Linux、Mac，语言Python。

TensorFlow：Linux、Mac，语言Python、C、C++。

PyTorch：Linux、Mac，Python、Lua

* 教程和资源的比较：

TensorFlow、Theano、MXNet、PyTorch。

* 架构和扩展的比较：

Theano、TensorFlow、PyTorch、MXNet。

* CNN建模能力的比较：

Caffe、MXNet、Theano，TensorFlow、PyTorch。

* RNN建模能力的比较：

CNTK、PyTorch、TensorFlow、MXNet。

* 多GPU支持：

MXNet、TensorFlow、PyTorch、Theano。

* 综合性能：

MXNet、Theano、TensorFlow、PyTorch。


    

    

    

    

    
