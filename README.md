## Group Members

Jiayong Lin

Yixuan Ren

Bo He

Junran Yang 

## Topic

Rprop Variants Implementation and Performance Comparison

## Abstract

The resilient back-propagation (Rprop), proposed by Riedmiller and Braun, is one of the most popular learning algorithms for neural networks in backpropagation. It overcomes the inherent disadvantages of pure gradient-descent by performing a local adaptation of the weight-updates according to the behavior of the error function. several further improved variants had also been proposed in the following years. In this project we implement the basic full-batch gradient descent and four variants of Rprop, and compare their performances on different tasks, including regressions and classifications. Following experiments show that Rprop has significant advantages over regular gradient descent on various tasks. The internal ranking of four Rprop variants is usually determined by task types and the network architectures. Empirically, iRprop+ algorithm is slightly
better.

## INTRODUCTION
   Back-propagation is the most widely used algorithm for supervised learning with multi-layered feed-forward networks, and gradient descent is the core procedure for applying the updates. Most gradient descent uses the sign of the gradient as well as the magnitude. The direction is completely determined by the sign. To decide the step size, most algorithms involved a scaled version of gradients. However, the original gradient descent still remains some problems: the choice of the learning rate, which scales the derivative, has an important effect on the convergence speed. In a word, the performance of standard error back propagation may depend largely on hyper-parameter selection. Moreover, when the network goes deep, which means it has too many layers, the gradient will vanish and the training may even almost stop. 
   
   There were some other algorithms proposed to deal with this problem. One of them is momentum. However, the choice of the momentum parameter is equally problem dependent as the learning rate. Another big bunch is parameter adaptation. These adaptive algorithms, including two main categories of global and local, improve the influence from the learning rate, but do nothing for the partial derivative. So they’re still halfway -the effort of carefully selecting a proper learning rate can be easily destroyed by other factors. 
   
   Rprop, standing for “Resilient backpropagation”, is a local adaptive learning scheme. Its principle is to eliminate the harmful influence of the size of the partial derivative on the weight step. This is done by controlling the weight update for every single connection individually during the learning process in order to minimize oscillations and to maximize the update step-size. Rprop does not require specifying any parameter values like learning rate or momentum value. Instead of using the magnitude of the gradient to determine W, it only uses its sign and the learning rates vary during learning. In addition, the robustness is another feature of the Rprop. The number of learning steps of the Rprop algorithm is significantly reduced in comparison to the original gradient-descent procedure, whereas the expense of computation of the Rprop adaptation process is held considerably small. 
   
## THE RPROP ALGORITHM

[Detail of all RPROP Algorithms](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.17.1332&rep=rep1&type=pdf)

## IMPLEMENTATION AND EXPERIMENTS

We have tested our model on both classification and regression tasks. Classification task includes MNIST database of handwritten digits, Breast Cancer Wisconsin data set. For the regression task, we test our model on regressing 3d sine wave. In each task, we compare the accuracy and convergence speed for each Rprop variants. In order to realize the full functions of each Rprop variant algorithms, we implemented backpropagation and data interfaces from scratch. Because the Rprop does not perform well on minibatch, we are using full batch learning for all variants. Full batch learning greatly increases the training time for a large dataset like MNIST. We implemented the fully connected network with sigmoid non-linearity, because we want to showthe Rprop can reduce the likelihood of gradient to vanish without ReLU.

### Evaluating on Breast Cancer Data
The Breast Cancer dataset is from a digitized image of a fine needle aspirate (FNA) of a breast mass. It includes Diagnosis(ground truth label) and nine numeric features for each cell nucleus. Totally, the data has 683 instances. For this classification task, the network will use cross-entropy loss and a sigmoid function at the output layer. ir+ converges faster than EBP and other Rprop variants. Although the difference seems small, Rprop variants generate better results than EBP.

| Algorithm        | Accuracy           | 
| ------------- |:-------------:|
|EBP      | 0.965 | 
| IR+      | 0.965    | 
| IR- | 0.971     | 
| R+      | 0.965    | 
| R- | 0.971     | 


### Evaluating on 3D Sine Wave
In order to test the algorithms performance on the regression task, we synthesize the 3d sine wave data. We trained the model with 2500 data points, and test on 1600 different data points. It is what the train data and test data are showing:

Firstly, we trained the model with 3 hidden layers [16, 32, 16]. From the loss plot, we can easily observe that Rprop converge significantly faster than EBP. in addition, Rprop’s results have less mean absolute error. In this specific task, r+ yields a better result.

##### 3D SINE WAVE REGRESSION WITH SHALLOW NETWORK ERROR TABLE:
|Algorithm| Mean L1 Loss|
| ------------- |:-------------:|
|EBP|0.143|
|IR+ |0.120|
|IR- |0.120|
|R+| 0.104|
|R-| 0.122|

After we train the model with a deeper network (5 hidden layers in this case ), it will have a higher chance for gradient vanishing happens. The training loss plot shows that the EBP stops learning halfway, but Rprop is not affected by the number of hidden layers changes. Rprop’s result sine plots are not very different from the shallow network, but it did have a higher Mean absolute error. r+ is still the best algorithm for this task.

##### 3D SINE WAVE REGRESSION WITH DEEP NETWORK ERROR TABLE:
|Algorithm| Mean L1 Loss|
| ------------- |:-------------:|
|EBP|0.748|
|IR+ |0.132|
|IR- |0.126|
|R+| 0.126|
|R-| 0.135|



## CONCLUSION

According to our experiments, the class of Rprop algorithms performs obviously better than classical batch gradient descent in all the tasks. In detail, Rprop can provide a faster speed toward convergence no matter how the other hyperparameters, such as the learning rate, number of layers or type of activation functions are set. Although we have more advanced techniques for fine-tuning a complex neural network nowadays, Rprop is still an effective and convenient choice for boosting its performance. As we can see from the experimental results, there is no great difference between four algorithms. However, the iRprop+ algorithm is slightly better than the other three algorithms. Their performances depends on the condition of the surface of manifolds in the high-dimensional space, whichis determined by the task types and the network architectures.


