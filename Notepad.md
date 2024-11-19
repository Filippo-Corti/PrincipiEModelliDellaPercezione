# A. Argomenti per la presentazione

- Cos'è il Rumore:
    - Come si forma
    - Come si classifica

- Tecniche per la Rimozione del Rumore (Denoising):
    - Filtri Tradizionali (Gaussian, Median, BM3D)
    - Reti Neurali
        - Cosa fanno meglio rispetto ai Filtri?
        - Due diversi casi per l'addestramento del modello:
            - Noisy + Clean Images
                - CNN
                - Auto Encoders
                - GAN
            - Noisy Images
                - N2N
                - N2V

- L'Algoritmo N2V
    - Caratteristiche e confronti rispetto agli altri
    - Settori dove è maggiormente utilizzato
    - Principio di Funzionamento dell'Algoritmo
        - Le Componenti Principali:
            - U-Net
            - Loss Function -> A che Grand Truth fa riferimento?
            - ...

- N2V in pratica:
    - Grand Challenge
    - Codice Python
    - Risultati con diversi valori per epochs e batch_size
    - Difetti del N2V e possibili risoluzioni

- Come l'N2V si ispira al nostro Sistema Nervoso nella Vista.

# B. Risorse

[CAREamics source code] https://github.com/CAREamics/careamics/blob/main/src/careamics/careamist.py#L423 

[Pytorch Lightning Trainer] https://lightning.ai/docs/pytorch/stable/common/trainer.html 

[CAREamics code for the Challenge - with comparison images] https://careamics.github.io/0.1/applications/Noise2Void/JUMP/ 

[CAREamics code to test N2V on natural images] https://careamics.github.io/0.1/applications/Noise2Void/BSD68/ 

[CAREamics Noise2Void Overview] https://careamics.github.io/0.1/algorithms/Noise2Void/ 

[Original Article presenting N2V] https://openaccess.thecvf.com/content_CVPR_2019/papers/Krull_Noise2Void_-_Learning_Denoising_From_Single_Noisy_Images_CVPR_2019_paper.pdf 

[ ??? ] https://proceedings.mlr.press/v97/batson19a/batson19a.pdf

[Fixing N2V Problems - Pagina 525] https://link.springer.com/chapter/10.1007/978-3-031-25069-9_33

[More on N2V] https://www.toolify.ai/ai-news/denoising-images-with-noise2void-a-deep-learning-approach-1150583

[N2V Code Implementation] https://github.com/juglab/n2v

[N2V2 - Contains details for N2V] https://www.researchgate.net/publication/365448865_N2V2_--_Fixing_Noise2Void_Checkerboard_Artifacts_with_Modified_Sampling_Strategies_and_a_Tweaked_Network_Architecture

[An example experiment with N2V with Architecture details] https://pmc.ncbi.nlm.nih.gov/articles/PMC8563445/pdf/nihms-1749463.pdf

[Another example of N2V with params] https://colab.research.google.com/github/r3gm/InsightSolver-Colab/blob/main/N2V_for_Image_Denoising_of_Single_Channel_Images__Training_and_Inference.ipynb

[Architecture Template] https://github.com/BiaPyX/BiaPy/blob/master/templates/denoising/2d_denoising.yaml

[Perplexity.ai answer to which params are important for N2V] https://www.perplexity.ai/search/i-have-zero-knowledge-of-cnns-R9QNnIGPSeGw.uHlJkE9yg#4

[U-Net explained] https://arxiv.org/pdf/1505.04597v1

[Youtube Video(s)] https://www.youtube.com/watch?v=nVKvGBq_-wQ

[Batch vs Epoch] https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/

# C. Appunti e Note

Le Immagini nel Dataset JUMP sono in formato .TIFF Hyperstack, con 4 canali per ogni immagine che rappresentano RGB + Alpha.

Noise2Void impara direttamente dalle Noisy Images, senza bisogno di Clean Images (like Auto-Encoders) o Multiple Noisy Images (like Noise2Noise). \
Assunzione:
    - Il Segnale ha una struttura, il Rumore non ha una struttura (ovvero i pixel di s non sono indipendenti mentre quelli di n sì)
      => Necessità di UNSTRUCTURED NOISE
    
Questo ci consente, guardando i pixel vicini, di predirre il segnale e non il rumore associato.

Si basa su una Blind-Spot Network, differente rispetto alle CNN tradizionali poiché il Campo Recettivo non considera il pixel centrale. Esso è sostituito da un valore estratto da un vicino scelto casualmente dal campo recettivo. Questo è l'Input Patch.

La predizione (Target Patch) è l'Input Patch modificata, ma avente come pixel centrale il precedente pixel centrale del segnale originale.

Il procedimento è poi quello tradizionale delle Reti Neurali, con minimizzazione dell'errore...

Vantaggi di N2V:
    + Richiede semplicemente Noisy Images
    + E' molto rapido, vedi confronto con BM3D


## Architettura di N2V

Come si arriva all'Architettura di N2V:

1. Perceptron

The perceptron is a basic neural network unit that models the functioning
of a biological neuron. It was developed by Frank Rosenblatt in the late 1950s
and has since been widely used in various pattern recognition and classification
tasks. The perceptron takes one or more inputs, which are features variables
and computes the weighted sum of the inputs, and applies an activation function to produce the output. 
The output is determined based on whether the
sum of the weighted inputs is greater than a certain threshold. It is a building
block for more complex neural networks, such as multi-layer perceptrons

The perceptron consists of 4 parts.

Input values or One input layer
Weights (and Bias)
Net sum
Activation Function -> Per le nostre applicazioni, ReLU

The choice of an
activation function σ is purely empirical such that there is always a room for
an alternative choice given the problem and the performance of a model using
a particular activation function σ

[Check out
https://www.researchgate.net/publication/369921211_Deep_Learning_Architectures
https://towardsdatascience.com/what-the-hell-is-perceptron-626217814f53
https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
]


2. Multi-Layer Perceptron (Artifical Neural Network or Feed-Forward Neural Network)

Deep Feed-Forward networks, commonly known as neural networks or multilayer perceptrons (MLPs), are considered the most representative models of
deep learning. Feed-Forward Neural Networks are a type of artificial neural
network that are widely used in machine learning and deep learning applications. A Feed-Forward network aims to approximate a function f*, which could
be the mapping of an input x to a category y. The network creates a mapping
y = f(x; θ) and optimizes the parameters θ to obtain the best function approximation f∗. 
In this case θ is the variable representing learning parameters
of the function f∗. 
These networks are called Feed-Forward because they
are designed to process information in a forward direction, from the input layer
through the hidden layers to the output layer. The input layer receives the input
data, which is then processed by the hidden layers using a set of weights and
biases, and the output is generated by the output layer. The model does not
have any internal connections that allow the output to be fed back into itself
in contrast to Recurrent Neural Networks. The model can be represented
by a directed acyclic graph that illustrates the composition of functions.



[Check out
https://www.researchgate.net/publication/369921211_Deep_Learning_Architectures
]

3. Convolutional Neural Network

Although Feed-Forward Neural Networks are capable of handling image recognition tasks, Convolutional Neural Networks (CNNs)
are more suitable for these types of problems due to their superior performance.
Unlike Feed-Forward Neural Networks, which utilize vectors and matrices as
learning parameters, CNNs use filters or volumes as learning parameters and
convolutions or volumes to process the data from the filters. Although CNNs
are designed specifically for image recognition tasks, they still rely on FeedForward Neural Networks as a key component in their architecture. In most
cases, FFNs form the last layers of a CNN, where they take in flattened data,
which is a transformed version of the initial input volume into a simple vector, and perform classification tasks on the images.

Convolutional Neural Network is a type of neural networks that have been used
in image classification tasks, and object recognition for a long period of time.
They are using the same architecture as Feed-Forward Neural Networks, with
the only difference in the introduction of convolutions as layers, filters as weights.
Beyond that they use back-propagation as Feed-Forward Neural Networks, activation functions, optimization functions and gradient descent as a their building
blocks.

As it was mentioned earlier, the Feed-Forward Neural Networks can also solve
image classification problems, but usually the training process take longer and
the accuracy is quite high on the real test datasets. The reason behind that is
the complexity of the objects represented in an image. Complexity includes an
object’s class features such as shape of an object, the distance between one point
to the another, the color, and so on. Therefore, feed-forward neural networks
take an image of size (N, N) as a flatten version of it. The flatten version of a
matrix is a vector V of size (N, 1).

In pratica prima applico CNN, con cui viene comodo il formato matriciale, poi la appiattisco e la mando alla FNN.

The output of a Convolutional Neural Network (CNN) is typically a set of predictions based on the input data, which is often an image or a sequence of images. Here’s a breakdown of the process leading to the output:
Input Layer: The CNN receives an input, usually an image represented in three dimensions (width, height, and depth for color channels).
Convolutional Layers: These layers apply filters (kernels) to the input image to extract features. Each filter detects specific patterns such as edges or textures. The output from this layer is known as a feature map, which highlights the presence of detected features in the image.
Activation Layers: After convolution, activation functions (like ReLU or sigmoid) are applied to introduce non-linearity, allowing the network to learn complex patterns.
Pooling Layers: These layers reduce the dimensionality of the feature maps while retaining important information, typically using operations like max pooling or average pooling. This helps in making the network more computationally efficient and reduces overfitting.
Flattening: The pooled feature maps are flattened into a one-dimensional vector to prepare them for input into fully connected layers1.
Fully Connected Layers: These layers take the flattened feature maps and perform classification or regression tasks by computing outputs based on learned weights from previous layers.
Output Layer: Finally, the output from the fully connected layer is passed through an activation function (such as softmax for multi-class classification) to produce probability scores for each class. This gives the final prediction, indicating which class the input image belongs to.
In summary, the output of a CNN is a probability distribution across different classes for classification tasks or specific values for regression tasks, depending on its design and intended application.

If the patch size is the same as that of the image it will be a regular neural network. Because of this small patch, we have fewer weights. 

NOTA: i Dense Layers (la FNN) alla fine della CNN richiede che l'input venga portato ad una dimensione fissa. 
Questo è evidente poiché una matrice W di pesi MxN può essere applicata solamente ad un vettore di input di lunghezza N, per ottenere
un vettore di output di lunghezza M. (Si noti: y = Wx + b, non y = xW + b) 

[Check out
https://www.researchgate.net/publication/369921211_Deep_Learning_Architectures
https://www.geeksforgeeks.org/introduction-convolution-neural-network/
https://www.andreaprovino.it/convolutional-neuralnetwork
]

4. Fully Convolutional Network
 
CONVOLUTIONAL networks are driving advances in
recognition. Convnets are not only improving for
whole-image classification, but also making
progress on local tasks with structured output. These include advances in bounding box object detection,
part and keypoint prediction, and local correspondence.
The natural next step in the progression from coarse to
fine inference is to make a prediction at every pixel. Prior
approaches have used convnets for semantic segmentation, in which each pixel is
labeled with the class of its enclosing object or region, but
with shortcomings that this work addresses

Semantic segmentation faces an inherent tension between semantics and location: global information resolves
what while local information resolves where. What can be
done to navigate this spectrum from location to semantics?
How can local decisions respect global structure? It is not
immediately clear that deep networks for image classification yield representations sufficient for accurate, pixelwise
recognition

Una FCN o Fully Convolutional Network è una deep neural network (rete neurale profonda) che supera le limitazioni delle convenzionali CNN eliminando il dense layer in favore di 1×1 convolutional layers.
L’assenza dei fully connected layers consente alle FCN Networks di:
 - elaborare immagini di diverse dimensioni, cosa non possibile con le strutture fisse convenzionali delle CNN.
 - avere una struttura più snella (lower parameters) e aumentare quindi la velocità computazionale, riducendo la latenza (low latency).

Feature	                        CNN (Convolutional Neural Network)	                                            FCN (Fully Convolutional Network)
Fully Connected Layers	        Yes, after convolution and pooling layers	                                    No, uses convolutional layers throughout the network
Spatial Information	            Spatial information lost after flattening (fixed-size output)	                Retains spatial information (output size depends on input)
Output	                        Fixed output size (e.g., class labels)	                                        Variable output size (e.g., pixel-level predictions in segmentation)
Input Size	                    Fixed input size (needs resizing)	                                            Variable input size (can handle different input sizes)
Use Cases	                    Image classification, regression	                                            Image segmentation, object localization, dense predictions

Instead of using fully connected layers, we can replace them with 1x1 convolutional layers. Here's why this works:

1x1 convolutions are similar to fully connected layers in the sense that each 1x1 filter connects to all channels at each spatial location. This allows the network to learn class-specific features at each pixel location, similar to how FC layers would learn overall image features, but in a more spatially-aware way.
This is done after the convolutional layers, and it allows the network to make pixel-wise predictions for each location in the image.
Now, instead of reducing the image to a single class label, the network produces an output map that retains the spatial dimensions of the original image. But initially, after passing through convolutions, the output will still be smaller than the input due to operations like max pooling or stride convolutions.
etup
Let's say your input feature map has dimensions 32x32x128. This means the spatial dimensions of the image are 32x32, and there are 128 channels (depth).

Input dimensions: 
32×32×128
32x32 are the spatial dimensions (height and width).
128 is the number of channels (depth).
1x1 Convolution Operation
A 1x1 convolution applies a small 1x1 filter across the entire spatial grid (32x32), but with a depth that spans the number of input channels. Essentially, the filter "sees" the entire set of channels at each spatial location (pixel), and combines them in a weighted way.

Filter size: 
1×1×128 (this is the size of the filter applied at each pixel location, with depth equal to the number of input channels).
Stride: Typically, stride=1 for 1x1 convolutions.
Padding: Padding is usually not needed for 1x1 convolutions since you're applying it only on the spatial dimensions (height and width).
How the 1x1 Convolution Works
For each pixel (location) in the spatial grid (32x32), the convolution filter looks at all the channels in the input feature map. The 1x1 filter has 128 weights, one for each channel, and these weights are applied to the respective channels at that spatial location.

Each spatial location (pixel) in the input feature map is processed by the 1x1 filter. The filter applies a weighted sum across all 128 channels at that pixel.
Weight matrix for each spatial location: For each pixel, you apply a filter that is 
1×1×128, so each pixel gets processed by a weight matrix that takes all 128 input channels into account.
Output of 1x1 Convolution
The output of this operation depends on how many filters (weights) you have in the 1x1 convolution layer.
If you have M filters in the 1x1 convolution, the output will have M channels for each spatial location.
Let's break down what happens during the 1x1 convolution:

Each of the 128 input channels contributes to the final output at that pixel location (spatial location).
If you apply M filters in the 1x1 convolution, each filter will generate one output value per pixel. So, for each spatial location, you will get M values, one for each filter.
Example: 
32×32×128 → 32×32×M
Input: 
32×32×128 (height x width x channels).
1x1 Convolution Filters: 
1×1×128 (for each pixel).
Number of Filters: Suppose you use M filters in the 1x1 convolution layer.
The result after the 1x1 convolution will be an output feature map of size 32x32xM, where M is the number of filters you have in the 1x1 convolution. Each filter produces one output value per spatial location, so the output has the same spatial dimensions (32x32) as the input, but the number of channels is now M instead of 128.

The Process at Each Spatial Location
Take each pixel in the input feature map (of size 32×32).
Look at the 128 channels (depth) at that pixel.
Apply each filter: Each filter has a set of 128 weights (one for each input channel at that pixel location). The filter computes a weighted sum of all the 128 input channels at that pixel.
Output: The result is one output value for that pixel location, which is repeated across all pixels in the spatial grid.
Key Points:
A 1x1 convolution does not change the spatial dimensions (height and width), but it changes the depth (the number of channels).
It combines features from all the channels at each pixel and outputs a new set of features.
The number of filters M determines how many output channels you have for each spatial location.
The 1x1 convolution is used to learn complex combinations of features without changing the spatial dimensions of the input.

3. Fusing the Output
After going through conv7 as below, the output size is small, then 32× upsampling is done to make the output have the same size of input image. But it also makes the output label map rough. And it is called FCN-32s:

This is because, deep features can be obtained when going deeper, spatial location information is also lost when going deeper. That means output from shallower layers have more location information. If we combine both, we can enhance the result.

To combine, we fuse the output (by element-wise addition):

FCN-16s: The output from pool5 is 2× upsampled and fuse with pool4 and perform 16× upsampling. Similar operations for FCN-8s as in the figure above.

FCN-32s result is very rough due to loss of location information while FCN-8s has the best result.

This fusing operation actually is just like the boosting / ensemble technique used in AlexNet, VGGNet, and GoogLeNet, where they add the results by multiple model to make the prediction more accurate. But in this case, it is done for each pixel, and they are added from the results of different layers within a model.


[Check out
https://arxiv.org/pdf/1605.06211v1
https://www.andreaprovino.it/fcn
https://towardsdatascience.com/review-fcn-semantic-segmentation-eb8c9b50d2d1
https://medium.com/@mohit_gaikwad/overview-fully-convolutional-network-for-semantic-segmentation-b4ef92eeb8c4
]

5. U-Net



6. N2V

Utilizza una U-Net particolare con:

- ReLu
- ADAM Optimization
- 

