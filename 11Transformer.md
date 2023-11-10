# Transformers


## Motivation

Deep Learning needs huge amounts of training data and correspondingly high processing effort for training. In order to cope with this processing complexity, GPUs/TPUs must be applied. However, GPUs and TPUs yield higher training speed, if operations can be **parallelized**. The drawback of RNNs (of any type) is that the recurrent connections can not be parallelized. **Transformers** {cite}`Vaswani2017` exploit only **Self-Attention**, without recurrent connections. So they can be trained efficiently on GPUs. In this section first the concept of Self-Attention is described. Then Transformer architectures are presented.

## Self Attention

In **Self-Attention** layers, the attention-coefficients 

$$
a_{i,j}=f(x_{i},x_{j})
$$

score the influence of the input $x_j$ at position $j$ on the input $x_i$ at postion $i$. In the image below the calculation of the outputs $y_i$ in a Self-Attention layer is depicted. Here, 

* $x_i * x_j$ is the scalar product of the two vectors. 
* $x_i$ and $x_j$ are learned such, that their scalar product yields a high value, if the output strongly depends on their correlation. 


```{figure} https://maucher.home.hdm-stuttgart.de/Pics/selfAttention1.png
---
align: center
width: 400pt
name:  selfattention1
---
Calculation of $y_1$.

```{figure} https://maucher.home.hdm-stuttgart.de/Pics/selfAttention2.png
---
align: center
width: 400pt
name:  selfattention2
---
Calculation of Self-Attention outputs $y_1$ (top) and $y_2$ (bottom), respectively.

```

### Contextual Embeddings

What is the meaning of the outputs of a Self-Attention layer? To answer this question, we focus on applications, where the inputs to the network $x_i$ are sequences of words. In this case, words are commonly represented by their embedding vectors (e.g. Word2Vec, Glove, Fasttext, etc.). The **drawback of Word Embeddings** is that they are **context free**. E.g. the word **tree** has an unique word embedding, independent of the context (tree as natural object or tree as a special type of graph). On the other hand, the elements $y_i$ of the Self-Attention-Layer output $\mathbf{y}=(y_1,y_2,\ldots y_{T})$ can be considerd to be contextual word embeddings! The representation $y_i$ is a contextual embedding of the input word $x_i$ in the given context.

### Queries, Keys and Values

As depicted in {ref}`figure Self-Attention <selfattention2>`, each input vector $x_i$ is used in **3 different roles** in the Self Attention operation:

- **Query:** It is compared to every other vector to establish the weights for its own output $y_i$ 
- **Key:** It is compared to every other vector to establish the weights for the output of the j-th vector $y_j$
- **Value:** It is used as part of the weighted sum to compute each output vector once the weights have been established.

In a Self-Attention Layer, for each of these 3 roles, a separate **version** of $x_i$ is learned:

* the **Query** vector is obtained by multiplying input vector $x_i$  with the learnable matrix $W_q$:

$$
q_i=W_q x_i
$$


* the **Key** vector is obtained by multiplying input vector $x_i$  with the learnable matrix $W_k$:

$$
k_i=W_k x_i
$$

* the **Value** vector is obtained by multiplying input vector $x_i$  with the learnable matrix $W_v$:

$$
v_i=W_v x_i
$$

Applying these three representations the outputs $y_i$ are calculated as follows:

$$
a'_{i,j} & = & q_i^T k_j \\
a_{i,j} & = & softmax(a'_{i,j})  \\
y_i & = & \sum_j a_{i,j} v_j  
$$ (qkv1)

The image below visualizes this calculation:


```{figure} https://maucher.home.hdm-stuttgart.de/Pics/selfAttention1qkv.png
---
align: center
width: 400pt
name:  selfattentionqkv
---
Calculation of Self-Attention outputs $y_1$ from queries, keys and values of the input-sequence

```

In the calculation, defined in {eq}`qkv1`, the problem is that the softmax-function is sensitive to large input values in the sense that for large inputs most of the softmax outputs are close to 0 and the corresponding gradients are also very small. The effect is very slow learning adaptations. In order to circumvent this, the inputs to the softmax are normalized:

$$
a'_{i,j} & = & \frac{q_i^T k_j}{\sqrt{d}} \\
a_{i,j} & = & softmax(a'_{i,j}) 
$$

### Multi-Head Attention and Positional Encoding

There are 2  drawbacks of the approach as introduced so far:

1. Input-tokens are processed as **unordered set**, i.e. order-information is ignored. For example the output $y_{passed}$ for the input **Bob passed the ball to Tim** would be the same as the output $y_{passed}$ for the input *Tim passed the ball to Bob*.
2. For a given pair of input-tokens $x_i, x_j$ query $q$ and key $k$ are always the same. Therefore their attention coefficien $a_{ij}$ is also always the same. However, the correlation between a given pair of tokens may vary. In some contexts their interdependence may be strong, in others weak.   

These problems can be circumvented by *Multi-Head-Attention* and *Positional Encoding*.

**Multi-Headed Self-Attention** provides an additional degree of freedom in the sense, that multiple (query,key,value) triples for each pair of positions $(i,j)$ can be learned. For each position $i$, multiple $y_i$ are calculated, by applying the attention mechanism, as introduced above, $h$ times in parallel. Each of the $h$ elements is called an *attention head*. Each attention head applies its own matrices $W_q^r, W_k^r, W_v^r$ for calculating individual queries $q^r$, keys $k^r$ and values $v^r$, which are combined to the output:    

$$
\mathbf{y}^r=(y^r_1,y^r_2,\ldots y^r_{T_y}).   
$$

The length of the input vectors $x_i$ is typically $d=256$. A typical number of heads is $h=8$. For combining outputs of the $h$ heads to the overall output-vector $\mathbf{y}^r$, there exists 2 different options: 

* **Option 1:** 
  - Cut vectors $x_i$ in $h$ parts, each of size $d_s$
  - Each of these parts is fed to one head
  - Concatenation of  $y_i^1,\ldots,y_i^h$ yields $y_i$ of size $d$
  - Multiply this concatenation with matrix $W_O$, which is typically of size $d \times d$

* **Option 2:**
  - Fed entire vector $x_i$ to each head. 
  - Matrices $W_q^r, W_k^r,W_v^r$ are each of size $d \times d$
  - Concatenation of  $y_i^1,\ldots,y_i^h$ yields $y_i$ of size $d \cdot h$
  - Multiply this concatenation with matrix $W_O$, which is typically of size $d \times (d \cdot h)$


```{figure} https://maucher.home.hdm-stuttgart.de/Pics/selfAttention1qkv.png
---
align: center
width: 400pt
name:  singlehead
---
Single-Head Self-Attention: Calculation of first element $y_1$ in output sequence.
```

```{figure} https://maucher.home.hdm-stuttgart.de/Pics/selfAttention1qkvMultipleHeads.png
---
align: center
width: 400pt
name:  multihead
---
Multi-Head Self-Attention: Combination of the individual heads to the overall output. 

```

**Positional Encoding:** In order to embed information to distinguish different locations of a word within a sequence, a **positonal-encoding-vector** is added to the word-embedding vector $x_i$. Certainly, each position $i$ has it's own positional encoding vector.

```{figure} https://maucher.home.hdm-stuttgart.de/Pics/positionalEncoding1.png
---
align: center
width: 400pt
name:  positionalencoding
---
Add location-specific positional encoding vector to word-embedding vector $x_i$. Image source: [http://jalammar.github.io/illustrated-transformer](http://jalammar.github.io/illustrated-transformer) 

```

The vectors for positional encoding are designed such that the similiarity of two vectors decreases with increasing distance between the positions of the tokens to which they are added. This is illustrated in the image below: 

```{figure} https://maucher.home.hdm-stuttgart.de/Pics/positionalEncoding2.png
---
align: center
width: 400pt
name:  positionalencoding2
---
Positional Encoding: To each position within the sequence a unique *positional-encoding-vector* is assigned. As can be seen the euclidean distance between vectors for further away positions is larger than the distance between vectors, which belong to positions close to each other.  [http://jalammar.github.io/illustrated-transformer](http://jalammar.github.io/illustrated-transformer) 

```


```{figure} https://maucher.home.hdm-stuttgart.de/Pics/BERTpositionalEncoding.png
---
align: center
width: 500pt
name:  positionalencoding3
---
Visualisation of true positional encoding in BERT

```


## Building Transformers from Self-Attention-Layers

As depicted in the image below, a Transformer in general consists of an Encoder and a Decoder stack. The Encoder is a stack of Encoder-blocks. The Decoder is a stack of Decoder-blocks. Both, Encoder- and Decoder-blocks are Transformer blocks. In general a **Transformer Block** is defined to be **any architecture, designed to process a connected set of units - such as the tokens in a sequence or the pixels in an image - where the only interaction between units is through self-attention.**

```{figure} https://maucher.home.hdm-stuttgart.de/Pics/transformerStack.png
---
align: center
width: 400pt
name:  stack
---
Encoder- and Decoder-Stack of a Transformer. Image source: [http://jalammar.github.io/illustrated-transformer](http://jalammar.github.io/illustrated-transformer) 

```

A typical Encoder block is depicted in the image below. In this image the *Self-Attention* module is the same as already depicted in {ref}`Image Multihead Self-attention<multihead>`. The outputs $z_i$ of the Self-Attention module are exactly the contextual embeddings, which have been denoted by $y_i$ in {ref}`Image Multihead Self-attention<multihead>`. Each of the outputs $z_i$ is passed to a Multi-Layer Perceptron (MLP). The outputs of the MLP are the new representations $r_i$ (one for each input token). These outputs $r_i$ constitute the inputs $x_i$ to the next Encoder block.
  
```{figure} https://maucher.home.hdm-stuttgart.de/Pics/transformerEncoder1.png
---
align: center
width: 400pt
name:  encoderblock
---
Encoder Block - simple variant: Self-Attention Layer followed by Feed Forward Network. Image source: [http://jalammar.github.io/illustrated-transformer](http://jalammar.github.io/illustrated-transformer) 

```

The image above depicts a simple variant of an Encoder block, consisting only of Self-Attention and a Feed Forward Neural Network. A more complex and more practical option is shown in the image below. Here, short-cut connections from the Encoder-block input to the output of the Self-Attention Layer are implemented. The concept of such short-cuts has been introduced and analysed in the context of ResNet. Moreover, the sum of the Encoder-block input and the output of the Self-Attention Layer is layer-normalized before it is passed to the Feed Forward Net. 

```{figure} https://maucher.home.hdm-stuttgart.de/Pics/normalisationEncoder.png
---
align: center
width: 400pt
name:  encoderblockprac
---
Encoder Block - practical variant: Short-Cut Connections and Layer Normalisation are applied in addition to Self-Attention and Feed Forward Network. Image source: [http://jalammar.github.io/illustrated-transformer](http://jalammar.github.io/illustrated-transformer) 

```

Image {ref}`Encoder-Decoder<decoder>` illustrates the modules of the Decoder block and the linking of Encoder and Decoder. As can be seen a Decoder block integrates two types of attention: 

* **Self-Attention in the Decoder:** Like the Encoder block, this layer calculates queries, keys and values from the output of the previous layer. However, since Self Attention in the Decoder is only allowed to attend to earlier positions[^fa2] in the output sequence future tokens (words) are masked out. 


* **Encoder-Decoder-Attention:** Keys and values come from the output of the Encoder stack. Queries come from the output of the previous layer. In this way an alignment between the input- and the output-sequence is modelled.

On the top of all decoder modules a Dense Layer with softmax-activation is applied to calculate the most probable next word. This predicted word is attached to the decoder input sequence for calculating the most probable word in the next time step, which is then again attached to the input in the next time-step ...

In the alternative **Beam Search** not only the most probable word in each time step is predicted, but the most probable *B* words can be predicted and applied in the input of next time-step. The parameter $B$ is called *Beamsize*.  

```{figure} https://maucher.home.hdm-stuttgart.de/Pics/transformerEncoderDecoder1.png
---
align: center
width: 400pt
name:  decoder
---
Encoder- and Decoder Stack in a Transformer [http://jalammar.github.io/illustrated-transformer](http://jalammar.github.io/illustrated-transformer) 

```

In the image below the iterative prediction of the tokens of the target-sequence is illustrated. In iteration $i=4$ the $4.th$ target token must be predicted. For this the decoder takes as input the $i-1=3$ previous estimations and the keys and the values from the Encoder stack.  

```{figure} https://maucher.home.hdm-stuttgart.de/Pics/transformerPrediction.png
---
align: center
width: 400pt
name:  transpredict
---
Prediction of the 4.th target word, given the 3 previously predictions . Image source: [http://jalammar.github.io/illustrated-transformer](http://jalammar.github.io/illustrated-transformer) 

```

# Appendix

```{figure} https://maucher.home.hdm-stuttgart.de/Pics/selfAttentionConcept.png
---
align: center
width: 400pt
name:  attention concept
---
Further explanation of self-attention concept 

```