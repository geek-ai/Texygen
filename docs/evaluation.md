## Evaluations:


### Metrics

Paper #TODO

#### NLL-oracle loss

NLL-oracle loss is originally proposed by [SeqGAN](https://arxiv.org/abs/1609.05473)

A randomly initialized LSTM is regarded as the true model, aka, the oracle. GAN models need to minimize average negative log-likelihood of generate data on oracle LSTM.

![](fig/math/nll.png)

where ![](fig/math/Gt.png) denotes generative model (generator LSTM), ![](fig/math/Go.png) denotes the true model (oracle LSTM). 

#### NLL-test loss

NLL-test loss is dual to NLL-oracle loss. i.e. it's average negative log-likelihood of real data on generator. It can only be applied to RNN-based generator.

![](fig/math/inll.png)

#### BLEU score

BLEU is a widely used metric evaluating the word similarity between two sentences or documents.

Please refer to [BLEU: a method for automatic evaluation of machine translation](https://dl.acm.org/citation.cfm?id=1073135)

Also refer to its python [nltk implementation with smooth function](http://www.nltk.org/_modules/nltk/translate/bleu_score.html). 
We use smooth function _method1_.


#### self-BLEU score

This is a metric we proposed in order to evaluate the mode collapse in each model.

It's average BLEU score, with every generator's one instance as hypothesis, the other instances be references.

#### EmbSim

Inspired by BLEU, we propose a new metric named EmbSim evaluating the similarity between two documents. Instead of comparing sentences words by words, we compare the word embedding.

First, word embedding is evaluated on real data using a skip-gram model.

For each word embedding, we compute its cosine distance with the other words. And then formulate it as a matrix ![](fig/math/W.png), where
![](fig/math/wij.png) with ![](fig/math/ei.png) 
 be the word embedding of the ith word on real data. 
 
We call ![](fig/math/W.png) the similarity matrix of real data.  

Similarly, evaluate word embedding on generate data, and get the similarity matrix ![](fig/math/wp.png) of generation data. ![](fig/math/wpij.png),  ![](fig/math/epi.png) be the word embedding of the ith word on generated data. 

The EmbSim is defined as ![](fig/math/embsim.png)

where ![](fig/math/wi.png) is the ith column of ![](fig/math/W.png)

### Experiment Results

* epoch 1-80: pretrain process

* epoch 81-181: adversarial training process
  - gsGAN has no pretrain process

nll loss on oracle:

![](fig/nll.png)

inverse nll loss on oracle:

![](fig/inll.png)

embedding similarity on image coco:

![](fig/embsim.png)

BLEU:

on original dataset:

|            | seqGAN | maliGAN | rankGAN | leakGAN | textGAN  |
|------------|--------|---------|---------|---------|----------|
| BLEU2      | 0.917  | 0.887   | 0.937   | 0.926   | 0.650    
| BLEU3      | 0.747  | 0.697   | 0.799   | 0.816   | 0.645
| BLEU4      | 0.530  | 0.482   | 0.601   | 0.660   | 0.596
| BLEU5      | 0.348  | 0.312   | 0.414   | 0.470   | 0.523

on test dataset:

|       | seqGAN | maliGAN | rankGAN | leakGAN | textGAN      | training set |
|-------|--------|---------|---------|---------|--------------|--------------|
| BLEU2 | 0.745  | 0.673   | 0.743   | 0.746   | 0.593        | 0.740        |
| BLEU3 | 0.498  | 0.432   | 0.467   | 0.528   | 0.463        | 0.520        |
| BLEU4 | 0.294  | 0.257   | 0.264   | 0.355   | 0.277        | 0.337        |
| BLEU5 | 0.180  | 0.159   | 0.156   | 0.230   | 0.207        | 0.218        |

Mode Collapse (self-BLEU):


|            | seqGAN | maliGAN | rankGAN | leakGAN | textGAN       | training set  |
|------------|--------|---------|---------|---------|---------------|---------------|
| BLEU2      | 0.950  | 0.918   | 0.959   | 0.966   | 0.942         |0.892         |
| BLEU3      | 0.840  | 0.781   | 0.882   | 0.913   | 0.931         |0.747         |
| BLEU4      | 0.670  | 0.606   | 0.762   | 0.848   | 0.804         |0.573         |
| BLEU5      | 0.489  | 0.437   | 0.618   | 0.780   | 0.746         |0.415         |

Instances on image coco:

seqGAN:
```text
a very tall pointy across the street 
a bowl full of various cooking in to black kitchen 
a parked car with a woman hanging over a motorcycle . 
a bowl full monitor with a monitor next to a couple is painted painted in it . 
this with an image of a white toilet 
an image of a motorcycle decorated with tall trees 
```

maliGAN:
```text
a parked motorcycle driving on a side of a beach . 
a picture of a blurry photo of shops air with benches in american spectators off . 
a toilet is on a very parked at an entirely new fashion . 
several delta skiers in shower in front of a traffic road next to the ocean . 
a beautiful kitchen in white kitchen counter sitting next on a park . 
a man in a butcher kitchen tying picture . 
```

rankGAN:
```text
a nice bathroom with a bathroom sink and a small toilet . 
a bathroom with a pink toilet and a sink and a wall . 
a yellow bike parked by a window next to a woman holding a great underneath a book . 
a large island where some various utensils and green walls and a pink seat . 
a small child is riding a green quadruple cows around . 
man on white and white umbrella together is down . 
```

leakGAN:
```text
a bathroom with a mirrors reflection on far up into a tray and cup a banana to it 's back on it . 
a kitchen with white porcelain toilet , open and a glass of wine 
a bathroom with a glass shower , sink and scale . 
a man tinkers with his ear . 
a cat laying on a tub near the wall . 
two people are preparing food from a bar with a red fire wallpaper . 
```

textGAN:
```text
is in a bathroom with a sink controls . 
a motorcycle . 
of a bathroom with a sink . 
is in a bathroom with a toilet and a sink . 
a motorcycle . 
is is flying very of a bathroom with a bathroom . 
```