![](Doc/fig/texygen-01.png)

TexyGen is a text generation model evaluation platform, which included most poplular GAN models in the text generation area 
along with some evaluation metric.

It's easy to run and get evaluation results.

## Requirement
We suggest you run the platform under Python 3.6+ with following libs:
* **tensorflow==1.3.0**
* numpy==1.12.1
* scipy==0.19.0
* nltk==3.2.3
* CUDA 7.5+ (Suggested for GPU speed up, not compulsory)    

Or just type `pip install requirements.txt` in your terminal.

## Paper

// TODO

#### Implemented Models and Original Paper:

seqGan--  [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/abs/1609.05473)

maliGan-- [Maximum-Likelihood Augmented Discrete Generative Adversarial Networks](https://arxiv.org/abs/1702.07983)

rankGan-- [Adversarial ranking for language generation](http://papers.nips.cc/paper/6908-adversarial-ranking-for-language-generation)

leakGan-- [Long Text Generation via Adversarial Training with Leaked Information](https://arxiv.org/abs/1709.08624)

textGan-- [Adversarial Feature Matching for Text Generation](https://arxiv.org/abs/1706.03850)
 
gsGan-- [GANS for Sequences of Discrete Elements with the Gumbel-softmax Distribution](https://arxiv.org/abs/1611.04051)


## Get Started

```bash
git clone 
cd apex-text-gen
# run SeqGAN with default setting
python3 main.py
```
[Document](Doc/doc.md)


## Evaluation Results

[evaluation results](Doc/evaluation.md)
