<h1><img src="docs/fig/texygen-01.png" width="250"></h1>

Texygen is a text generation model benchmark platform for text generation researchers to easily conduct repeatable experiments and compare performance.
It includes the most poplular GAN models in the text generation area along with some evaluation metrics.

Should you have any questions and enquiries, please feel free to contact Yaoming Zhu (ym-zhu [AT] outlook.com) and [Weinan Zhang](http://wnzhang.net) (wnzhang [AT] sjtu.edu.cn).

## Requirement
We suggest you run the platform under Python 3.6+ with following libs:
* **tensorflow==1.3.0**
* numpy==1.12.1
* scipy==0.19.0
* nltk==3.2.3
* CUDA 7.5+ (Suggested for GPU speed up, not compulsory)    

Or just type `pip install requirements.txt` in your terminal.

## Implemented Models and Original Papers

SeqGan --  [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/abs/1609.05473)

MaliGan -- [Maximum-Likelihood Augmented Discrete Generative Adversarial Networks](https://arxiv.org/abs/1702.07983)

RankGan -- [Adversarial ranking for language generation](http://papers.nips.cc/paper/6908-adversarial-ranking-for-language-generation)

LeakGan -- [Long Text Generation via Adversarial Training with Leaked Information](https://arxiv.org/abs/1709.08624)

TextGan -- [Adversarial Feature Matching for Text Generation](https://arxiv.org/abs/1706.03850)
 
GSGan -- [GANS for Sequences of Discrete Elements with the Gumbel-softmax Distribution](https://arxiv.org/abs/1611.04051)


## Get Started

```bash
git clone https://github.com/geek-ai/Texygen.git
cd Texygen
# run SeqGAN with default setting
python3 main.py
```
More detailed documentation for the platform and code setup is provided [here](docs/doc.md).


## Evaluation Results

More detailed benchmark settings and evaluation results are provided [here](docs/evaluation.md).


