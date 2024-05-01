<div align="center">
  
# LM-OGlc-NAc-Site
Integrating Embeddings from Multiple Protein Language Models to Improve Protein O-GlcNAc Site Prediction

</div>
<p align="center">
<a href="https://www.python.org/"><img alt="python" src="https://img.shields.io/badge/Python-3.10.0-yellow.svg"/></a>
<a href="https://github.com/agemagician/Ankh"><img alt="Ankh" src="https://img.shields.io/badge/Ankh-1.10.0-teal.svg"/></a>
<a href="https://biopython.org/"><img alt="Bio" src="https://img.shields.io/badge/Bio-1.7.0-brightgreen.svg"/></a>
<a href="https://pypi.org/project/fair-esm/"><img alt="fair-esm" src="https://img.shields.io/badge/fair--esm-2.0.0-purple.svg"/></a>
<a href="https://keras.io/"><img alt="Keras" src="https://img.shields.io/badge/Keras-2.8.0-red.svg"/></a>
<a href="https://numpy.org/"><img alt="numpy" src="https://img.shields.io/badge/numpy-1.26.4-white.svg"/></a>
<a href="https://pandas.pydata.org/"><img alt="pandas" src="https://img.shields.io/badge/pandas-2.2.2-orange.svg"/></a>
<a href="https://scikit-learn.org/"><img alt="scikit_learn" src="https://img.shields.io/badge/scikit_learn-1.4.2-blue.svg"/></a>
<a href="scipy.org"><img alt="SciPy" src="https://img.shields.io/badge/SciPy-1.13.0-navy.svg"/></a>
<a href="https://www.tensorflow.org/"><img alt="tensorflow" src="https://img.shields.io/badge/TensorFlow-2.8.0-orange.svg"/></a>
<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.3.0-orange.svg"/></a>
<a href="https://tqdm.github.io/"><img alt="tqdm" src="https://img.shields.io/badge/tqdm-4.66.2-blue.svg"/></a>
<a href="https://huggingface.co/transformers/"><img alt="Transformers" src="https://img.shields.io/badge/Transformers-4.40.1-yellow.svg"/></a>
<div align="center">
<a href="https://github.com/KCLabMTU/LM-OGlcNAc-Site/commits/main"><img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/KCLabMTU/LM-OGlcNAc-Site.svg?style=flat&color=blue"></a>
<a href="https://github.com/KCLabMTU/LM-OGlcNAc-Site/pulls"><img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/KCLabMTU/LM-OGlcNAc-Site.svg?style=flat&color=blue"></a>
</div>

</p>

## Webserver
http://kcdukkalab.org/LMOGlcNAcSite

## Authors
Suresh Pokharel<sup>1</sup>, Pawel Pratyush<sup>1</sup>, Hamid D. Ismail<sup>1</sup>, Junfeng Ma<sup>2</sup>, Dukka B KC<sup>1*</sup>
<br>
<sup>1</sup>Department of Computer Science, Michigan Technological University, Houghton, MI, USA
<br>
<sup>2</sup>
Department of Oncology, Lombardi Comprehensive Cancer Center, Georgetown University Medical Center, Georgetown University, Washington, DC 20057, USA
<br>
<sup>*</sup> Corresponding Author: dbkc@mtu.edu

## Clone the Repository

If Git is installed on your system, clone the repository by running the following command in your terminal:

```shell
git clone github.com:KCLabMTU/LM-OGlcNAc-Site.git
```
###  - Or -
## Download the Repository
If you do not have Git or perfer to download directly:
Download the repository directly from GitHub. [Click Here](https://github.com/KCLabMTU/LMCrot/archive/refs/heads/main.zip) to download the repository as a zip file.

### Install Libraries 
Python version: `3.10.0`

To intall the required libraries, run the following command:
```shell
pip install -r requirements.txt
```
Required libraries and versions:
<code>
ankh==1.10.0
Bio==1.7.0
biopython==1.83
datasets==2.19.0
fair_esm==2.0.0
keras==2.8.0
numpy==1.26.4
pandas==2.2.2
protobuf==3.20.*
scikit_learn==1.4.2
scipy==1.13.0
tensorflow==2.8.0
torch==2.3.0
tqdm==4.66.2
transformers==4.40.1
</code>

## To run `LM-OGlcNAc-Site` model on your own sequences 

In order to predict succinylation site using your own sequence, you need to have two inputs:
1. Copy sequences you want to predict to `input/sequence.fasta`
2. Run `python predict.py`
3. Find results inside the `output` folder in a csv file named `results.csv`

## Citation
Pokharel, S.; Pratyush, P.; Ismail, H.D.; Ma, J.; KC, D.B. Integrating Embeddings from Multiple Protein Language Models to Improve Protein O-GlcNAc Site Prediction. Int. J. Mol. Sci. 2023, 24, 16000. [https://doi.org/10.3390/ijms242116000](https://doi.org/10.3390/ijms242116000)

Paper Link: [https://www.mdpi.com/1422-0067/24/21/16000](https://www.mdpi.com/1422-0067/24/21/16000)


## Contact
Please send an email to [sureshp@mtu.edu](sureshp@mtu.edu) (CC: [dbkc@mtu.edu](mailtodbkc@mtu.edu), [ppratyus@mtu.edu](mailto:ppratyus@mtu.edu) for any kind of queries and discussions.
