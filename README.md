# LM-OGlc-NAc-Site
Integrating Embeddings from Multiple Protein Language Models to Improve Protein O-GlcNAc Site Prediction

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
<!-- Is there a specific Python Library required? -->

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
3. Find results inside the `output` folder named `results.csv`

## Citation
Pokharel, S.; Pratyush, P.; Ismail, H.D.; Ma, J.; KC, D.B. Integrating Embeddings from Multiple Protein Language Models to Improve Protein O-GlcNAc Site Prediction. Int. J. Mol. Sci. 2023, 24, 16000. [https://doi.org/10.3390/ijms242116000](https://doi.org/10.3390/ijms242116000)

Paper Link: [https://www.mdpi.com/1422-0067/24/21/16000](https://www.mdpi.com/1422-0067/24/21/16000)


## Contact
Please send an email to [sureshp@mtu.edu](sureshp@mtu.edu) (CC: [dbkc@mtu.edu](mailtodbkc@mtu.edu), [ppratyus@mtu.edu](mailto:ppratyus@mtu.edu) for any kind of queries and discussions.
