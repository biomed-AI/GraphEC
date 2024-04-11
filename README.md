# Introduction
Here, we proposed GraphEC, a geometric graph learning-based EC number predictor established on the active sites and predicted structures. Specifically, the enzyme active sites were first identified because of their critical role in enzyme function. Guided by the active sites, GraphEC was trained using the geometric graph learning with the ESMFold-predicted protein structures. To augment the features, informative sequence embeddings were generated using a pre-trained language model (ProtTrans). Finally, a label diffusion algorithm was utilized to further improve the prediction by incorporating homology information. Additionally, the optimum pH of enzymes was predicted to reflect the enzyme-catalyzed reactions. 
![figure1](https://github.com/YidongSong/GraphEC/assets/42714970/a4bbacbe-72d3-4884-9d94-7924f63a8aea)

# System requirement
GraphEC is developed under Linux environment with:
Python 3.8.16, numpy v1.24.3, pyg v2.3.0, pytorch v1.13.1, biopython v1.81, debugpy v1.6.7, decorator v5.1.1, filelock, v3.12.1, gmp v6.2.1, idna v3.4, ipython v8.12.0, openfold v1.0.1, scipy 1.10.1, and six v1.16.0

# Install and set up GPSite
**1.** Clone this repository by https://github.com/YidongSong/GraphEC  
**2.** Install the packages required by GraphEC. The [ESMFold](https://github.com/facebookresearch/esm) and [ProtTrans](https://github.com/agemagician/ProtTrans) can be installed followed by their official tutorials. The pre-trained ProtT5-XL-UniRef50 model can be downloaded [here](https://zenodo.org/record/4644188)
