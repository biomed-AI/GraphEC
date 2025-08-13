# Introduction
Here, we proposed GraphEC, a geometric graph learning-based EC number predictor established on the active sites and predicted structures. Specifically, the enzyme active sites were first identified because of their critical role in enzyme function. Guided by the active sites, GraphEC was trained using the geometric graph learning with the ESMFold-predicted protein structures. To augment the features, informative sequence embeddings were generated using a pre-trained language model (ProtTrans). Finally, a label diffusion algorithm was utilized to further improve the prediction by incorporating homology information. Additionally, the optimum pH of enzymes was predicted to reflect the enzyme-catalyzed reactions. 
![figure1](https://github.com/YidongSong/GraphEC/assets/42714970/a4bbacbe-72d3-4884-9d94-7924f63a8aea)

# System requirement
GraphEC is developed under Linux environment with:
Python 3.8.16, numpy v1.24.3, pyg v2.3.0, pytorch v1.13.1, biopython v1.81, debugpy v1.6.7, decorator v5.1.1, filelock, v3.12.1, gmp v6.2.1, idna v3.4, ipython v8.12.0, openfold v1.0.1, scipy 1.10.1, and six v1.16.0

# Install and run the program
**1.** Clone this repository by `git clone https://github.com/biomed-AI/GraphEC.git`. 
      
**2.** Install the packages required by GraphEC. The [ESMFold](https://github.com/facebookresearch/esm) and [ProtTrans](https://github.com/agemagician/ProtTrans) can be installed, followed by their official tutorials. The pre-trained ProtT5-XL-UniRef50 model can be downloaded [here](https://zenodo.org/record/4644188) .   
      
**3.** Run GraphEC with the following command:    
      
```
bash run.sh EC_number ./Data/fasta/EC_number.fasta gpu_num
```
where ```EC_number``` represents the prediction task; ```./Data/fasta/EC_number.fasta``` represents the data needed to be predicted in fasta format; and ```gpu_num``` represents the GPU used to complete the prediction.   
      
The results can be found in ```./EC_number/results```, including the full predictions and top K predictive scores (K is defaulted to 5).   

### Note: 
If there are permission issues, please use the following code：   
```
chmod -R 755 ./EC_number/tools/
```

**4.** Run GraphEC-AS by the following command:    
```
bash run.sh ActiveSite ./Data/fasta/Active_sites.fasta gpu_num
```
where ```ActiveSite``` represents the prediction of active sites; ```./Data/fasta/Active_sites.fasta``` represents the data needed to be predicted in fasta format; and ```gpu_num``` represents the GPU used to complete the prediction. 

The results are saved in ```./Active_sites/results```

**5.** Run GraphEC-pH by the following command:

```
bash run.sh Optimum_pH ./Data/fasta/optimum_pH.fasta gpu_num
```
where ```Optimum_pH``` indicates the prediction of optimum pH; ```./Data/fasta/optimum_pH.fasta``` represents the data needed to be predicted in fasta format; and ```gpu_num``` represents the GPU used to complete the prediction.     

The results are saved in ```./Optimum_pH/results```

      
# Dataset and model   
**1.** EC number prediction    
The training set is provided in ```./EC_number/data/datasets/Training_set.csv```, with the two independent tests located in ```./EC_number/data/datasets/NEW-392.csv``` and ```./EC_number/data/datasets/Price-149.csv```   
The trained models are saved in ```./EC_number/model```     
       
**2.** Active site prediction   
The training set is saved in ```./Active_sites/data/datasets/train.pkl```, and the test set is saved in ```./Active_sites/data/datasets/test.pkl```    
The trained models can be found in  ```./Active_sites/model```

**3.** Optimum pH prediction   
The training set can be found in ```./Optimum_pH/data/datasets/train.pkl```, and the test set can be found in ```./Optimum_pH/data/datasets/test.pkl```     
The trained models are saved in ```./Optimum_pH/model```     

# Citation and contact
Citation: preparing

In case you have questions, please contact Yidong Song (songyd6@mail2.sysu.edu.cn).  







