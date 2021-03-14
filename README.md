# Adversarial Deep Ensemble for Malware Detection
This code repository is for the paper, entitled **Adversarial Deep Ensemble: Evasion Attacks and Defenses for Malware Detection** by Deqiang Li and Qianmu Li (IEEE TIFS). Please check out the early access version [here](https://ieeexplore.ieee.org/document/9121297). 

## Overview
Our research question is how effective the ensemble attack and how robust the ensemble defense when they combat with each other. 
we enhance the robustness of deep neural network (DNN) by incorporating two defense techniques: adversarial training and ensemble (i.e., adversarial deep ensemble for short).
The hardened DNNs are applied to an interesting context: adversarial mawlare detection. More specifically, we consider the Android malware examples. The main features of this repository are listed in the following:
* Combat ensemble-based defense models with ensemble-based attacks;
* Implement 5 defense methods for malware detection.
* Implement more than 13 attacks, including gradient-based attacks, gradient-free attacks, transfer attacks, and mixture of attacks (ensemble based).
* Generate the executable adversarial malware examples (APKs) automatically at scale.
* Perturb a mawlare example using a large degree of manipulations such as Java reflection, Activities renaming, and etc.
  
## Dependencies:
We develop codes on the system of **Ubuntu**. The leveraged packages are as follows:
* python 2.7
* tensorflow-gpu==1.9.0 or 1.14.0
* numpy >= 1.15.4
* scikit-Learn >= 0.20.3
* [androguard 3.3.5](https://github.com/androguard/androguard/releases/tag/v3.3.5)
* [apktool](https://ibotpeaches.github.io/Apktool/)

Most of dependencies can be installed by 'pip' (e.g., pip install -r requirements.txt), except for the toolkit of [apktool](https://ibotpeaches.github.io/Apktool/) which shall be installed by following the official document of its own. Though we also cope with some incompatible issues to accommodate python 3.6, a thorough test is never conducted. 


## Usage
  #### 1. Dataset
  * For apk files, we recommend the [Drebin](https://www.sec.cs.tu-bs.de/~danarp/drebin/) and [Androzoo](https://androzoo.uni.lu/). Note that both datasets are required to follow the policies of their own to obtain the apks. We re-compose the benign data of Drebin, of which the sha256s are available at [here](https://drive.google.com/drive/folders/1AHnNhtE2-YLWj8jeyciW52lFqFGdEmTB?usp=sharing). Correspondingly, these apks files can be download from [Androzoo](https://androzoo.uni.lu/).
  * For the preprocessed data, we provide the pre-processed via [drebin feature extraction](https://www.sec.cs.tu-bs.de/pubs/2014-ndss.pdf), which can be found [here](https://drive.google.com/open?id=1AHnNhtE2-YLWj8jeyciW52lFqFGdEmTB).  
  * For waging attacks on the Drebin dataset, we randomly select 800 malware examples, of which a list of sha256s, named `attack.list`, is available [here](https://drive.google.com/open?id=1AHnNhtE2-YLWj8jeyciW52lFqFGdEmTB)
  #### 2. Configure
  We are required to change the `conf` by `project_root=/absolute/path/to/adv-dnn-ens-malware/` and `database_dir = /absolute/path/to/drebin/` to accommodate the current project and dataset paths. To be spical, in the folder of `database_dir`, the structure shall be:
  ```
  drebin
  |   attack.list % sha256 of 800 apks
  |---drebin % the folder saves information about pre-processed data
        |   normalizer
        |   vocabulary.pkl
        |   vocabulary_info.pkl
        |   X.pkl
        |   y.pkl
  |---benign_samples % the folder contains benign apk files (optional if 'drebin' feature exists)
  |---malicious_samples % the folder contains malicious apk files (at least contains 800 APKs corresponding to the attack.list)
  |---attack % this folder contains attack results and will be created by default
  ```
 #### 3. Run some scripts
We suggest the following motions to perform the code: Learn a basic dnn; Generate adversarial malware examples; Learn a defense model.

&emsp; (1). Learn a basic model (i.e., no defensive effort is put on the model):
```
python main.py learner -t
``` 
&emsp; (2). Generate adversarial representation against the basic model 
```
python main.py attack -v basic_dnn -m fgsm
```
&emsp; More commands for performing other attack methods (e.g., `gdkde`, `pgdl1`, `pgdl2`, `pgdlinf`, `jsma`, `bca_k`, `max`, etc) against other models can be found in [`main.py`](./main.py). This means we can wage 
other attacks conveniently by an instruction, for example `gdkde`:
```
python main.py attack -v basic_dnn -m gdkde
```
All the hyper-parameters for the attack methods can be found in [`attack_manager.py`](./attacker/attack_manager.py).

&emsp; (2.1). Furthermore, we can generate executable adversarial examples by appending an extra `-r`, for example waging `fgsm` attack against the basic model:
```
python main.py attack -v basic_dnn -m fgsm -r
```

&emsp; (3). Learn the hardened model for example using `adversarial training` with the attack `rfgsm`:
```
python main.py defender -d atrfgsm -t
```
&emsp; Similarly, more commands for instantiating other adversarial training defenses incorporating an attack (e.g., adversarial training using adam, mixture of attacks, adversarial deep ensemble) can be found in [`main.py`](./main.py).
In addition, we can wage attack against the defense model once we finish the corresponding training process:
```
python main.py attacker -v atrfgsm -m fgsm 
```

&emsp; (4). Test defense model on pristine test set:
```
python main.py defender -d atrfgsm -p 
python main.py learner -p
```
&emsp; (5). Test defense model on adversarial representation/examples set:
```
python main.py defender -d atrfgsm -a
python main.py learner -a
``` 
We can specify a set of adversarial example by assigning a directory to the variable `adv_sample_dir` in the [config](./conf) file.

## Learned Parameters

All learned model will be saved into the current directory under `save` folder that can be redirected by settings in the file of `conf`. We also provides some defenses models, which can be obtained [here](https://drive.google.com/open?id=1AHnNhtE2-YLWj8jeyciW52lFqFGdEmTB)

## Adversarial APKs
Following the nice suggestion from researcher Teenu S. John, we share some of the generated APKs via a shared link for research purposes ([request form](https://forms.gle/3iVpg6vGtRZBqPhGA)). 

## Acknowledgement

We adapt some codes from the following repositories:
* [Drebin feature extraction](https://github.com/MLDroid/drebin)
* [CleverHans](https://github.com/tensorflow/cleverhans)
* [avpass](https://github.com/sslab-gatech/avpass)

## Contacts

Welcome to dedicate yourselves into adversarial mawlare detection! If you have any questions or would like to make contributions to this repository such as [issuing](https://github.com/deqangss/adv-dnn-ens-malware/issues) for us, please do not hesitate to contact us: `lideqiang@njust.edu.cn`.

## License

* For ethical consideration, all the code presented on this repository is for educational/research proposes solely. The illegal or misuse of the code can lead to criminal behaviours. We (our organization and authors) will not be held responsible in any criminal charges.

* This project is released under the [GPL license](./LICENSE).


## Citation

If you'd like to cite us in a project or publication, please include a reference to the IEEE TIFS paper:
```buildoutcfg
@ARTICLE{9121297,
  author={D. {Li} and Q. {Li}},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Adversarial Deep Ensemble: Evasion Attacks and Defenses for Malware Detection},
  year={2020},
  volume={15},
  number={},
  pages={3886-3900},
  doi={10.1109/TIFS.2020.3003571}
}
```
