# FGSM adversarial attack for both of white and black box attack.

This is about the adversarial attack and defense
Main content is attack, defense will be uploaded soon

python version 3.7.5


* make a new anaconda enviroment
conda create -n name python=3.7.5

* pytorch, torchvision, cudatoolkit download : 
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch (pytorch hompage explains details)

* matplotlib
conda install matplotlib

you can write the code in the directory where you download the file.

python WhiteBox.py
python BlackBox.py

The WhiteBox.py file is about the fgsm attack.
The code was made with reference to the pytorch site.
* paper : Explaining and harnessing adversarial examples.(https://arxiv.org/abs/1412.6572)
* pytorch site
  adversarial attack : https://tutorials.pytorch.kr/beginner/fgsm_tutorial.html
  cnn classification : https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py

The BlackBox.py file is about the blacbox attack.
* paper : Practical Black-Box Attacks against Machine Learning (https://arxiv.org/abs/1602.02697)
