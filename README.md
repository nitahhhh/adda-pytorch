# adda-pytorch
Implementation of "Adversarial Discriminative Domain Adaptation"(https://arxiv.org/abs/1702.05464) on pytorch 


### dataset 
mnist -> usps</br>
usps -> mnist
</br>

### command
##### $ python train_src.py --config "A->B config-file"
#training a model on source domain data </br>
"A->B config-file" should be usps2mnist.yaml or mnist2usps.yaml</br>
val reports the accuracy on target data
</br></br>

##### $ python main.py --config "A->B config-file"
#adapt the model trained on source data to fit target data with ADDA method</br>
"A->B config-file" should be usps2mnist.yaml or mnist2usps.yaml
</br></br>


### performance on testing set
| Method       | mnist -> usps   | usps -> mnist | 
| ------------- |:-------------:|:-------------:| 
| source only      | 84% (1560/1860)| 78% (7820/10000)|
| adapted      | 92% (1709/1860)    |91% (9074/10000)|
