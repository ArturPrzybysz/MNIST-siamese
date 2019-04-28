## SIAMESE CNN  
After an attempt to solve dog vs cats with siamese net, I have moved to MNIST dataset.  
Solving MNIST was surprisingly easy. After a few epochs on random triplets the problem seemed to be solved.

### Relevant papers and other sources
During my work on this project I have found help in these works:  
https://arxiv.org/pdf/1503.03832.pdf  
https://arxiv.org/pdf/1706.07567.pdf
https://github.com/omoindrot/tensorflow-triplet-loss/issues/6 (discussion on stuck loss)  

### Results
TSNE for test set:  
![](http://github.com/ArturPrzybysz/MNIST-siamese/img/testTSNE.png)

TSNE for train set:
![](http://github.com/ArturPrzybysz/MNIST-siamese/img/trainTSNE.png)