
This file is best viewed in Markdown reader (eg. https://jbt.github.io/markdown-editor/)  


  
  
# Installation  
  No Extra installation is needed. 

  
  
# Data  
  No extra pre processing is done. The data is used in an exactly similar fashion while training the Advanced Model as it was used earlier in training the Basic Model.

  
  
# Code Overview  
 The Advanced model is implemented by using only 1 CNN layer, followed by a GlobalMaxPooling2D layer. These two layers are available in `tensorflow.keras.layers` package and no other extra code source other than this is used. 
The model is initialized in the `__init__()` method of the `class MyAdvancedModel(models.Model):` class. `vocab_size` and `embed_dim` are passed to this class through a dictionary defined in the  `train_advanced.py` script. 
The forward pass is defined in the `call` method of the same class. The `regularization` and `loss` functions are completed in the script `train_lib.py` as per the given directions.
The code is also well commented for ease in readability.
  
## Train and Predict  
#### Train a model  
```  
python train_advanced.py --embed-file data/glove.6B.100d.txt --embed-dim 100 --batch-size 10 --epochs 5 
# stores the model by default at : serialization_dirs/advanced/  
```  
  
#### Predict with model  
```  
python predict.py --load-serialization-dir ./serialization_dirs/advanced/ --prediction-file ./predictions/advanced_test_pre
diction_1.txt --batch-size 10 
```  
## Extra Scripts  
  No other extra scripts or libraries have been used for the implementation of the Advanced Model. 
 
  
## What is turned in?  
  
A single zip file containing the following files:  
  
1. model.py  
2. train_advanced.py  
3. train_lib.py  
4. basic_test_prediction.txt  
5. advanced_test_prediction_1.txt  
6. advanced_test_prediction_2txt  
7. advanced_test_prediction_3.txt  
8. gdrive_link.txt

serialization_dirs.zip contains 4 different models. 3 models pertaining to the Advanced model and 1 basic model.
Most advanced model being placed in the 'advanced' folder.
  
# References  -

[1]  Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification ([https://www.aclweb.org/anthology/P16-2034.pdf](https://www.aclweb.org/anthology/P16-2034.pdf))

[2] Combining Recurrent and Convolutional Neural Networks for Relation Classification ([https://www.aclweb.org/anthology/C14-1220.pdf](https://www.aclweb.org/anthology/C14-1220.pdf))

[3] Relation Classification via Convolutional Deep Neural Network ([https://www.aclweb.org/anthology/C14-1220.pdf](https://www.aclweb.org/anthology/C14-1220.pdf))

[4] How to implement a basic CNN with tensorflow - ([https://towardsdatascience.com/tensorflow-2-0-create-and-train-a-vanilla-cnn-on-google-colab-c7a0ac86d61b](https://towardsdatascience.com/tensorflow-2-0-create-and-train-a-vanilla-cnn-on-google-colab-c7a0ac86d61b) )
