**ASSIGNMENT 1**

Our implementation is heavily inspired from keras implementation for the same.
Our constructed modules have been aimed to maintain the same API.

The assignment consists of the following files - 
  1) models.py - Contains the basic framework for our Feed Forward Neural Network.
                 Implementation of the Sequential Class, Dense Layer Class And Softmax Class
                 Contains implementation for forward and back propgation algorithms.
                 Other functions implemented include the compile function, fit function, predict function, 
                 predict_probab function and the evaluate function.
  
  2) optimizers.py - Contains the implementation of the generic optimizer class.
                     Various optimizers are implemented including SGD (with support for momentum and nesterov acceleration), 
                     RMSprop, Adam and Nadam.
                     Each of the optimzers can be incorporated easily with the models created.
