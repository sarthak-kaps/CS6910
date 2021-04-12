## Part A
This part consists of the following files -
* `data_gen.py` - This file contains the code to read the `iNaturalist` dataset and returns the train, test and validation dataset. <br>
                 
  - The function that achieves this is `generate_dataset(config)`. <br>
                  It takes in a configuration file which specifies whether data augmentation needs to be done or not and what batch sizes 
                  are considered. <br>
  - We do a 90 - 10 split of train and validation data.
                  
* `model_maker.py` - This is the file in which we make our model.
  - The file contains a function `make_model(config)` that reads a configuration file and constructs the model according to the configuration specified. 
   - This function returns the constructed model.

* `sweep.py` - This the file which performs a sweep on various hyperparameters of the model and logs the results and features in wandb.
               This file makes call to `data_generator` in `data_gen.py` to get the train, validation and test data.
               It then calls `make_model` in `model_maker.py` to prepare the model. The model is the compiled and trained on the dataset.

* `all_config.yaml`, `new_config.yaml` and `special_config.yaml` - These specify the set of configurations we consider for training. They are 
                                                                   specified in detail in the report. Each of these makes calls to `sweep.py` 
                                                                   with a configuration. <br>
                                                                   To run a sweep use the following command - `wandb sweep (name of config file)`

* `best_model_run.py` - This file trains the best model we found and saves it for use in testing. <br>
                        We used the `model.save(<model name>)` functionality. <br>
                        Our saved model is present in the folder `org_linear_inc_nfLayer1_32_dataAug_False_drpout_0_bs_32_opt_nadam` <br>
                        You can load it using the command - `tf.keras.models.load_model(org_linear_inc_nfLayer1_32_dataAug_False_drpout_0_bs_32_opt_nadam)`

* `test.py` - This file tests our best model on the test dataset. It also generates all the images and visualisations seen in the report. In particular we saw the performance of our model on some sample images and plotted it with some details. This is done by the function `plot_sample_images()`. We also visualized the layer filters with the function `visualize_layer_1_filter()`. We also visualized the effect of layer 1 filters on a random image using the function `visualize_layer_1_filter_on_image`.

* `guided_back_propagation.py` - This file applies the idea of guided back propagation on our best model to get some interesting visualisations 
                                 for the last convolution layer. <br>
                                 We used the `@tf.custom_gradient` functionality to define our own gradient during
                                 backpropagation. We zeroed all negative gradients and allowed the positive gradients to flow. <br>
                                 We later overrode the `ReLU` activation with our new activation layer with the above gradient properties. <br>
                                 We then considered 10 random neurons and ran the model over all the images and visualized the inputs for which the neurons 
                                 fired and saved them. <br>
                                 We observed some interesting patterns.

## Results

Our best model which we saved gave the following accuracies - 
| DATASET          | ACCURACY | LOSS |
|------------------|-----------------------------------------------|----------------------|
| Training            |             50.24 %                    | 1.43                    |
| Validation | 42.06 %                 | 1.754        | 
| Test    | 42.50 %                                   | 1.752                    |

The visualizations can be found in the wandb report.
