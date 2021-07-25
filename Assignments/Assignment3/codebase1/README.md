# Sequence To Sequence Model without Attention

In this codebase we implemented a sequence to sequence model that took in a word from the English language and translated it to an equivalent word in the Hindi language
such that the pronunciation of the word does not change when read in both the languages.

For example our model will do the following - <br>
Input word - pichde,  Output word - पिछड़े

Codebase1 consists of the following files - 
* model_maker.py - 
  * This file implements the `make_model()` function which takes in a configuration file and generates a seq2seq model according to the specifications of the configuration.
  * The seq2seq model contains of 2 major components - the encoder and decoder.
  * The function also generates the encoder and decoder models separately so that we can use them for inference later.
  * The `make_model()` function returns the model and along with it returns the individual encoder and decoder models. 
* encode_input.py - 
  *  This file implements a one hot encoder which reads the data given in the train, validation and test files and constructs a one hot encoding of the characters for both the languages.
  *  This implementation is done in `class one_hot_encoder`.
  *  The function `vectorize_data` in the class performs this encoding.
  *  We only considered one hot encoding for character level translation as other encodings were not of much relevance to the job in hand.
* sweep.py - 
  * This file runs the sweep on a particular given configuration.
  * It constructs the one hot encoding of the data by using the functionalities present in the `encode_input` file.
  * It constructs the model by using the `make_model()` function in the `model_maker` file. It does this by passing the configutation dictionary to `model_maker`. 
  * Then model is then compiled and fit to the data.
  * We finally do the inference on the validation data.
* sweep_with_beam_search.py - 
  * This file does the exact same job as `sweep.py`.
  * The only difference is that we consider another parameter called `beam_width`, which is used during validation.
  * During inference on the validation data this file performs beam search with given beam width.
* beam_search.py - 
  * This file implements beam search.
  * This is a more basic version and runs slow, it is given here for clarity.
* beam_search_fast.py -
 * This file is a more efficient implementation of beam search that we use for inference.
 * The implementation is carried out through 2 classes - 
   * `class Sequence` - This class maintains the sequence and its meta data during decoding.
   * `class BeamSearch` - This class contains the core of the implementation of beam search.
* test.py - 
   * This file runs the given model on the test data.
   * The path to the directory where the model is saved needs to be provided as input.
   * We have uploaded our best model here so that you can run the best model on the test data.
* test_with_beam_search.py - 
   * This file does the same job as `test.py`.
   * The only difference is that testing is done using beam search.
   * Along with the path to the model we need to supply the beam width as input.  

## Best Models
We have uploaded 2 of our best models - 
* Good_model_GRU_beam_search - This is uploaded on wandb report as well. Test Accuracy - 41 %
* Good_model_GRU_beam_5 - This model is similar to above but with beam width 5. Test Accuracy - 41 %.

## Testing
You can run the test files as follows - 
* test.py - 
  * Use the command : `$ python test.py`. 
  * It will ask the name of the model. Enter one of the 2 best models as name and it will run the test data over them without beam search.
* test_with_beam_search.py - 
  * Use the command : `$ python test_with_beam_search.py`.
  * It will ask the name of the model. Enter one of the 2 best models as name.
  * It will ask for beam width. Enter beam width value (preferably 2, 3, 4, 5).
  * It will run the test data over the model with beam search.
