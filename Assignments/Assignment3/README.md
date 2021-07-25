# Recurrent transliteration system
Built a recurrent neural networks to build a transliteration system.

We experimented with the Dakshina dataset released by Google. This dataset contains pairs of the following form: `xxx yyy` 
Ex : `ajanabee अजनबी`

i.e., a word in the native script and its corresponding transliteration in the Latin script (the way we type while chatting with our friends on WhatsApp etc)

This is the problem of mapping a sequence of characters in one language to a sequence of characters in another language. Notice that this is a scaled down version of the problem of translation where the goal is to translate a sequence of words in one language to a sequence of words in another language (as opposed to sequence of characters here)

The Project has been implemented in 3 folders - 
* Codebase1 - Contains code to build Recurrent model without attention, Read the README of codebase1 for more details.
* Codebase2 - Contains code to build Recurrent mdoel with attention, Read the README of codebase2 for more details.
* sweep_configs - Contains the sweep configurations used for hyperparamters search - 
  * sweep_without_attention.yaml - Contains configurations for models that do not have attention and beam search
  * sweep_config_beamsearch.yaml - Contains configurations for models that do not have attention but have beam search
  * sweep_config_attention.yaml - Contains configurations for models that use attention 
  * sweep_config_attention_final.yaml - Contains more configurations for models that use attention
