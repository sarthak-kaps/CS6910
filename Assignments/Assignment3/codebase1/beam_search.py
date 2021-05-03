import tensorflow as tf
import numpy as np
import copy

class Sequence :

    def __init__(self, character, probability) :
        self.characters = [character]
        self.sum_log_probability = tf.math.log(probability) 

    def add(self, character, probability) :
        self.characters.append(character)
        self.sum_log_probability += tf.math.log(probability)

    def get_top(self) :
        return self.characters[-1]

class BeamSearch :

    best_seq_live = []
    best_seq_dead = []
    max_length = 0

    def __init__(self, beam_width, max_decoder_seq_length, decoder_character_map, sos = '\t', eos = '\n') :
        self.beam_width = beam_width
        self.sos = sos
        self.best_seq_live = [Sequence(self.sos, 1.0)] 
        self.best_seq_dead = []
        self.max_length = max_decoder_seq_length
        
        # define character to one hot integer and one hot integer to character maps
        self.decoder_character_map = decoder_character_map
        self.reverse_decoder_char_map = dict((i, char) for char, i in self.decoder_character_map.items())
        self.eos = eos
        
    def apply(self, encoder, decoder, input_seq) :
        states_values = [encoder(input_seq)] * self.beam_width
        final_seq_list = []
        
        for _ in range(0, self.max_length) :
            new_best_seq_live = []
            
            if len(self.best_seq_live) == 0 :
                break

            for i, seq in enumerate(self.best_seq_live) :
                target_seq = np.zeros((1, 1, len(self.decoder_character_map) + 1))
                curr_character = seq.get_top()
            
                target_seq[0, 0, self.decoder_character_map[curr_character]] = 1

                
                if type(states_values[i]) != list:
                    to_split = decoder.predict([target_seq] + [states_values[i]])
                else:
                    to_split = decoder.predict([target_seq] + states_values[i])
                output_tokens, states_values[i] = to_split[0], to_split[1:]

                top_probabilities, indexes = tf.nn.top_k(output_tokens[0, 0], self.beam_width)
#                 print(top_probabilities)
#                 print(len(indexes.numpy()))
                
                for i, index in enumerate(indexes) :
                    new_seq = copy.deepcopy(seq)
#                     print(self.reverse_decoder_char_map[index.numpy()])
                    new_seq.add(self.reverse_decoder_char_map[index.numpy()], top_probabilities[i].numpy())
                    if new_seq.get_top() == self.eos :
                        self.best_seq_dead.append(new_seq)
                    else :
                        new_best_seq_live.append(new_seq)

            self.best_seq_live = []
            new_best_seq_live = sorted(new_best_seq_live, key = lambda seq1 : -seq1.sum_log_probability.numpy() / (len(seq1.characters) ** (0.7)))
            
            self.best_seq_live = new_best_seq_live[:self.beam_width]
#             print(list(map(lambda x : x.characters, self.best_seq_live)))

        
        final_seq_list = self.best_seq_dead + self.best_seq_live
        final_seq_list = sorted(final_seq_list, key = lambda seq1 : -seq1.sum_log_probability.numpy() / (len(seq1.characters) ** (0.7)))
    
#         print(list(map(lambda x : x.characters, final_seq_list))) 
      
        new_best_seq_live = []
        return final_seq_list[:self.beam_width]

'''
Sample Usage
bs = BeamSearch(3, data_encoder.max_decoder_seq_length, data_encoder.target_token_index)
'''

