import tensorflow as tf
import numpy as np
import copy

class Sequence :

    def __init__(self, character, probability) :
        self.characters = [character]
        self.sum_log_probability = tf.math.log(probability) 
        self.dead = False

    def add(self, character, probability) :
        if self.dead :
          return
        self.characters.append(character)
        self.sum_log_probability += tf.math.log(probability)
        if character == '\n' :
          self.dead = True
        
    def set_state(self, state) :
      self.state = state

    def set_enc_out(self, enc_out) :
      self.enc_out = enc_out

    def get_state(self) :
      return self.state

    def get_enc_out(self) :
      return self.enc_out

    def get_top(self) :
        return self.characters[-1]

class BeamSearch :

    best_seq_live = []
    best_seq_dead = []
    max_length = 0

    def __init__(self, beam_width, max_decoder_seq_length, decoder_character_map, batch_size, sos = '\t', eos = '\n') :
        self.beam_width = beam_width
        self.sos = sos
        self.best_seq_live = [[Sequence(self.sos, 1.0)] * beam_width] * batch_size

        for i in range(0, batch_size) :
          self.best_seq_live[i] = [Sequence(self.sos, 1.0)] * beam_width

        self.best_seq_dead = [[]] * batch_size

        for i in range(0, batch_size) :
          self.best_seq_dead[i] = []
        self.max_length = max_decoder_seq_length
        self.batch_size = batch_size

        # define character to one hot integer and one hot integer to character maps
        self.decoder_character_map = decoder_character_map
        self.reverse_decoder_char_map = dict((i, char) for char, i in self.decoder_character_map.items())
        self.eos = eos

    def get_individual_states(self, states_value, nlayers) :
      new_states_value = [0] * len(states_value[0])
      for i in range(0, nlayers) :
        for j in range(0, len(states_value[0])) :
          if i == 0 :
            new_states_value[j] = []
          new_states_value[j].append(states_value[i][j])
      return new_states_value

    def combine_states(self, states, nlayers) :
      states_values = [0] * nlayers
      for i in range(0, len(states)) :
        for j in range(0, len(states[0])) :
          if i == 0 :
            states_values[j] = []
          states_values[j].append(states[i][j])

      return states_values

    def apply(self, encoder, decoder, input_seq) :
        states_value, enc_out = encoder.predict(input_seq)
        nlayers = len(states_value)

        orig_states_value = copy.copy(states_value)
        orig_enc_out = copy.copy(enc_out)



        new_states_value = self.get_individual_states(states_value, nlayers)

        for i in range(0, self.batch_size) :
          self.best_seq_live[i][0].set_state(new_states_value[i])
          self.best_seq_live[i][0].set_enc_out(enc_out[i])

        final_seq_list = [[]] * self.batch_size
        for i in range(0, self.batch_size) :
          final_seq_list[i] = []


        flag = True

        for _ in range(0, self.max_length) :

            if flag :
              states = []
            if flag :
              enc_out = []

            total_seq = 0
            for seq_batch in self.best_seq_live :
              for seq in seq_batch :
                if flag :
                  states.append(seq.get_state())
                if flag :
                  enc_out += [seq.get_enc_out()]

                total_seq += 1

            if total_seq == 0 :
              break

            if flag :
              enc_out = np.array(enc_out)
            if flag :
              states_value = self.combine_states(states, nlayers)

            if flag :
              for i in range(nlayers) :
                states_value[i] = np.array(states_value[i])

            flag = False


            target_seq = np.zeros((total_seq, 1, len(self.decoder_character_map) + 1))

            index = 0
            for seq_batch in self.best_seq_live :
              for seq in seq_batch :
                curr_character = seq.get_top()
                target_seq[index, 0, self.decoder_character_map[curr_character]] = 1
                index += 1


            to_split = decoder.predict([target_seq, states_value, enc_out])

            output_tokens, new_states_values, attn_weights = to_split[0], list(
            to_split[1:-1]), to_split[-1]


            states_value = new_states_values

            itr = 0
            new_best_seq_live = []
            for (j, seq_batch) in enumerate(self.best_seq_live) :
              new_best_seq_live.append([])
              for seq in seq_batch :
                top_probabilities, indexes = tf.nn.top_k(output_tokens[itr, 0], self.beam_width)

                for i, index in enumerate(indexes) :
                    new_seq = copy.deepcopy(seq)
                    new_seq.add(self.reverse_decoder_char_map[index.numpy()], top_probabilities[i].numpy())
                    new_seq.set_state(states_value)
                    new_seq.set_enc_out(seq.get_enc_out())

                    if new_seq.get_top() == self.eos :
                        new_best_seq_live[j].append(new_seq)
                    else :
                        new_best_seq_live[j].append(new_seq)

                itr += 1

              seen = set()
              temp = []
              for e in new_best_seq_live[j] :
                if tuple(e.characters) not in seen :
                  temp.append(e)
                  seen.add(tuple(e.characters))
              new_best_seq_live[j] = temp

              new_best_seq_live[j] = sorted(new_best_seq_live[j], key = lambda seq1 : -seq1.sum_log_probability.numpy() / (len(seq1.characters) ** 0.7))


            for (j, seq_batch) in enumerate(self.best_seq_live) :
              self.best_seq_live[j] = new_best_seq_live[j][:self.beam_width]


        for i in range(0, len(final_seq_list)) :
          final_seq_list[i] = self.best_seq_dead[i] + self.best_seq_live[i]
          final_seq_list[i] = sorted(final_seq_list[i], key = lambda seq1 : -seq1.sum_log_probability.numpy() / (len(seq1.characters) ** 0.7))


        new_best_seq_live = []
        for i in range(0, len(final_seq_list)) :
          final_seq_list[i] = final_seq_list[i][:self.beam_width]
        return final_seq_list

'''
Sample Usage
bs = BeamSearch(3, data_encoder.max_decoder_seq_length, data_encoder.target_token_index, BATCH_SIZE)
bs.apply(encoder, decoder, input_seqs)
'''

 
