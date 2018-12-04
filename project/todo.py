from torch.nn import functional as F
from collections import defaultdict
from config import config
import torch

_config = config()
    

def evaluate(golden_list, predict_list):
    correct_labels = 0
    golden_labels, predicted_labels = 0, 0
    for i in range(len(golden_list)):
        # store tag position, phrase type and length
        g_dict, p_dict = defaultdict(int), defaultdict(int)
        # memory of previous tag
        g_key,  p_key  = (), ()
        for j in range(len(golden_list[i])):
            # check golden_list
            # if the previous tag is a "B" tag
            if g_key:
                # if it’s an "I" tag of the same phrase
                if golden_list[i][j][0] == 'I' and golden_list[i][j][2] == g_key[2]:
                    g_dict[g_key] += 1
                # else if it's a different "B" tag
                elif golden_list[i][j][0] == 'B' and golden_list[i][j][2] != g_key[2]:
                    g_key = (i, j, golden_list[i][j][2])
                    g_dict[g_key] += 1
                # else clear memory of previous tag
                else:
                    g_key = ()
            # if it's a "B" tag
            elif golden_list[i][j][0] == 'B':
                g_key = (i, j, golden_list[i][j][2])
                g_dict[g_key] += 1
            # check predict_list
            # if the previous tag is a "B" tag
            if p_key:
                # if it’s an "I" tag of the same phrase
                if predict_list[i][j][0] == 'I' and predict_list[i][j][2] == p_key[2]:
                    p_dict[p_key] += 1
                # else if it's a different "B" tag
                elif predict_list[i][j][0] == 'B' and predict_list[i][j][2] != p_key[2]:
                    p_key = (i, j, predict_list[i][j][2])
                    p_dict[p_key] += 1
                # else clear memory of previous tag
                else:
                    p_key = ()
            # if it's a "B" tag
            elif predict_list[i][j][0] == 'B':
                p_key = (i, j, predict_list[i][j][2])
                p_dict[p_key] += 1
        # for each key in p_dict that is also found in g_dict, and with the same value,
        # add to the number of correct labels
        for key in p_dict:
            if key in g_dict and p_dict[key] == g_dict[key]:
                correct_labels += 1
        golden_labels    += len(g_dict)
        predicted_labels += len(p_dict)
    if correct_labels == 0:
        return 1 if predicted_labels == 0 and golden_labels == 0 else 0
    else:
        precision = correct_labels / predicted_labels
        recall    = correct_labels / golden_labels
        return 2 * precision * recall / (precision + recall)


def new_LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + ((1-forgetgate) * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy


def get_char_sequence(model, batch_char_index_matrices, batch_word_len_lists):
    # Given an input of the size [s, w, c], we will convert it into a minibatch of the
    # shape [s*w, c] to represent s*w words (w in each sentence), and c characters in
    # each word.
    n_sentences, n_words, n_characters = batch_char_index_matrices.shape
    batch_char_index_matrices = batch_char_index_matrices.view(n_sentences*n_words, n_characters)
    batch_word_len_lists = batch_word_len_lists.view(n_sentences*n_words)
    
    # Get corresponding char_embeddings, we will have a Final Tensor of the shape
    # [s*w, c, 50]
    input_char_embeds = model.char_embeds(batch_char_index_matrices)
    # Sort the mini-batch wrt word-lengths, to form a pack_padded sequence.
    perm_idx, sorted_batch_word_len_list = model.sort_input(batch_word_len_lists)
    sorted_input_char_embeds = input_char_embeds[perm_idx]
    output_sequence = torch.nn.utils.rnn.pack_padded_sequence(sorted_input_char_embeds, lengths=sorted_batch_word_len_list.data.tolist(), batch_first=True)
    # Feed the pack_padded sequence to the char_LSTM layer.
    output_sequence, state = model.char_lstm(output_sequence)
    
    # Get hidden state of the shape [2, s*w, 50].
    hidden_state = state[0]
    # Concatenate the output of the forward-layer and the backward-layer.
    hidden_state = torch.cat((hidden_state[0], hidden_state[1]), dim=-1)
    # Recover the hidden_states corresponding to the unsorted index.
    _, desorted_indices = torch.sort(perm_idx, descending=False)
    hidden_state = hidden_state[desorted_indices]
    # Re-shape it to get a Tensor the shape [s, w, 100].
    hidden_state = hidden_state.view(n_sentences, n_words, 100)
    
    return hidden_state

