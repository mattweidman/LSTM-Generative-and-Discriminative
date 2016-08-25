import numpy as np

empty_char = '\00'

# returns a sorted list of distinct characters found in filename
# removes the newline character and adds the '\x00' character
def get_char_list(filename):
    with open(filename) as f:
        char_set = set(c for c in f.read())
    try:
        char_set.remove('\n')
    except KeyError:
        pass
    char_set.add(empty_char)
    return sorted(char_set)


# take a newline-separated corpus and converts it to a numpy tensor
# each line is an example
# returns size (num lines, max line length, num distinct characters)
def load_data(filename):
    char_list = get_char_list(filename)
    char_to_ind = dict((c,i) for i,c in enumerate(char_list))
    empty_char_ind = char_to_ind[empty_char]
    with open(filename) as f:
        lines = f.read().split('\n')
    maxlen = max(len(l) for l in lines)
    tensor = np.zeros((len(lines), maxlen, len(char_list)))
    for i, l in enumerate(lines):
        for j in range(len(l)):
            ind = char_to_ind[l[j]]
            tensor[i,j,ind] = 1
        for j in range(len(l), maxlen):
            tensor[i,j,empty_char_ind] = 1
    return tensor
