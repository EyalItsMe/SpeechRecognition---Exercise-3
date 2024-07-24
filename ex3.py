import numpy as np
import sys

def ctc_loss(matrix_path, label_string, output_tokens):
    network_outputs = np.load(matrix_path)

    epsilon = " "
    output_tokens = epsilon + output_tokens
    new_string = " "
    for letter in label_string:
        new_string += letter + " "

    z = np.zeros(len(new_string), dtype=int)
    for i, p in enumerate(label_string):
        z[2 * i + 1] = output_tokens.index(p)

    graph_matrix = np.zeros((network_outputs.shape[0], len(new_string)))
    graph_matrix[0][0] = network_outputs[0][z[0]] # May be a problem, check it again
    graph_matrix[0][1] = network_outputs[0][z[1]]

    for i in range (1, len(graph_matrix)): # I represents the time
        for j in range(len(graph_matrix[i])): # J represents the Phoneme
            if j == 0:
                graph_matrix[i][j] = graph_matrix[i-1][j] * network_outputs[i][z[j]]
            else:
                graph_matrix[i][j] = (graph_matrix[i-1][j-1] + graph_matrix[i-1][j]) * network_outputs[i][z[j]]
                if j > 1 and new_string[j] != " " and new_string[j] != new_string[j-2]:
                    graph_matrix[i][j] += graph_matrix[i - 1][j - 2] * network_outputs[i][z[j]]

    return graph_matrix[-1][-1] + graph_matrix[-1][-2]


def print_p(p: float):
    print("%.3f" % p)

if __name__ == '__main__':
    matrix_path = sys.argv[1]
    label_string = sys.argv[2]
    output_tokens = sys.argv[3]
    print_p(ctc_loss(matrix_path, label_string, output_tokens))

