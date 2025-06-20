import numpy as np


def output_misses(data_list, network):

    for i in range(len(data_list)-1):
        values = data_list[i].split(',')
        correct_label = int(values[0])
        scaled_inputs = (np.asfarray(values[1:])/ 255.0 * 0.99) + 0.1
        outputs = network.query(scaled_inputs)

        label = np.argmax(outputs)
        if label != correct_label:
            print("Network guess:", label, "correct Label:", correct_label, "index:", i)