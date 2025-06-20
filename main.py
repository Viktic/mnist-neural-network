import Train
import Test
import pickle
import numpy as np

#defines path for the trained-network file
nn_trained_file = "nn.pk1"

#checks if there is a saved, trained network and loads it, if thats the case
try:
    with open(nn_trained_file, "rb") as file:
        trained_network_loaded = pickle.load(file)
        file.close()
    print("loaded trained network")

#if there is no saved, trained network, a network instance is created, trained and tested
except FileNotFoundError:

    Train.miniBatchGradientDescent(*Train.serializeData())
    accuracy = Train.testAccuracy()
    #if the testing does satisfy the accuracy criterion, the trained network will be saved
    if accuracy > 0.95:
        with open(nn_trained_file, "wb") as file:
            pickle.dump(Train.nn, file)
            file.close()
        print ("saved trained network")
    else:
        print ("Network did not meet training standarts")


training_data = Train.data_list_test

#outputs the falsely classified images for better understanding of the networks perfomance
Test.output_misses(training_data, trained_network_loaded)
