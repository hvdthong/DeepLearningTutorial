import numpy as np
from dbn import UnsupervisedDBN
from sklearn.externals import joblib
import os
import multiprocessing


def running_unDBN(files, mypath):
    for f in files:
        train_data = np.array([line.split(",") for line in open(mypath + "/" + f + '/train.txt')], dtype=int)
        # train_data = train_data / float((max([max(l) for l in train_data])))  # data scaling

        test_data = np.array([line.split(",") for line in open(mypath + "/" + f + '/test.txt')], dtype=int)
        # test_data = test_data / float((max([max(l) for l in test_data])))  # data scaling

        data = np.concatenate((train_data, test_data), axis=0)
        data = data / float((max([max(l) for l in data])))  # data scaling

        train_size, test_size = train_data.shape[0], test_data.shape[0]
        print data.shape, train_data.shape, test_data.shape

        train_data, test_data = data[0:train_size], data[train_size:]
        print train_data.shape, test_data.shape

        # exit()

        if (len(data.shape) == 2) and (len(data[0]) > 100):
            print f, data.shape
            model = UnsupervisedDBN(hidden_layers_structure=[100, 100],
                                    activation_function="sigmoid",
                                    optimization_algorithm="sgd",
                                    learning_rate_rbm=0.001,
                                    n_epochs_rbm=10,
                                    verbose=True)
            model.fit(train_data)
            # joblib.dump(model, "./model/" + f)
            # print "Saving model %s for features generation" % f

            # train = np.array([line.split(",") for line in open(mypath + "/" + f + '/train.txt')], dtype=int)
            # data = data / float((max([max(l) for l in data])))  # data scaling
            # test = np.array([line.split(",") for line in open(mypath + "/" + f + '/test.txt')], dtype=int)
            # test = test / float((max([max(l) for l in test])))  # data scaling

            train_trans, test_trans = model.transform(train_data), model.transform(test_data)
            print test_trans[0]
            print test_trans[1]
            print len(train_trans), len(test_trans)


def writing_file(data, name):
    wf = open("./Features/" + file, "w+")
    for l in data:
        line = ",".join([str(round(value, 3)) for value in l])
        wf.write(line + "\n")
    wf.close()


def generating_ftr(file):
    model = joblib.load("./model/" + file)
    train = np.array([line.split(",") for line in open("./SemanticFiles/" + file + '/train.txt')], dtype=int)
    test = np.array([line.split(",") for line in open("./SemanticFiles/" + file + '/test.txt')], dtype=int)

    train_trans, test_trans = model.transform(train), model.transform(test)

    # for value in train_trans:
    #     print value
    # for value in test_trans:
    #     print value

    print test_trans[0]
    print test_trans[1]
    print len(train_trans)
    # writing_file(train_trans, file + "_train")



if __name__ == "__main__":
    # mypath = "./SemanticFiles_v2"
    # folders = [x[0].replace("./SemanticFiles_v2/", "") for x in os.walk(mypath)]
    # del folders[0]
    # new_folders = []
    # # get all the folders which have 2 dimensions and the length of columns is larger than 100
    # for f in folders:
    #     data = np.array([line.split(",") for line in open(mypath + "/" + f + '/train.txt')], dtype=int)
    #     if (len(data.shape) == 2) and (len(data[0]) > 100): new_folders.append(f)
    #
    # num_ = 15  # number of processes used for running unsupervised DBN = (len(new_folders) / num_)
    # group_folders = [new_folders[i:i + num_] for i in range(0, len(new_folders), num_)]
    #
    # for i in xrange(0, len(group_folders)):
    #     print "Running the %i process" % i
    #     p = multiprocessing.Process(target=running_unDBN, args=(group_folders[i],))
    #     p.start()

    file = ["Activiti_Activiti"]
    running_unDBN(file, mypath="./SemanticFiles_v2")
    # generating_ftr("linkedin_indextank-engine_test")