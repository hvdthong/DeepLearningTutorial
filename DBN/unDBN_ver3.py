import os
import numpy as np
from dbn import UnsupervisedDBN
from sklearn.externals import joblib
from os import listdir
from os.path import isfile, join
import multiprocessing
from unDBN_ftr_ver2 import writing_file

# pathFile = "./Data/SemanticFiles_v2/"
# pathFeatures, pathModel = "./OutputFeatures_ver2/Features_v2/", "./OutputModel_ver2/model_v2/"

# pathFile = "./Data/SemanticFilesMethods_v2/"
# pathFeatures, pathModel = "./OutputFeatures_ver2/FeaturesMethods_v2/", "./OutputModel_ver2/modelMethods_v2/"

pathFile = "./Data/SemanticFilesMethods_v3/"
pathFeatures, pathModel = "./OutputFeatures_ver2/FeaturesMethods_v3/", "./OutputModel_ver2/modelMethods_v3/"
layerStruct, numFtrs, nepochs = [100, 100], 100, 200


def writing_file(data, name):
    wf = open(pathFeatures + name, "w+")
    for l in data:
        line = ",".join([str(round(value, 4)) for value in l])
        wf.write(line + "\n")
    wf.close()


def running_DBN(files):
    for f in files:
        train_data = np.array([line.split(",") for line in open(pathFile + "/" + f + '/train.txt')], dtype=int)
        # train_data = train_data / float((max([max(l) for l in train_data])))  # data scaling

        test_data = np.array([line.split(",") for line in open(pathFile + "/" + f + '/test.txt')], dtype=int)
        # test_data = test_data / float((max([max(l) for l in test_data])))  # data scaling

        data = np.concatenate((train_data, test_data), axis=0)
        data = data / float((max([max(l) for l in data])))  # data scaling

        train_size, test_size = train_data.shape[0], test_data.shape[0]
        train_data, test_data = data[0:train_size], data[train_size:]
        print f, train_data.shape, test_data.shape

        model = UnsupervisedDBN(hidden_layers_structure=layerStruct,
                                activation_function="sigmoid",
                                optimization_algorithm="sgd",
                                learning_rate_rbm=0.001,
                                n_epochs_rbm=nepochs,
                                verbose=True)
        model.fit(train_data)
        train_trans, test_trans = model.transform(train_data), model.transform(test_data)
        print "Saving model %s for features generation" % f
        joblib.dump(model, pathModel + f)
        print 'Writing features of file: %s' %f
        writing_file(train_trans, f + "_train.txt")
        writing_file(test_trans, f + "_test.txt")


if __name__ == "__main__":
    folders = [x[0].replace(pathFile, "") for x in os.walk(pathFile)]
    del folders[0]
    new_folders = []

    # # get all the folders which have 2 dimensions and the length of columns is larger than output features
    # # just for checking
    # for f in folders:
    #     try:
    #         data = np.array([line.split(",") for line in open(pathFile + f + '/train.txt')], dtype=int)
    #         # data = np.array([line.split(",") for line in open(pathFile + f + '/test.txt')], dtype=int)
    #     except ValueError:
    #         print "Wrong format %s" % f
    #         exit()
    #     else:
    #         if (len(data.shape) == 2) and (len(data[0]) > numFtrs):
    #             new_folders.append(f)

    new_folders = folders
    # n = 0
    # while True:
        # check all models in features folders
        # files = set([f.replace("_train.txt", "").replace("_test.txt", "") for f in listdir(pathFeatures) if
        #              isfile(join(pathFeatures, f))])
        # new_folders = [f for f in new_folders if f not in files]

    # if len(new_folders) == 0:
    #     break
    # else:

    num_, jobs = 15, []  # number of processes used for running unsupervised DBN = (len(new_folders) / num_)
    group_folders = [new_folders[i:i + num_] for i in range(0, len(new_folders), num_)]

    for i in xrange(0, len(group_folders)):
        print "Running the %i process" % i
        # running_DBN(group_folders[i])

        p = multiprocessing.Process(target=running_DBN,
                                    args=(group_folders[i], ))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()

            # if len(new_folders) >= num_:
            #     for i in xrange(0, len(group_folders)):
            #         print "Running the %i process" % i
            #         p = multiprocessing.Process(target=running_DBN,
            #                                     args=(group_folders[i], ))
            #         jobs.append(p)
            #         p.start()
            # else:
            #     for i in xrange(0, len(new_folders)):
            #         print "Running the %s process" % i
            #         p = multiprocessing.Process(target=running_DBN,
            #                                     args=([new_folders[i]]))
            #         jobs.append(p)
            #         p.start()
            # for p in jobs:
            #     p.join()
        # n += 1
