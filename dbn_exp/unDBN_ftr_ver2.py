import numpy as np
import rbm
from sklearn.externals import joblib
import multiprocessing
import os
from os import listdir
from os.path import isfile, join
import logging

# construct directory for files, features and models
# pathFile, pathFeatures, pathModel, numFtrs  = "./SemanticFilesMethods/", "./FeaturesMethods/", "./modelMethods/", 30
# pathFile, pathFeatures, pathModel, numFtrs = "./SemanticFiles/", "./Features/", "./model/", 100
pathFile, pathFeatures, pathModel, numFtrs = "./SemanticFiles_v2/", "./Features_v2/", "./model_v2/", 100
# pathFile, pathFeatures, pathModel, numFtrs = "./SemanticFilesMethods_v2/", "./FeaturesMethods_v2/", "./modelMethods_v2/", 100

nunits, nhids, nepochs = numFtrs, 30, 200
hid_struct = [nunits for i in xrange(0, nhids)]
options = [1 for i in xrange(0, nhids + 1)]


# logging.basicConfig(filename="wrongfile_%s.log" % (pathFile.replace(".", "").replace("/", "")), level=logging.DEBUG)

def writing_file(data, name):
    wf = open(pathFeatures + name, "w+")
    for l in data:
        line = ",".join([str(round(value, 4)) for value in l])
        wf.write(line + "\n")
    wf.close()


def running_DBN(files, hidden_structure, nepochs, options):
    for f in files:
        data = np.array([line.split(",") for line in open(pathFile + f + "/train.txt")], dtype=int)
        data = data / float((max([max(l) for l in data])))  # data scaling
        print f, data.shape
        rows, cols = data.shape
        dbn = rbm.DBN(cols, hidden_structure, options)
        try:
            dbn.train(data, max_epochs=nepochs)
        except:
            logging.error("wrong file: %s" % f)  # print the wrong files to log files
        else:  # if not get an error
            test = np.array([line.split(",") for line in open(pathFile + f + '/test.txt')], dtype=int)
            test = test / float((max([max(l) for l in test])))
            train_trans, test_trans = dbn.run_visible(data), dbn.run_visible(test)
            joblib.dump(dbn, pathModel + f)
            print "Saving model %s for features generation" % f
            writing_file(train_trans, f + "_train.txt")
            writing_file(test_trans, f + "_test.txt")


def generating_ftr(files):
    for f in files:
        model = joblib.load("./model/" + f)
        train = np.array([line.split(",") for line in open("./SemanticFiles/" + f + '/train.txt')], dtype=int)
        test = np.array([line.split(",") for line in open("./SemanticFiles/" + f + '/test.txt')], dtype=int)

        # data scaling
        train, test = train / float((max([max(l) for l in train]))), test / float((max([max(l) for l in test])))
        train_trans, test_trans = model.run_visible(train), model.run_visible(test)
        # for t in train_trans:
        #     print t
        for t in test_trans:
            print t
        print train_trans.shape, test_trans.shape
    return train_trans, test_trans


if __name__ == '__main__':
    # running_DBN(["larsgeorge_hbase-book"], hid_struct, nepochs, options)
    # exit()

    # running_DBN(["linkedin_indextank-engine"], hid_struct, nepochs, options)  # some problem here
    # exit()

    folders = [x[0].replace(pathFile, "") for x in os.walk(pathFile)]
    del folders[0]
    new_folders = []

    # get all the folders which have 2 dimensions and the length of columns is larger than 100
    for f in folders:
        try:
            data = np.array([line.split(",") for line in open(pathFile + f + '/train.txt')], dtype=int)
            data = np.array([line.split(",") for line in open(pathFile + f + '/test.txt')], dtype=int)
        except ValueError:
            print "Wrong format %s" % f
        else:
            if (len(data.shape) == 2) and (len(data[0]) > numFtrs):
                new_folders.append(f)

    print len(new_folders)

    # n = 0
    # while True:
    #     files = set([f.replace("_train.txt", "").replace("_test.txt", "") for f in listdir(pathFeatures) if
    #                  isfile(join(pathFeatures, f))])
    #     new_folders = [f for f in new_folders if f not in files]
    #
    #     if len(new_folders) == 0:
    #         break
    #     else:
    #         num_ = 10  # number of processes used for running unsupervised DBN = (len(new_folders) / num_)
    #         jobs = []
    #         group_folders = [new_folders[i:i + num_] for i in range(0, len(new_folders), num_)]
    #
    #         if len(new_folders) >= num_:
    #             for i in xrange(0, len(group_folders)):
    #                 print "Running the %i process" % i
    #                 p = multiprocessing.Process(target=running_DBN,
    #                                             args=(group_folders[i], hid_struct, nepochs, options))
    #                 jobs.append(p)
    #                 p.start()
    #         else:
    #             for i in xrange(0, len(new_folders)):
    #                 print "Running the %s process" % i
    #                 p = multiprocessing.Process(target=running_DBN,
    #                                             args=([new_folders[i]], hid_struct, nepochs, options))
    #                 jobs.append(p)
    #                 p.start()
    #         for p in jobs:
    #             p.join()
    #     n += 1
