import numpy as np
import rbm
import sys, pickle
from PIL import Image
from sklearn.externals import joblib


def readImage(dfile):
    output = []
    for line in dfile:
        l = line.split()
        im = Image.open(l[0])

        imdata = im.getdata()
        row = list(imdata)
        rowbinary = [0 if (pixel >= 100) else 1 for pixel in row]
        output.append(rowbinary)

    data = np.array(output)
    return data


def readRBM():
    pkl_file = open(sys.argv[5], 'rb')
    r = pickle.load(pkl_file)
    return r


def readdata():
    dfile = open("./datacontinuous.txt")
    readrbm = 0
    imageflag = 0
    opt = 1

    if imageflag:
        opt = 0
        data = readImage(dfile)
    else:
        output = []
        for line in dfile:
            l = line.split()
            output.append([float(bit) for bit in l])
        data = np.array(output)
    return data, readrbm, imageflag, opt


def printRes(res):
    f = open(sys.argv[1] + '.res', 'w')
    for row in res:
        for c in row:
            f.write(str(c))
            f.write(' ')
        f.write('\n')
    f.close()


def saveRBM(r):
    f = open(sys.argv[1] + '.pkl', 'wb')
    pickle.dump(r, f, -1)
    f.close()


if __name__ == '__main__':
    data, readrbm, imageflag, opt = readdata()
    N, M = data.shape
    print data.shape

    option = [1, 1, 1, 1, 1]

    dbn = rbm.DBN(M, [5, 5, 5, 5], option)
    dbn.train(data, max_epochs=10)
    # joblib.dump(dbn, "./dbn_test")
    # model = joblib.load("./dbn_test")

    res = dbn.run_visible(data[0:10])
    print len(res), len(res[0])

    print res[0]
    print res[1]

    f = "kpelykh_docker-java"
    data = np.array([line.split(",") for line in open("./SemanticFiles/" + f + "/train.txt")], dtype=int)
    data = data / float((max([max(l) for l in data])))  # data scaling
    print f, data.shape
    rows, cols = data.shape
    nunits, nhids, nepochs = 100, 10, 200
    hid_struct = [nunits for i in xrange(0, nhids)]
    options = [1 for i in xrange(0, nhids + 1)]
    dbn = rbm.DBN(cols, hid_struct, options)
    dbn.train(data, max_epochs=nepochs)
    train_trans = dbn.run_visible(data)
    print train_trans.shape
    print train_trans[0]
    print train_trans[1]



    # res = dbn.run_hidden(np.array([[1,0,0],[0,1,0],[0,0,1]]))
    # if imageflag:
    #     img = Image.new("1",(50,60))
    #     k = 1
    #     for r in res:
    #         pixels = [255 if pixel == 0 else 0 for pixel in r ]
    #         img.putdata(pixels)
    #         img.save(sys.argv[1]+str(k)+'-guess.JPEG',"JPEG")
    #         k = k+1
    # else:
    #     print data.shape, len(res)

    # vis_ = dbn.run_hidden(data[0])
    # print vis_
