import os
import glob
import numpy as np
import math

wd = '/work/05863/mgarciat/stampede2/'

paths = glob.glob(wd + "allgames/*.ulx")
files = [os.path.basename(x) for x in paths]
files = np.random.permutation(files)

N = len(files)
trainratio = 0.9

Nt = math.trunc(trainratio * N)

trainfiles = files[:Nt]
testfiles = files[Nt:]

alldir = wd + "allgames/"
traindir = wd + "train/"
testdir = wd + "test/"

sh = ""
for fn in trainfiles:
    json = fn.replace(".ulx", ".json")
    z8 = fn.replace(".ulx", ".z8")
    sh += "cp {}/{} {}/{}\n".format(alldir, fn, traindir, fn)
    sh += "cp {}/{} {}/{}".format(alldir, json, traindir, json)
    sh += "cp {}/{} {}/{}".format(alldir, z8, traindir, z8)

for fn in testfiles:
    json = fn.replace(".ulx", ".json")
    z8 = fn.replace(".ulx", ".z8")
    sh += "cp {}/{} {}/{}".format(alldir, fn, testdir, fn)
    sh += "cp {}/{} {}/{}".format(alldir, json, testdir, json)
    sh += "cp {}/{} {}/{}".format(alldir, z8, testdir, z8)

with open(wd + "copybash")