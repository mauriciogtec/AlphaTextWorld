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

for fn in trainfiles:
    json = fn.replace(".ulx", ".json")
    z8 = fn.replace(".ulx", ".z8")
    cmd = "cp {}{} {}{}"
    cmd = cmd.format(alldir, fn, traindir, fn)
    os.system(cmd)
    cmd = cmd.format(alldir, json, traindir, json)
    os.system(cmd)
    cmd = cmd.format(alldir, z8, traindir, z8)
    os.system(cmd)

for fn in testfiles:
    json = fn.replace(".ulx", ".json")
    z8 = fn.replace(".ulx", ".z8")
    cmd = "cp {}/{} {}/{}"
    cmd = cmd.format(alldir, fn, testdir, fn)
    os.system(cmd)
    cmd = cmd.format(alldir, json, testdir, json)
    os.system(cmd)
    cmd = cmd.format(alldir, z8, testdir, z8)
    os.system(cmd)
