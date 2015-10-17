#!/usr/bin/python

import os
import sys
import re
import math
import pandas as pd
import scipy  as sp
import numpy  as np
from collections import defaultdict

def parseFile(fname, filetype):
    sampleDict = defaultdict(float)

    if filetype is None: # TCGA database
        skipLines = 2 
        header = ("Comp_El_REF", "Exp_val(DOUBLE)")
        keyIndex = 0
        valueIndex = 1 
        if 'DESCRIPTION' in fname:
            return None, None
    elif filetype is fileExtensions[1]: # miRNA database 
        skipLines = 1
        header = ("miRNA_ID", "RPKM(DOUBLE)")
        keyIndex = 0
        valueIndex = 1 # 1 is read_count, 2 is RPKM
        if False == fname.endswith(filetype) or \
           True  == fname.endswith(fileExtensions[2]):
            return None, None
    elif filetype is fileExtensions[2]: # miRNA database 
        skipLines = 1
        header = ("miRNA_ID", "RPKM(DOUBLE)")
        keyIndex = 0
        valueIndex = 1 # 1 is read_count, 2 is RPKM
        if False == fname.endswith(filetype):
            return None, None
    else:
        return None, None
    print "Parsing ", fname
    
    with open(fname, 'r') as f:
        for line in f.readlines():
            if skipLines > 0:
                #print "Skipping " + line
                skipLines -= 1
                continue 
            fields = line.split('\t');
            #print fields[keyIndex], fields[valueIndex]
            sampleDict[fields[keyIndex]] = fields[valueIndex] 
            #raw_input()
    return header, sampleDict
    

def  parseAllFiles(dirPath, fileType):
    allSampleDf = None
    for dirpath, dirnames, filenames in os.walk(dirPath):
        try:
            for fname in filenames:
                header, sampleDict = parseFile(os.path.join(dirpath, fname), fileType)

                if header is None:
                    continue
                
                if allSampleDf is None:
                    allSampleDf = pd.DataFrame.from_dict([sampleDict], orient='columns')
                else:
                    tempdf = pd.DataFrame.from_dict([sampleDict], orient='columns')
                    allSampleDf = allSampleDf.append(tempdf, ignore_index=True)
            
                print allSampleDf.shape
        except:
            print "Exception", fname

    return allSampleDf

fileExtensions = [None, 'mirna.quantification.txt', 'hg19.mirna.quantification.txt']

if __name__ == "__main__":
    print "There are three type of extensions to the miRNA files"
    print "1) mirna.quantification.txt"
    print "2) hg19.mirna.quantification.txt"
    print "Usage: ./getMatrix.py <Path Directory of files> <file name to write> <TCGA = 0|miRNA = 1 or 2 >\n"
    print "Provide one of the numbers as input if matrix has to generated of miRNA"

    if len(sys.argv) != 4:
        print "Usage: ./getMatrix.py <Path Directory of files> <file name to write> <TCGA = 0|miRNA = 1 or 2>\n"
        exit()

    print "Parsing files of type ", fileExtensions[int(sys.argv[3])]
    dataFrame = parseAllFiles(sys.argv[1], fileExtensions[int(sys.argv[3])])

    if 0 == int(sys.argv[3]):
        filename = "TCGA.csv"
    elif 1 == int(sys.argv[3]):
        filename = "mirna.quantification.csv"
    elif 2 == int(sys.argv[3]):
        filename = "hg19.mirna.quantification.csv"
    else:
        print "Option is doesn't exist. Check program inputs"
        exit()

    with open(filename, 'w') as destFile:
        dataFrame.to_csv(destFile, na_rep='-NA-', sep= ',')



