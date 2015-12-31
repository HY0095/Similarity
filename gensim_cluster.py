#!/usr/local/bin/python

#coding:-*-utf-8-*-

#*******************************************************************************
# gensim_cluster.py
#
# Update Similarity Clusters  
# 
# Usage: 
#     Parameters: seedcorpus; seedindex
# 
#                                                Author: dzn    Date: 25-11-2015
#*******************************************************************************

import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import os
import shutil
import jieba
import time
import linecache
import datetime 
import itertools
import getopt
import logging
import operator
from gensim import corpora, similarities, models
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)


def validateopts(opts):
	#Stopword     = '/home/aaa/py_tools/stopwords.txt'     # Default StopWord File
	#Punctuation  = '/home/aaa/py_tools/punctuations.txt'  # Default Punctuation File
	threshold    = 0.15                                    # Default similarity threshold
	for option, value in opts:
		if option  in ["-h", "--help"]:
			tips()
		elif option in ["--indexpath", "-i"]:
			indexpath = value
		elif option in ["--dirpath", "-d"]:
			dirpath = value			
		elif option in ["--corpuspath", "-c"]:
			corpuspath = value
		elif option in ["--threshold", "-t"]:
		    threshold = value
		elif option in ["--corpusday", "-p"]:
		    corpusday = value
		elif option in ["--rundate", "-r"]:
		    rundate = value
		elif option == "-u":
			print "usage -u"
	return indexpath, corpuspath, dirpath, threshold, rundate, corpusday

def Txt2Dict(file_name):
    dict_name = {}
    for line in open(file_name) :
        kv = line.strip().split('\t')
        dict_name[int(kv[0].strip())] = [ int(v.strip()) for v in kv[1:]]        
    return dict_name

def Dict2Txt2(dict_name, file_name):
    txt = open(file_name, 'w')
    for key, value in dict_name.items():
        txt.write('%s \t' % (key))
        for i in xrange(len(value)):
            txt.write('%s \t' % value[i])
        txt.write('\n')
    txt.close()

def DictUnin(dict1, dict2):
    for k1, v1 in dict1.items():
        step = 1
        for k2, v2 in dict2.items():
            #print k1
            #print "*******"
            #print v2
            if k1 in v2 :
                dict2[k2] = list(set(v1+v2))
                break
            else :
                if step == len(dict2) :
                    dict2[k1] = v1  
            step += 1        
    return dict2

def UpdateCluter(dict1, dict2):
    dict3 = {}
    #print "dict1"
    #print dict1
    #print "dict2"
    #print dict2
    for k1, v1 in dict1.items():
        step = 1
        for k2, v2 in dict2.items():
            if k2 in v1 :
                #print dict1[k1]
                tmpdict   = ExtractDict(dict2, dict1[k1])
                #print tmpdict
                dict3[k1] = tmpdict[0]
                del dict1[k1]
                break
            step += 1
            #print step
    return dict3

def ExtractDict(Dictionary, keylist):
    outdict = {}
    step    = 1
    key     = keylist[0]
    for keyiter in keylist :
        if step == 1 :
            outdict[key] = Dictionary[keyiter]
        else :
            outdict[key] = list(set(outdict[key] + Dictionary[keyiter]))
        step += 1
    return outdict.values()       

# Read column, and return a list
def ReadColumn(infile, col_index):
    column = list()
    for text in open(infile):
        text = text.replace('\n','').split()
        column.append(int(text[col_index]))
    return column

def CorpusCluster(indexpath, corpuspath, dirpath, threshold, corpusday):
    
    global corekeys
    global indexlist
    indexlist = ReadColumn(indexpath, 0) # Read Index of Seeds
    print "indexlist == "
    print indexlist
    # Read Corpus when needed !
    class ReadCorpus(object):
        def __iter__(self):
            for line in open(corpuspath):
                yield line.split()

    corp        = ReadCorpus()  # Read corpus as Corp
    dictionary  = corpora.Dictionary(corp) # Create dictionary
    corpus      = [dictionary.doc2bow(text) for text in corp] # get bag-of-word
    tfidf       =  models.TfidfModel(corpus)   # Create TF-IDF Model    
    corpus_tfidf= tfidf[corpus]
    # Calculate Similarities 
    index       = similarities.Similarity(dirpath+'/index',corpus,num_features = len(dictionary))
    i           = 0
    dictcluster = {}
    for text in corpus_tfidf:
        dict_tmp = {}
        sims_tmp = list(index[text])
        #print sims_tmp
        # Use Threshold carefully !
        dict_tmp[int(indexlist[i])]  = [int(indexlist[k]) for k,x in enumerate(sims_tmp) if x >= (max(sims_tmp)-threshold)]
        #print dict_tmp
        if i == 0 :
            dictcluster = dict_tmp   # Initialization Dictionary Cluster      
        else :
            dictcluster = DictUnin(dictcluster, dict_tmp)   #Update Dictionary Cluster
        i += 1
	    #print dictcluster
    print "Dict_Cluster" 
    print dictcluster
    corekeys = dictcluster.keys()
    initialindex = Txt2Dict(indexpath)    
    print "initialindex"
    print initialindex
    updateindex = UpdateCluter(dictcluster, initialindex)
    print "updateindex"
    print updateindex
    corekeys    = updateindex.keys()
    
    #print "updateindex +++++ "
    #print updateindex

    Dict2Txt2(updateindex, indexpath) # Output seedindex into txt
    
    WriteSeedCorpus(corpuspath, dirpath+'/seedcorpus_'+rundate+'.txt')  # Output seedcorpus into txt

# Extract Seed Corpus 
def WriteSeedCorpus(infile, outfile):
    coreindex = list()
    for key in corekeys :
        coreindex.append(indexlist.index(key)+1)                
    out  = open(outfile, 'w')
    print coreindex
    print "=================================================="
    for k in coreindex :
        text = linecache.getline(infile,k)
        out.writelines(text)
    #for line in open(infile).readlines()[coreindex]:
    #    out.writelines(line.encode('utf-8')+'\n')
    out.close()           

def tips():
	"""Display the usage tips"""
	print "Please use: "+sys.argv[0]+" [options]"
	print "usage:%s --indexpath=value --corpuspath=value --dirpath=value --threshold=value --rundate=value --corpusday=value"
	print "usage:%s -i value -c value -d value -t threshold -r value -p value"
	sys.exit(2)

def main():
	global rundate
	try:
		opts,args = getopt.getopt(sys.argv[1:],"hi:c:d:t:r:p:u",["indexpath=","corpuspath=","dirpath=","threshold=","rundate=","corpusday=","help"])
	except getopt.GetoptError:
		tips()
	if len(opts) >= 5:
		indexpath, corpuspath, dirpath, threshold, rundate, corpusday = validateopts(opts)
	else:
		print "ErrorMessage: Please Check What Your Input !"
		tips()
		raise SystemExit	

	if not (os.path.isfile(indexpath)):
		print "ErrorMessage: Please Check Your Index_file !"
		raise SystemExit

	if not (os.path.isfile(corpuspath)):
		print "ErrorMessage: Please Check Your Corpus_file !"
		raise SystemExit

	if not (os.path.isdir(dirpath)):
		print "ErrorMessage: Please Check Your dir_path !"
		raise SystemExit
			
	print '*****************************************'
	print 'dirpath    = ' + dirpath
	print 'indexpath  = ' + indexpath
	print 'corpuspath = ' + corpuspath
	print 'corpusday  = ' + corpusday
	print 'rundate    = ' + rundate	
	print '*****************************************'

	CorpusCluster(indexpath, corpuspath, dirpath, threshold, corpusday)

if __name__ == '__main__':
	main()


             


