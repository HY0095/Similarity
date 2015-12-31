#!/usr/local/bin/python

#coding:-*-utf-8-*-

#*******************************************************************************
# jieba_gensim.py
# Use Jieba to cut words from documents, and then use Gensim to 
# compare similarities of these documents.
# 
# --Usage: python jieba_gensim.py -i '/home/dzn/hd' -t 0.8 > '/home/aaa/log.txt'
#
# ---- Default Similarity Threshold = 0.8
#                                                Author: dzn    Date: 04-11-2015
#*******************************************************************************

import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import os
import shutil
import jieba
import time
import numpy as np
import linecache
import datetime 
import itertools
import getopt
import logging
import operator
from gensim import corpora, similarities, models
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)


# Step_01: Split raw documents words by word, and remove stopwords, punctuations 
def Doc2Word(dirpath,stopword,punctuation, corpusdir):
	
	global tmpindex
	global indexfile
	print "Step 1. ***********************************"		
	print "        Process Remove Stopwords and Punctuations Started..."
	Outfile    = open(os.path.join(corpusdir, 'doc2words.txt'), 'w')
	Stopwords  = set([word.decode('utf-8') for word in open(stopword).read().split('\n')])   # load stopwords.txt into a nodupkey list 
	Punctuations = set([word.decode('utf-8') for word in open(punctuation).read().split('\n')])   # load punctuations.txt into nodupkey list
	file_num   = 1
	fileindex  = {}
	for dirpaths, dirnames, files in os.walk(dirpath):	    
	    for item in files:
	        file_name  = os.path.join(dirpaths,item)
	        #file_index = rundate+ str(file_num)
	        #fileindex[file_index] = file_name
	        #######
	        for line in open(file_name).readlines():
	            content = line.replace("\n", "").split("\t")[1]
	            #print ' '.join(content)
	            file_index = rundate+str(file_num)
	            filenum = line.replace("\n", "").split("\t")[0]
	            fileindex[int(file_index)] = filenum.strip()
	            file_num+= 1
	            texts   = jieba.cut(content)
	            texts   = [word.strip() for word in texts if not word in Stopwords]
	            texts   = [word.strip() for word in texts if not word in Punctuations]
	            outtexts= ' '.join(texts)
	            Outfile.writelines(outtexts.encode("utf-8") + "\n")
	            	            
	        #line       = open(file_name).read()
	        #content    = str(line).strip()      
	        #file_num  += 1	        
	        #texts      = jieba.cut(content)
	        #texts      = [word.strip() for word in texts if not word in Stopwords]   # texts filtered stopwords
	        #texts      = [word.strip() for word in texts if not word in Punctuations]   # texts filtered punctuations
	        #outtexts   = ' '.join(texts)   # Prepare to write into file
	        #print outtexts
	        #Outfile.writelines(outtexts.encode('utf-8') + '\n')
	Outfile.close()
	tmpindex = sorted(fileindex.keys())
	print fileindex
	print tmpindex
		
	indexfile  = corpusdir+'/doc2index_'+rundate+'.txt'
	Dict2Txt1(fileindex, indexfile)
	
	infile = os.path.join(corpusdir, 'doc2words.txt')
	cmd = "sed -i '" + "1d' '" + infile + "'"
	#print "============"+cmd
	#os.system(cmd)
	print "***********************************"
	print "Process Remove Stopwords and Punctuations Finished..."
	return infile


 # Step_02: Calculate similatities of documents, 

def Doc2Similarity(infile, threshold, corpusdir):   
	
	global corekeys 
	print "Step 2. ***********************************"
	print "        Process Compute Documents Similarities Started..."
	contexts   = [[word for word in line.split()] for line in open(infile)]
	# Remove unique words from context
	all_stems  = sum(contexts, [])
	stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
	texts      = [[stem for stem in text if stem not in stems_once] for text in contexts]
		
	# Write Corpus into txt file
	corpuspath = corpusdir+'/corpus_'+rundate+'.txt'
	Outfile    = open(corpuspath, 'w')
	
	for stem in texts:
	    outtexts  =  ' '.join(stem)   # Prepare to write into file
	    Outfile.writelines(outtexts.encode('utf-8')+'\n')
	Outfile.close()
	
	class FTCorpus(object):	    
	    def __iter__(self):
	        for line in open(corpuspath):  # self.
	            # football corpus new in document per line,tokens separated by whitespace
	            yield line.split() #
	
	Corp        = FTCorpus() #corpus_path,dictionary
	dictionary  = corpora.Dictionary(Corp)
	print "*************"
	corpus      = [dictionary.doc2bow(text) for text in Corp] # get bag-of-word
	tfidf       =  models.TfidfModel(corpus)   # Create TF-IDF Model
	corpus_tfidf= tfidf[corpus]
	
	lsi = models.LsiModel(corpus_tfidf, id2word = dictionary, num_topics = 10)
	corpus_lsi = lsi[corpus_tfidf]
	for doc in corpus_lsi:
	    print doc
	
	# Use Similarity Package to calculate similarities, instead of using MatrixSimilarity, which needn't load corpus into memory
	index       = similarities.Similarity(corpusdir+'/index',corpus, num_features = len(dictionary))	
	i           = 0
	print '*** threshold = ' + str(threshold)
	dictcluster = {}
	for text in corpus_tfidf:
	    dict_tmp = {}
	    sims_tmp = list(index[text])
	    print sims_tmp
	    # Use Threshold carefully ! 
	    #dict_tmp[int(tmpindex[i])]  = [tmpindex[k] for k,x in enumerate(sims_tmp) if x >= (max(sims_tmp)-threshold)]
	    #dict_tmp[int(tmpindex[i])]  = [tmpindex[k] for k,x in enumerate(sims_tmp) if x >= max(np.percentile(sims_tmp, 99.9), max(sims_tmp)*0.9)]
	    dict_tmp[int(tmpindex[i])] = [tmpindex[k] for k,x in enumerate(sims_tmp) if x >= max(sims_tmp)*0.9]
	    print dict_tmp
	    if i == 0 :
	        dictcluster = dict_tmp   # Initialization Dictionary Cluster
	    else :
	        dictcluster = DictUnin(dictcluster, dict_tmp)   #Update Dictionary Cluster
	    i += 1
	    #print dictcluster
	print "Dict_Cluster"
	print dictcluster
	corekeys  = dictcluster.keys()

	txt_name  = corpusdir+'/seedindex_'+rundate+'.txt'
	Dict2Txt2(dictcluster, txt_name) #Output cluster_dictionary into txt file
	
	WriteSeedCorpus(corpuspath, corpusdir+'/seedcorpus_'+rundate+'.txt')  #Output seedcorpus into txt
	
	#ReadSeedWrite(txt_name, 0, seedindex)

	print "Process Compute Documents Similarities Finished..."

# Extract Seed Corpus 
def WriteSeedCorpus(infile, outfile):
    coreindex = list()
    for key in corekeys :
        coreindex.append(tmpindex.index(key)+1)                
    out  = open(outfile, 'w')
    print coreindex
    print "=========================================="
    for k in coreindex :
        text = linecache.getline(infile,k)
        out.writelines(text)
    #for line in open(infile).readlines()[coreindex]:
    #    out.writelines(line.encode('utf-8')+'\n')
    out.close()           

def position(rawlist, threshold):
    return [i for i,x in enumerate(rawlist) if x >= threshold]

# Read column, and return a list
def ReadColumn(infile, col_index):
    column = list()
    for text in open(infile):
        text = text.replace('\n','').split()
        column.append(int(text[col_index]))
    return column

def ReadSeedWrite(infile, col_index, outfile):
    out = open(outfile, 'a')
    for text in open(infile):
        text = text.replace('\n','').split()
        out.writelines(text[col_index] + '\n')
    out.close()

def Dict2Txt1(dict_name, file_name):
    txt = open(file_name, 'w')
    for key, value in dict_name.items():
        txt.write('%s \t %s \n' % (key, value))
    txt.close()

def Dict2Txt2(dict_name, file_name):
    txt = open(file_name, 'w')
    for key, value in dict_name.items():
        txt.write('%s \t' % (key))
        for i in xrange(len(value)):
            txt.write('%s \t' % value[i])
        txt.write('\n')
    txt.close()    

def Txt2Dict(file_name):
    dict_name = {}
    for line in open(file_name) :
        kv = line.strip().split('\t')
        dict_name[int(kv[0])] = [ int(v) for v in kv[1:]]        
    return dict_name

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
             
def tips():
	"""Display the usage tips"""
	print "Please use: "+sys.argv[0]+" [options]"
	print "usage:%s --input=value --stopword=value --Punctuation=value --Threshold=value --Rundate=value --Corpusdir=value --Seedcorpus=value"
	print "usage:%s -i value -s value -p value -t value -c value -r rundate -o seedcorpus"
	sys.exit(2)

def validateopts(opts):
	Stopword     = '/home/dzn/Similarity/stopwords.txt'     # Default StopWord File
	Punctuation  = '/home/dzn/Similarity/punctuations.txt'  # Default Punctuation File
	Threshold    = 0.15                                    # Default similarity threshold
	for option, value in opts:
		if option  in ["-h", "--help"]:
			tips()
		elif option in ["--input", "-i"]:
			Input = value
		elif option in ["--stopword", "-s"]:
			Stoppath = value
		elif option in ["--Punctuation", "-p"]:
			Punctuation = value
		elif option in ["--threshold", "-t"]:
		    Threshold = float(value)
		elif option in ["--corpusdir", "-c"]:
		    Corpusdir = value
		elif option in ["--rundate", "-r"]:
		    Rundate =value
		elif option in ["--seedindex", "-o"]:
		    Seedindex = value
		elif option == "-d":
			print "usage -d"
	return Input, Stopword, Punctuation, Threshold, Corpusdir, Rundate, Seedindex
	
def main():
	global rundate
	global seedindex
	try:
		opts,args = getopt.getopt(sys.argv[1:],"hi:s:p:t:c:r:o:d",["input=","stopword=","punctuation=","threshold=","corpusdir=","rundate=","seedindex=","help"])
	except getopt.GetoptError:
		tips()
	if len(opts) >= 1:
		dirpath,stopword,punctuation,threshold,corpusdir,rundate,seedindex = validateopts(opts)
	else:
		print "ErrorMessage: Please Check What Your Input !"
		tips()
		raise SystemExit	
	if not (os.path.isdir(dirpath)):
		print "ErrorMessage: Please Check Your Dir. Path !"
		raise SystemExit
	
	print '***************************'
	print 'dirpath    = ' + dirpath
	print 'stopword   = ' + stopword
	print 'punctuation= ' + punctuation
	print 'Corpusdir  = ' + corpusdir
	print 'Rundate    = ' + rundate
	print 'Seedindex  = ' + seedindex
	print 'rundate    = ' + rundate	
	print '***************************'
	
	start_CPU = time.clock()
	tmpfile = Doc2Word(dirpath, stopword, punctuation, corpusdir)
	end_CPU = time.clock()
	print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
	print("Doc2Word Costs: %f CPU seconds" % (end_CPU - start_CPU))
	
	start_CPU = time.clock()
	Doc2Similarity(tmpfile, threshold, corpusdir)
	end_CPU = time.clock()
	print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
	print("Doc2Similarity Costs: %f CPU seconds" % (end_CPU - start_CPU))
	

if __name__ == '__main__':
	main()

