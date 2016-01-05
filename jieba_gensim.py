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
from scipy import sparse
import scipy as sp
#import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import metrics
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
	            #  &&&&&&&&&&&&
	            #file_index = rundate+str(file_num)
	            if file_num >= 100000 :
	                print "**** ErrorMessage: file_num < 100000 is not satisfied ! ****"
	                raise SystemExit
	            else :
	                file_index = int(rundate)*100000+file_num
	                filenum = line.replace("\n", "").split("\t")[0]
	                fileindex[file_index] = filenum.strip()
	                file_num+= 1
	                texts   = jieba.cut(content)
	                texts   = [word.strip() for word in texts if not word in Stopwords]
	                texts   = [word.strip() for word in texts if not word in Punctuations]
	                outtexts= ' '.join(texts)
	                Outfile.writelines(outtexts.encode("utf-8") + "\n")
	Outfile.close()
		
	indexfile  = corpusdir+'/doc2index_'+rundate+'.txt'
	Dict2Txt1(fileindex, indexfile)
	
	infile = os.path.join(corpusdir, 'doc2words.txt')
	#update: 20160104
	#cmd = "sed -i '" + "1d' '" + infile + "'"
	#print "============"+cmd
	#os.system(cmd)
	print "***********************************"
	print "Process Remove Stopwords and Punctuations Finished..."

# Step_02: Add Corpus

def AddCorpus(newpath):
    print "Step 2. **********************************"
    print "         Add Corpus"
    contexts   = [[word for word in line.split()] for line in open(newpath)]
    # Remove unique words from context
    all_stems  = sum(contexts, [])
    stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
    texts      = [[stem for stem in text if stem not in stems_once] for text in contexts]
    corpuspath = corpusdir+'/corpus_'+rundate+'.txt'
    Outfile    = open(corpuspath, 'w')
    for stem in texts :
        outtexts  = ' '.join(stem)
        Outfile.writelines(outtexts.encode('utf-8')+'\n')
    Outfile.close()
    
    cmds1 = 'cat '+corpuspath+' >> '+corpusdir+'/corpus.txt'
    print "cmds1 = " + cmds1
    os.system(cmds1)
    corpuspath = corpusdir+'/corpus.txt'
    print ' ==================== '
    print corpuspath
    cmds2 = 'cat '+corpusdir+'/doc2index_'+rundate+'.txt'+'>> '+corpusdir+'/corpus_index.txt'
    print cmds2
    os.system(cmds2)
    indexfile = corpusdir+'/corpus_index.txt'

 # Step_03: Calculate similatities of documents, 

def Doc2Similarity(threshold, corpusdir):   
	
	global corekeys
	global tmpindex
	print "Step 3. ***********************************"
	print "        Process Compute Documents Similarities Started..."
	contexts   = [[word for word in line.split()] for line in open(corpusdir+'/corpus.txt')]
	# Remove unique words from context
	#all_stems  = sum(contexts, [])
	#stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
	texts      = [[stem for stem in text ] for text in contexts]
	
	fileindex = {}
	for line in open(corpusdir+'/corpus_index.txt'):
	    text = line.strip().split('\t')
	    fileindex[int(text[0])] = text[1].strip()
	    
	abc  = sorted(fileindex.keys())
	#print 'filenidex = '
	#print fileindex
	#print 'abc'
	#print abc
	
	class FTCorpus(object):	    
	    def __iter__(self):
	        for line in open(corpusdir+'/corpus.txt'):  # self.
	            # football corpus new in document per line,tokens separated by whitespace
	            yield line.split() #
	
	Corp        = FTCorpus() #corpus_path,dictionary
	dictionary  = corpora.Dictionary(Corp)
	print "*************"
	corpus      = [dictionary.doc2bow(text) for text in Corp] # get bag-of-word
	tfidf       =  models.TfidfModel(corpus)   # Create TF-IDF Model
	corpus_tfidf= tfidf[corpus]
	lda         = models.LdaMulticore(corpus_tfidf, num_topics=50, passes=1, iterations=50, id2word=dictionary)
	ldamodel    = corpusdir+'/lda_'+rundate+'_model'
	lda.save(ldamodel)
	aa          = lda[corpus_tfidf]
	s_l         = []
	for ll in aa :
	    s_l.append(ll)
	s_m  = List2SP(s_l)
	data = scale(s_m, with_mean = False)
	#reduced_data = PCA(n_components = 5).fit_transform(data.toarray())
	data = data.toarray()
	kmeans = KMeans(init='k-means++', n_clusters=10,n_init=10)
	#kmeans.fit(reduced_data)
	kmeans.fit(data)
	#clus_pred = kmeans.predict(reduced_data)
	clus_pred = kmeans.predict(data)
	centroids = kmeans.cluster_centers_
	
	k_s = {}
	for ii, p in enumerate(clus_pred):
	    if p in k_s :
	        k_s[p].append(ii) 
	    else :
	        k_s[p] = []
	        k_s[p].append(ii)
	
	result = []
	
	for ii, cent in enumerate(centroids):
	    ll  = k_s[ii]
	    dis = []
	    ind = []
	    for index in ll:
	        #dis_index = dist(cent, reduced_data[index])
	        dis_index = dist(cent, data[index])
	        dis.append(dis_index)
	        ind.append(index)
	    dis_min = np.min(dis)
	    index_min = ll[dis.index(dis_min)]  
	    result.append([ii, index_min, ll])
	    
	# Write out kmeans_cluster_result
	#topic_cluster = corpusdir+'/topic_cluster_'+rundate+'.txt'
	
	for ii in range(len(result)):
	    tmpcmd = 'mkdir '+corpusdir+'/topic_cluster_'+str(ii)
	    os.system(tmpcmd)
	    topic_cluster = corpusdir+'/topic_cluster_'+str(ii)+'/cluster.txt'
	    fout = open(topic_cluster, 'w')
	    fout.write('%s \t' % result[ii][0])
	    fout.write('%s \t' % abc[result[ii][1]])
	    fout.write('%s \t' % [abc[ij] for ij in result[ii][2]])
	    fout.write('\n')
	    fout.write('\n')
	    fout.write('%s \t' % result[ii][0])
	    fout.write('%s \t' % result[ii][1])
	    fout.write('%s \t' % result[ii][2])
	    fout.close()
	    topic_corpus = corpusdir+'/topic_cluster_'+str(ii)+'/corpus.txt'
	    subcorpustext = []
	    tmpindex = []
	    fout = open(topic_corpus, 'w')
	    if len(result[ii][2]) >= 1:
	        for jj in result[ii][2]:
	            subcorpustext.append(texts[jj])
	            tmpindex.append(abc[jj])
	            tmp_corpus = ' '.join(texts[jj])
	            fout.writelines(tmp_corpus.encode('utf-8')+'\n')
	    fout.close()
	    
	    print "^^^^^^^ ii ^^^^^^^^^^^^^"
	    print "ii = " + str(ii)
	    print "^^^^^^^^^^^^tmpindex^^^^^^^^^^^^^"
	    print tmpindex
	    
	    dictionary  = corpora.Dictionary(subcorpustext)
	    tmpcorpus   = [dictionary.doc2bow(text) for text in subcorpustext]
	    tfidf       =  models.TfidfModel(tmpcorpus)
	    corpus_tfidf= tfidf[tmpcorpus]
	    # Use Similarity Package to calculate similarities, instead of using MatrixSimilarity, which needn't load corpus into memory
	    index       = similarities.Similarity(corpusdir+'/topic_cluster_'+str(ii)+'/index',tmpcorpus, num_features = len(dictionary))
	    i           = 0
	    print '*** threshold = ' + str(threshold)
#####
######	
	    dictcluster = {}
	    for text in corpus_tfidf:
	        dict_tmp = {}
	        sims_tmp = list(index[text])
	        print "******* sims_tmp ***********"
	        print sims_tmp
	        # Use Threshold carefully ! 
	        #dict_tmp[int(tmpindex[i])]  = [tmpindex[k] for k,x in enumerate(sims_tmp) if x >= (max(sims_tmp)-threshold)]
	        #dict_tmp[int(tmpindex[i])]  = [tmpindex[k] for k,x in enumerate(sims_tmp) if x >= max(np.percentile(sims_tmp, 99.9), max(sims_tmp)*0.9)]
	        dict_tmp[int(tmpindex[i])] = [tmpindex[k] for k,x in enumerate(sims_tmp) if x >= max(sims_tmp)*threshold]
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
	    
	    txt_name  = corpusdir+'/topic_cluster_'+str(ii)+'/seedindex.txt'
	    Dict2Txt2(dictcluster, txt_name) #Output cluster_dictionary into txt file
	    WriteSeedCorpus(topic_corpus, corpusdir+'/topic_cluster_'+str(ii)+'/seedcorpus.txt')  #Output seedcorpus into txt
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

def dist(x,y):
    return np.sqrt(np.sum((x-y)**2))       

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
                   
def List2SP(List, dim = 0):
    """Convert List to Sparse Matrix."""
    list_row   = list()
    list_col   = list()
    list_value = list()
    for row, t in enumerate(List):
        for tt in t:
            list_row.append(row)
            list_col.append(tt[0])
            list_value.append(tt[1])
    if dim == 0:
        sparse_matrix = sparse.coo_matrix((list_value, (list_row, list_col)))
    else :
        row_set = set(list_row)
        sparse_matrix = sparse.coo_matrix((list_value, (list_row, list_col)), shape = [len(row_set), dim])        
    return sparse_matrix

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
	Threshold    = 0.9                                   # Default similarity threshold
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
		elif option == "-d":
			print "usage -d"
	return Input, Stopword, Punctuation, Threshold, Corpusdir, Rundate
	
def main():
	global rundate
	global corpuspath
	global corpusdir
	try:
		opts,args = getopt.getopt(sys.argv[1:],"hi:s:p:t:c:r:o:d",["input=","stopword=","punctuation=","threshold=","corpusdir=","rundate=","help"])
	except getopt.GetoptError:
		tips()
	if len(opts) >= 1:
		dirpath,stopword,punctuation,threshold,corpusdir,rundate = validateopts(opts)
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
	print 'rundate    = ' + rundate	
	print '***************************'
	
	corpuspath = corpusdir+'/corpus_'+rundate+'.txt'
	
	start_CPU = time.clock()
	Doc2Word(dirpath, stopword, punctuation, corpusdir)
	end_CPU = time.clock()
	print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
	print("Doc2Word Costs: %f CPU seconds" % (end_CPU - start_CPU))

	start_CPU = time.clock()
	AddCorpus(corpusdir+'/doc2words.txt')
	end_CPU = time.clock()
	print "corpuspath = "+corpuspath
	print "indexfile  = "+indexfile
	print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
	print("Doc2Word Costs: %f CPU seconds" % (end_CPU - start_CPU))
	
	start_CPU = time.clock()
	Doc2Similarity(threshold, corpusdir)
	end_CPU = time.clock()
	print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
	print("Doc2Similarity Costs: %f CPU seconds" % (end_CPU - start_CPU))
	

if __name__ == '__main__':
	main()

