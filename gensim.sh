#!/bin/bash

for((i=29;i>=29;i--));do

run_date=`date +%Y-%m-%d --date="-${i} day"`
rundate=`date +%Y%m%d --date="-${i} day"` 
day=`date +%d --date="-${i} day"`
month=`date +%m --date="-${i} day"`
week_day=`date +%w --date="-${i} day"`
week_number=`date +%W --date="-${i} day"`
week=`date +%W --date="-${i} day"`
corpusdir="/home/dzn/hd/corpus"
corpusbackup="/home/dzn/hd/corpus_backup"
docsdir="/home/dzn/Hd_news"
pyscrips="/home/dzn/py_tools"
corpuspath=${corpusdir}/seedcorpus.txt
indexpath=${corpusdir}/seedindex.txt

# Create Corpus File 
if [ ${day} == 01 ];then
rm -rf ${corpusdir}/month_${month}
mkdir ${corpusdir}/month_${month}
rm -rf ${corpuspath} ${indexpath}
touch ${corpuspath} ${indexpath}
fi

corpusday=${corpusdir}/month_${month}/${rundate}
rm -rf ${corpusday}
mkdir ${corpusday}
cd ${corpusday}
filepath=${docsdir}/month_${month}/${rundate}
${pyscrips}/jieba_gensim.py -i ${filepath} -c ${corpusday} -r ${rundate} -o ${indexpath} > ${corpusday}/gensim_${rundate}.log

if [ $? == 0 ];then
cat ${corpusbackup}/seedindex.txt ${corpusday}/seedindex_${rundate}.txt > ${indexpath}
cat ${corpusbackup}/seedcorpus.txt ${corpusday}/seedcorpus_${rundate}.txt > ${corpuspath}
${pyscrips}/gensim_cluster.py -i ${indexpath} -c ${corpuspath} -d ${corpusdir} -r ${rundate} -p ${corpusday} > ${corpusday}/updatecluster_${rundate}.log
mv -f ${corpusdir}/seedcorpus_${rundate}.txt ${corpuspath}
rm -rf ${corpusdir}/index.0

#if [ $? == 0 ];then
#cp -f ${indexpath} ${corpusbackup}/seedindex.txt
#cp -f ${corpuspath} ${corpusbackup}/seedcorpus.txt
#fi

fi

done





