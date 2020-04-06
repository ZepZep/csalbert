#!/usr/bin/env python2
#coding=utf-8
from __future__ import unicode_literals
from __future__ import print_function

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import argparse
import functools
import sqlite3
import logging
import json

import manatee
manatee.setEncoding('UTF-8')

import numpy as np
import pandas as pd

import joblib

import datetime
import os.path

# from treeinterpreter import treeinterpreter as ti

log = logging.getLogger("train")

pstructure = "p"
gstructure = "g"

def text(corpus, docid, attrname='word'):                                                          
    doc = corpus.get_struct('doc')                                         
    word = corpus.get_attr(attrname)                                                
                                                                                  
    beg = doc.beg(docid)                                                          
    end = doc.end(docid)                                                          
                                                                                  
    p = corpus.get_struct(pstructure)                                             
    g = corpus.get_struct(gstructure)                                             
                                                                                  
    grng = g.whole()                                                              
    prng = p.whole()                                                              
                                                                                  
    it = word.textat(beg)                                                         
                                                                                  
    paragraphs = []                                                               
    tokens = None                                                                 
    for i in range(beg, end):                                                     
        if prng.find_beg(i) == i:                                                 
            if tokens is not None:                                                
                paragraphs.append(tokens)                                         
            tokens = []                                                           
        if grng.find_beg(i) != i and tokens:                                      
            tokens.append(" ")                                                    
        tokens.append(it.next())                                                  
    if tokens:                                                                    
        paragraphs.append(tokens)                                                                                                                            
    out = ''
    for p in paragraphs:
        for t in p:
            out += t
        out += ' '
    return out.strip()

def gen_text(corpus, attrname):
    doc = corpus.get_struct('doc')
    attr = corpus.get_attr(attrname)
    for i in xrange(doc.size()):
        beg, end = doc.beg(i), doc.end(i)
        it = attr.textat(beg)
        yield [it.next() for pos in xrange(beg, end)]

def read_annots(corpus, db):
    annots = pd.read_sql("select file, name, value from document_annotations", db)
    annots = annots[annots.name.isin(simple_attributes)]
    doc = corpus.get_struct("doc")
    sa = doc.get_attr("filename")
    lex = {sa.pos2str(i): i for i in xrange(doc.size())}
    annots['docid'] = annots.file.map(lambda docname: lex.get(docname, -1))
    return annots[annots.docid != -1].drop(columns="file")
 

default_annot_values = {                                                      
    'zanr':  ['zpravodajství', 'rozhovor', 'komentář'],                       
    'tema': ['migrační krize', 'domácí politika',                             
        'zahraniční politika / diplomacie',                                   
        'společnost / společenská situace', 'jiné', 'energetika',             
        'sociální politika', 'konflikt na Ukrajině', 'kultura',               
        'konflikt v Sýrii', 'zbrojní politika', 'ekonomika / finance',        
        'konspirace'],                                                        
    'zamereni': ['zahraniční', 'domácí', 'obojí', 'nelze určit'],             
    'lokace': ['EU', 'Česká republika', 'USA', 'jiná země',                   
        'jiné / nelze určit', 'Rusko', 'NATO', 'Rusko + USA'],                
    'argumentace': ['ne', 'ano'],                                             
    'emoce': ['missing', 'rozhořčení', 'soucit', 'strach', 'nenávist', 'jiná'],
    'vyzneni_celku': ['neutrální', 'negativní', 'pozitivní'],                 
    'rusko': ['missing', 'pozitivní příklad', 'neutrální', 'oběť',            
        'negativní příklad', 'hrdina'],                                       
    'vyzneni1': ['neutrální', 'negativní', 'missing', 'pozitivní', 'velebící',
        'nenávistné'],                                                        
    'vyzneni2': ['neutrální', 'negativní', 'missing', 'pozitivní', 'velebící',
        'nenávistné'],                                                        
    'vyzneni3': ['neutrální', 'negativní', 'missing', 'pozitivní', 'velebící',
        'nenávistné'],                                                        
    'obrazek': ['ne', 'ano'],                                                 
    'video': ['ne', 'ano'],                                                   
    'nazor': ['ne', 'ano'],                                                   
    'odbornik': ['ne', 'ano'],                                                
    'zdroj': ['ne', 'ano'],                                                   
    'strach': ['ne', 'ano'],                                                  
    'vina': ['ne', 'ano'],                                                    
    'nalepkovani': ['ne', 'ano'],                                             
    'demonizace': ['ne', 'ano'],                                              
    'relativizace': ['ne', 'ano'],                                                                                                                           
    'fabulace': ['ne', 'ano'],                                                
    'year': '2016,2017,2018'.split(',')                                       
}                     

simple_attributes = ['vina', 'nalepkovani', 'argumentace', 'emoce',
    'demonizace', 'relativizace', 'strach', 'fabulace', 'nazor', 'lokace',
    'zdroj', 'rusko', 'odbornik', 'tema', 'zanr', 'zamereni',
    'vyzneni_celku', 
]

binary_attributes = [x for x in default_annot_values\
        if len(default_annot_values[x]) == 2]

def main():
    fmt = '[%(asctime)-15s] %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=fmt)

    m = argparse.ArgumentDefaultsHelpFormatter
    p = argparse.ArgumentParser(description="", formatter_class=m)
    p.add_argument("-c", "--corpus", type=str, required=True)
    p.add_argument("-d", "--db", type=str, required=True)
    p.add_argument("-o", "--outfile", type=str, required=True)

    args = p.parse_args()

    log.info("opening database {}".format(args.db))
    db = sqlite3.connect(args.db)
    db.isolation_level = None  # I want to handle transactions myself
    log.info("opening corpus {}".format(args.corpus))
    corp = manatee.Corpus(args.corpus)
    log.info("corpus has %d positions" % corp.size())

    log.info("reading annotations")
    attrs = read_annots(corp, db)
    #if (args.attr is None) or (sum(attrs.name == args.attr) == 0):
    #    log.error("you must specify valid attribute to train on")
    #    log.info("available attributes are: " + " ".join(attrs.name.unique()))
    #    return 1
    #log.info("read %d annotations for %s" % (sum(attrs.name == args.attr), args.attr))

    headers_simple = []
    headers_multi = []

    for k, v in default_annot_values.iteritems():
        if k not in simple_attributes:
            continue
        if len(v) > 2:
            for vv in v:
                headers_multi.append((k, vv, (k + "_" + vv).replace(' ', '-')))
        else:
            headers_simple.append(k)

    log.info("reading corpus text")

    doc = corp.get_struct('doc')
    docsize = corp.get_struct('doc').size()
    fn = doc.get_attr('filename')
    
    with open('labels.csv', 'w') as lf:
        for x in headers_simple: print(x, file=lf)
        for x,y,z in headers_multi: print(z.encode('utf-8'), file=lf)

    #of = open('data.csv', 'w')
    
    #row = 'index\ttext\t' + '\t'.join(headers_simple) + '\t' + '\t'.join(ma for k, v, ma in headers_multi)
    #print(row.encode('UTF-8'), file=of)

    print("Grouping most common answer")
    most_common = attrs.groupby(["docid", "name"]).agg(lambda x: pd.Series.mode(x)[0])

    print("Converting dataset")
    df = most_common.unstack()
    df = df[~df.isna().any(axis=1)]
    df.columns = df.columns.droplevel()
    df.columns.name=None
    df.index.name = None

    print("Getting text")
    df["text"] = df.index.map(lambda docid: text(corp, docid, 'word'))
    
    print("Writing dataset")
    df.to_csv("data.csv", index=False)


if __name__ == "__main__":
        exit(main())

