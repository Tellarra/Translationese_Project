import json
from collections import defaultdict
from math import *
import nltk
import nltk.translate.bleu_score as bleu
import warnings
warnings.filterwarnings('ignore')

def readText(filename) :
    #fileDict = defaultdict()
    with open(filename, 'r', encoding="utf8") as file :
        fileDict = json.load(file)

    # corpus est une liste de dictionnaires
    # les cles du dictionnaires permettent d
    # identifier la nature de chaque information
    return fileDict

#Affiche le nombres de direction de traduction 
def countLang(corpusDict) :

    dicoTrad = defaultdict(float)
    
    for key in corpusDict :
        dicoTrad[key["src_lang"], key["tgt_lang"]] += 1.0


    print(len(dict(dicoTrad).keys()))
    return dicoTrad

#Compte le nombre d'exemples présents dans le corpus
def countExem(corpusDict) :
    nbOfExem = 0

    for key in corpusDict :
        if key["hyp"] :
            nbOfExem += 1

    return nbOfExem

#Evalue les scores moyens des traductions
def countScore(corpusDict) :
    
    scoreDict = defaultdict(float)
    totalFrom = 0.0
    totalTo = 0.0

    for key in corpusDict :
        if key["score"] and key["tgt_lang"] == "en" :
            scoreDict["toEn"] += key["score"]
            totalTo += 1.0
        elif key["score"] and key["src_lang"] == "en" :
            scoreDict["fromEn"] += key["score"]
            totalFrom += 1.0

    print(scoreDict["fromEn"] / totalFrom)
    print(scoreDict["toEn"] / totalTo)

    print(totalFrom * 100 / countExem(corpusDict))
    print(totalTo * 100 / countExem(corpusDict))

def constructDicoBleu(dicoCorpus, directTrad, dicoTrad) :
    """
        Méthode qui construit le dictionnaire de chaque direction 
        de traduction
        key : phrase ref, value : liste phrase hypothèses
    """
    listeHypoth = set()
    for key in corpusDict :
        if key['src_lang'] == directTrad and key['tgt_lang'] == directTrad :
            listeHypoth.add(key['hyp'])
            dicoTrad[key['src']] = listeHypoth
    
    return dicoTrad



def scoreBleu(corpusDict, dicoTrad) :
    """
        Méthode qui calcule le score bleu
        de chaque direction de traduction
    """

    listeScore = []
    for directTrad in dicoTrad :
        listeHyp = []
        listeRef = []
        #print(directTrad)
        for key in corpusDict :
            if key['src_lang'] in directTrad and key['tgt_lang'] in directTrad :
                ref = key['ref']
                hyp = key['hyp']
                """ print(ref.split())
                print(hyp.split()) """
                listeRef.append(ref.split())
                listeHyp.append(hyp.split())
            
            #listeScore.append(bleu.sentence_bleu(ref.split(), hyp.split()))
        """
            NOTE: Pour chaque ref qui est la même -> faire une liste de toutes les hypothèses ?
            Et faire sentence_bleu() et additionner tous les scores ?
            Faire dictionnaire, phrase src : liste de hypothèses
        """ 
         
        #break 
        
        """ print(listeRef)
        print(listeHyp) """
        #break
        dicoTrad[directTrad] = bleu.corpus_bleu(listeRef, listeHyp)
    #print(listeScore)
    print(dicoTrad)
    

if __name__ == "__main__":
    corpusDict = readText("da_newstest2016.json")
    dicoTrad = countLang(corpusDict)
    countExem(corpusDict)
    countScore(corpusDict)
    scoreBleu(corpusDict, dicoTrad)
