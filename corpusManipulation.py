import json
from collections import defaultdict
import math

def readText(filename) :
    with open(filename, 'r', encoding="utf8") as file :
        fileDict = json.load(file)

    """ for key in fileDict :
        print(key["src_lang"]) """

    # corpus est une liste de dictionnaires
    # les cles du dictionnaires permettent d
    # identifier la nature de chaque information
    return fileDict

#Affiche le nombres de langues 
def countLang(corpusDict) :

    listLang = set()

    for key in corpusDict :
        """ if key["src_lang"] not in listLang :
            listLang.append()
        print(key) """
        listLang.add(key["src_lang"])

    print(listLang)

def countExem(corpusDict) :
    nbOfExem = 0

    for key in corpusDict :
        if key["hyp"] :
            nbOfExem += 1

    print(nbOfExem)
    

if __name__ == "__main__":
    corpusDict = readText("da_newstest2016.json")
    countLang(corpusDict)
    countExem(corpusDict)
