import json
from collections import defaultdict
from math import *
import nltk
import nltk.translate.bleu_score as bleu
import warnings
import edit_distance
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
        if key["tgt_lang"] == "en" :
            scoreDict["toEn"] += key["score"]
            totalTo += 1.0
        elif key["src_lang"] == "en" :
            scoreDict["fromEn"] += key["score"]
            totalFrom += 1.0

    print(scoreDict["fromEn"] / totalFrom)
    print(scoreDict["toEn"] / totalTo)

    print(totalFrom * 100 / countExem(corpusDict))
    print(totalTo * 100 / countExem(corpusDict))

def createDicoRefHyp(dicoCorpus, directTrad) :
    """
        Méthode qui construit le dictionnaire de chaque direction 
        de traduction
        key : phrase ref, value : liste phrase hypothèses
    """
    listeHypoth = []
    listeRef = []
    totalBleu = 0.0
    dicoBleu = defaultdict(list)
    dicoTest = defaultdict(float)
    totalRef = 0.0

    for key in corpusDict :
        ref = key['ref']
        #splitRef = ref.split()
        hyp = key['hyp']
        hypSplit = hyp.split()
        #print(directTrad[0], " ", directTrad[1])
        #print(key['src_lang'], " ", key['tgt_lang'])
        #print(key['src_lang'] == directTrad[0])
        #print("ici")
        #print(key['tgt_lang'] == directTrad[1])
        if key['src_lang'] == directTrad[0] and key['tgt_lang'] == directTrad[1] :
            dicoBleu[ref] += hypSplit
    for key in dicoBleu :
        totalBleu += bleu.sentence_bleu(dicoBleu[key], key)
            
        totalRef += 1
    
    """ for key in corpusDict :
        if key['src_lang'] in directTrad and key['tgt_lang'] in directTrad :
            ref = key['ref']
            if ref not in listeRef :
                listeRef.append(ref.split())
    
    for ref in listeRef :
        for key in corpusDict :
            ref2 = key['ref']
            hyp = key['hyp']
            if ref == ref2.split() and hyp.split() not in listeHypoth :
                hyp = key['hyp']
                listeHypoth.append(hyp.split())

        totalBleu += bleu.sentence_bleu(listeHypoth, ref) """

    #print(totalBleu / totalRef)
    return totalBleu / totalRef



def scoreBleu(corpusDict, dicoTrad) :
    """
        Méthode qui calcule le score bleu
        de chaque direction de traduction
    """

    listeScore = []
    for directTrad in dicoTrad :
        
        #print(directTrad)
        #for key in corpusDict :
        dicoTrad[directTrad] = createDicoRefHyp(corpusDict, directTrad)
        
        #if key['src_lang'] in directTrad and key['tgt_lang'] in directTrad :
        """ ref = key['ref']
        hyp = key['hyp']
        print(ref.split())
        print(hyp.split())
        listeRef.append(ref.split())
        listeHyp.append(hyp.split()) """
            
        """
            NOTE: Pour chaque ref qui est la même -> faire une liste de toutes les hypothèses ?
            Et faire sentence_bleu() et additionner tous les scores ?
            Faire dictionnaire, phrase src : liste de hypothèses
        """ 
        
         
    
        """ print(listeRef)
        print(listeHyp) """
        #dicoTrad[directTrad] = bleu.corpus_bleu(listeRef, listeHyp)
    
    print(dicoTrad)
    
#def editDist(dicoRefHyp) :




"""PARTIE JULIETTE EX 4 ET 5"""


def minmaxDA (corpusDict) :

    """
    méthode qui trouve le minimum et le maximum des scores DA
    arg : dictionnaire du corpus
    return : tuple de float (max, min)
    """
    listescore = []


    for key in corpusDict : 
        if key["score"] :
            listescore.append(key["score"])

    return (listescore[numpy.argmax(listescore)], listescore[numpy.argmin(listescore)])

def distribScore (corpusDict) :
    """
    méthode qui met dans deux listes le score des phrases selon leur type (source ou ref)
    arg : dictionnaire du corpus
    return : deux listes de float (scores_src, scores_ref)
    """
    scores_src = []
    scores_ref = []

    for key in corpusDict :
        if key["src_lang"] == key["orig_lang"] :
            scores_src.append(key["score"])
        else :
            scores_ref.append(key["score"])

    return scores_src, scores_ref



def impact_longueur (corpusDict) :
    """
    méthode qui crée un dictionnaire de listes avec pour clé le nombre de mots et en valeur la liste des scores associés aux phrases ayant ce nombre de mot
    arg : dictionnaire du corpus
    renvoie : dictionnaire de listes {nb_mots : [scores]}
    """

    dic_scores = defaultdict(list)

    for key in corpusDict :
        nb_words = len(key["src"].split())
        dic_scores[nb_words].append(key["score"])

    return dic_scores

def moyenne_scores (dicLongueur) :

    """Méthode qui crée un dictionnaire avec pour clé le nombre de mots d'une phrase et en valeur la moyenne des scores associés à ces phrases
    arg : dictionnaire de listes {nb_mots : [scores]}
    return : dictionnaire {nb_mots : moyenne_scores}"""

    dic_moyenne = defaultdict(float)

    for key in dicLongueur.keys() :
        somme = 0
        compteur = 0
        for elt in range(len(dicLongueur[key])) :
            somme += dicLongueur[key][elt]
            compteur +=1
        dic_moyenne[key] = somme/compteur
    
    return dic_moyenne 



if __name__ == "__main__":
    corpusDict = readText("da_newstest2016.json")
    dicoTrad = countLang(corpusDict)
    countExem(corpusDict)
    countScore(corpusDict)
    scoreBleu(corpusDict, dicoTrad)

    distribution = distribScore(corpusDict)
    dic_impact = impact_longueur(corpusDict)
    dic_moyenne = moyenne_scores(dic_impact)
    
    """
    #graphiques influence longueur phrase sur score moyen & nombre de phrases pour chaque nombre de mots
    data = []
    for key in dic_moyenne :
        data.extend({"nombre de mots": nb_words  , "Score moyen": score}
            for nb_words, score in dic_moyenne.items())
    data = pd.DataFrame(data)
    ax = sns.barplot(x="nombre de mots", y="Score moyen", data=data)
    for item in ax.get_xticklabels():
        item.set_rotation(45)
    ax.set_ylabel("Score moyen")
    plt.show()

    data = []
    for key in dic_impact :
        data.extend({"nombre de mots par phrase": nb_words  , "Nombre de phrases": len(score)}
            for nb_words, score in dic_impact.items())
    data = pd.DataFrame(data)
    ax = sns.barplot(x="nombre de mots par phrase", y="Nombre de phrases", data=data)
    for item in ax.get_xticklabels():
        item.set_rotation(45)
    ax.set_ylabel("Nombre de phrases")
    plt.show()

    """

    """
    #graphique distribution score DA selon type
    x1 = sns.distplot(distribution[0], label ="Source") #bleu
    x1 = sns.distplot(distribution[1], label ="Référence") #orange
    plt.show()
    """
