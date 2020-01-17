import json
from collections import defaultdict
from math import *
import nltk.translate.bleu_score as bleu
import warnings
import edit_distance
import spacy
from spacy import displacy
#from nltk import Tree
import en_core_web_sm
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

def readText(filename) :

    with open(filename, 'r', encoding="utf8") as file :
        fileDict = json.load(file)

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
    dicoBleu = defaultdict(list)
    
    for key in corpusDict :
        ref = key['ref']
        hyp = key['hyp']
        hypSplit = hyp.split()

        if key['src_lang'] == directTrad[0] and key['tgt_lang'] == directTrad[1] :
            dicoBleu[ref] += hypSplit

    return dict(dicoBleu)



def scoreBleu(corpusDict, dicoTrad) :
    """
        Méthode qui calcule le score bleu
        de chaque direction de traduction
    """
    
    dicoBleu = {}
    chencherry = bleu.SmoothingFunction()
    for directTrad in dicoTrad :
        totalBleu = 0.0
        totalBleu2 = 0.0

        dicoBleu = createDicoRefHyp(corpusDict, directTrad)

        for ref in dicoBleu.keys() :
            
            totalBleu += bleu.sentence_bleu(dicoBleu[ref], ref.split(), smoothing_function=chencherry.method3)
            
        dicoTrad[directTrad] = totalBleu / len(dicoBleu.keys())
    
    return(dicoTrad)
    
def editDist(dicoCorpus) :

    dicoDist = dicoTrad
    for directTrad in dicoDist :
        totalDist = 0.0
        total = 0.0
        dicoRefHyp = createDicoRefHyp(dicoCorpus, directTrad)
        for key in dicoRefHyp.keys() :
            for hyp in dicoRefHyp[key] :
                sm = edit_distance.SequenceMatcher(key.split(), hyp)
                totalDist += sm.distance()
                total += 1
        dicoDist[directTrad] = totalDist / total

    return dicoDist

def moyScoreDirectTrad(corpusDict, dicoTrad) :
    
    dicoScore = {}
    for directTrad in dicoTrad :
        totalScore = 0.0
        total = 0.0
        for key in corpusDict :
            if key['src_lang'] == directTrad[0] and key['tgt_lang'] == directTrad[1] :
                totalScore += key['score']
                total += 1
        dicoScore[directTrad] = totalScore / total

    print(dicoScore)

def listesRefHyp(corpusDict) :

    listeRefHyp = [[], []]
    

    for key in corpusDict :
        listeRefHyp[0].append(key['ref'])
        listeRefHyp[1].append(key['hyp'])

    return listeRefHyp

""" def poS(listeRefHyp) :

    nlp = spacy.load("en_core_web_sm")
    dicoPoS = defaultdict(defaultdict(str))

    for elt in listeRefHyp :
        doc = nlp(elt)
        for token in doc :

            print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                    token.shape_, token.is_alpha, token.is_stop) """

def tok_format(tok):
    return "_".join([tok.orth_, tok.tag_])

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(tok_format(node), [to_nltk_tree(child) for child in node.children])
    else:
        return tok_format(node)  


def dependance(corpusDict) :
    nlp = spacy.load("en_core_web_sm")
    listeRefHyp = listesRefHyp(corpusDict)
    
    
    #dicoRefToken = poS(listeRefHyp[0])
    #dicoHypToken = poS(listeRefHyp[1])

    for ref in listeRefHyp[0] :
        doc = nlp(ref)
        #[to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]
        displacy.render(doc, style='dep')
        for token in doc:
            print(token.text, token.dep_, token.head.text, token.head.pos_,
            [child for child in token.children])
            """ print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                    token.shape_, token.is_alpha, token.is_stop) """
            """ fig = go.Figure(data=[go.Table(header=dict(values=['text', 'lemma', 'pos', 'tag', 'dep', 'shape', 'ALPHA', 'STOP']),
                    cells=dict(values=[[token.text], [token.lemma_], [token.pos_], [token.tag_], [token.dep_], [token.shape_],
                        [token.is_alpha], [token.is_stop]]))
                        ])
        fig.show() """
                    
        break




##################### PARTIE JULIETTE EX 4 ET 5 ###############


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
        Méthode qui met dans deux listes le score des phrases selon leur type (source ou ref)
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
        Méthode qui crée un dictionnaire de listes avec pour clé le nombre de mots et en valeur la liste des scores associés aux phrases ayant ce nombre de mot
        arg : dictionnaire du corpus
        renvoie : dictionnaire de listes {nb_mots : [scores]}
    """

    dic_scores = defaultdict(list)

    for key in corpusDict :
        nb_words = len(key["src"].split())
        dic_scores[nb_words].append(key["score"])

    return dic_scores

def moyenne_scores (dicLongueur) :

    """
        Méthode qui crée un dictionnaire avec pour clé le nombre de mots d'une phrase et en valeur la moyenne des scores associés à ces phrases
        arg : dictionnaire de listes {nb_mots : [scores]}
        return : dictionnaire {nb_mots : moyenne_scores}
    """

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
    #print(scoreBleu(corpusDict, dicoTrad))
    #print(editDist(corpusDict))
    #print(moyScoreDirectTrad(corpusDict, dicoTrad))
    dependance(corpusDict)

    distribution = distribScore(corpusDict)
    dic_impact = impact_longueur(corpusDict)
    print(dic_moyenne = moyenne_scores(dic_impact))
    
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
