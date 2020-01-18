import json
from collections import defaultdict
from math import *
import nltk.translate.bleu_score as bleu
import warnings
import edit_distance
import spacy
from spacy import displacy
from nltk import Tree
import en_core_web_sm
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

def readText(filename) :
    """
        Méthode qui lis le fichier da_newstest2016.json
    """

    with open(filename, 'r', encoding="utf8") as file :
        fileDict = json.load(file)

    return fileDict

def countLang(corpusDict) :
    """
        Méthode qui compte le nombre de direction de traduction
        dans le corpus
        Arg : dicoTrad = dictionnaire key : direction de traduction
        value : float

        return : Le dictionnaire de direction de traduction
    """
    dicoTrad = defaultdict(float)
    
    for key in corpusDict :
        dicoTrad[key["src_lang"], key["tgt_lang"]]


    print(len(dict(dicoTrad).keys()))
    return dicoTrad

def countExem(corpusDict) :
    """
        Méthode qui compte le nombre d'exemples dans le corpus
        Arg : nbOfExem = nombre total de phrases

        return : Le nombre total de phrases dans le corpus
    """
    
    nbOfExem = 0

    for key in corpusDict :
        nbOfExem += 1

    return nbOfExem

def countScore(corpusDict) :
    """
        Méthode qui calcule le score moyen des traductions
        depuis l'Anglais et vers l'Anglais
        Arg : - ScoreDict = dictionnaire du total des scores
        - totalFrom = le total des scores DA de traduction depuis l'Anglais
        - totalTo = le total des scores DA de traduction vers l'Anglais

    """
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

    print("Score DA depuis l'Anglais : ", scoreDict["fromEn"] / totalFrom)
    print("Score DA vers l'Anglais : ", scoreDict["toEn"] / totalTo)

    print("Pourcentage des scores depuis l'Anglais par rapport au total de phrases : ", totalFrom * 100 / countExem(corpusDict), "%")
    print("Pourcentage des scores vers l'Anglais par rapport au total de phrases : ", totalTo * 100 / countExem(corpusDict), "%")

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
    
def distribScoreBleu(corpusDict, dicoTrad) :

    chencherry = bleu.SmoothingFunction()
    dicoScore = defaultdict(list)
    dicoPhrases = dicoDirIndir(corpusDict)
    for directTrad in dicoTrad :
        dicoBleu = createDicoRefHyp(corpusDict, directTrad)

        for ref in dicoBleu.keys() :

            if dicoBleu[ref] in dicoPhrases['direct_hyp'] :
                
                bleuScore = bleu.sentence_bleu(dicoBleu[ref], ref.split(), smoothing_function=chencherry.method3)
                if scoreBleu != 0.0 :
                    dicoScore['direct_hyp'] += [bleuScore]
            elif dicoBleu[ref] in dicoPhrases['indirect_hyp'] :
                bleuScore = bleu.sentence_bleu(dicoBleu[ref], ref.split(), smoothing_function=chencherry.method3)
                if scoreBleu != 0 :
                    dicoScore['indirect_hyp'] += [bleuScore]
            
    return dicoScore    

def editDist(dicoCorpus) :
    """
        Méthode qui calcule la distance d'édition
        pour chaque direction de traduction
    """
    dicoDist = dicoTrad
    dicoDistDirIndir = {}

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

def editDistDirIndir(dicoCorpus) :
    """
        Méthode qui construit le dictionnaire
        de distance d'édition pour les phrases direct
        et indirect
        Arg : - dicoPhrases = dictionnaires avec les phrases direct et indirect 
    """
    dicoPhrases = dicoDirIndir(corpusDict)
    dicoScore = defaultdict(list)

    for key in dicoCorpus :
        if key['hyp'] in dicoPhrases['direct_hyp'] :
            sm = edit_distance.SequenceMatcher(key['ref'].split(), key['hyp'])
            dicoScore['direct_hyp'] += [sm.distance()]
        elif key['hyp'] in dicoPhrases['indirect_hyp'] :
            sm = edit_distance.SequenceMatcher(key['ref'].split(), key['hyp'])
            dicoScore['indirect_hyp'] += [sm.distance()]

    return dicoScore

def moyScoreDirectTrad(corpusDict, dicoTrad) :
    """
        Méthode qui calcule la moyenne des scores
        de chaque direction de traduction
    """
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

def dicoDirIndir(corpusDict) :

    dicoDirIndir = defaultdict(list)

    for key in corpusDict :
        if key['orig_lang'] == key['src_lang'] :
            dicoDirIndir["direct_hyp"] += [key['hyp'].split() ]
        else :
            dicoDirIndir["indirect_hyp"] += [key['hyp'].split()]

    return dicoDirIndir

""" def poS(listeRefHyp) :

    nlp = spacy.load("en_core_web_sm")
    dicoPoS = defaultdict(defaultdict(str))

    for elt in listeRefHyp :
        doc = nlp(elt)
        for token in doc :

            print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                    token.shape_, token.is_alpha, token.is_stop) """

def dependance(corpusDict) :
    nlp = spacy.load("en_core_web_sm")
    dicDirIndir = dicoDirIndir(corpusDict)
    dicoDept = defaultdict(float)
 
    for key in dicDirIndir :
        total = 0.0
        for hyp in dicDirIndir[key] :
            doc = nlp(hyp)
            root = [token for token in doc if token.head == token][0]
            total += treeDepth(root)
        dicoDept[key] = total / len(dicDirIndir[key])

    return dicoDept                      

def treeDepth(root) :
    if not list(root.children):
        return 1
    else:
        return 1 + max(treeDepth(x) for x in root.children)

##################### PARTIE JULIETTE EX 4 ET 5 ###############


def distribScore (corpusDict) :
    """
        Méthode qui met dans deux listes distinctes le score des phrases selon leur type de traduction (directe : src, indirecte : ref)
        arg : liste de dictionnaires des phrases du corpus
        return : deux listes de float (scores_src, scores_ref)
    """
    scores = [[],[]]

    #on parcourt chaque dictionnaire associé à une traduction du corpus 
    for key in corpusDict :
        #si la langue source et la langue d'origine sont la même (trad. directe)
        #on ajoute à score_src le score DA, sinon dans score_ref
        if key["src_lang"] == key["orig_lang"] :
            scores[0].append(key["score"])
        else :
            scores[1].append(key["score"])

    return scores

def impact_longueur (corpusDict) :

    """
        Méthode qui crée un dictionnaire de listes avec pour clé le nombre de mots et en valeur la liste 
        des scores associés aux phrases ayant ce nombre de mot
        arg : liste de dictionnaires des phrases du corpus
        renvoie : dictionnaire de listes {nb_mots : [scores]}
    """

    dic_scores = defaultdict(list)

    #on parcourt chaque traduction du corpus
    for key in corpusDict :
        #on récupère le nombre de mots de la phrase et on l'ajoute à dic_scores
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

    #pour chaque clé de dicLongueur
    for key in dicLongueur.keys() :
        #on initialise les variables
        somme = 0
        compteur = 0
        #on parcours les scores associés à key pour en faire la moyenne qu'on met dans dic_moyenne
        for elt in dicLongueur[key] :
            somme += elt
            compteur +=1
        dic_moyenne[key] = somme/compteur
    
    return dic_moyenne 

def long_moyenne (corpusDict) :
    """
        Méthode qui calcule la longueur moyenne des phrases hypothèse selon leur type de traduction
        Arg : liste de dictionnaires des phrases du corpus
        Return : tuple de float correspondants aux longueurs moyennes directe, indirecte
    """

    somme_src = 0
    somme_ref = 0
    compteur_src = 0
    compteur_ref = 0

    #pour chaque traduction du corpus, on vérifie si c'est une traduction directe ou indirecte
    #on compte le nombre de mots de la phrase et selon son type, on met à jour les variables
    #pour pouvoir renvoyer la moyenne
    for key in corpusDict :
        if key["src_lang"] == key["orig_lang"] :
            nb_mots = len(key["hyp"].split())
            somme_src += nb_mots
            compteur_src += 1
        else :
            nb_mots = len(key["hyp"].split())
            somme_ref += nb_mots
            compteur_ref +=1

    return somme_src/compteur_src, somme_ref/compteur_ref


def diff_mots_moyenne (corpusDict) :
    """
        Méthode qui calcule la moyenne de la différence de mots entre l'hypothèse et la phrase source selon le type de traduction
        Arg : liste de dictionnaires des phrases du corpus
        Return : tuple de floats correspondants aux différentes moyennes selon le type
    """

    somme_src = 0
    somme_ref = 0
    compteur_src = 0
    compteur_ref = 0

    #pour chaque traduction du corpus on vérifie si c'est une traduction direct ou indirecte
    #on calcule la différence de mots entre la phrase hypothèse et la phrase source
    #selon le type de phrase on met à jour les valeurs pour pouvoir renvoyer les moyennes
    for key in corpusDict : 
        if key["src_lang"] == key["orig_lang"] :
            nb_mots = len(key["hyp"].split())-len(key["src"].split())
            somme_src += nb_mots
            compteur_src += 1
        else :
            nb_mots = len(key["hyp"].split())-len(key["src"].split())
            somme_ref += nb_mots
            compteur_ref +=1

    return somme_src/compteur_src, somme_ref/compteur_ref


def type_longueur (corpusDict) :
    """
        Méthode qui crée deux dictionnaires (traduction directe, indirecte) d'int , avec pour clé 
        le nombre de mots par phrase et en valeur le nombres de phrases avec ce nombre de mots
        renvoie : tuple de deux dictionnaires (traduction directe, indirecte) {n mots : nombres de phrases avec n mots}
    """

    dic_src= defaultdict(int)
    dic_ref = defaultdict(int)

    #pour chaque traduction du corpus on compte le nombre de mots de la phrase hypothèse
    #on vérifie le type de traducion et selon le type on incrémente la valeur associée au nombre de mots
    for key in corpusDict :
        nb_words = len(key["hyp"].split())
        if key["src_lang"] == key["orig_lang"] :
            dic_src[nb_words] += 1
        else :
            dic_ref[nb_words] +=1

    return dic_src, dic_ref

def mot_moyen (corpusDict) :
    """
        Méthode qui calcule le nombre de lettres moyen par mot selon le type de phrase
        Arg : liste de dictionnaires des phrases du corpus
        Return : liste de tuples de float correspondants aux moyennes du nombre de lettres par mot selon le type
    """

    somme_src = 0
    somme_ref = 0
    total_src = 0
    total_ref = 0

    #pour chaque traduction dans le corpus on met dans une liste le nombre de mots dans hypothèse
    for key in corpusDict :
        words = key["hyp"].split()
        #on parcourt la liste de mots de la phrase et selon le type de phrase on met à jour les valeurs 
        #pour renvoyer la moyenne selon le type
        for elt in words :
            if key["src_lang"] == key["orig_lang"] :
                somme_src += len(elt)
                total_src +=1
            else :
                somme_ref += len(elt)
                total_ref += 1
    return (somme_src/total_src , somme_ref/total_ref)


if __name__ == "__main__":
    corpusDict = readText("da_newstest2016.json")
    dicoTrad = countLang(corpusDict)
    countExem(corpusDict)
    countScore(corpusDict)
    #print(scoreBleu(corpusDict, dicoTrad))
    #print(editDist(corpusDict))
    #editDistDirIndir(corpusDict)
    distribScoreBleu(corpusDict, dicoTrad)
    #print(moyScoreDirectTrad(corpusDict, dicoTrad))
    #dependance(corpusDict)

    #données 
    #distribution = distribScore(corpusDict)
    #dic_impact = impact_longueur(corpusDict)
    #dic_moyenne = moyenne_scores(dic_impact)
    #data_long_moy = long_moyenne(corpusDict)
    #data_type_longueur = type_longueur(corpusDict)
    #data_diff_mots = diff_mots_moyenne(corpusDict)
    #data_mot_moyen = mot_moyen(corpusDict)
    
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

    
    #graphique distribution score DA selon type
    """ x1 = sns.distplot(distribution[0], label ="Source") #bleu
    x1 = sns.distplot(distribution[1], label ="Référence") #orange
    plt.show() """

    #graphique distribution score BLEU selon type
    x1 = sns.distplot(distribScoreBleu(corpusDict,dicoTrad)['direct_hyp'], label ="Source") #bleu
    x1 = sns.distplot(distribScoreBleu(corpusDict, dicoTrad)['indirect_hyp'], label ="Référence") #orange
    plt.show()

    #graphique distribution DE selon type
    """ x1 = sns.distplot(editDistDirIndir(corpusDict)['direct_hyp'], label ="Source") #bleu
    x1 = sns.distplot(editDistDirIndir(corpusDict)['indirect_hyp'], label ="Référence") #orange
    plt.show() """
   
