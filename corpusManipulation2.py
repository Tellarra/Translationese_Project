import json
from collections import defaultdict
from math import *
import nltk.translate.bleu_score as bleu
import warnings
import editdistance
import spacy
from spacy import displacy
from nltk import Tree
import en_core_web_sm
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random

warnings.filterwarnings('ignore')

def readText(filename) :
    """
        Méthode qui lis le fichier da_newstest2016.json
    """

    with open(filename, 'r', encoding="utf8") as file :
        fileDict = json.load(file)

    return fileDict

#Partie 3
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
            dicoBleu[ref] += [hypSplit]
    
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
    
def editDistDirection(dicoCorpus) :

    """
    Méthode qui renvoie la distance d'edition moyenne pour chaque direction de langue du corpus
    arg : liste de dictionnaires des phrases du corpus
    return : dictionnare avec pour clé la direction de traduction et en valeur la moyenne de la distance d'édition
   """

    directions = defaultdict(float)
    count = defaultdict(float)

    #pour chaque phrase traduite du dictionnaire on regarde la direction de traduction
    #on calcule la distance d'édition entre hypothèse et ref
    #on incrémente le compteur de phrase dans cette direction
    for key in dicoCorpus :
        trad_dir = key["src_lang"] + ">" + key["tgt_lang"]
        directions[trad_dir] += editdistance.eval(key["hyp"], key["ref"])
        count[trad_dir] +=1


    #pour chaque direction 
    #on fait la moyenne
    for elt in directions :
        count[elt] = directions[elt]/count[elt]

    return count

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

#Partie 4

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

def distribScoreBleu(corpusDict, dicoTrad) :

    chencherry = bleu.SmoothingFunction()
    dicoScore = defaultdict(list)
    dicoPhrases = dicoDirIndir(corpusDict)

    for directTrad in dicoTrad :
        dicoBleu = createDicoRefHyp(corpusDict, directTrad)

        for ref in dicoBleu.keys() :
            if dicoBleu[ref][0] in dicoPhrases['direct_hyp'] :
                bleuScore = bleu.sentence_bleu(dicoBleu[ref], ref.split(), smoothing_function=chencherry.method3)
                dicoScore['direct_hyp'] += [bleuScore]
            elif dicoBleu[ref][0] in dicoPhrases['indirect_hyp'] :
                bleuScore = bleu.sentence_bleu(dicoBleu[ref], ref.split(), smoothing_function=chencherry.method3)
                dicoScore['indirect_hyp'] += [bleuScore]

    return dicoScore  

def editDistDirIndir(dicoCorpus) :
    """
        Méthode qui construit le dictionnaire
        de distance d'édition pour les phrases direct
        et indirect
        Arg : - dicoPhrases = dictionnaires avec les phrases direct et indirect 
        - dicoScore = le dictionnaire de tous les scores de distance d'édition
        pour direct et indirect

        Return : Le dictionnaire dicoScore
    """

    liste_dir = []
    liste_ind = []

    for key in dicoCorpus :
        if key["orig_lang"] == key["src_lang"] :
            liste_dir.append(editdistance.eval(key["hyp"], key["ref"]))
        else :
            liste_ind.append(editdistance.eval(key["hyp"], key["ref"]))

    return liste_dir, liste_ind  


def dicoSrc(corpusDict) :

    dicoPhrases = defaultdict(list)

    for key in corpusDict :
        if key['orig_lang'] == key['src_lang'] :
            dicoPhrases["direct_src"] += [key['src']]
        else :
            dicoPhrases["indirect_src"] += [key['src']]

    return dicoPhrases

def dicoDirIndir(corpusDict) :

    dicoPhrases = defaultdict(list)

    for key in corpusDict :
        if key['orig_lang'] == key['src_lang'] :
            dicoPhrases["direct_hyp"] += [key['hyp'].split() ]
        else :
            dicoPhrases["indirect_hyp"] += [key['hyp'].split()]

    return dicoPhrases


#Partie 5 

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
    dicoSources = dicoSrc(corpusDict)
    listeMoyDepth = []
    listeDepth = [[], []]


    for key in dicoSources :
        total = 0
        for src in dicoSources[key] :
            doc = nlp(src)
            root = [token for token in doc if token.head == token][0]
        
            if key == 'direct_src' :
                listeDepth[0].append((src, treeDepth(root)))
            else :
                listeDepth[1].append((src, treeDepth(root)))

    for liste in listeDepth :
        count = 0
        for src in liste :
            count += src[1]
        listeMoyDepth.append(count / len(liste))

    return listeDepth, listeMoyDepth                      

def treeDepth(root) :
    if not list(root.children):
        return 1
    else:
        return 1 + max(treeDepth(x) for x in root.children)



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
            nb_mots = len(key["src"].split())
            somme_src += nb_mots
            compteur_src += 1
        else :
            nb_mots = len(key["src"].split())
            somme_ref += nb_mots
            compteur_ref +=1

    return somme_src/compteur_src, somme_ref/compteur_ref

def long_moyenne (corpusDict) :

    """
    Méthode qui calcule la longueur moyenne des phrases sources selon leur type de traduction
    Arg : liste de dictionnaires des phrases du corpus
    Return : tuple de float correspondants aux longueurs moyennes directe, indirecte
    """

    somme_dir = 0
    somme_ind = 0
    compteur_dir = 0
    compteur_ind = 0

    #pour chaque traduction du corpus, on vérifie si c'est une traduction directe ou indirecte
    #on compte le nombre de mots de la phrase et selon son type, on met à jour les variables
    #pour pouvoir renvoyer la moyenne
    for key in corpusDict :
        if key["src_lang"] == key["orig_lang"] :
            nb_mots = len(key["src"].split())
            somme_dir += nb_mots
            compteur_dir += 1
        else :
            nb_mots = len(key["src"].split())
            somme_ind += nb_mots
            compteur_ind +=1

    return somme_dir/compteur_dir, somme_ind/compteur_ind


def mot_moyen (corpusDict) :
    """
        Méthode qui calcule le nombre de lettres moyen par mot selon le type de phrase
        Arg : liste de dictionnaires des phrases du corpus
        Return : liste de tuples de float correspondants aux moyennes du nombre de lettres par mot selon le type
    """

    somme_dir = 0
    somme_ind = 0
    total_dir= 0
    total_ind = 0

    #pour chaque traduction dans le corpus on met dans une liste le nombre de mots dans hypothèse
    for key in corpusDict :
        words = key["src"].split()
        #on parcourt la liste de mots de la phrase et selon le type de phrase on met à jour les valeurs 
        #pour renvoyer la moyenne selon le type
        for elt in words :
            if key["src_lang"] == key["orig_lang"] :
                somme_dir += len(elt)
                total_dir +=1
            else :
                somme_ind += len(elt)
                total_ind += 1
    return (somme_dir/total_dir , somme_ind/total_ind)

def repetition (corpusDict) :
    """
    Méthode qui renvoie la moyenne selon chaque type de traduction du nombre d'occurences d'un mot dans une phrase
    arg : liste de dictionnaires des phrases du corpus
    return : tuple de float représentants les moyennes pour les phrases directes et indirectes
    """
    words_dir = []
    words_ind = []
    values_dir = []
    values_ind = []

    #pour chaque traduction du corpus on récupère les mots de la phrase source dans une liste et on crée un dictionnaire d'int
    for key in corpusDict :
        words = key["src"].split()
        dico = defaultdict(int)
        #pour chaque mot de la phrase on incrémente sa valeur dans le dictionnaire
        for elt in words :
                dico[elt] +=1
        #si c'est une traduction directe ou indirecte on ajoute le dictionnaire dans la liste associée
        if key["src_lang"] == key["orig_lang"] :
            words_dir.append(dico)
        else : 
            words_ind.append(dico)

    #pour chaque élément dans la liste de phrases sources directes on récupère toute les valeurs des dictionnaires dans une liste
    for elt in words_dir :
        for values in elt.values() :
            values_dir.append(values)
        
    #pareil pour les phrases indirectes
    for elt in words_ind :
        for values in elt.values() :
            values_ind.append(values)
        
    #on retourne la moyenne de l'occurence d'un mot dans une phrase selon le type de traduction
    return sum(values_dir)/len(values_dir), sum(values_ind)/len(values_ind)


#CLASSIFIEUR
def find_nearest(array, value):
	#méthode qui permet de trouver la valeure la plus proche d'une liste
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()

    return array[idx]


def profondeur (phrase, data_prof) :
    """Méthode qui va rechercher la profondeur de la phrase dans les données"""

    profondeurPh = 0
    for liste in data_prof[0] :
        for src in liste :
            if phrase['src'] == src[0] :
                profondeurPh = src[1]
    return profondeurPh

def vectorisation (phrase, data_long_moy, data_prof) :
    """Méthode qui à partir d'une phrase source, va renvoyer un tuple représentant une phrase traduite
    avec son label (direct indirect) et son vecteur associé de type [-1, 1]"""

    
    data_prof_moy = data_prof[1]
    vecteur = []
    #on calcule la longueur et la profondeur de la phrase
    longueur = len(phrase["src"].split())
    profondeurPh = profondeur(phrase, data_prof)
    type = -1

    #on trouve la valeur moyenne la plus proche de la longueur/profondeur de la phrase
    near_value_long = find_nearest(data_long_moy, longueur)
    near_value_prof = find_nearest(data_prof_moy, profondeurPh)


    #si la valeur moyenne la plus proche c'est la moyenne de traduction directe on met 1 dans le vecteur sinon -1
    if near_value_long == data_long_moy[0] :
        vecteur.append(1)
    elif near_value_long == data_long_moy[1] : 
        vecteur.append(-1)
    
    if near_value_prof == data_prof_moy[0] :
        vecteur.append(1)
    elif near_value_prof == data_prof_moy[1] :
        vecteur.append(-1)

    #on regarde de quel type est la phrase
    if phrase["src_lang"] == phrase["orig_lang"] :
        type = +1

    return (type, vecteur)

def classify (vect_obs, vect_par) :

    """
    Méthode qui calcule le produit scalaire du vecteur d'observation et de paramètre et renvoie l'étiquette associée au vecteur d'observation
    
    Args :
        - vect_obs (liste d'int) vecteur d'observation
        - vect_par (liste d'int) vecteur de paramètre
    
    return : int (étiquette)
    """
    vect_obs = np.array(vect_obs)
    if vect_obs.dot(vect_par) >= 0 :
        return 1
    return -1

def learn (corpus, vect_par, n) :
    """
    Méthode qui va apprendre le vecteur de parametre
    
    Args : 
        -corpus (liste de tuples d'int), liste d'observations
        - vect_par (liste d'int) vecteur de paramètre
        -n (int) nombre de passes maximum souhaitées
    
    return : 
        vect_par (liste d'int) vecteur de parametre modifié
    """
    
    #on créee et initialise le vecteur qui va servir d'historique du vecteur paramètre (la sauvegarde d'avant)
    old_vect_par = [0]*len(vect_par)
    
    #tant que le vecteur de paramètres ne se stabilise pas ou après 200 tours on continue 
    while n != 0 and np.array_equal(vect_par, old_vect_par) == False:
    
        #on met à jour old_vect_par
        old_vect_par = vect_par
        #pour chaque élément dans le corpus on vérifie si les prédictions sont correctes
        for elt in corpus :
            #si elles ne le sont pas, on met à jour le vecteur de paramètres
            if classify(elt[1], vect_par) != elt[0] :
                vect_par = vect_par + elt[0] * elt[1].astype(int)
                    
        #on décrémente le compteur de passe
        n = n-1
    
    return vect_par


def teste (corpus, vect_par) :
    """
    Methode qui calcule le pourcentage d'erreurs sur le corpus
    
    Args :
        - corpus (liste de  tuples d'int) liste d'observations
        - vect_par (liste d'int) vecteur de paramètre
    
    return : int pourcentage d'erreurs
    
    """
    
    #on crée une variable pour compter le nombre d'erreur
    nb_err = 0
    
    #pour chaque élément du corpus, on verifie si la prédiction faite pour l'élément est bonne, si elle ne l'est pas, on incrémente le compteur d'erreurs
    for elt in corpus :
        if classify(elt[1], vect_par) != elt[0] :
            nb_err +=1
    #on retourne le pourcentage d'erreur
    return nb_err/len(corpus)


if __name__ == "__main__":
    corpusDict = readText("da_newstest2016.json")
    dicoTrad = countLang(corpusDict)
    #countExem(corpusDict)
    #countScore(corpusDict)
    #print(scoreBleu(corpusDict, dicoTrad))
    #print(editDist(corpusDict))
    #print(editDistDirIndir(corpusDict))
    #distribScoreBleu(corpusDict, dicoTrad)
    #print(moyScoreDirectTrad(corpusDict, dicoTrad))
    dicoListProf= dependance(corpusDict)
    #données 
    #distribution = distribScore(corpusDict)
    #dic_impact = impact_longueur(corpusDict)
    #dic_moyenne = moyenne_scores(dic_impact)
    data_long_moy = long_moyenne(corpusDict)
    data_mot_moyen = mot_moyen(corpusDict)
    #data_repetition = repetition(corpusDict)
    #data_edit_dist = editDistDirIndir(corpusDict)
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
    """ x1 = sns.distplot(distribScoreBleu(corpusDict,dicoTrad)['direct_hyp'], label ="Source") #bleu
    x1 = sns.distplot(distribScoreBleu(corpusDict, dicoTrad)['indirect_hyp'], label ="Référence") #orange
    plt.show() """

    #graphique distribution DE selon type
    """ x1 = sns.distplot(data_edit_dist[0], label ="Source") #bleu
    x1 = sns.distplot(data_edit_dist[1], label ="Référence") #orange
    plt.show() """

    #graphique distribution profondeur des arbres selon phrases direct ou indirect
    """ listeprofondeur = [[],[]]
    count = 0
    for liste in dicoListProf[0] :
        for src in liste :
            listeprofondeur[count].append(src[1])
        count += 1
    
    x1 = sns.distplot(listeprofondeur[0], label ="direct") #bleu
    x1 = sns.distplot(listeprofondeur[1], label ="indirect") #orange
    plt.show() """

    ######## Classifieur ######
    """ listeVecteurs = []
    for elt in corpusDict :
        listeVecteurs.append(vectorisation(elt, data_mot_moyen, dicoListProf))
    random.shuffle(listeVecteurs)

    tailleListe = len(listeVecteurs)/2
    train = listeVecteurs[:int(tailleListe)]
    test = listeVecteurs[int(tailleListe):]

    w = [0,0]
    w = learn(train, w, 200)
    print(w,teste(test,w)) """


   
