#!/usr/bin/env python
# coding: utf-8

# # Projet 10

# In[ ]:


"""
Mettre en place une modélisation pour distinguer les vrais/faux billets
- Création d'un algorithme supervisé

Compléments : 
- Valeurs manquantes présentes (margin_low) 

Soutenance : 
- Cheminement pour création de l'algorithme et modèle finale retenu
- Test de l'aglorithme en direct
"""


# ## Importation des données & Nettoyage des données

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st


# In[7]:


df_bill = pd.read_csv("/Users/adpro/Desktop/Projet 10/billets.csv", sep=";")


# In[8]:


df_bill.info()


# In[9]:


df_bill


# In[ ]:


"""
- 1500 valeurs / 1463 si on enlève les VM
- variables : billet vrai/faux + 6 variables de dimensions du billets. 

Etape : 
- 1 Données : 1 nettoyer (VM) + transformer (Z?)
- 2 Modélisation : 
    Modèle 1 : Clustering > méthode des Kmeans
    Modèle 2 : Regression logistique
"""


# ### Traitement des valeurs manquantes

# In[ ]:


"""
On va prédire les valeurs manquantes de margin low en utilisant une regression linéaire
"""


# In[74]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import t, shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels


# In[141]:


#on commence par enlever la variable is genuine qualitative
df_manq = df_bill.drop("is_genuine",1)
data = df_manq.dropna(subset=["margin_low"])


# In[260]:


data.corr()
#Forte corrélation entre margin_low et lenght -.66
#Potentiel problème de colinéarité


# In[433]:


data = df_manq.dropna(subset=["margin_low"])
reg_multi = smf.ols('margin_low~diagonal+height_left+height_right+margin_up+length', data=data).fit()
print(reg_multi.summary())


# In[57]:


#indépendance des  résidus : test de durbin watson
# 2 idéal, 0 autocorrélation positive, 4 auto corrélation négative 
statsmodels.stats.stattools.durbin_watson(reg_multi.resid)


# In[72]:


X = data[['diagonal', 'height_left', 'height_right', 'margin_up','length']]

x_temp = sm.add_constant(X)

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(x_temp.values, i) for i in range(x_temp.values.shape[1])]
vif["features"] = x_temp.columns
print(vif.round(1))

#cependant il est normal d'avoir des corrélations importantes entre nos variables car 
#il s'agit de dimensions de billets qui sont part nature interdépendantes


# In[76]:


#normalité des résidus : 
name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
test = sms.jarque_bera(reg_multi.resid)
lzip(name, test)

#résidus non normaux > asymetrie


# In[79]:


#Running plot & giving it a title
stats.probplot(reg_multi.resid, dist="norm", plot= plt)
plt.title("Model1 Residuals Q-Q Plot")

#Saving plot as a png
plt.savefig("Model1_Resid_qqplot.png")


# In[91]:


print("la taille de l'échantillon étant importante (", data.shape[0], ") cela ne va pas poser de problème pour la suite")


# In[80]:


import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from scipy import stats
from statsmodels.compat import lzip
import statsmodels
import matplotlib.pyplot as plt

# test de l'homocesdascticité 
# si significatif > hypothèse d'homoscedasticité rejeté > implique une hétéroscédasticité
name = ['Lagrange multiplier statistic', 'p-value', 
        'f-value', 'f p-value']
test = sms.het_breuschpagan(reg_multi.resid, model.model.exog)
lzip(name, test)

#on va donc rajouter cov_type='HC1' dans notre modèle de regression


# In[108]:


reg_multi = smf.ols('margin_low~diagonal+height_left+height_right+margin_up+length', data=data).fit(cov_type='HC1')
print(reg_multi.summary())
print("AIC:", reg_multi.aic)


# In[119]:


reg_multi_1 = smf.ols('margin_low~height_left+height_right+margin_up+length', data=data).fit(cov_type='HC1')
reg_multi_2 = smf.ols('margin_low~height_left+height_right+length', data=data).fit(cov_type='HC1')
reg_multi_3 = smf.ols('margin_low~height_right+length', data=data).fit(cov_type='HC1')
reg_multi_4 = smf.ols('margin_low~length', data=data).fit(cov_type='HC1')
print("Modèle complet :", "AIC:", "{:.2f}".format(reg_multi.aic), "  R2 adj :", "{:.2f}".format(reg_multi.rsquared_adj))
print("Modèle -1 :", "AIC:", "{:.2f}".format(reg_multi_1.aic), "  R2 adj :", "{:.2f}".format(reg_multi_1.rsquared_adj))
print("Modèle -2 :", "AIC:", "{:.2f}".format(reg_multi_2.aic), "  R2 adj :", "{:.2f}".format(reg_multi_2.rsquared_adj))
print("Modèle -3 :", "AIC:", "{:.2f}".format(reg_multi_3.aic), "  R2 adj :", "{:.2f}".format(reg_multi_3.rsquared_adj))
print("Modèle -4 :", "AIC:", "{:.2f}".format(reg_multi_4.aic), "  R2 adj :", "{:.2f}".format(reg_multi_4.rsquared_adj))


# In[ ]:


"""
Ainsi le modèle la variable qui prédit le mieux la margin_low est la longueur
Le modèle le plus parcimonieux semble être le modèle _3 
>> on va donc prédire la marge à partir de la lenght et de la height_right
"""


# In[120]:


reg_multi_3 = smf.ols('margin_low~height_right+length', data=data).fit(cov_type='HC1')


# In[434]:


X = data[['height_right','length']]
marg_pred = reg_multi_3.predict(X)
data_t=data.copy()
data_t["margepred"] = marg_pred
data_t[["margin_low","margepred"]]


# In[435]:


#prédiction des valeurs manquantes
X = df_bill[['height_right','length']]
marg_pred = reg_multi_3.predict(X)
data_temp=df_bill.copy()
data_temp["margepred"] = marg_pred
data_temp[["margin_low","margepred"]]

#on remplace les valeurs manquante par la valeur prédite :
data_temp.margin_low = np.where(data_temp.margin_low.isnull(), data_temp.margepred, data_temp.margin_low)
data_temp.info()


# In[232]:


DF_Final = data_temp.drop("margepred",1)


# ### Description des billets 

# In[261]:


# prediction margin low avec regression linéaire (quanti)
# description billet vrai/faux (kmeans ? box plot, ACP ?)
DF_Final['is_genuine'] = DF_Final['is_genuine'].astype(int)
# 1 = Vrai
# 0 = Faux
DF_Final


# In[247]:


plt.figure(figsize=(8, 5))
plt.title("Box-plot de la distribution du prix")
sns.set_style("whitegrid")
ax = sns.boxplot(x= DF_Final["is_genuine"], y =DF_Final["diagonal"])
plt.show()


# In[249]:


plt.figure(figsize=(8, 5))
plt.title("Box-plot de la distribution du prix")
sns.set_style("whitegrid")
ax = sns.boxplot(x= DF_Final["is_genuine"], y =DF_Final["height_left"])
plt.show()


# In[250]:


plt.figure(figsize=(8, 5))
plt.title("Box-plot de la distribution du prix")
sns.set_style("whitegrid")
ax = sns.boxplot(x= DF_Final["is_genuine"], y =DF_Final["height_right"])
plt.show()


# In[251]:


plt.figure(figsize=(8, 5))
plt.title("Box-plot de la distribution du prix")
sns.set_style("whitegrid")
ax = sns.boxplot(x= DF_Final["is_genuine"], y =DF_Final["margin_low"])
plt.show()


# In[252]:


plt.figure(figsize=(8, 5))
plt.title("Box-plot de la distribution du prix")
sns.set_style("whitegrid")
ax = sns.boxplot(x= DF_Final["is_genuine"], y =DF_Final["margin_up"])
plt.show()


# In[253]:


plt.figure(figsize=(8, 5))
plt.title("Box-plot de la distribution du prix")
sns.set_style("whitegrid")
ax = sns.boxplot(x= DF_Final["is_genuine"], y =DF_Final["length"])
plt.show()


# #

# ## Modèle 1 : Clustering avec les KMeans

# In[ ]:


"""
Etape 1 : Le principe va être de réaliser un clustering avec les Kmeans en visant 2 clusters 
qui distingueront les billets selon leur authenticité. 

Etape 2 : on crée une fonction appliquant le modèle sur un nouveau jeu de donnée afin de prédire l'authenticité

Etape 3 : On applique la fonction sur les données de production
"""


# ### Création du modèle

# In[424]:


# importation des fichiers et des librairies et construction des fonctions
from scipy.cluster.hierarchy import linkage, fcluster , dendrogram
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

## importation functions utiles ##
from sklearn.cluster import KMeans
from sklearn import decomposition
from matplotlib.collections import LineCollection
import numpy as np


# In[263]:


#calibrage des labels/indexs sur les pays : 
X = DF_Final.drop("is_genuine",1)
df_dendrogramme_ok = X

#Centrage et Réduction
std_scale = preprocessing.StandardScaler().fit(df_dendrogramme_ok)
df_dendrogramme_cr = std_scale.transform(df_dendrogramme_ok)


#générer la matrice des liens
Z = linkage(df_dendrogramme_cr,method='ward')
#print(Z)


# In[265]:


## Confirmation du nombre de partitions ( méthode du coude des K-means)
## pour cela on observe de  la décroissance de l’inertie intra-classe pour déterminer la “meilleure” valeur de k

from sklearn import cluster
a=[]
for i in range(2,40):
    kmeans = cluster.KMeans(n_clusters=i) 
    kmeans.fit(X) 
    a.append(kmeans.inertia_)
plt.scatter(range(2,40),a)


# In[ ]:


"""
à ce stade on constate que l'inertie suggère 3 voir 4 clusters.
Cependant nous allons forcer une solution à 2 clusters pour cibler les 2 catégories cibles (Vrai/Faux billet)
Nous allons ainsi forcer une perte d'information
Mais nous vérifierons ensuite  que ce modèle est suffisament précis pour prédire l'authencité des billets
"""


# In[269]:


#dendrogramme en couleur avec les 2 classes (obtenu en prenant une hauteur t = 30)
# on va forcer à 2 classes pour avoir vrai vs faux billets

plt.figure(figsize=(15,30))
plt.title('CAH avec matérialisation des 2 classes')
dendrogram(Z,labels=None,orientation='left',color_threshold=35)
plt.show()


# In[324]:


# réalisation des Kmeans à 2 clusters : 
n_clust = 2

X = DF_Final.drop("is_genuine",1)
km = KMeans(n_clusters = n_clust)
km.fit(X)

clusters = km.labels_


# In[326]:


#ajout d'une colonne indiquant le cluster attribué au billet corrrespondant à la prédiction d'authenticité
DF_KM = DF_Final.copy()
DF_KM["Classe"]= clusters
DF_KM


# In[482]:


matrice_confusion_km = pd.crosstab(df_bill.is_genuine, clusters)
matrice_confusion_km


# In[496]:


#matrice de confusion permettant de voir le recoupement entre la prédiction et l'authencité réelle
# avec les différentes classes pour voir à quoi elles correspondent
print("Matrice de confusion")
matrice_confusion_km = pd.crosstab(DF_KM.is_genuine, clusters)
matrice_confusion_km.columns= ['Classe_0','Classe_1']
matrice_confusion_km.index = ['Faux','Vrai']
matrice_confusion_km


# In[497]:


matrice_confusion_km


# In[498]:


#Changement automatique des noms des clusters : 

if matrice_confusion_km.Classe_0[1] > matrice_confusion_km.Classe_0[0]:
    matrice_confusion_ok = matrice_confusion_km.rename(columns={'Classe_0': 'Prédit Vrai', 'Classe_1': 'Predit Faux'})
else :
    matrice_confusion_ok = matrice_confusion_km.rename(columns={'Classe_1': 'Prédit Vrai', 'Classe_0': 'Predit Faux'})

matrice_confusion_ok

#la classe peut s'inverser lorsque l'on réexcute les kmneans. 
#vérifier lors de l'interprétation. 


# In[466]:


print("Pour info dans notre jeu de données nous avons :")
print("Nombre de billet vrais :", DF_Final.loc[DF_Final["is_genuine"] == 1,:].shape[0])
print("Nombre de billet faux :", DF_Final.loc[DF_Final["is_genuine"] == 0,:].shape[0])


# In[483]:


#on peut ainsi calculer un indice de précision du modèle sur les données d'entrainement: 
from sklearn import metrics
preci_km = metrics.adjusted_rand_score(clusters, DF_KM.is_genuine)
print ("Le modèle utilisant les Kmeans a donc une précision de :", "{:.2f}".format(100*preci_km) ,"%") 


# In[297]:


centroids = km.cluster_centers_
centroids


# In[368]:


#description des centroides : 
DF_KM.groupby("Classe").mean()


# ### Création de l'algorithme

# In[554]:


"""
Pour prédire l'authenticité des billets on va : 
Etape 1 : Prédire le cluster des billets en utilisant notre modèle utilisant les kmeans
Etape 2 : interpréter les cluster selon le modèle pour caractériser l'authenticité des billets

Pour attribuer le cluster au billet on va utiliser la fonction km.predict
Cette fonction prédit le cluster le plus proche possible en utilisant la distance aux centroides.

Pour plus d'infos sur la fonction : 
> https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html pour plus d'infos
> ou encore : https://stackoverflow.com/questions/54511542/what-is-the-use-of-predict-method-in-kmeans-implementation-of-scikit-learn

Ainsi l'algorithme : 
    - Utilise le clustering "km" crée a partir des données disponibles et labélisées
    - On applique ce modèle pour prédire l'authenticité avec km.predict sur les data à tester
    - On liste les billets prédits comme vrai/faux
    - On rappelle la précision du modèle
"""


# In[499]:


#Algorithme : 
def test_km(data) :
    data_algo_km = km.predict(data)
    data = data.assign(Prediction = data_algo_km)
    
    if matrice_confusion_km.Classe_0[1] > matrice_confusion_km.Classe_0[0]:
        v = data.query("Prediction == 0").index.tolist()
        f = data.query("Prediction == 1").index.tolist()
    
    else :
        v = data.query("Prediction == 1").index.tolist()
        f = data.query("Prediction == 0").index.tolist()
        
    print("les Billets suivants sont vraies :", v)
    print("Les Billets suivants sont des faux : ", f)
    print("La précision de la prédiction est de :", "{:.2f}".format(100* preci_km) ,"%")


# #### Test de l'algo sur des données connues

# In[490]:


#test de l'algorithme sur les 5 première valeurs du data test
algo_test = DF_Final.drop("is_genuine",1)


# In[491]:


algo_test = algo_test.iloc[[1,59,1300,1456,765,345],:]


# In[493]:


algo_test


# In[500]:


test_km(algo_test)


# In[492]:


#vérification de la prédiction pour rappel 1 = Vrai, 0 = Faux
df_bill.iloc[[1,59,1300,1456,765,345],:]


# #### Test de l'algo sur les données de production

# In[366]:


df_prod = pd.read_csv("/Users/adpro/Desktop/Projet 10/billets_production.csv", sep=",")


# In[367]:


df_prod = df_prod.set_index("id")


# In[364]:


df_prod


# In[501]:


test_km(df_prod)


# ## Modèle 2 : Régression Logistique 

# ### Split des données

# In[369]:


"""
On va réaliser un split de nos données afin d'avoir : 
    - un set d'entrainement pour construire nos algorithmes
    - un set test pour vérifier la précision de notre algorithmes   
"""
#Split

#valeurs caractéristiques et valeur cible
X = DF_Final[["diagonal","height_left","height_right","margin_low","margin_up","length" ]]
Y = DF_Final["is_genuine"]

#fractionner dataset
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .20, random_state = 0) # random state ?

#création des deux df train & test
df_train = X_train.copy()
df_train ["is_genuine"] = Y_train

df_test = X_test.copy()
df_test ["is_genuine"] = Y_test


# In[370]:


#X_train
#X_test
#Y_train
#Y_test


# In[ ]:


"""
La variable à prédire est une variable boolean : Vrai / Faux. 
Nous allons donc utiliser une regression logistique multiple. 

Etape 1 : Regression logistique pour prédire la variable d'intérêt et 
analyse du modèle de regression sur les données d'entrainement

Etape 2 : Utilisation du modèle de regression pour prédire l'authenticité des billets tests
Et comparaison avec les labels connus vérifier la précision de la prédiction

Etape 3 : création de l'algorithme utilisant la régression logistique
"""


# In[377]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.metrics import accuracy_score


# In[43]:


#première regression logistique utilisant l'ensemble des variables
res = smf.glm("is_genuine ~ diagonal + height_left + height_right + margin_low + margin_up + length",
                   data=df_train, family=sm.families.Binomial()).fit()
print(res.summary())


# In[372]:


#print (res.params)
#print(res.conf_int(alpha = 0.05))


# In[47]:


#accès à la log-vraisemblance du modèle
print("Log-vraisemblance du modèle : %.4f" % (res.llf))

#log-vraisemblance du null modèle
print("Log-vraisemblance du null modèle : %.4f" % (res.llnull))

#R2 de McFadden
R2MF = 1 - res.llf / res.llnull
print("R2MF : ", R2MF)

#exponenielle de LL_null
L0 = np.exp(res.llnull)
#exponentielle de LL_modèle
La = np.exp(res.llf)
#taille de l'échantillon
n = df_train.shape[0]
#R2 de Cox et Snell
R2CS = 1.0 - (L0 / La)**(2.0/n)
print("R2 de Cox - Snell : %.4f" % (R2CS))

#max du R2 de COx-Snell
maxR2CS = 1.0 - (L0)**(2.0/n)
#R2 de Nagelkerke
R2N = R2CS / maxR2CS
print("R2 de Nagelkerke : %.4f" % (R2N))


# In[ ]:


"""
On constate que les variables diagnoal et height_left n'ont pas de poids significatif
dans la prédiction de l'authenticité

On va tester deux modèles : 

>>on va donc enlever ces deux variables dans notre prédiction
"""


# #### Model - Regression logistique

# In[374]:


res = smf.glm("is_genuine ~ height_right + margin_low + margin_up + length",
                   data=df_train, family=sm.families.Binomial()).fit()
print(res.summary())


# In[375]:


#accès à la log-vraisemblance du modèle
print("Log-vraisemblance du modèle : %.4f" % (res.llf))

#log-vraisemblance du null modèle
print("Log-vraisemblance du null modèle : %.4f" % (res.llnull))

#R2 de McFadden
R2MF = 1 - res.llf / res.llnull
print("R2MF : ", R2MF)

#exponenielle de LL_null
L0 = np.exp(res.llnull)
#exponentielle de LL_modèle
La = np.exp(res.llf)
#taille de l'échantillon
n = df_train.shape[0]
#R2 de Cox et Snell
R2CS = 1.0 - (L0 / La)**(2.0/n)
print("R2 de Cox - Snell : %.4f" % (R2CS))

#max du R2 de COx-Snell
maxR2CS = 1.0 - (L0)**(2.0/n)
#R2 de Nagelkerke
R2N = R2CS / maxR2CS
print("R2 de Nagelkerke : %.4f" % (R2N))


# In[409]:


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

x2 = X_train.drop("diagonal",1)
x2 = x2.drop("height_left",1)

xt2 = X_test.drop("diagonal",1)
xt2 = xt2.drop("height_left",1)

y = Y_train

model = LogisticRegression(solver="liblinear", random_state=0).fit(x2, y)

#sortie du modèle : 
## classes du modèle
print("classes :", model.classes_)

## intercept
print("intercept : ", model.intercept_)

## intercept
print("coefficients :", model.coef_)


# #### Evaluation du modèle sur les données d'entrainement

# In[415]:


#Evaluation du modèle : 
##Probabilité pour chaque billet d'être vrai (première valeur) et faux (2ème valeur)
p_pred = model.predict_proba(x2)

##valeur de y prédite par le modèle
y_pred = model.predict(x2)

## précision du modèle
score_ = model.score (x2,y)

## Matrice de confusion
conf_m = confusion_matrix (y, y_pred)

## classification détailée
report = classification_report(y,y_pred)


# In[386]:


y_pred


# In[405]:


print(report)


# In[516]:


matrice_train = pd.crosstab(Y_train, y_pred)
matrice_train


# In[517]:


print("Matrice de confusion sur les données d'entrainement : ")
matrice_train.columns = ['Predit Faux',"Predit Vrais"]
matrice_train.index = ['Faux','Vrais']
matrice_train


# In[514]:


print("Pour Info nous avons dans les données d'entrainement :")
print("Nombre de billets vrais :", df_train.loc[df_train["is_genuine"] == 1,:].shape[0])
print("Nombre de billets faux :", df_train.loc[df_train["is_genuine"] == 0,:].shape[0])


# In[388]:


p = metrics.adjusted_rand_score(y_pred, Y_train)
print("Précision du modèle sur les données d'entrainement :", "{:.2f}".format(100* p) ,"%")


# #### Test du modèle sur les données Test

# In[502]:


conf_m


# In[506]:


y_pred_t = model.predict(xt2)

matrice_test = pd.crosstab(Y_test, y_pred_t)
matrice_test


# In[508]:


print("Matrice de confusion sur les données tests : ")
matrice_test.columns= ['Predit Faux',"Predit Vrai"]
matrice_test.index = ['Faux','Vrai']
matrice_test


# In[518]:


print("Pour Info nous avons dans les données test:")
print("Nombre de billets vrais :", df_test.loc[df_test["is_genuine"] == 1,:].shape[0])
print("Nombre de billets faux :", df_test.loc[df_test["is_genuine"] == 0,:].shape[0])


# In[393]:


p = metrics.adjusted_rand_score(y_pred_t, Y_test)
print("Précision du modèle sur les données test :", "{:.2f}".format(100* p) ,"%")


# In[292]:


#probabilité pour chaque billet d'être vrai (première valeur) et faux (2ème valeur)
#model.predict_proba(xt2)


# In[414]:


#Evaluation du modèle : 
##Probabilité pour chaque billet d'être vrai (première valeur) et faux (2ème valeur)
p_pred_t = model.predict_proba(xt2)

##valeur de y prédite par le modèle
y_pred_t = model.predict(xt2)

## précision du modèle
score_t = model.score (xt2,Y_test)

## Matrice de confusion
conf_m_t = confusion_matrix (Y_test, y_pred_t)

## classification détailée
report_t = classification_report(Y_test,y_pred_t)
print(report_t)


# ### Création de l'algorithme

# In[520]:


def test_rlogistic(data): 
    #suppression des variables inutilisées
    data=data.drop("diagonal",1)
    data=data.drop("height_left",1)
    
    ### Même chose mais avec le modèle 2
    ##prédiction des probabilité
    a = model.predict_proba(data)
    b = model.predict(data)
    ## prediction de l'authenticité du billet
    data = data.assign(Prediction = b)
    #séparation de l'array en 2 colonnes
    temp_prob = np.hsplit(a,2)
    data["Proba_Vrai"] = temp_prob[1]
    data["Proba_Faux"] = temp_prob[0]
    
    #edition des sorties
    vrai = data.loc[data["Prediction"] == 1,("Prediction","Proba_Vrai")]
    vrai = vrai.rename(columns={"Proba_Vrai": 'Probabilité %'}) 
    faux = data.loc[data["Prediction"] == 0,("Prediction","Proba_Faux")]
    faux = faux.rename(columns={"Proba_Faux": 'Probabilité %'}) 
    
    df = vrai.append(faux)
    df['Prediction'] = df['Prediction'].replace([1], 'Authentique')
    df['Prediction'] = df['Prediction'].replace([0], 'Faux') 
    df['Probabilité %'] = df['Probabilité %'] *100
    
    print("Voici les prediction du modèle utilisant la régression logistique :")
    print("\n")
    print(df)


# #### Test de l'algorithme sur des données connues

# In[521]:


test_rlogistic(algo_test)


# In[522]:


#vérification de la prédiction
df_bill.iloc[[1,59,1300,1456,765,345],:]


# #### Test de l'algorithme sur les billets de la production

# In[523]:


test_rlogistic(df_prod)


# # Algorithme de test

# # Kmeans

# In[425]:


def test_km(data) :
    data_algo_km = km.predict(data)
    data = data.assign(Prediction = data_algo_km)
    
    if matrice_confusion_km.Classe_0[0] > matrice_confusion_km.Classe_1[0]:
        v = data.query("Prediction == 1").index.tolist()
        f = data.query("Prediction == 0").index.tolist()
    
    else :
        v = data.query("Prediction == 0").index.tolist()
        f = data.query("Prediction == 1").index.tolist()
        
    print("les Billets suivants sont vraies :", v)
    print("Les Billets suivants sont des faux : ", f)
    print("La précision de la prédiction est de :", "{:.2f}".format(100* preci_km) ,"%")


# # Regression logistique

# In[426]:


def test_rlogistic(data): 
    #suppression des variables inutilisées
    data=data.drop("diagonal",1)
    data=data.drop("height_left",1)
    
    ### Même chose mais avec le modèle 2
    ##prédiction des probabilité
    a = model.predict_proba(data)
    b = model.predict(data)
    ## prediction de l'authenticité du billet
    data = data.assign(Prediction = b)
    #séparation de l'array en 2 colonnes
    temp_prob = np.hsplit(a,2)
    data["Proba_Vrai"] = temp_prob[1]
    data["Proba_Faux"] = temp_prob[0]
    
    #edition des sorties
    vrai = data.loc[data["Prediction"] == 1,("Prediction","Proba_Vrai")]
    vrai = vrai.rename(columns={"Proba_Vrai": 'Probabilité %'}) 
    faux = data.loc[data["Prediction"] == 0,("Prediction","Proba_Faux")]
    faux = faux.rename(columns={"Proba_Faux": 'Probabilité %'}) 
    
    df = vrai.append(faux)
    df['Prediction'] = df['Prediction'].replace([1], 'Authentique')
    df['Prediction'] = df['Prediction'].replace([0], 'Faux') 
    df['Probabilité %'] = df['Probabilité %'] *100
    
    print("Voici les prediction du modèle utilisant la régression logistique :")
    print("\n")
    print(df)


# In[524]:


df_prod = pd.read_csv("/Users/adpro/Desktop/Projet 10/billets_production.csv", sep=",")
df_prod = df_prod.set_index("id")


# In[ ]:




