#**1. Importing the Datasets*

library(ranger)
library(caret)
library(data.table)

setwd("C:\\Users\\user\\R projects\\Detect Credit Card Fraud")

df <- read.csv("creditcard.csv")
df


#** 2. Data Exploration*

dim((df))

head(df,6)

tail(df, 6)


table(df$Class)

summary(df$Amount)

names(df)

var(df$Amount)

sd(df$Amount)


#** Data Manipulation*

# Dans cette section, nous allons mettre à l'échelle nos données à l'aide de la fonction scale(). 
# Nous l'appliquerons sur la variable Amount. 
# La mise à l'échelle est également connue sous le nom de normalisation des caractéristiques(feature standardization. en anglais). 
# Grâce à la mise à l'échelle, les données sont structurées en fonction d'une plage spécifiée. 
# Par conséquent, il n'y a pas de valeurs extrêmes dans notre jeu de données qui pourraient 
# interférer avec le fonctionnement de notre modèle. 

head(df)

df$Amount = scale (df$Amount)
newdf= df[,-c(1)]
head(newdf)

#** 4. Data Modeling

# Après avoir normalisé notre jeu de données, nous allons le diviser 2: Train set et test avec un
# ratio de division de 0,80. Cela signifie que 80% de nos données seront attribuées aux données 
# d'entraînement et 20% aux données de test. Nous trouverons ensuite les dimensions à l'aide de la 
# fonction dim().


library(caTools)

set.seed(123)

datasample = sample.split(newdf$Class,SplitRatio=0.80)

train_data = subset(newdf,datasample==TRUE)
test_data = subset(newdf,datasample==FALSE)

dim(train_data)

dim(test_data)

#**5. Ajustement du modèle de régression logistique*

# Dans cette section, nous allons ajuster notre premier modèle en commencant par une régression 
# logistique. 

# La régression logistique est utilisée pour modéliser la probabilité de résultat d'une
# classe telle que réussite/échec, positif/négatif et dans notre cas - fraude/non-fraude. 
# Nous implémentons ce modèle sur nos données de test de la manière suivante:

LogisticModel=glm(Class~., test_data, family=binomial()) # glm=generalized linear model
# family = binomial,indique à R que nous voulons faire une régression logistique.

summary(LogisticModel)

# Après avoir résumé notre modèle, nous allons le visualiser à travers les graphiques suivants:

plot(LogisticModel)


# Afin d'évaluer la performance de notre modèle, nous allons délimiter la courbe ROC. ROC est 
# également connu sous le nom de Receiver Optimistic Characteristics. Pour cela, nous allons d'abord
# importer le package ROC, puis tracer notre courbe ROC pour analyser ses performances.

library(pROC)

LogisticModel()

lr.predict <- predict(LogisticModel,test_data, probability = TRUE)

auc.gbm = roc(test_data$Class, lr.predict, plot = TRUE, col = "blue")


#** 6. Ajustement d'un modèle d'arbre de décision*

# Dans cette section, nous allons mettre en œuvre un algorithme d'arbre de décision. Les arbres de 
# décision permettent de représenter les résultats d'une décision. Ces résultats sont 
# essentiellement une conséquence grâce à laquelle nous pouvons conclure à quelle classe appartient
# l'objet. 
# Implémenter maintenat notre modèle et tracons le à l'aide de la fonction rpart.plot(). 
# Nous utiliserons spécifiquement la séparation récursive pour tracer l'arbre de décision.

library(rpart)
library(rpart.plot)

decisionTree_model <- rpart(Class ~ . , df, method = 'class')

predicted_val <- predict(decisionTree_model, df, type = 'class')

probability <- predict(decisionTree_model, df, type = 'prob')

rpart.plot(decisionTree_model)



#** 7. Réseau neuronal artificiel (Artificial Neural Network)*

# Les réseaux neuronaux artificiels sont un type d'algorithme de Machine Learning qui s'inspire du 
# système nerveux humain. Les modèles ANN sont capables d'apprendre les modèles en utilisant les 
# données historiques et sont capables d'effectuer une classification sur les données d'entrée. 

# Nous allons importer le package neuralnet qui nous permettra d'implémenter nos ANN. Nous allons 
# ensuite procéder au traçage à l'aide de la fonction plot(). 

# Dans le cas des réseaux neuronaux artificiels, la plage de valeurs est comprise entre 1 et 0. 
# Nous fixerons un seuil de 0,5, c'est-à-dire que les valeurs supérieures à 0,5 correspondent à 1 et
# les autres à 0. 


library(neuralnet)

ANN_model = neuralnet (Class~.,train_data,linear.output=FALSE)

plot(ANN_model)

predANN=compute(ANN_model,test_data)
resultANN=predANN$net.result
resultANN=ifelse(resultANN>0.5,1,0)


#**8. Gradient Boosting (GBM)*

# Le  Gradient Boosting est un algorithme populaire de Machine Learning qui est utilisé 
# pour effectuer des tâches de classification et de régression. Ce modèle comprend plusieurs modèles 
# d'ensemble sous-jacents comme des arbres de décision faibles. Ces arbres de décision se combinent
# ensemble pour former un modèle fort de gradient boosting. Nous implémenterons l'algorithme de 
# descente de gradient dans notre modèle de la manière suivante:


library(gbm, quietly=TRUE)

# Obtention du temps d'entraînement du modèle GBM

system.time(
  model_gbm <- gbm(Class ~ .
                   , distribution = "bernoulli"
                   , data = rbind(train_data, test_data)
                   , n.trees = 500
                   , interaction.depth = 3
                   , n.minobsinnode = 100
                   , shrinkage = 0.01
                   , bag.fraction = 0.5
                   , train.fraction = nrow(train_data) / (nrow(train_data) + nrow(test_data))
  )
)

# Détermination de la meilleure itération en fonction des données de test.

gbm.iter = gbm.perf(model_gbm, method = "test")


model.influence = relative.influence(model_gbm, n.trees = gbm.iter, sort. = TRUE)

plot(model_gbm)
