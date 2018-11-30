########33
#######  DECISION TREE Classifier ###################################################################################################3
##############

from sklearn import tree
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

for i in range(1,11):
    decision_tree = DecisionTreeClassifier(max_depth = i)
    decision_tree.fit(train_X, train_y)
    print(i,"Accuracy:", decision_tree.score(val_X, val_y))

from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, average_precision_score, precision_score, recall_score

decision_tree = DecisionTreeClassifier(max_depth = 9)
decision_tree.fit(train_X, train_y)
y_pred_tree = decision_tree .predict(val_X)

DecisionTree=[decision_tree.score(val_X, val_y),precision_score(val_y, y_pred_tree),recall_score(val_y, y_pred_tree),f1_score(val_y, y_pred_tree)]
results=pd.DataFrame(DecisionTree,columns=['DecisionTree'],index=['Accuracy:', 'Precision:', 'Recall:','F1:'])


# Export our trained model as a .dot file
with open("treeisfraud.dot", 'w') as f:
     #f = Source(
         f=tree.export_graphviz(decision_tree,
                              out_file=f,
                              max_depth = 9,
                              impurity = True,
                              feature_names = ['amount','oldbalanceOrg', 'newbalanceOrig',
            'type_CASH_IN','type_CASH_OUT','type_DEBIT','type_PAYMENT','type_TRANSFER',
        'oldbalanceDest', 'newbalanceDest'],
                              class_names = ['No Fraud', 'Fraud'],
                              rounded = True,
                              filled= True )#)
        
#Convert .dot to .png to allow display in web notebook
check_call(['dot','-Tpng','treeisfraud.dot','-o','treeisfraud.png'])

# Annotating chart with PIL
img = Image.open("treeisfraud.png")
draw = ImageDraw.Draw(img)
font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf', 26)
draw.text((10, 0), # Drawing offset (position)
          '"Title <= Is Fraud', # Text to draw
          (0,0,255), # RGB desired color
          font=font) # ImageFont object with desired font
img.save('tree_isfraud.png')
PImage("tree_isfraud.png")

########33
#######  KNeighborsClassifier ###################################################################################################3
##############

from sklearn.neighbors import KNeighborsClassifier

neiclassifier = KNeighborsClassifier(n_neighbors=5)  
neiclassifier.fit(train_X, train_y)
y_pred_nei = neiclassifier.predict(val_X)

KNeighbors=[neiclassifier.score(val_X, val_y),precision_score(val_y, y_pred_nei),recall_score(val_y, y_pred_nei),f1_score(val_y,y_pred_nei)]


results['KNeighbors']=KNeighbors
results

########33
#######  RANDOM FOREST CLASSIFIER###################################################################################################3
##############
In [16]:
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(random_state=1)
forest.fit(train_X, train_y)
y_pred_for = forest.predict(val_X)

RandomForest=[forest.score(val_X, val_y),precision_score(val_y, y_pred_for),recall_score(val_y, y_pred_for),f1_score(val_y,y_pred_for)]

results['RandomForest']=RandomForest
results

########33
####### LOGISTIC REGRESSION ###################################################################################################3
##############

from sklearn.linear_model import LogisticRegression
import warnings
#warnings.filterwarnings("ignore", category=FutureWarning)

logreg = LogisticRegression(solver='lbfgs')
logreg.fit(train_X, train_y)
y_pred_log =logreg.predict(val_X)

LogisticRegression=[logreg.score(val_X, val_y),precision_score(val_y, y_pred_log),recall_score(val_y, y_pred_log),f1_score(val_y,y_pred_log)]

results['LogisticRegression']=LogisticRegression
results

########33
#######  VISUALIZATION RESULTS ###################################################################################################3
##############

import matplotlib.pyplot as plt
import random

preds=pd.DataFrame()
preds['validation']=val_y
preds['tree']=y_pred_tree
preds['neighbors']=y_pred_nei
preds['forest']=y_pred_for
preds['logistic']=y_pred_log

preds.describe()
pr=preds.sample(n=100)
ind = np.linspace(0,200,100)# len(val_y),len(val_y))
plt.xlabel('Data index')
plt.ylabel('Fraud or No?') 

plt.plot(ind, pr['validation'],'X', markersize=6,label='Test data' )
plt.plot(ind,pr['tree'], '.', markersize=5,label='Decision Tree')
plt.plot(ind,pr['neighbors'], '.', markersize=5,label='K Neighbors')
plt.plot(ind,pr['forest'], '.', markersize=5,label='Random Forest')
plt.plot(ind,pr['logistic'], '.', markersize=5,label='Logistic Regression')
plt.legend(loc='right')
#plt.xlim(-2,1012)
plt.ylim(-.1,1.12)
plt.show()
