#! python3.6

#---------------------------imports---------------------------
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import preprocessing
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#---------------------------data---------------------------
dataset = pd.read_csv("iris.csv")

#--------------------Label Encoder-----------------------------------
label_encoder = preprocessing.LabelEncoder()
dataset['class_cat'] = label_encoder.fit_transform(dataset['species'])

#---------------split train and test data---------------
X = dataset.iloc[:,0:4]
y = dataset.iloc[:,4]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 1)#42

#---------------compare models with KFold and StratifiedKFold---------------
models=[]
models.append(("KNN",KNeighborsClassifier()))
models.append(("CART",DecisionTreeClassifier()))
models.append(("GNB",GaussianNB()))
models.append(("SVM",SVC(gamma='auto')))
models.append(("LDA",LinearDiscriminantAnalysis()))


results=[]
#resultdict = dict()
cvscoredict = dict()

for k in (KFold,StratifiedKFold):
    cvscoreset = []
    names=[]
    for name,model in models:
        kfold = k(n_splits = 10, random_state = None)
        cvscore = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
        results.append([name,cvscore.mean(),cvscore.std()])
        cvscoreset.append(cvscore.tolist())
        names.append(name)
    cvscoredict.update({k:cvscoreset})    
    #resultdict.update({k:results})

fig2 = plt.figure()
gs = GridSpec(2,2, figure=fig2)
ax1 = fig2.add_subplot(gs[0,0],)
ax2 = fig2.add_subplot(gs[0,1],sharey=ax1)
ax3 = fig2.add_subplot(gs[1, 0:2])

ax1.boxplot(cvscoredict[KFold], labels =  names)
ax1.title.set_text('Model Comparison: KFold')
ax2.boxplot(cvscoredict[StratifiedKFold], labels =  names)
ax2.title.set_text('Model Comparison: StratifiedKFold')

means = [ele[1] for ele in results]
std = [ele[2]*30000 for ele in results]
#for Kfold(first5)
ax3.scatter(names,means[:5],s=std[:5],alpha=0.3)
ax3.scatter(names,means[:5])
#for StratifiedKfold(last5)
ax3.scatter(names,means[5:],s=std[5:],alpha=0.3)
ax3.scatter(names,means[5:])
ax3.title.set_text('Mean and Standard Deviation with blue for Kfold and green for StratifiedKfold')

fig2.tight_layout(pad = 1)

#---------------------Scatter Matrix----------------------------
pd.plotting.scatter_matrix(dataset.iloc[:,:4])

#--------------------part3----------------
fig3 = plt.figure(constrained_layout=False)
gs = GridSpec(1, 3, figure=fig3)
ax = [fig3.add_subplot(gs[0, 0]),fig3.add_subplot(gs[0,1:])]

def get_acc(features):
    model = SVC(gamma='auto')
    model.fit(X_train.iloc[:,features],y_train)
    
    y_pred = model.predict(X_test.iloc[:,features])
    
    return accuracy_score(y_test,y_pred)

colnames = list(dataset.iloc[:,:4].columns)
collen = len(colnames)
f2=[]
f2acc=[]
f3=[]
f3acc=[]
for i in range(collen):
    for j in range(i+1,collen):
        f2.append([i,j])
        f2acc.append(get_acc([i,j]))
        for k in range(j+1,collen):
            f3.append([i,j,k])
            f3acc.append(get_acc([i,j,k]))

#feature accuracy matrix for 2 features
matrix = np.zeros((collen,collen))
for i,ele in enumerate(f2):
    matrix[ele[0]][ele[1]] = matrix[ele[1]][ele[0]] = f2acc[i]

splot = ax[0]
#plot matrix
image = splot.imshow(matrix,cmap=plt.cm.Blues)
#splot.figure.colorbar(image, ax=splot )
splot.set(xticks=np.arange(matrix.shape[1]),
         yticks=np.arange(matrix.shape[0]),
         title='Accuracy Matrix for 2 features',
         xticklabels=colnames,
         yticklabels=colnames,
         )
splot.set_yticklabels(labels=colnames, rotation=90, va='center', fontsize=8)
splot.set_xticklabels(labels=colnames, fontsize=8)

#set annotations i.e. text in square
fmt = '.2f' #if normalise else 'd'
threshold = matrix.max()/2
for i in range(matrix.shape[0]):
    for j in range (matrix.shape[1]):
        splot.text(j,i,format(matrix[i,j],fmt),
                  ha='center',va='center',
                  color='white' if matrix[i,j]>threshold else 'black')
#plt.imshow(matrix,cmap=plt.cm.Blues)
#plt.show()

f4acc=get_acc(range(4))
f=f2+f3
ypts=f2acc+f3acc
lenypts = len(ypts)
ax[1].plot(range(lenypts),ypts,'-o');   # line (-), circle marker (o)
ax[1].plot(range(lenypts),[f4acc]*lenypts,color='red')


d={0:'sl',1:'sw',2:'pl',3:'pw'}
for i in range(lenypts):
    ax[1].annotate([d[i] for i in f[i]],(i,ypts[i]),(i+0.1,ypts[i]-0.01),fontsize=8,rotation=90)

plt.show()