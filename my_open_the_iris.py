import pandas as pd
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

def load_dataset():
    url = 'https://storage.googleapis.com/qwasar-public/track-ds/iris.csv'
    dataset = pd.read_csv(url)
    return(dataset)

def summarize_dataset():
    print(dataset.shape)
    # print(dataset.head(10))
    # print(dataset.describe())
    # print(dataset.groupby('class').size())

def print_plot_univariate():
    dataset.hist()
    pyplot.show()

def print_plot_multivariate():
    scatter_matrix(dataset)
    pyplot.show()

def my_print_and_test_models():
    array = dataset.values
    X = array[:,0:4]
    y = array[:,4]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size = 0.20, random_state = 1)
    
    model = DecisionTreeClassifier()
    cv_results = cross_val_score(model, X_train, Y_train, cv = 5, scoring = 'accuracy')
    print('%s: %f (%f)' % ('DecisionTree', cv_results.mean(), cv_results.std()))

    model = GaussianNB()
    cv_results = cross_val_score(model, X_train, Y_train, cv = 5, scoring = 'accuracy')
    print('%s: %f (%f)' % ('GaussianNB', cv_results.mean(), cv_results.std()))

    model = KNeighborsClassifier()
    cv_results = cross_val_score(model, X_train, Y_train, cv = 5, scoring = 'accuracy')
    print('%s: %f (%f)' % ('KNeighbors', cv_results.mean(), cv_results.std()))

    model = LogisticRegression()
    cv_results = cross_val_score(model, X_train, Y_train, cv = 5, scoring = 'accuracy')
    print('%s: %f (%f)' % ('LogisticRegression', cv_results.mean(), cv_results.std()))

    model = LinearDiscriminantAnalysis()
    cv_results = cross_val_score(model, X_train, Y_train, cv = 5, scoring = 'accuracy')
    print('%s: %f (%f)' % ('LinearDiscriminant', cv_results.mean(), cv_results.std()))

    model = SVC(gamma = 'auto')
    cv_results = cross_val_score(model, X_train, Y_train, cv = 5, scoring = 'accuracy')
    print('%s: %f (%f)' % ('SVM', cv_results.mean(), cv_results.std()))
    
dataset = load_dataset()
summarize_dataset()
print_plot_univariate()
print_plot_multivariate()
my_print_and_test_models()
