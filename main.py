import project1 as p1
import utils
import numpy as np
#import pandas as pd

from timeit import default_timer as timer #timer

# sklearn linear classifiers
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

#-------------------------------------------------------------------------------
# Data loading
#-------------------------------------------------------------------------------

train_data = utils.load_data('reviews_train.tsv')
val_data = utils.load_data('reviews_val.tsv')
test_data = utils.load_data('reviews_test.tsv')

#stopwords
with open('stopwords.txt') as f:
    stopwords = f.read()
    #print("stopwords:", stopwords)


train_texts, train_labels = zip(*((sample['text'], sample['sentiment']) for sample in train_data))
val_texts, val_labels = zip(*((sample['text'], sample['sentiment']) for sample in val_data))
test_texts, test_labels = zip(*((sample['text'], sample['sentiment']) for sample in test_data))



#-------------------------------------------------------------------------------
# Creating bag-of-words dictionary (timed)
#-------------------------------------------------------------------------------
start = timer() #timer

dictionary = p1.bag_of_words(train_texts, stopwords)
print("Size of dictionary:", len(dictionary))

end = timer() #timer
print("time to create BOW dictionary:", end - start) #timer

#-------------------------------------------------------------------------------
# Extracting features from text using dictionary
#-------------------------------------------------------------------------------


train_bow_features = p1.extract_bow_feature_vectors(train_texts, dictionary)
val_bow_features = p1.extract_bow_feature_vectors(val_texts, dictionary)
test_bow_features = p1.extract_bow_feature_vectors(test_texts, dictionary)

print("features", train_bow_features[:5,:15])

#-------------------------------------------------------------------------------
# Toy data - calculating thetas, plotting data & decision boundary (Problem 5)
#-------------------------------------------------------------------------------

toy_features, toy_labels = toy_data = utils.load_toy_data('toy_data.tsv')

# Checking for convergence (using T=100 instead of T=10)
T = 10
L = 0.2

thetas_perceptron = p1.perceptron(toy_features, toy_labels, T)
thetas_avg_perceptron = p1.average_perceptron(toy_features, toy_labels, T)
thetas_pegasos = p1.pegasos(toy_features, toy_labels, T, L)




def plot_toy_results(algo_name, thetas):
    print('theta for', algo_name, 'is', ', '.join(map(str,list(thetas[0]))))
    print('theta_0 for', algo_name, 'is', str(thetas[1]))
    utils.plot_toy_data(algo_name, toy_features, toy_labels, thetas)

plot_toy_results('Perceptron', thetas_perceptron)
plot_toy_results('Average Perceptron', thetas_avg_perceptron)
plot_toy_results('Pegasos', thetas_pegasos)

#-------------------------------------------------------------------------------
# Calculating train and validation accuracy using Amazon reviews data (Problem 7)
#-------------------------------------------------------------------------------

T = 10
L = 0.01

# Perceptron

start = timer()

pct_train_accuracy, pct_val_accuracy = \
    p1.classifier_accuracy(p1.perceptron, train_bow_features,val_bow_features,train_labels,val_labels,T=T)
print("{:35} {:.4f}".format("Training accuracy for perceptron:", pct_train_accuracy))
print("{:35} {:.4f}".format("Validation accuracy for perceptron:", pct_val_accuracy))

end = timer()
print("time to run Perceptron:", end - start)


# Avg perceptron
start = timer()

avg_pct_train_accuracy, avg_pct_val_accuracy = \
   p1.classifier_accuracy(p1.average_perceptron, train_bow_features,val_bow_features,train_labels,val_labels,T=T)
print("{:43} {:.4f}".format("Training accuracy for average perceptron:", avg_pct_train_accuracy))
print("{:43} {:.4f}".format("Validation accuracy for average perceptron:", avg_pct_val_accuracy))

end = timer()
print("time to run Avg perceptron:", end - start)

# Pegasos
start = timer()
avg_peg_train_accuracy, avg_peg_val_accuracy = \
   p1.classifier_accuracy(p1.pegasos, train_bow_features,val_bow_features,train_labels,val_labels,T=T,L=L)
print("{:50} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
print("{:50} {:.4f}".format("Validation accuracy for Pegasos:", avg_peg_val_accuracy))

end = timer()
print("time to run Pegasos:", end - start)

# LDA
start = timer()

clf1 = LinearDiscriminantAnalysis()
clf1.fit(train_bow_features, train_labels)
parameters_LDA = clf1.get_params()
print("!", parameters_LDA)

pred1=clf1.predict(val_bow_features)

LDA_train_accuracy=clf1.score(train_bow_features, train_labels)
LDA_val_accuracy=clf1.score(val_bow_features, val_labels)

print("{:50} {:.4f}".format("Training accuracy for LDA:", LDA_train_accuracy))
print("{:50} {:.4f}".format("Validation accuracy for LDA:", LDA_val_accuracy))

end = timer()
print("time to run LDA:", end - start)


# Logistic regression
start = timer()

clf2 = LogisticRegression(random_state=0).fit(train_bow_features, train_labels)
parameters_LR = clf2.get_params()
print("!!", parameters_LR)

pred2=clf2.predict(val_bow_features)

LR_train_accuracy=clf2.score(train_bow_features, train_labels)
LR_val_accuracy=clf2.score(val_bow_features, val_labels)

print("{:50} {:.4f}".format("Training accuracy for Logistic regression:", LR_train_accuracy))
print("{:50} {:.4f}".format("Validation accuracy for Logistic regression:", LR_val_accuracy))

end = timer()
print("time to run Logistic regression:", end - start)

# Naive Bayes
start = timer()

clf3 = GaussianNB()
clf3.fit(train_bow_features, train_labels)
parameters_NB= clf3.get_params()
print("!!!", parameters_NB)

pred3=clf3.predict(val_bow_features)

NB_train_accuracy=clf3.score(train_bow_features, train_labels)
NB_val_accuracy=clf3.score(val_bow_features, val_labels)

print("{:50} {:.4f}".format("Training accuracy for Naive Bayes:", NB_train_accuracy))
print("{:50} {:.4f}".format("Validation accuracy for Naive Bayes:", NB_val_accuracy))

end = timer()
print("time to run Naive Bayes:", end - start)

#-------------------------------------------------------------------------------
# Hyperparameter tuning, accuracy on test set (Problem 8)
#-------------------------------------------------------------------------------

data = (train_bow_features, train_labels, val_bow_features, val_labels)

# values of T and lambda to try
Ts = [1, 5, 10, 15, 25, 50]
Ls = [0.001, 0.01, 0.1, 1, 10]

pct_tune_results = utils.tune_perceptron(Ts, *data)
print('perceptron valid:', list(zip(Ts, pct_tune_results[1])))
print('best = {:.4f}, T={:.4f}'.format(np.max(pct_tune_results[1]), Ts[np.argmax(pct_tune_results[1])]))

avg_pct_tune_results = utils.tune_avg_perceptron(Ts, *data)
print('avg perceptron valid:', list(zip(Ts, avg_pct_tune_results[1])))
print('best = {:.4f}, T={:.4f}'.format(np.max(avg_pct_tune_results[1]), Ts[np.argmax(avg_pct_tune_results[1])]))

# fix values for L and T while tuning Pegasos T and L, respective
fix_L = 0.01
peg_tune_results_T = utils.tune_pegasos_T(fix_L, Ts, *data)
print('Pegasos valid: tune T', list(zip(Ts, peg_tune_results_T[1])))
print('best = {:.4f}, T={:.4f}'.format(np.max(peg_tune_results_T[1]), Ts[np.argmax(peg_tune_results_T[1])]))

fix_T = Ts[np.argmax(peg_tune_results_T[1])]
peg_tune_results_L = utils.tune_pegasos_L(fix_T, Ls, *data)
print('Pegasos valid: tune L', list(zip(Ls, peg_tune_results_L[1])))
print('best = {:.4f}, L={:.4f}'.format(np.max(peg_tune_results_L[1]), Ls[np.argmax(peg_tune_results_L[1])]))

utils.plot_tune_results('Perceptron', 'T', Ts, *pct_tune_results)
utils.plot_tune_results('Avg Perceptron', 'T', Ts, *avg_pct_tune_results)
utils.plot_tune_results('Pegasos', 'T', Ts, *peg_tune_results_T)
utils.plot_tune_results('Pegasos', 'L', Ls, *peg_tune_results_L)

#-------------------------------------------------------------------------------
# Use the best method (perceptron, average perceptron or Pegasos) along with
# the optimal hyperparameters according to validation accuracies to test
# against the test dataset. The test data has been provided as
# test_bow_features and test_labels.
#-------------------------------------------------------------------------------

T = 25
L = 0.01

avg_peg_train_accuracy, avg_peg_test_accuracy = \
   p1.classifier_accuracy(p1.pegasos, train_bow_features, test_bow_features, train_labels, test_labels, T=T, L=L)
print("{:50} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
print("{:50} {:.4f}".format("Test accuracy for Pegasos:", avg_peg_test_accuracy))

#-------------------------------------------------------------------------------
# Assign to best_theta, the weights (and not the bias!) learned by your most
# accurate algorithm with the optimal choice of hyperparameters.
#-------------------------------------------------------------------------------


T = 25
L = 0.01

# extract the best theta w/o bias from selected algorithm (pegasos)
TT = p1.pegasos(train_bow_features, train_labels, T,L)


best_theta = TT[0] # Your code here
wordlist   = [word for (idx, word) in sorted(zip(dictionary.values(), dictionary.keys()))]
sorted_word_features = utils.most_explanatory_word(best_theta, wordlist)
print("Most Explanatory Word Features")
print(sorted_word_features[:10])
