from sklearn.svm import SVC
from run_exp import exp

# model1 = SVC(kernel='linear')
# exp("svm_linear", model1)

model2 = SVC(kernel='rbf')
exp("svm_rbf", model2)