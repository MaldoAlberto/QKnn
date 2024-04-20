
from qknn.functions import index_positions, diffuser, qram, oracle_st, qknn,qknn_experiments
from sklearn import datasets
from sklearn.model_selection import train_test_split

# import numpy for postprocessing to find the k-nn label
import numpy as np

# the qustom qknn methods
from qknn.functions import qknn_experiments

def main():
    
    # read the dataset
    iris = datasets.load_iris()
    
    #split the dataset
    x_train, x_test, y_train, y_test =train_test_split(iris['data'], iris['target'], train_size=0.7, test_size=0.3, random_state=2)
    
    print("Number of original training examples:", len(x_train))
    print("Number of original test examples:", len(x_test))
    
    
    #initial conditions of the experiments of the qknn
    experiment_size = 10
    min_QRAM = 3
    max_QRAM = 8
    features = len(x_train[0])
    
    for k in [1,3,5,7]:
        print(f'Select the {k}-nn' )
        
        #qknn rotation in ry
        qknn_e = qknn_experiments(
            x_test=x_test,
            x_train=x_train,
            y_test=y_test,
            y_train=y_train,
            features=features,
            min_QRAM=min_QRAM,
            max_QRAM=max_QRAM,
            max_trials=1,
            rotation="ry",
            experiment_size=experiment_size)
        
        print(qknn_e.experiments_knn(k=k))
        qknn_e.print_results()
    
        # qknn rotation in rz
        qknn_e = qknn_experiments(
            x_test=x_test,
            x_train=x_train,
            y_test=y_test,
            y_train=y_train,
            features=features,
            min_QRAM=min_QRAM,
            max_QRAM=max_QRAM,
            max_trials=1,
            rotation="rz",
            experiment_size=experiment_size)
        print(qknn_e.experiments_knn(k=k))
        qknn_e.print_results()
    


#if __name__ == '__main__':
   
main()
