# scikit-learn MLPClassifier() with dropout
# forked from https://github.com/glennq/scikit-learn/tree/mlp_dropout_new  and modified for python 3.7 
#  
mlp = MLPClassifier(activation=activation,
                                    hidden_layer_sizes=10,
                                    solver='lbfgs', alpha=1e-5,
                                    learning_rate_init=0.2, max_iter=1,
                                    random_state=1, dropout=dropout)
