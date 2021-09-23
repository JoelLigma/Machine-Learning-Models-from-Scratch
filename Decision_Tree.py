"""
Decision Tree Classifier from scratch
-------------------------------------

- Supports Gini Impurity and Information Gain 
- Supports both categorical and numerical features as input
- Stopping criteria: max_depth, min_samples_leaf, min_IG_or_GG, n_classes < 2
"""

import numpy as np
import pandas as pd

## TO DO
# code feature importance retrieval


class my_DecisionTreeClassifier:
    """
    Parameters:
    ----------
    criterion: str ("gini" and "entropy" are supported) - To assess the quality of the splits.
    max_depth: int (default=100) - The maximum depth of the tree.
    min_samples_leaf: int (default=1) - The minimum number of examples allowed in each leaf node.
    min_IG_or_GG: float (default=0) - The minimum Information Gain or Gini Gain allowed to continue splitting.
    """
    def __init__(self, criterion="gini", max_depth=100, min_samples_leaf=1, min_IG_or_GG=0):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_IG_or_GG = min_IG_or_GG

    def fit(self, X, y):
        """
        Parameters:
        ----------
        X: pd.DataFrame - The input data matrix
        y: pd.Series - The input label vector

        Returns:
        -------
        Self: Fitted classifier object
        """
        self.tree = self.build_tree(X, y)
        return self

    def build_tree(self, X, y, depth=0):
        label_list = list(y)
        #print("label list length:", len(label_list))
       # print("n_classes:", len(set(label_list)))
        classes = list(set(y))
        counter = 0
        for i in range(y.shape[0]):
            if label_list[i] == classes[0]:
                counter += 1
        try:
            d = {f"{classes[0]}": counter, f"{classes[1]}": y.shape[0] - counter}
            output_label = max(d, key=d.get)
        except IndexError:
            output_label = classes[0]

        node = Node(output_label=output_label)

        # recursive split
        if (depth < self.max_depth) and (len(classes) > 1) and (y.shape[0] > self.min_samples_leaf):
            best_split_dictionary = self.find_best_split(X, y)
            
            if best_split_dictionary:
                node.feature = best_split_dictionary["column"]
                node.splitting_value = best_split_dictionary["splitting_value"]
                node.categorical = best_split_dictionary["categorical"]
                node.criterion = best_split_dictionary["criterion"]
                # split
                node.left = self.build_tree(X=best_split_dictionary["data"][0].iloc[:,:-1], y=best_split_dictionary["data"][0].iloc[:,-1], depth=depth+1)
                node.right = self.build_tree(X=best_split_dictionary["data"][1].iloc[:,:-1], y=best_split_dictionary["data"][1].iloc[:,-1], depth=depth+1)
            
        return node


    def find_best_split(self, X, y):

        # check if X and y are pandas dataframe and series as well as of equal length
        if not isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
            raise TypeError("X must be of type pd.DataFrame and y of type pd.Series")
        elif X.shape[0] != y.shape[0]:
            raise ValueError("X and y must be of equal length")
        else:    
            # get parent node impurity
            if self.criterion == "entropy":
                current_entropy = self.compute_entropy(y)
            elif self.criterion == "gini":
                current_impurity = self.gini_impurity(y)
            else:
                raise ValueError("Only entropy or gini are valid criteria")

            # combine current subset
            current_subset = pd.concat([X, y], axis=1).copy()

            # keep track of best overall candidate_split in order to use it to create the child nodes
            best_split = {"column": None, "splitting_value": None, "categorical": None, "criterion": 0, "data": None}
            
            # start search for best split
            for i in range(X.shape[1]):
                # order values and get unique entries only
                unique_values = sorted(list(set(X.iloc[:, i])))

                for j in range(len(unique_values)):  

                    if len(unique_values) <= 20: # it is categorical
                        categorical = True
                        value = unique_values[j]
                        # create candidate split (True splits left and False splits right)
                        left = current_subset[current_subset.iloc[:, i] == value].copy()
                        right = current_subset[current_subset.iloc[:, i] != value].copy()

                    else: # it is numerical
                        categorical = False
                        value = sum(unique_values[j:j+2])/2
                        # create candidate split (True splits left and False splits right)
                        left = current_subset[current_subset.iloc[:, i] <= value].copy()
                        right = current_subset[current_subset.iloc[:, i] > value].copy()

                    # get child node impurities (-1 to access label column)
                    if self.criterion == "entropy":
                        left_node_impurity = self.compute_entropy(left.iloc[:,-1])
                        right_node_impurity = self.compute_entropy(right.iloc[:,-1]) 
                        # aggregate both (weighted) entropies to compute IG                    
                        new_entropy = left.shape[0] / current_subset.shape[0] * left_node_impurity \
                                    + right.shape[0] / current_subset.shape[0] * right_node_impurity   
                        # compute Information Gain
                        information_gain = current_entropy - new_entropy

                        if information_gain > best_split["criterion"]:
                            best_split = {"column": i, "splitting_value": value, "categorical": categorical, "criterion": information_gain, "data": (left, right)}

                    else: # it is gini
                        left_node_probability = self.gini_impurity(left.iloc[:,-1])
                        right_node_probability = self.gini_impurity(right.iloc[:,-1]) 
                        # aggregate both weighted gini impurities
                        weighted_gini_impurity = left.shape[0] / current_subset.shape[0] * left_node_probability \
                                               + right.shape[0] / current_subset.shape[0] * right_node_probability   
                        # compute gini gain
                        gini_gain = current_impurity - weighted_gini_impurity

                        if gini_gain > best_split["criterion"]:
                            best_split = {"column": i, "splitting_value": value, "categorical": categorical, "criterion": gini_gain, "data": (left, right)}
                        
                        
        # stopping criteria 
        if best_split["criterion"] <= self.min_IG_or_GG:
            return None          

        #if best_split["categorical"] == True:
        #    print(f'X{best_split["column"]} == {best_split["splitting_value"]} {self.criterion}={best_split["criterion"]}')
        #else:
        #    print(f'X{best_split["column"]} <= {best_split["splitting_value"]} {self.criterion}={best_split["criterion"]}')
               
        return best_split
        
    def compute_entropy(self, label_col):

        labels = np.unique(label_col, return_counts=True)
        entropy = 0
        for i in range(len(labels[0])):
            probability = labels[1][i]/len(label_col)
            entropy += -probability * np.log2(probability)
        return entropy

    def gini_impurity(self, label_col):

        labels = np.unique(label_col, return_counts=True)
        gini = 0
        for i in range(len(labels[0])):
            probability = labels[1][i] / labels[1].sum()
            gini += probability**2
        return 1 - gini

    def predict(self, X):
        output_list = []
        for i in range(X.shape[0]):
            output_list += [self.predict_per_row(row=X.iloc[i:i+1,:])]
        return output_list

    def predict_per_row(self, row):

        node = self.tree

        while node.left:
            # handle categorical vs. numerical
            if node.categorical: 
                if list(row.iloc[:, node.feature])[0] == node.splitting_value:
                    # True goes left
                    node = node.left
                else: 
                    # False goes right
                    node = node.right
            else:
                if list(row.iloc[:, node.feature])[0] <= node.splitting_value:
                    # True goes left
                    node = node.left
                else: 
                    # False goes right
                    node = node.right

        return node.output_label

    def get_params(self):
        return {
            "criterion": self.criterion,
            "max_depth" : self.max_depth,
            "min_samples_leaf" : self.min_samples_leaf,
            "min_IG" : self.min_IG_or_GG
        } 

    def feature_importance(self):
        return self.tree

class Node:
    def __init__(self, output_label):
        self.output_label = output_label
        self.feature = None
        self.splitting_value = None
        self.categorical = None
        self.criterion = None
        self.left = None
        self.right = None
        self.rule = None
        self.type = None
        self.samples = None


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

    # test model
    df = pd.read_csv("C:/users/joell/Desktop/Old Projects/citrus.csv")
    X = df.iloc[:,1:]
    X = pd.get_dummies(X, drop_first=True) # to avoid dummy trap
    y = df.name.copy()
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # fit classifier
    y_hat = my_DecisionTreeClassifier().fit(X_train, y_train).predict(X_test)
    print("My Decision Tree:", accuracy_score(y_true=y_test, y_pred=y_hat))

    # SKLEARN BASELINE
    sklearn_y_hat = DecisionTreeClassifier(random_state=0).fit(X_train, y_train).predict(X_test)
    print("sklearn Decision Tree:", accuracy_score(y_true=y_test, y_pred=sklearn_y_hat))