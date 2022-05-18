"""
ID3 Decision Tree contructor
Takes in a csv file with the training data.

Carla Pérez-Gavilán
Jorge Palacios Hristova
Marcela Fuentes Lecuona
Yann Le Lorier Bárcena
"""
import pandas as pd
import math
import pprint 

def total_entropy(ds, label):
    """
    Function to calculate total (initial) entropy of a system
    """
    total_entropy = 0 
    total = ds.shape[0]
    target_var = ds[label]
    unique_target_var = list(set(target_var))
    for var in target_var:
        var_count = ds[ds[label] == var].shape[0]
        var_entropy = - (var_count/total)*math.log2(var_count/total) 
        total_entropy += var_entropy
    return total_entropy

def entropy_feature(ds_feature, label):
    """
    Function to calculate entropy for a single attribute
    """
    all_var = ds_feature[label]
    total = ds_feature.shape[0]
    result = 0 
    for v in all_var:
        var_count = ds_feature[ds_feature[label]== v].shape[0]
        entropy_var = 0
        if var_count != 0:
            entropy_var = - (var_count/total)*math.log2(var_count/total) 
        result += entropy_var
    return result

def info_gain(ds, feature_name, label):
    """
    Gain information obtained from the total entropy and the entropy for a single attribute

    """
    info_gain = 0 
    unique_features = ds[feature_name].unique()
    class_data = ds[label].unique()
    total = ds.shape[0]
    for f in unique_features:
        values_feature = ds[ds[feature_name] == f]
        num_features = values_feature.shape[0]
        info_gain +=  (num_features / total) * entropy_feature(values_feature, label)
    return total_entropy(ds, label) - info_gain

def max_info_gain(ds, label):
    """
    Getting the maximum gain from different attibutes to choose the one that give the most information
    """
    feature_list = ds.columns.drop(label)
    max = -1 
    max_feature = None
    for f in feature_list:
        temp_info_gain = info_gain(ds, f, label)
        if max < temp_info_gain:
            max = temp_info_gain
            max_feature = f
    return max_feature



def generate_child(feature_name, ds, label):
    """
    The function that generates a child after an attribute
    """
    dict_value_count =  ds[feature_name].value_counts(sort=False)
    subtree = {}
    class_data = ds[label].unique()
    for v, c in dict_value_count.iteritems():
        data = ds[ds[feature_name] == v] 
        assigned_flag = False
        for cl in class_data: 
            count = data[data[label]== cl].shape[0]
            if count == c: 
                subtree[v] = cl
                ds = ds[ds[feature_name] != v]
                assigned_flag = True
        if not assigned_flag: 
            subtree[v] = "unassigned"
    return subtree, ds

def construct_tree(root, prev, ds, label):
    """
    The actual construction of the tree
    """
    class_data = ds[label].unique()
    if ds.shape[0] != 0:
        max_info = max_info_gain(ds, label)
        tree, ds = generate_child(max_info, ds, label)
        next_ = None

        if prev != None: 
            root[prev] = dict()
            root[prev][max_info] = tree
            next_ =  root[prev][max_info]
        else: 
            root[max_info] = tree
            next_ = root[max_info]

        for node, branch in list(next_.items()):
            if branch == "unassigned":
                new_data = ds[ds[max_info] == node]
                construct_tree(next_, node, new_data, label)

def id3(ds, label):
    """
    Main function call
    """
    ds_copy = ds.copy()
    tree = {}
    construct_tree(tree, None, ds_copy, label)
    return tree

def print_pretty(d, indent=0):
    """
    Prints the resulting decision tree in a human-readable way
    """
    for key, value in d.items():
        print('  ' * indent + str(key))
        if isinstance(value, dict):
            print_pretty(value, indent+1)
        else:
            print('  ' * (indent+1) + str(value))

if __name__ == "__main__":
    #tenis data
    df = pd.read_csv("tenis.csv")

    #restaurant data
    #df = pd.read_csv("./../weka/weka-3-8-6-azul-zulu-linux/weka-3-8-6/data/restaurant-data.arff.csv")

    # for tenis data:
    tree = id3(df, 'play')
    #for restaurant data:
    #tree = id3(df, 'class')

    print_pretty(tree)
