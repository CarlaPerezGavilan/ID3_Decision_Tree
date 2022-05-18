from enum import unique
import pandas as pd
import math

from pip import main

def total_entropy(ds, label):
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
    class_data = ds[label].unique()
    if ds.shape[0] != 0:
        max_info = max_info_gain(ds, label)
        tree, ds = generate_child(max_info, ds, label)
        next = None

        if prev != None: 
            root[prev] = dict()
            root[prev][max_info] = tree
            next =  root[prev][max_info]
        else: 
            root[max_info] = tree
            next = root[max_info]

        for node, branch in list(next.items()):
            if branch == "unassigned":
                new_data = ds[ds[max_info] == node]
                construct_tree(next, node, new_data, label)

def id3(ds, label):
    ds_copy = ds.copy()
    tree = {}
    construct_tree(tree, None, ds_copy, label)
    return tree

def print_tree(tree):
    str_tree = str(tree)
    print(tree)
    final = ""
    sangria_counter = 0 
    for char in str_tree:
        if char == '{' :
            sangria_counter += 1
            final =  final + (' '*sangria_counter)
        elif char == '}':
            sangria_counter -= 1
            final =  final + '\n' + (' '*sangria_counter)
        elif char == ":" or char == ",": 
            final =  final + '\n' + (' '*sangria_counter)
        else:
            final += char
    print(final)

if __name__ == "__main__":
    df = pd.read_csv("tenis.csv")
    tree = id3(df, 'play')
    print_tree(tree)