def reassign_labels(sf):
    sf['safe_loans'] = sf['bad_loans'].apply(lambda x: +1 if x == 0 else -1)
    return sf.remove_column('bad_loans')


def sub_samples(df, target, seed=1):
    safe_df_raw = df[df[target] == 1]
    risky_df_raw = df[df[target] == -1]
    ## Since there are less risky df than safe df, find the ratio of the sizes
    ## and use that percentage to undersample the safe df.
    perc = len(risky_df_raw) / len(safe_df_raw)
    safe_df = safe_df_raw.sample(perc, seed=seed)
    risky_df = risky_df_raw
    df_data = risky_df.append(safe_df)
    return (df_data, safe_df, risky_df)

def hot_encode(df, features):
    for f in features:
        df_ohe = df[f].apply(lambda x: {x: 1})
        df_unp = df_ohe.unpack(column_name_prefix=f)

        # Change None's to 0's
        for col in df_unp.column_names():
            df_unp[col] = df_unp[col].fillna(0)

        df = df.remove_column(f)
        df = df.add_columns(df_unp)
    return df

def inter_node_num_mistakes(labels_in_node):
    if len(labels_in_node) == 0: return 0      ## Corner case
    n_pos = (labels_in_node == 1).sum()        ## Count the number of 1's (safe loans)
    n_neg = len(labels_in_node)  - n_pos       ## or (labels_in_node == -1).sum()
    return n_neg if n_pos >= n_neg else n_pos  ## Num. of mistakes that the majority classifier makes.

def best_splitting_feature(data, features, target):
    best_feature = None
    best_error = 10
    ## Note: Since error is always <= 1, therefore init. with something > 1.

    num_data_points = len(data)
    for feature in features:
        l_split = data[data[feature] == 0]  ## The left split will have all data points where feature value is 0
        r_split = data[data[feature] == 1]  ## The right split "    "    "    "     "      "     "    value is 1

        l_mistakes = inter_node_num_mistakes(l_split[target]) ## Calc. num. of misclassified ex. in left split. (SArray)
        r_mistakes = inter_node_num_mistakes(r_split[target]) ## Calc. num. of misclassified ex. in right split.

        # Error = (# of mistakes (left) + # of mistakes (right)) / (# of data points)
        error = (l_mistakes + r_mistakes) / num_data_points   ## Compute the classification error of this split.

        if error < best_error:  ## update best...
          best_error = error
          best_feature = feature
    ##
    return best_feature

def create_leaf(target_values,
                splitting_feature=None, left=None, right=None, is_leaf=True):
    ## Create a leaf node
    leaf = {
      'prediction': None,
      'splitting_feature' : splitting_feature,
      'left' : left,
      'right' : right,
      'is_leaf': is_leaf,
    }

    if is_leaf:
      ## Count the number of data points that are +1 and -1 in this node.
      num_1s = len(target_values[target_values == +1])
      num_minus_1s = len(target_values[target_values == -1])

      ## For the leaf node, set the prediction to be the majority class.
      ## Store the predicted class (1 or -1) in leaf['prediction']
      leaf['prediction'] = 1 if num_1s > num_minus_1s else -1
    ##
    return leaf

def count_nodes(tree):
    return 1 if tree is None or tree['is_leaf'] else \
        1 + count_nodes(tree['left']) + count_nodes(tree['right'])

def count_leaves(tree):
    return 1 if tree is None or tree['is_leaf'] else \
        count_leaves(tree['left']) + count_leaves(tree['right'])

def classify(tree, x, annotate=False):
    if tree['is_leaf']:  ## if the node is a leaf node.
      if annotate:
        print("At leaf, predicting %s" % tree['prediction'])
      return tree['prediction']
    else:
      ## split on feature.
      split_feature_val = x[tree['splitting_feature']]
      if annotate:
        print("Split on %s = %s" % (tree['splitting_feature'], split_feature_val))
      return classify(tree['left'], x, annotate) if split_feature_val == 0 else \
          classify(tree['right'], x, annotate)
