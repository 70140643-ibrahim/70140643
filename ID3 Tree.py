
import pandas as pd
import math
from collections import Counter

# Dataset inside the same Python file
data = {
    'Outlook': ['Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast',
                'Sunny','Sunny','Rain','Sunny','Overcast','Overcast','Rain'],
    'Temperature': ['Hot','Hot','Hot','Mild','Cool','Cool','Mild',
                    'Cool','Mild','Mild','Mild','Mild','Hot','Mild'],
    'Humidity': ['High','High','High','High','Normal','Normal','Normal',
                 'High','Normal','Normal','Normal','High','Normal','High'],
    'Wind': ['Weak','Strong','Weak','Weak','Weak','Strong','Strong',
             'Weak','Weak','Weak','Strong','Strong','Weak','Strong'],
    'PlayTennis': ['No','No','Yes','Yes','Yes','No','Yes',
                    'No','Yes','Yes','Yes','Yes','Yes','No']
}

df = pd.DataFrame(data)

def entropy(data):
    labels = data.iloc[:, -1]
    counts = Counter(labels)
    total = len(labels)
    ent = 0
    for count in counts.values():
        p = count / total
        ent -= p * math.log2(p)
    return ent

def information_gain(data, attribute):
    total_entropy = entropy(data)
    values = data[attribute].unique()
    weighted_entropy = 0
    for value in values:
        subset = data[data[attribute] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset)
    return total_entropy - weighted_entropy

def id3(data, attributes):
    labels = data.iloc[:, -1]
    if len(set(labels)) == 1:
        return labels.iloc[0]
    if len(attributes) == 0:
        return labels.mode()[0]
    gains = {attr: information_gain(data, attr) for attr in attributes}
    best_attr = max(gains, key=gains.get)
    tree = {best_attr: {}}
    for value in data[best_attr].unique():
        subset = data[data[best_attr] == value]
        remaining_attrs = [attr for attr in attributes if attr != best_attr]
        tree[best_attr][value] = id3(subset, remaining_attrs)
    return tree

def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree
    attr = next(iter(tree))
    value = sample[attr]
    return predict(tree[attr][value], sample)

attributes = list(df.columns[:-1])
decision_tree = id3(df, attributes)

print("Decision Tree:")
print(decision_tree)

sample = {
    'Outlook': 'Sunny',
    'Temperature': 'Cool',
    'Humidity': 'High',
    'Wind': 'Strong'
}

print("\nPrediction for sample:")
print(predict(decision_tree, sample))
