import numpy as np
import pandas as pd

# Load data
data = pd.read_csv('enjoysport.csv')

# Split data into concepts and target
concepts = np.array(data.iloc[:, :-1])
target = np.array(data.iloc[:, -1])

def learn(concepts, target):
    # Initialize specific and general hypotheses
    specific_h = concepts[0].copy()
    general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]

    

    # Iterate over concepts and target
    for i, (concept, label) in enumerate(zip(concepts, target)):
        if label == "yes":
            # Update specific hypothesis
            for x in range(len(specific_h)):
                if concept[x] != specific_h[x]:
                    specific_h[x] = '?'
           
        elif label == "no":
            # Update general hypothesis
            for x in range(len(specific_h)):
                if concept[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'
          


    # Remove redundant general hypotheses
    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]
    for i in indices:
        general_h.remove(['?', '?', '?', '?', '?', '?'])

    return specific_h, general_h

s_final, g_final = learn(concepts, target)

print("Final Specific_h:")
print(s_final)
print("Final General_h:")
print(g_final)