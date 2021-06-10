import platform
import pandas as pd
import numpy as np

os_name = platform.system()

if os_name == 'Windows': # Windows
    wine_data = pd.read_csv('D:/OneDrive/22. Python/Wine/wine_dataset.csv')
elif os_name == 'Darwin': # MacOS
    wine_data = pd.read_csv('/Users/bbergamim/OneDrive/22. Python/Wine/wine_dataset.csv')

wine_data['style'] = wine_data['style'].replace('red', 0)
wine_data['style'] = wine_data['style'].replace('white', 1)

# Separate variables (predictors and target):
y = wine_data['style']
x = wine_data.drop('style', axis = 1)

from sklearn.model_selection import train_test_split

# Create try and test groups:
x_try, x_test, y_try, y_test = train_test_split(x, y, test_size = 0.3)

from sklearn.ensemble import ExtraTreesClassifier

# Create the model:
model = ExtraTreesClassifier()
model.fit(x_try, y_try)

# Accurace:
result = model.score(x_test, y_test)
result = "{:.2%}".format(result)
print("Accurace: ", result)

# Example:
n = []
original_ex = []
for n in y_test[300:305]:
    original_ex.append(n)

prediction = model.predict(x_test[300:305])
n = []
prediction_ex = []
for n in prediction:
    prediction_ex.append(n)

print('Example (original): ', original_ex)
print('Example (prediction): ', prediction_ex)