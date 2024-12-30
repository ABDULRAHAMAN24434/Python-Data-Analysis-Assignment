from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()

# Create a DataFrame
try:
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    df['species'] = df['target'].map(species_map)
except Exception as e:
    print(f"An error occurred: {e}")


pd.set_option("display.max_columns", None)



print(df.head())
print(df.tail())
print(df.describe())


print(df.isna())
print(df.isnull())
df.dropna

# Plotting sepal length (cm) vs sepal width (cm) 
plt.plot(df["sepal length (cm)"], df["sepal width (cm)"])
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.title("sepal length (cm) vs sepal width (cm)")
plt.show()

# Plotting a bar chart for species by petal length (cm)
df.groupby("species")["petal length (cm)"].mean().plot(kind='bar')
plt.xlabel("species")
plt.ylabel("petal length (cm)")
plt.title("species by petal length (cm)")
plt.show()

# Plotting a histogram for petal length cm
df['petal length (cm)'].plot(kind='hist', bins=10)
plt.xlabel("petal length (cm)")
plt.title("petal length (cm) distribution")
plt.show()

# Plotting a scatter plot of sepal length vs petal length.
df.plot(kind="scatter", x="sepal length (cm)", y="petal length (cm)")
plt.title("sepal length vs petal length")
plt.show()

