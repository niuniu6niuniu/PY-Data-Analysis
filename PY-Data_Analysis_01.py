import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("countries.csv")
us = data[data.country == "United States"]
ch = data[data.country == "China"]
ind = data[data.country == "India"]
plt.plot(us.year, us.population / us.population.iloc[0] * 100, 'b')
plt.plot(ch.year, ch.population / ch.population.iloc[0] * 100, 'g')
plt.plot(ind.year, ind.population / ind.population.iloc[0] * 100, 'r')
plt.xlabel("year")
plt.ylabel("population")
plt.legend(["United States", "China", "India"])
plt.show()
