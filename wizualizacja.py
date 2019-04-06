import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


dane = 'ekstrakcja_przyklad_clean.pickle'
dane_table = pd.read_pickle(dane)

print(dane_table.head())
print(list(dane_table))


count = Counter(dane_table['klasa'])
print(count.most_common())
mck=count.keys()
mcv=count.values()

sum =0
for i in mcv:
    sum = sum + i
srednia = sum/(len(mcv))
# histogram = plt.figure(figsize = (6, 4))
# plt.bar(mck, mcv, orientation = 'vertical', color  = 'orange')
# plt.xticks(rotation=90, size = 8)
# plt.subplots_adjust(bottom = 0.28, top = 0.99)
# plt.axhline(y = srednia, color = 'r', ls = 'dashed')
# plt.show(histogram)

dane_x = dane_table.loc[16238, 'ekstrakcja']
dane_y = np.arange(1, 431, 1)
print(dane_x, dane_y)
plt.figure()
plt.plot(dane_y, dane_x)
plt.show()