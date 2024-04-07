import matplotlib.pyplot as plt
import numpy as np

quad = []
expo = []
quad2 = []

for x in np.arange(1,5,0.1):
    quad.append(-0.41726 + 1.1792*x + 0.16915 * x**2)
    expo.append(x**(64/63))
    quad2.append(-0.13436 + 0.80541 * x + 0.28833 * x**2)

plt.plot(np.arange(1,5,0.1), quad, label="-0.41726 + 1.1792*x + 0.16915 * x**2")
plt.plot(np.arange(1,5,0.1), quad2, label="-0.13436 + 0.80541 * x + 0.28833 * x**2")
plt.plot(np.arange(1,5,0.1), expo, label="x**(64/63)")
plt.legend()
plt.show()
