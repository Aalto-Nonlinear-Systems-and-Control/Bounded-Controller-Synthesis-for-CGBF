import numpy as np
import matplotlib.pyplot as plt

s = 1
# Define grid
x = np.linspace(-4.2/s, 4.2/s, 1000)
y = np.linspace(-4.2/s, 4.2/s, 1000)
X, Y = np.meshgrid(x, y)

R=3.5
a=2.0
b=1.5

# Define the ring-shaped region
# banana = (X**2 + Y**2 - R**2)**2 + b*X - a*Y**3
banana = a*Y**3 - b*X - (X**2 + Y**2 - R**2)**2

# banana2 = a*( - s*y)**2 - b*(s*X) - ((s*X)**2 + (s*Y)**2 - R**2)**2
banana3 = a * (R - Y)**2 - b * X - (X**4 + Y**4 - R**2)**2
banana_masked = np.ma.masked_where(banana3 < 0, banana3)


ellipse = ((X + 0.5)**2 / 1**2) + ((Y - (2.1-4.0))**2 / 0.5**2) - 1

# Plot the banana shape
# plt.contourf(X, Y, banana, levels=[-np.inf, 0], colors="gold")
contour = plt.contourf(X, Y, banana_masked, levels=20, cmap="plasma")
plt.contourf(X, Y, ellipse, levels=[-np.inf, 0], colors="skyblue", alpha=0.7)
plt.colorbar(contour)
plt.gca().set_aspect("equal")  # Keep aspect ratio correct]
plt.grid()
plt.title("Ring-shaped Region Using a Single Equation")
plt.show()