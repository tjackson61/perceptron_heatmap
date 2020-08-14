import codecademylib3_seaborn
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

labels = [0,0,0,1]
data = [[0,0],
        [0,1],
        [1,0],
        [1,1]]



plt.scatter([point[0] for point in data],labels, c=labels)


classifier = Perceptron(max_iter=40)
classifier.fit(data,labels)
#Creates 100 evenly spaced points for x and y
#list of 100 points in 1 list for x
#list of 100 points in 1 list for y
x_values = np.linspace(0, 1, 100)
y_values = np.linspace(0, 1, 100)

#Saves x and y points in a combined list of 10,000 points
#Every combination of x value with y value
#[(x,y),(x,y)...(x,y)] | [(1,1),(1,2)...(1,100)] ...
point_grid = list(product(x_values, y_values))

#Calls .decision_function on the 10,000 point (x,y) list showing how far each point is from decision boundary closer to 0 means closer to boundary.
distances = classifier.decision_function(point_grid)
#Makes distance absolute value
abs_distances = [abs(dis) for dis in distances]
#Reshapes the 10,000 distances into a list of 100 lists that are 100 elements long
#i.e distances_matrix[0] has len() = 100
distances_matrix = np.reshape(abs_distances, (100,100))

heatmap = plt.pcolormesh(x_values,y_values,distances_matrix)
plt.colorbar(heatmap)
plt.show()
