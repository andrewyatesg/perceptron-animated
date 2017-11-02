__author__ = 'ayates'

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import animation

class Perceptron():
    def __init__(self, training_set=None, testing_set=None, weights=None):

        self.training_set = training_set
        self.testing_set = testing_set

        fig, ax = plt.subplots()

        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))

        self.fig = fig
        self.ax = ax

        if weights is None:
            self.weights = np.array([[0., 0., 0.]]).T
        else:
            self.weights = weights

        # points that define the target function
        self.point1 = (random.uniform(-1, 1), random.uniform(-1, 1))
        self.point2 = (random.uniform(-1, 1), random.uniform(-1, 1))

    def target(self, feature):
        x, y = feature[1], feature[2]
        x1, y1 = self.point1
        x2, y2 = self.point2
        slope = (y2 - y1) / (x2 - x1)
        # simple check to see if point (x, y) is above or below the line
        return 1 if y > (slope * (x - x1) + y1) else -1

    def hypothesis(self, feature):
        return np.sign(np.dot(feature, self.weights))

    def test(self):
        mismatches = 0

        for feature in self.testing_set:
            if self.hypothesis(feature) != self.target(feature):
                mismatches += 1

        return mismatches / float(len(self.testing_set))

    def graph(self):

        xs = np.array([i[1] for i in self.training_set])
        ys = np.array([i[2] for i in self.training_set])

        # Graph target function
        x1, y1 = self.point1
        x2, y2 = self.point2
        slope = (y2 - y1) / (x2 - x1)
        self.ax.plot([np.amin(xs), np.amax(xs)], slope * (np.array([np.amin(xs), np.amax(xs)]) - x1) + y1, "g")

        # Graph the testing set as a scatterplot
        self.ax.scatter(xs, ys)

    def animate(self):
        # training data
        x = np.array([i[1] for i in self.training_set])

        minX = np.amin(x)
        maxX = np.amax(x)

        line, = self.ax.plot([], [], lw=2)

        self.test_data = True # Flag for printing test error, after finished animating

        def init():
            line.set_data([], [])
            return line,

        '''
        Called each frame. Runs one 'step' of the algorithm every frame.
        '''
        def animate(i):
            misclassified = []

            # Loops through each point in the training set and finds misclassified points
            for feature in self.training_set:
                intended = self.target(feature)

                if self.hypothesis(feature) != intended:
                    misclassified += [(feature, intended)]

            # pick a random misclassified point
            # and adjust the perceptron in its direction
            if misclassified:
                feature, intended = random.choice(misclassified)
                adapt = np.array([feature]).T * intended
                self.weights += adapt
            else:
                # We're done. Print error over test data.
                if self.test_data:
                    print("Error: " + str(np.around(self.test() * 100, decimals=3)) + "%")
                    self.test_data = False

            # Updates line based on updated weights
            lineDom = np.linspace(minX, maxX)
            line.set_data(lineDom, (-self.weights[1] * lineDom - self.weights[0]) / self.weights[2])

            return line,

        return init, animate


def test_run(data_size, testing_size):
    training_size = data_size

    training_set = [[1., random.uniform(-1, 1), random.uniform(-1, 1)]
                    for i in range(training_size)]
    testing_set = [[1., random.uniform(-1, 1), random.uniform(-1, 1)]
                   for i in range(testing_size)]

    pla = Perceptron(training_set=training_set, testing_set=testing_set)

    pla.graph()

    init, animate = pla.animate()
    anim = animation.FuncAnimation(pla.fig, animate, init_func=init,
                               frames=100000000000000000, interval=100, blit=True)

    plt.show()

test_run(50, 200)