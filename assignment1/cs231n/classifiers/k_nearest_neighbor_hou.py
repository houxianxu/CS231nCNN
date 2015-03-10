import numpy as np

class KNearestNeighbor(object):
    """ a KNN classifier with L2 distance """
    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just 
        memorizing the training data.

        Input:
        X - A num_train x dimension array where each row is a training point.
        y - A vector of length num_train, where y[i] is the label for X[i, :]
        """

        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict the lables for test data using classifier

        Input:
        X - A num_test x dimension array where each row is a test point.
        k - The number of nearest neighbors that vote for predicted label
        num_loops - Determines which method to use to compute distances
                between training points and test points.

        Output:
        y - A vector of length num_test, where y[i] is the predicted label for the
            test point X[i, :].
        """
        print '11'  
        num_test = X.shape[0] # number of test data
        num_train = self.X.shape[0] # number of train data

        # Compute the distance
        if num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 0:
            dists = self.compute_distances_no_loops(X)

        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the 
        test data.

        Input:
        X - An num_test x dimension array where each row is a test point.

        Output:
        dists - A num_test x num_train array where dists[i, j] is the distance
            between the ith test point and the jth training point.      
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        for i in xrange(num_test):
            for j in xrange(num_train):
                dists[i][j] = np.sum(np.square(self.X_train[i] - X[j]))

        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """

    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))

    for i in xrange(num_test):
        dists[i] = np.sum(np.square(self.X_train - X[i]), axis=1)    
    return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        # Tile the X_train, and repeat the test X
        tile_X_train = np.tile(self.X_train, (num_test, 1), axis=0) # [num_test*num_train, ~]
        repeat_X = np.repeat(X, num_train, axis=0) # [num_test*num_train, ~]

        dists_all = np.sum(np.square(tile_X_train - repeat_X), axis=1) # [num_test*num_train, ~]
        # reshape the matrix
        dists = np.reshape(dists_all, (2, 6))

        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Input:
        dists - A num_test x num_train array where dists[i, j] gives the distance
        between the ith test point and the jth training point.

        Output:
        y - A vector of length num_test where y[i] is the predicted label for the
        ith test point.
        """

        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)

        for i in xrange(num_test):
            closest_y = []  # store the index of the k nearest neighbor
            
            # use the np.argsort to get the index
            k_nearest_neighbor = np.argsort(dists[i], axis=1)[0:k]
            closest_y = self.y_train[k_nearest_neighbor].tolist()

            
            y_pred[i] = max(closest_y, key=closest_y.count)

            
















