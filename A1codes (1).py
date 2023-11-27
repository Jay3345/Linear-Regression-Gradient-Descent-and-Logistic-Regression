import numpy as np
from cvxopt import matrix, solvers
from matplotlib import pyplot as plt
import math

########################
### HELPER FUNCTIONS ###((We did not write these))
########################
def _plotReg():

	# simple 2D example
	n = 30  # number of data points
	d = 1  # dimension
	noise = 0.2  # noise level
	X = np.random.randn(n, d)  # input matrix
	y = X + np.random.randn(n, 1) * noise + 2  # ground truth label

	plt.scatter(X, y, marker='x', color='k')  # plot data points

	# learning
	X = np.concatenate((np.ones((n, 1)), X),  axis=1)  # augment input
	w_L2 = minimizeL2(X, y)
	y_hat_L2 = X @ w_L2
	w_L1 = minimizeL1(X, y)
	y_hat_L1 = X @ w_L1
	w_Linf = minimizeLinf(X, y)
	y_hat_Linf = X @ w_Linf

	# plot models
	plt.plot(X[:, 1], y_hat_L2, 'b', marker=None, label='$L_2$')
	plt.plot(X[:, 1], y_hat_L1, 'r', marker=None, label='$L_1$')
	plt.plot(X[:, 1], y_hat_Linf, 'g', marker=None, label='$L_\infty$')
	plt.legend()
	plt.show()


def _plotCls():

	# 2D classification example
	m = 100
	d = 2
	c0 = np.array([[1, 1]])  # cls 0 center
	c1 = np.array([[-1, -1]])  # cls 1 center
	X0 = np.random.randn(m, 2) + c0
	X1 = np.random.randn(m, 2) + c1

	# plot data points
	plt.scatter(X0[:, 0], X0[:, 1], marker='x', label='Negative')
	plt.scatter(X1[:, 0], X1[:, 1], marker='o', label='Positive')

	X = np.concatenate((X0, X1), axis=0)
	X = np.concatenate((np.ones((2*m, 1)), X),  axis=1)  # augment input
	y = np.concatenate([np.zeros([m, 1]), np.ones([m, 1])], axis=0)  # class labels

	# gradient descent
	w_gd = gd(logisticRegObj, np.random.randn(d + 1, 1), X, y, 0.1, 1000, 1e-10)

	# plot models
	x_grid = np.arange(-4, 4, 0.01)
	plt.plot(x_grid, (-w_gd[0]-w_gd[1]*x_grid) / w_gd[2], '--k')
	plt.legend()
	plt.show()

########################
########################
########################

#########################
## My(And Alex's) Work ##
#########################


#Question 1a
def minimizeL2(X, y):
    product1 = np.dot(X.T, X)
    inverse = np.linalg.inv(product1)
    product2 = np.dot(inverse, X.T)
    return (np.dot(product2, y))



#Question 1b
def minimizeL1(X, y):
    dimensions = X.shape #2D Tuple showing dimensions of X as (rows(n), columns(d))
    n = dimensions[0]
    d = dimensions[1]
    negative_identity = np.negative(np.identity(n))

    # Constructing G
    G_1 = np.concatenate([np.zeros([n, d]), negative_identity], axis=1) # Attach them along x axis
    G_2 = np.concatenate([X, negative_identity], axis=1)
    G_3 = np.concatenate([np.negative(X), negative_identity], axis=1)
    G = np.concatenate([G_1, G_2, G_3], axis=0) # Attach them along y axis

    # Constructing h
    h_1 = np.zeros([n, 1])
    h_2 = y
    h_3 = np.negative(y)
    h = np.concatenate([h_1, h_2, h_3], axis=0)

    # Constructing c
    c_1 = np.zeros([d, 1])
    c_2 = np.ones([n, 1])
    c = np.concatenate([c_1, c_2], axis=0)

    # Solving for x
    c = matrix(c)
    G = matrix(G)
    h = matrix(h)
    solvers.options['show_progress'] = False
    sol = solvers.lp(c, G, h)

    return np.array(sol['x'])[0:d]



#Question 1c
def minimizeLinf(X, y):
    dimensions = X.shape #2D Tuple showing dimensions of X as (rows(n), columns(d))
    n = dimensions[0]
    d = dimensions[1]
    negative_ones = np.negative(np.ones([n, 1]))

    # Constructing G
    G_1 = np.concatenate([np.zeros([n, d]), negative_ones], axis=1) # Attach them along x axis
    G_2 = np.concatenate([X, negative_ones], axis=1)
    G_3 = np.concatenate([np.negative(X), negative_ones], axis=1)
    G = np.concatenate([G_1, G_2, G_3], axis=0) # Attach them along y axis

    # Constructing h
    h_1 = np.zeros([n, 1])
    h_2 = y
    h_3 = np.negative(y)
    h = np.concatenate([h_1, h_2, h_3], axis=0)

    # Constructing c
    c_1 = np.zeros([d, 1])
    c_2 = np.ones([1, 1])
    c = np.concatenate([c_1, c_2], axis=0)

    # Solving for x
    c = matrix(c)
    G = matrix(G)
    h = matrix(h)
    solvers.options['show_progress'] = False
    sol = solvers.lp(c, G, h)

    return np.array(sol['x'])[0:d]



# Question 1d
def synRegExperiments():
    def genData(n_points):
        '''
        This function generate synthetic data
        '''
        X = np.random.randn(n_points, d) # input matrix
        X = np.concatenate((np.ones((n_points, 1)), X), axis=1) # augment input
        y = X @ w_true + np.random.randn(n_points, 1) * noise # ground truth label
        return X, y

    n_runs = 100
    n_train = 30
    n_test = 1000
    d = 5
    noise = 0.2
    train_loss = np.zeros([n_runs, 3, 3]) # n_runs * n_models * n_metrics
    test_loss = np.zeros([n_runs, 3, 3]) # n_runs * n_models * n_metrics

    for r in range(n_runs):
        w_true = np.random.randn(d + 1, 1)
        Xtrain, ytrain = genData(n_train)
        Xtest, ytest = genData(n_test)

        # Learn different models from the training data
        w_L2 = minimizeL2(Xtrain, ytrain)
        w_L1 = minimizeL1(Xtrain, ytrain)
        w_Linf = minimizeLinf(Xtrain, ytrain)

        # Functions to calculate a specific loss type
        def lossL2(X, y, w):
            y_hat_L2 = X @ w
            diff_L2 = y_hat_L2 - y
            square_L2 = np.square(diff_L2)
            n = square_L2.shape[0] # The number of rows in square_L2
            loss_L2 = np.sum(square_L2)/(2*n)
            return loss_L2

        def lossL1(X, y, w):
            y_hat_L1 = X @ w
            diff_L1 = np.absolute(y_hat_L1 - y)
            n = diff_L1.shape[0] # The number of rows in diff_L1
            loss_L1 = np.sum(diff_L1)/n
            return loss_L1

        def lossLinf(X, y, w):
            final_vector = np.absolute((X @ w) - y)
            loss_Linf = np.max(final_vector)
            return loss_Linf


        # TODO: Evaluate the three models' performance (for each model,
        # calculate the L2, L1 and L infinity losses on the training
        # data). Save them to `train_loss`
        # Calculate losses
        loss_L2_train_M2 = lossL2(Xtrain, ytrain, w_L2)
        loss_L1_train_M2 = lossL1(Xtrain, ytrain, w_L2)
        loss_Linf_train_M2 = lossLinf(Xtrain, ytrain, w_L2)

        loss_L2_train_M1 = lossL2(Xtrain, ytrain, w_L1)
        loss_L1_train_M1 = lossL1(Xtrain, ytrain, w_L1)
        loss_Linf_train_M1 = lossLinf(Xtrain, ytrain, w_L1)

        loss_L2_train_Minf = lossL2(Xtrain, ytrain, w_Linf)
        loss_L1_train_Minf = lossL1(Xtrain, ytrain, w_Linf)
        loss_Linf_train_Minf = lossLinf(Xtrain, ytrain, w_Linf)

        # Record losses
        train_loss[r][0][0] = loss_L2_train_M2
        train_loss[r][0][1] = loss_L1_train_M2
        train_loss[r][0][2] = loss_Linf_train_M2

        train_loss[r][1][0] = loss_L2_train_M1
        train_loss[r][1][1] = loss_L1_train_M1
        train_loss[r][1][2] = loss_Linf_train_M1

        train_loss[r][2][0] = loss_L2_train_Minf
        train_loss[r][2][1] = loss_L1_train_Minf
        train_loss[r][2][2] = loss_Linf_train_Minf


        # TODO: Evaluate the three models' performance (for each model,
        # calculate the L2, L1 and L infinity losses on the test
        # data). Save them to `test_loss`
        # Calculate losses
        loss_L2_test_M2 = lossL2(Xtest, ytest, w_L2)
        loss_L1_test_M2 = lossL1(Xtest, ytest, w_L2)
        loss_Linf_test_M2 = lossLinf(Xtest, ytest, w_L2)

        loss_L2_test_M1 = lossL2(Xtest, ytest, w_L1)
        loss_L1_test_M1 = lossL1(Xtest, ytest, w_L1)
        loss_Linf_test_M1 = lossLinf(Xtest, ytest, w_L1)

        loss_L2_test_Minf = lossL2(Xtest, ytest, w_Linf)
        loss_L1_test_Minf = lossL1(Xtest, ytest, w_Linf)
        loss_Linf_test_Minf = lossLinf(Xtest, ytest, w_Linf)

        # Record losses
        test_loss[r][0][0] = loss_L2_test_M2
        test_loss[r][0][1] = loss_L1_test_M2
        test_loss[r][0][2] = loss_Linf_test_M2

        test_loss[r][1][0] = loss_L2_test_M1
        test_loss[r][1][1] = loss_L1_test_M1
        test_loss[r][1][2] = loss_Linf_test_M1

        test_loss[r][2][0] = loss_L2_test_Minf
        test_loss[r][2][1] = loss_L1_test_Minf
        test_loss[r][2][2] = loss_Linf_test_Minf

    # TODO: compute the average losses over runs
    # TODO: return a 3-by-3 training loss variable and a 3-by-3 test loss variable
    training_average = np.zeros([3, 3])
    test_average = np.zeros([3, 3])

    # Helper function that computes average losses over runs
    def find_average(X, loss_func):
        L2M2Average = 0
        L1M2Average = 0
        LinfM2Average = 0

        L2M1Average = 0
        L1M1Average = 0
        LinfM1Average = 0

        L2MinfAverage = 0
        L1MinfAverage = 0
        LinfMinfAverage = 0

        # Add up all the recorded losses from the train_loss matrix or test_loss matrix for every single loss for every single model
        for i in range(n_runs):
            L2M2Average += loss_func[i][0][0]
            L1M2Average += loss_func[i][0][1]
            LinfM2Average += loss_func[i][0][2]

            L2M1Average += loss_func[i][1][0]
            L1M1Average += loss_func[i][1][1]
            LinfM1Average += loss_func[i][1][2]

            L2MinfAverage += loss_func[i][2][0]
            L1MinfAverage += loss_func[i][2][1]
            LinfMinfAverage += loss_func[i][2][2]

        # Record the average of each kind of loss for each kind of model
        X[0][0] = L2M2Average/n_runs
        X[0][1] = L1M2Average/n_runs
        X[0][2] = LinfM2Average/n_runs

        X[1][0] = L2M1Average/n_runs
        X[1][1] = L1M1Average/n_runs
        X[1][2] = LinfM1Average/n_runs

        X[2][0] = L2MinfAverage/n_runs
        X[2][1] = L1MinfAverage/n_runs
        X[2][2] = LinfMinfAverage/n_runs

    find_average(training_average, train_loss)
    find_average(test_average, test_loss)

    return training_average, test_average



# Question 1f
def preprocessAutoMPG(dataset_folder):
    start = True
    datafile = open(dataset_folder, "r")
    lines = datafile.readlines()
    for line in lines:
        # Don't record entry if it has missing horsepower
        if(line[22] == '?'):
            continue

        # Extract elements of every entry
        currLine = line[:53]
        displacement = currLine[:4]
        mpg = currLine[7]
        cylinders = currLine[11:16]
        horsepower = currLine[22:27]
        weight = currLine[33:37]
        acceleration = currLine[44:48]
        model_year = currLine[51:53]

        # Record entry in array
        row = np.array([displacement, cylinders, horsepower, weight, acceleration, model_year])
        if(start):
            X = np.array([row])
            y = np.array([mpg])
            start = False
        else:
            X = np.vstack([X, row])
            y = np.vstack([y, mpg])

    datafile.close()
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    return X, y



# Question 1g
def runAutoMPG(dataset_folder):
    X, y = preprocessAutoMPG(dataset_folder)
    n, d = X.shape
    X = np.concatenate((np.ones((n, 1)), X), axis=1) # augment
    n_runs = 100
    train_loss = np.zeros([n_runs, 3, 3]) # n_runs * n_models * n_metrics
    test_loss = np.zeros([n_runs, 3, 3]) # n_runs * n_models * n_metrics

    for r in range(n_runs):
        # TODO: Randomly partition the dataset into two parts (50%
        # training and 50% test)
        np.random.shuffle(X)
        np.random.shuffle(y)
        Xsplit = np.array_split(X, 2)
        ysplit = np.array_split(y, 2)
        Xtrain = Xsplit[0]
        ytrain = ysplit[0]
        Xtest = Xsplit[1]
        ytest = ysplit[1]


        # TODO: Learn three different models from the training data
        # using L1, L2 and L infinity losses
        w_L2 = minimizeL2(Xtrain, ytrain)
        w_L1 = minimizeL1(Xtrain, ytrain)
        w_Linf = minimizeLinf(Xtrain, ytrain)

        # Functions to calculate a specific loss type
        def lossL2(X, y, w):
            y_hat_L2 = X @ w
            diff_L2 = y_hat_L2 - y
            square_L2 = np.square(diff_L2)
            n = square_L2.shape[0] # The number of rows in square_L2
            loss_L2 = np.sum(square_L2)/(2*n)
            return loss_L2

        def lossL1(X, y, w):
            y_hat_L1 = X @ w
            diff_L1 = np.absolute(y_hat_L1 - y)
            n = diff_L1.shape[0] # The number of rows in diff_L1
            loss_L1 = np.sum(diff_L1)/n
            return loss_L1

        def lossLinf(X, y, w):
            final_vector = np.absolute((X @ w) - y)
            loss_Linf = np.max(final_vector)
            return loss_Linf


        # TODO: Evaluate the three models' performance (for each model,
        # calculate the L2, L1 and L infinity losses on the training
        # data). Save them to `train_loss`
        # Calculate losses
        loss_L2_train_M2 = lossL2(Xtrain, ytrain, w_L2)
        loss_L1_train_M2 = lossL1(Xtrain, ytrain, w_L2)
        loss_Linf_train_M2 = lossLinf(Xtrain, ytrain, w_L2)

        loss_L2_train_M1 = lossL2(Xtrain, ytrain, w_L1)
        loss_L1_train_M1 = lossL1(Xtrain, ytrain, w_L1)
        loss_Linf_train_M1 = lossLinf(Xtrain, ytrain, w_L1)

        loss_L2_train_Minf = lossL2(Xtrain, ytrain, w_Linf)
        loss_L1_train_Minf = lossL1(Xtrain, ytrain, w_Linf)
        loss_Linf_train_Minf = lossLinf(Xtrain, ytrain, w_Linf)

        # Record losses
        train_loss[r][0][0] = loss_L2_train_M2
        train_loss[r][0][1] = loss_L1_train_M2
        train_loss[r][0][2] = loss_Linf_train_M2

        train_loss[r][1][0] = loss_L2_train_M1
        train_loss[r][1][1] = loss_L1_train_M1
        train_loss[r][1][2] = loss_Linf_train_M1

        train_loss[r][2][0] = loss_L2_train_Minf
        train_loss[r][2][1] = loss_L1_train_Minf
        train_loss[r][2][2] = loss_Linf_train_Minf


        # TODO: Evaluate the three models' performance (for each model,
        # calculate the L2, L1 and L infinity losses on the test
        # data). Save them to `test_loss`
        # Calculate losses
        loss_L2_test_M2 = lossL2(Xtest, ytest, w_L2)
        loss_L1_test_M2 = lossL1(Xtest, ytest, w_L2)
        loss_Linf_test_M2 = lossLinf(Xtest, ytest, w_L2)

        loss_L2_test_M1 = lossL2(Xtest, ytest, w_L1)
        loss_L1_test_M1 = lossL1(Xtest, ytest, w_L1)
        loss_Linf_test_M1 = lossLinf(Xtest, ytest, w_L1)

        loss_L2_test_Minf = lossL2(Xtest, ytest, w_Linf)
        loss_L1_test_Minf = lossL1(Xtest, ytest, w_Linf)
        loss_Linf_test_Minf = lossLinf(Xtest, ytest, w_Linf)

        # Record losses
        test_loss[r][0][0] = loss_L2_test_M2
        test_loss[r][0][1] = loss_L1_test_M2
        test_loss[r][0][2] = loss_Linf_test_M2

        test_loss[r][1][0] = loss_L2_test_M1
        test_loss[r][1][1] = loss_L1_test_M1
        test_loss[r][1][2] = loss_Linf_test_M1

        test_loss[r][2][0] = loss_L2_test_Minf
        test_loss[r][2][1] = loss_L1_test_Minf
        test_loss[r][2][2] = loss_Linf_test_Minf

    # TODO: compute the average losses over runs
    # TODO: return a 3-by-3 training loss variable and a 3-by-3 test loss variable
    training_average = np.zeros([3, 3])
    test_average = np.zeros([3, 3])

    # Helper function that computes average losses over runs
    def find_average(X, loss_func):
        L2M2Average = 0
        L1M2Average = 0
        LinfM2Average = 0

        L2M1Average = 0
        L1M1Average = 0
        LinfM1Average = 0

        L2MinfAverage = 0
        L1MinfAverage = 0
        LinfMinfAverage = 0

        # Add up all the recorded losses from the train_loss matrix or test_loss matrix for every single loss for every single model
        for i in range(n_runs):
            L2M2Average += loss_func[i][0][0]
            L1M2Average += loss_func[i][0][1]
            LinfM2Average += loss_func[i][0][2]

            L2M1Average += loss_func[i][1][0]
            L1M1Average += loss_func[i][1][1]
            LinfM1Average += loss_func[i][1][2]

            L2MinfAverage += loss_func[i][2][0]
            L1MinfAverage += loss_func[i][2][1]
            LinfMinfAverage += loss_func[i][2][2]

        # Record the average of each kind of loss for each kind of model
        X[0][0] = L2M2Average/n_runs
        X[0][1] = L1M2Average/n_runs
        X[0][2] = LinfM2Average/n_runs

        X[1][0] = L2M1Average/n_runs
        X[1][1] = L1M1Average/n_runs
        X[1][2] = LinfM1Average/n_runs

        X[2][0] = L2MinfAverage/n_runs
        X[2][1] = L1MinfAverage/n_runs
        X[2][2] = LinfMinfAverage/n_runs

    find_average(training_average, train_loss)
    find_average(test_average, test_loss)

    return training_average, test_average



# Question 2a
def linearRegL2Obj(w, X, y):
  XwY=np.dot(X, w)-y
  n=len(X)

  Norm = np.linalg.norm(XwY, 2)
  Norm *= Norm

  obj_val = (1/(2*n))*Norm

  grad= (1/n)*np.dot((X.T), (XwY))

  return obj_val, grad;



# Question 2b
def gd(obj_func, w_init, X, y, eta, max_iter, tol):
  w=w_init
  for i in range(max_iter):
    # TODO: Compute the gradient of the obj_func at current w
    obj_val, grad = obj_func(w, X, y)
    # TODO: Break if the L2 norm of the gradient is smaller than tol
    if np.linalg.norm(grad) < tol:
      break

    # TODO: Perform gradient descent update to w
    w=w.astype("float64")-(eta*grad)
  return w



# Question 2c
def logisticRegObj(w, X, y):
  #matrix multiplication X and w
  Xw= X @ w

  #n
  n=len(X)

  #JW as specified above using log rules
  Jw_part = (-y.T @ -np.logaddexp(0, -Xw)) - ((1-y).T @ -np.logaddexp(0, Xw))
  #getting the obj_val
  obj_val=(Jw_part)/n

  #Lets me pass the and retrieve the sigmoid of the matrix
  vectorizedSigmoid=np.vectorize(sigmoid)
  XwSigmoid=vectorizedSigmoid(Xw)
  #Getting the grad
  grad=(X.T/n) @ (XwSigmoid-y)

  return obj_val, grad

def sigmoid(Xw):
  return 1/(1+pow(math.e, -Xw))

# Question 2d
def synClsExperiments():
    def genData(n_points, d):
        '''
        This function generate synthetic data
        '''
        c0 = np.ones([1, d]) # class 0 center
        c1 = -np.ones([1, d]) # class 1 center
        X0 = np.random.randn(n_points, d) + c0 # class 0 input
        X1 = np.random.randn(n_points, d) + c1 # class 1 input
        X = np.concatenate((X0, X1), axis=0)
        X = np.concatenate((np.ones((2 * n_points, 1)), X), axis=1) # augmentation
        y = np.concatenate([np.zeros([n_points, 1]), np.ones([n_points, 1])], axis=0)
        return X, y


    def runClsExp(m=100, d=2, eta=0.1, max_iter=1000, tol=1e-10):
        '''
        Run classification experiment with the specified arguments
        '''
        Xtrain, ytrain = genData(m, d)
        n_test = 1000
        Xtest, ytest = genData(n_test, d)

        w_init = np.zeros([d + 1, 1])
        w_logit = gd(logisticRegObj, w_init, Xtrain, ytrain, eta, max_iter, tol)

        ytrain_hat=(Xtrain @ w_logit)>=0
        train_acc = np.mean(ytrain_hat == ytrain)

        ytest_hat=(Xtest @ w_logit)>=0
        test_acc = np.mean(ytest_hat == ytest)
        return train_acc, test_acc

    n_runs = 100
    train_acc = np.zeros([n_runs, 4, 3])
    test_acc = np.zeros([n_runs, 4, 3])
    for r in range(n_runs):
        for i, m in enumerate((10, 50, 100, 200)):
            train_acc[r, i, 0], test_acc[r, i, 0] = runClsExp(m=m)
        for i, d in enumerate((1, 2, 4, 8)):
            train_acc[r, i, 1], test_acc[r, i, 1] = runClsExp(d=d)
        for i, eta in enumerate((0.1, 1.0, 10., 100.)):
            train_acc[r, i, 2], test_acc[r, i, 2] = runClsExp(eta=eta)

    # TODO: compute the average accuracies over runs
    # TODO: return a 4-by-3 training accuracy variable and a 4-by-3 test accuracy variable
    avg_train_acc = np.mean(train_acc, axis=0)
    avg_test_acc = np.mean(test_acc, axis=0)
    return avg_train_acc, avg_test_acc


# Question 2f
def preprocessSonar(dataset_folder):
    start = True
    datafile = open(dataset_folder, "r")
    lines = datafile.readlines()

    for line in lines:
        line = line.strip()
        row = line.split(",")

        # Convert R to 0 and M to 1
        if(row[60] == 'R'):
            row[60] = 0
        else:
            row[60] = 1

        label = row.pop() # Also simultaneously removes label from row

        # Record current row that we've extraced and processed
        if(start):
            X = np.array([row])
            y = np.array([label])
            start = False
        else:
            X = np.vstack([X, row])
            y = np.vstack([y, label])

    datafile.close()

    # Changes entries of the matrix to be floats
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    return X, y



# Question 2g
def runSonar(dataset_folder):
    X, y = preprocessSonar(dataset_folder)
    n, d = X.shape
    X = np.concatenate((np.ones((n, 1)), X), axis=1) # augment

    eta_list = [0.1, 1, 10, 100]
    train_acc = np.zeros([len(eta_list)])
    val_acc = np.zeros([len(eta_list)])
    test_acc = np.zeros([len(eta_list)])

    # TODO: Randomly partition the dataset into three parts (40%
    # training (use the round function), 40% validation and
    # the remaining ~20% as test)

    # Define the proportions for the splits(Won't change here but is an option)
    split_training = 0.4  # 40%
    split_validation = 0.4  # 40%
    split_test = 0.2  # 20%

    # Calculate the sizes of each split
    training_size = int(n * split_training)
    validation_size = int(n * split_validation)
    test_size = n- training_size - validation_size
    # Generate random datapoints to shuffle the data
    shuffled_datapoints = np.arange(n)
    np.random.shuffle(shuffled_datapoints)

    # Split matrices X and Y into three partitions
    training_datapoints = shuffled_datapoints[:training_size ]
    validation_datapoints = shuffled_datapoints[training_size :(training_size  + validation_size)]
    test_datapoints = shuffled_datapoints[(training_size + validation_size):]

    Xtrain, ytrain = X[training_datapoints], y[training_datapoints]
    XValidation, yValidation = X[validation_datapoints], y[validation_datapoints]
    Xtest, ytest = X[test_datapoints], y[test_datapoints]


    for i, eta in enumerate(eta_list):
        w_init = np.zeros([d + 1, 1])
        w = gd(logisticRegObj, w_init, Xtrain, ytrain, eta, max_iter=1000, tol=1e-8)
        # Evaluate the model's accuracy on the training
        #data. Save it to `train_acc`
        ytrain_hat=(Xtrain @ w)>=0
        train_acc[i] = np.mean(ytrain_hat == ytrain)

        # Evaluate the model's accuracy on the validation
        # data. Save it to `val_acc`
        yValidation_hat=(XValidation @ w)>=0
        val_acc[i] = np.mean(yValidation_hat == yValidation)

        # Evaluate the model's accuracy on the test
        # data. Save it to `test_acc`
        ytest_hat=(Xtest @ w)>=0
        test_acc[i] = np.mean(ytest_hat == ytest)

    return train_acc, val_acc, test_acc

MPGlocation = "/content/auto-mpg.data.txt"
sonarLocation = "/content/sonar.all-data.txt"

def main():
    # Testing synRegExperiments()
    print("synRegExperiments: ")
    train_loss, test_loss = synRegExperiments()
    print(train_loss)
    print(test_loss)
    print("\n\n")

    # Testing preprocessAutoMPG
    X, y = preprocessAutoMPG(MPGlocation)
    #I am not showing this because it shows a 1xn matrix

    # Testing runAutoMPG
    print("runAutoMPG: ")
    train_loss, test_loss = runAutoMPG(MPGlocation)
    print(train_loss)
    print(test_loss)
    print("\n\n")

    # testing synClsExperiments
    print("synClsExperiments: ")
    X, y = synClsExperiments()
    print(X)
    print(y)
    print("\n\n")

    # testing preprocessSonar
    X, y = preprocessSonar(sonarLocation)
    #I am not showing this because it shows a 1xn matrix

    # testing runSonar
    print("runSonar: ")
    train_acc, val_acc, test_acc = runSonar(sonarLocation)
    print(train_acc)
    print(val_acc)
    print(test_acc)
    print("\n\n")

    print("Plot Linear Regression: ")
    _plotReg()
    print("Plot Gradient Descent & Logistic Regression: ")
    _plotCls()


if __name__ == "__main__":
    main()
