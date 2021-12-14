import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """

        # initiate the weight of Perceptron using the given Parameter in nn class and pass dimension into it
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        # return the current weight
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        # calculate the score using the DotProduct method and return the score
        score = nn.DotProduct(x, self.get_weights())

        return score

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"

        # first initiate class result
        class_result = None

        # run the model
        score = self.run(x)

        # if the score is less than zero return -1 otherwise return 1
        if nn.as_scalar(score) < 0:
            class_result = -1

        else:
            class_result = 1

        return class_result


    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        # set converge status to false at begining
        is_converge = False

        # using while loop to train the model until the model is converge
        while is_converge != True:
            is_converge = True

            # using for loop to iterate through the dataset
            for x_train, y_train in dataset.iterate_once(1):
                scalar_score_y = nn.as_scalar(y_train)

                # if the prediction is not equal to the expect result, it means the model has not converge yet
                if self.get_prediction(x_train) != scalar_score_y:
                    nn.Parameter.update(self.w, x_train, scalar_score_y)
                    is_converge = False





class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        # initiate the batch size and learning rate
        self.batch_size = 0
        self.learning_rate = -0.02

        # initiate the layer size
        self.l1_size =50
        self.l2_size = 50
        self.l3_size = 1

        # initiate the weight and bias for each layer using given Parameter method
        self.w1 = nn.Parameter(1, self.l1_size)
        self.b1 = nn.Parameter(1, self.l1_size)
        self.w2 = nn.Parameter(self.l1_size, self.l2_size)
        self.b2 = nn.Parameter(1, self.l2_size)
        self.w3 = nn.Parameter(self.l2_size, self.l3_size)
        self.b3 = nn.Parameter(1, self.l3_size)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"

        # set up  the batch size
        self.batch_size = x.data.shape[0]

        # start to run  the model by first linear the x and weight, then add bias to it and then call given ReLU method execept the last layer
        l_1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        l_2 = nn.ReLU(nn.AddBias(nn.Linear(l_1, self.w2), self.b2))
        return nn.AddBias(nn.Linear(l_2 , self.w3), self.b3)


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        # using given squareloss function to calcuate the loss
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        # using while loop to train the model until the error is less than 0.02 as required
        while True:

            # using for loop to iterate through the dataset
            for x_train, y_train in dataset.iterate_once(self.batch_size):

                # calcuate the loss
                loss = self.get_loss(x_train, y_train)
                w_b_list = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

                # get gradient
                gradient = nn.gradients(loss, w_b_list)

                # using gradient to update the weight and bias
                self.w1.update(gradient[0], self.learning_rate)
                self.b1.update(gradient[1], self.learning_rate)
                self.w2.update(gradient[2], self.learning_rate)
                self.b2.update(gradient[3], self.learning_rate)
                self.w3.update(gradient[4], self.learning_rate)
                self.b3.update(gradient[5], self.learning_rate)

            # check the new loss after weight and bias update
            if nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))) < 0.02:
                return

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        # initiate the batch size learning rate and input layer dimension and output layer dimension
        self.batch_size = 0
        self.learning_rate = -0.008
        self.input_dimension = 784
        self.out_dimension = 10

        # set up the hidden layer dimension
        self.l1_size = 100
        self.l2_size = 50
        self.l3_size = self.out_dimension

        # based on the layer dimension to set up the weight and bias of each layer
        self.w1 = nn.Parameter(self.input_dimension, self.l1_size)
        self.b1 = nn.Parameter(1, self.l1_size)
        self.w2 = nn.Parameter(self.l1_size, self.l2_size)
        self.b2 = nn.Parameter(1, self.l2_size)
        self.w3 = nn.Parameter(self.l2_size, self.l3_size)
        self.b3 = nn.Parameter(1, self.l3_size)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        # run the model which is completely the same as the linaer regression class
        if self.batch_size == 0:
            self.batch_size = x.data.shape[0]

        l_1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        l_2 = nn.ReLU(nn.AddBias(nn.Linear(l_1, self.w2), self.b2))
        return nn.AddBias(nn.Linear(l_2, self.w3), self.b3)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        # calcuate the loss using given SoftmaxLoss method
        predicted_result = self.run(x)
        loss = nn.SoftmaxLoss(predicted_result, y)

        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        goal_state = False

        # train the dataset until the 97 percent accuracy rate is achieved
        while goal_state != True:

            # iterate through  the dataset
            for x_train, y_train in dataset.iterate_once(self.batch_size):

                # get the loss
                loss = self.get_loss(x_train, y_train)
                w_b_list = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

                # calcuate the gradient
                gradient = nn.gradients(loss, w_b_list)

                # using gradient to update the weight and bias for each layer
                self.w1.update(gradient[0], self.learning_rate)
                self.b1.update(gradient[1], self.learning_rate)
                self.w2.update(gradient[2], self.learning_rate)
                self.b2.update(gradient[3], self.learning_rate)
                self.w3.update(gradient[4], self.learning_rate)
                self.b3.update(gradient[5], self.learning_rate)

            accuracy_score = dataset.get_validation_accuracy()

            # adjust the learning rate after reach certain accuracy score
            if  accuracy_score >= 0.95 and accuracy_score < 0.96:
                self.learning_rate = -0.005

            if  accuracy_score >= 0.96:
                print(1)
                self.learning_rate = -0.002


            # if the required accuracy score is achieved, set the goal_state to True which can break the while loop
            if accuracy_score >= 0.971:
                goal_state = True

class DeepQModel(object):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        "*** YOUR CODE HERE ***"
        self.learning_rate = None
        self.numTrainingGames = None
        self.batch_size = None

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a node with shape (batch_size x state_dim)
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"

    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a node with shape (batch_size x state_dim)
        Output:
            result: a node with shape (batch_size x num_actions) containing Q-value
                scores for each of the actions
        """
        "*** YOUR CODE HERE ***"

    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a node with shape (batch_size x state_dim)
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        "*** YOUR CODE HERE ***"