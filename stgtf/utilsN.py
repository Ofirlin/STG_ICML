import numpy as np

# Dataset class: adapted from TensorFlow source example:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.
    # Returns
        A tensor.
    """
    return tf.convert_to_tensor(x, dtype=dtype)
class DataSet:
    """Base data set class
    """

    def __init__(self, shuffle=True, labeled=True, **data_dict):
        assert '_data' in data_dict
        if labeled:
            assert '_labels' in data_dict
            assert data_dict['_data'].shape[0] == data_dict['_labels'].shape[0]
        self._labeled = labeled
        self._shuffle = shuffle
        self.__dict__.update(data_dict)
        self._num_samples = self._data.shape[0]
        self._index_in_epoch = 0
        if self._shuffle:
            self._shuffle_data()

    def __len__(self):
        return len(self._data)

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def labeled(self):
        return self._labeled

    @property
    def valid_data(self):
        return self._valid_data

    @property
    def valid_labels(self):
        return self._valid_labels

    @property
    def test_data(self):
        return self._test_data

    @property
    def test_labels(self):
        return self._test_labels

    @classmethod
    def load(cls, filename):
        data_dict = np.load(filename)
        return cls(**data_dict)

    def save(self, filename):
        data_dict = self.__dict__
        np.savez_compressed(filename, **data_dict)

    def _shuffle_data(self):
        shuffled_idx = np.arange(self._num_samples)
        np.random.shuffle(shuffled_idx)
        self._data = self._data[shuffled_idx]
        if self._labeled:
            self._labels = self._labels[shuffled_idx]

    def next_batch(self, batch_size):
        assert batch_size <= self._num_samples
        start = self._index_in_epoch
        if start + batch_size > self._num_samples:
            data_batch = self._data[start:]
            if self._labeled:
                labels_batch = self._labels[start:]
            remaining = batch_size - (self._num_samples - start)
            if self._shuffle:
                self._shuffle_data()
            start = 0
            data_batch = np.concatenate([data_batch, self._data[:remaining]],
                                        axis=0)
            if self._labeled:
                labels_batch = np.concatenate([labels_batch,
                                               self._labels[:remaining]],
                                              axis=0)
            self._index_in_epoch = remaining
        else:
            data_batch = self._data[start:start + batch_size]
            if self._labeled:
                labels_batch = self._labels[start:start + batch_size]
            self._index_in_epoch = start + batch_size
        batch = (data_batch, labels_batch) if self._labeled else data_batch
        return batch


def get_date_time():
    import time
    timestr = time.strftime("%Y%m%d_%H%M%S")
    return timestr
import tensorflow as tf
import numpy as np
import os

class Model(object):
    def __init__(self, input_node, hidden_layers_node, output_node, learning_rate, batch_size, display_step, activation,
            seed=1,
            feature_selection=False,
            a = 1,
            sigma = 0.1,
            lam=0.5,
            param_search=False
        ): #Note: a, sigma, lam should be set by params dict that will be passed to this class.
        self.param_search = param_search
        # Register hyperparameters for feature selection
        self.a = a
        self.sigma = sigma
        self.lam = lam
        # Register regular hyperparameters
        self.lr = learning_rate
        self.batch_size = batch_size
        self.display_step = display_step # to print loss/acc information during training

        G = tf.Graph()
        with G.as_default():
            self.sess = tf.Session(graph=G)
            # tf Graph Input
            X = tf.placeholder(tf.float32, [None, input_node]) # X.shape == [batch_size, feature_size]
            y = tf.placeholder(tf.float32, [None, output_node])
            thresh = tf.placeholder(tf.float32, [1])
            train_gates = tf.placeholder(tf.float32, [1], name='train_gates')
            self.nnweights = []
            prev_node = input_node
            prev_x = X
            with tf.variable_scope('gates', reuse=tf.AUTO_REUSE):
                self.alpha = tf.get_variable('alpha', [prev_node,],
                                          initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                prev_x = self.feature_selector(prev_x, train_gates)

            layer_name = 'layer' + str(1)
            for i in range(len(hidden_layers_node)):
                layer_name = 'layer' + str(i+1)
                with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
                    weights = tf.get_variable('weights', [prev_node, hidden_layers_node[i]],
                                              initializer=tf.truncated_normal_initializer(stddev=0.1))
                    self.nnweights.append(weights)
                    biases = tf.get_variable('biases', [hidden_layers_node[i]],
                                             initializer=tf.constant_initializer(0.0))
                    layer_out = (tf.matmul(prev_x, weights) + biases) # Softmax
                    if activation == 'relu':
                        layer_out = tf.nn.relu(layer_out)
                    elif activation == 'sigmoid':
                        layer_out = tf.nn.sigmoid(layer_out)
                    elif activation == 'tanh':
                        layer_out = tf.nn.tanh(layer_out)
                    elif activation == 'none':
                        layer_out =(layer_out)
                    else:
                        raise NotImplementedError('activation not recognized')

                    prev_node = hidden_layers_node[i]
                    prev_x = layer_out

            # Output of model
            pred = tf.nn.softmax(layer_out)
            # Minimize error using cross entropy
            loss_fun = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=layer_out))
            if feature_selection:
                ## gates regularization
                input2cdf = self.alpha
                reg = 0.5*(1 + tf.erf(input2cdf/(self.sigma*np.sqrt(2))))                
                reg_gates = self.lam*tf.reduce_mean(reg) 
                loss = loss_fun  +  reg_gates
                self.reg_gates = reg_gates # for debugging
            else:
                loss = loss_fun
            # Get optimizer
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
            # For evaluation
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # Initialize the variables (i.e. assign their default value)
            init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

        # Save into class members
        self.X = X
        self.y = y
        self.pred = pred
        self.train_gates = train_gates
        self.thresh = thresh
        self.loss = loss
        self.train_step = train_step
        self.correct_prediction = correct_prediction
        self.accuracy = accuracy

        # set random state
        tf.set_random_seed(seed)
        self.sess.run(init_op)

    def _to_tensor(self, x, dtype):
        """Convert the input `x` to a tensor of type `dtype`.
        # Arguments
            x: An object to be converted (numpy array, list, tensors).
            dtype: The destination type.
        # Returns
            A tensor.
        """
        return tf.convert_to_tensor(x, dtype=dtype)

    def hard_sigmoid(self, x, a):
        """Segment-wise linear approximation of sigmoid.
        Faster than sigmoid.
        Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
        In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.
        # Arguments
            x: A tensor or variable.
        # Returns
            A tensor.
        """
        x = a * x + 0.5
        zero = self._to_tensor(0., x.dtype.base_dtype)
        one = self._to_tensor(1., x.dtype.base_dtype)
        x = tf.clip_by_value(x, zero, one)
        return x

    def feature_selector(self, prev_x, train_gates):
        '''
        feature selector - used at training time (gradients can be propagated)
        :param prev_x - input. shape==[batch_size, feature_num]
        :param train_gates (bool) - 1 during training, 0 during evaluation
        :return: gated input
        '''
        # gaussian reparametrization
        base_noise = tf.random_normal(shape=tf.shape(prev_x), mean=0., stddev=1.)
        z = tf.expand_dims(self.alpha, axis=0) + self.sigma * base_noise * train_gates
        stochastic_gate = self.hard_sigmoid(z, self.a)
        new_x = prev_x * stochastic_gate
        return new_x

    def eval(self, new_X, new_y):
        acc, loss = self.sess.run([self.accuracy, self.loss], feed_dict={self.X: new_X,
                                                        self.y: new_y,
                                                        self.train_gates: [0.0],
                                                        self.thresh: [float('inf')]})
        return np.squeeze(acc), np.squeeze(loss)

    def get_raw_alpha(self):
        """
        evaluate the learned parameter for stochastic gates 
        """
        dp_alpha = self.sess.run(self.alpha)
        return dp_alpha

    def get_prob_alpha(self):
        """
        convert the raw alpha into the actual probability
        """
        dp_alpha = self.get_raw_alpha()
        prob_gate = self.compute_learned_prob(dp_alpha)
        return prob_gate

    def hard_sigmoid_np(self, x, a):
        return np.minimum(1, np.maximum(0,a*x+0.5))

    def compute_learned_prob(self, alpha):
        z = alpha
        stochastic_gate = self.hard_sigmoid_np(z, self.a)
        return stochastic_gate

    def load(self, model_path=None):
        if model_path == None:
            raise Exception()
        self.saver.restore(self.sess, model_path)

    def save(self, step, model_dir=None):
        if model_dir == None:
            raise Exception()
        try:
            os.mkdir(model_dir)
        except:
            pass
        model_file = model_dir + "/model"
        self.saver.save(self.sess, model_file, global_step=step)

    def train(self, trial, dataset, output_dir, num_epoch=100, plot_loss=False):
        losses = []
        accuracies = []
        print("num_samples : {}".format(dataset.num_samples))
        for epoch in range(num_epoch):
            avg_loss = 0.
            total_batch = int(dataset.num_samples/self.batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = dataset.next_batch(self.batch_size)
                _, c, reg_fs = self.sess.run([self.train_step, self.loss, self.reg_gates], feed_dict={self.X: batch_xs,
                                                              self.y: batch_ys,
                                                              self.train_gates: [1.0]})
                avg_loss += c / total_batch
            losses.append(avg_loss)
            # Display logs per epoch step
            if (epoch+1) % self.display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.9f}".format(avg_loss))
                print("reg_fs: {}".format(reg_fs))
                
                #losses.append(valid_loss)
                # Report intermediate objective value.
                if self.param_search:
                    valid_acc, valid_loss = self.eval(dataset.valid_data, dataset.valid_labels)
                    accuracies.append(valid_acc)
                    intermediate_value = valid_loss #1.0 - clf.score(test_x, test_y)
                    trial.report(intermediate_value, epoch)
                    # Handle pruning based on the intermediate value.
                    if trial.should_prune(epoch):
                        raise optuna.structs.TrialPruned()
        print("Optimization Finished!")
        if not self.param_search:
            test_acc, test_loss = self.eval(dataset.test_data, dataset.test_labels)
            print("test loss: {}, test acc: {}".format(test_loss, test_acc))
            #print("Saving model..")
            #self.save(step=1, model_dir=output_dir)
           # my_acc=tf.losses.sigmoid_cross_entropy
            self.acc=test_acc
        return accuracies
    
    def evaluate(self, X, y):
        acc, loss = self.eval(X, y)
        print("test loss: {}, test acc: {}".format(loss, acc))
        print("Saving model..")
        #self.save(step=1, model_dir=output_dir)
        #self.acc=test_acc
        return loss

def get_date_time():
    import time
    timestr = time.strftime("%Y%m%d_%H%M%S")
    return timestr
def convertToOneHot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.

    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v

        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """

    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)
