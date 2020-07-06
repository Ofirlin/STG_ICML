import numpy as np
import tensorflow as tf
import optuna
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
            # Minimize error using cross entropy
            if output_node==1:
               # pred = layer_out
                weights = tf.get_variable('weights', [1, 1],
                                              initializer=tf.truncated_normal_initializer(stddev=0.1))
                self.nnweights.append(weights)
                biases = tf.get_variable('biases', [1],
                                         initializer=tf.constant_initializer(0.0))
                pred = (tf.matmul(layer_out, weights) + biases)
                loss_fun = tf.reduce_mean(tf.squared_difference(pred, y))
            else:
                pred = tf.nn.softmax(layer_out)
                loss_fun = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=layer_out))
            if feature_selection:
                ## gates regularization
                input2cdf = self.alpha
                #reg = 0.5*(1 + tf.erf(input2cdf/(self.sigma*np.sqrt(2))))
                reg = 0.5 - 0.5*tf.erf((-1/(2*self.a) - input2cdf)/(self.sigma*np.sqrt(2)))
                reg_gates = self.lam*tf.reduce_mean(reg)
                loss = loss_fun  +  reg_gates
                self.reg_gates = reg_gates # for debugging
            else:
                loss = loss_fun
                self.reg_gates = 0
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
        self.loss = loss
        self.train_step = train_step
        self.correct_prediction = correct_prediction
        self.accuracy = accuracy
        self.output_node=output_node
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
                                                        self.train_gates: [0.0]
                                                        })
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
        train_losses, train_accuracies = [], []
        val_losses = []
        val_accuracies = []
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
            train_losses.append(avg_loss)
            # Display logs per epoch step
            if (epoch+1) % self.display_step == 0:
                valid_acc, valid_loss = self.eval(dataset.valid_data, dataset.valid_labels)
                val_accuracies.append(valid_acc)
                val_losses.append(valid_loss)
                if self.output_node!=1:
                    print("Epoch: {} train loss={:.9f} valid loss= {:.9f} valid acc= {:.9f}".format(epoch+1,\
                                                                                                    avg_loss, valid_loss, valid_acc))
                else:
                    print("Epoch: {} train loss={:.9f} valid loss= {:.9f}".format(epoch+1,\
                                                                                  avg_loss, valid_loss))
                print("train reg_fs: {}".format(reg_fs))
                if self.param_search:
                    # Report intermediate objective value.
                    intermediate_value = valid_loss #1.0 - clf.score(test_x, test_y)
                    trial.report(intermediate_value, epoch)
                    # Handle pruning based on the intermediate value.
                    if trial.should_prune(epoch):
                        raise optuna.structs.TrialPruned()
        print("Optimization Finished!")
        if not self.param_search:
            test_acc, test_loss = self.eval(dataset.test_data, dataset.test_labels)
            print("test loss: {}, test acc: {}".format(test_loss, test_acc))
            self.acc=test_acc # used for recording test acc for figures
        return train_accuracies, train_losses, val_accuracies, val_losses 
    def test(self,X_test):
        prediction = self.sess.run([self.pred], feed_dict={self.X: X_test,self.train_gates: [0.0]})
        if self.output_node!=1:
            prediction=np.argmax(prediction[0],axis=1)
        return prediction
    def evaluate(self, X, y):
        acc, loss = self.eval(X, y)
        print("test loss: {}, test acc: {}".format(loss, acc))
        print("Saving model..")
        #self.save(step=1, model_dir=output_dir)
        #self.acc=test_acc
        return acc, loss

