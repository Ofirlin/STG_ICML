from __future__ import print_function
import numpy as np
import tensorflow as tf
from lifelines.utils import concordance_index
import os
import pickle

import vision, utils 
from scipy.special import expit




class L2DeepSurv(object):
    def __init__(self, X, label, valid_X, valid_label,
        input_node, output_node,hidden_layers_node,
        learning_rate=0.001, learning_rate_decay=1.0, 
        activation='tanh', 
        L2_reg=0.0, L1_reg=0.0, optimizer='sgd', 
        dropout_keep_prob=1.0,
        feature_selection=False,
        seed=1,
        sigma=0.5,
        lam=0.005,
        standardize=False
        ):
        """
        L2DeepSurv Class Constructor.
        Parameters:
            X: np.array, covariate variables.
            label: dict, like {'e': event, 't': time}, Observation and Time in survival analyze.
            input_node: int, number of covariate variables.
            hidden_layers_node: list, hidden layers in network.
            output_node: int, number of output.
            learning_rate: float, learning rate.
            learning_rate_decay: float, decay of learning rate.
            activation: string, type of activation function.
            L1_reg: float, coefficient of L1 regularizate item.
            L2_reg: float, coefficient of L2 regularizate item.
            optimizer: string, type of optimize algorithm.
            dropout_keep_prob: float, probability of dropout.
            seed: set random state.
        Returns:
            L2DeepSurv Class.
        """
        # Register gates hyperparameters
        self.lam = lam
        self.sigma = sigma
        
        # Prepare data
        '''
        self.train_data = {}
        self.train_data['X'], self.train_data['E'], \
            self.train_data['T'], self.train_data['failures'], \
            self.train_data['atrisk'], self.train_data['ties'] = utils.parse_data(X, label) 
        self.valid_data = {}
        self.valid_data['X'], self.valid_data['E'], \
            self.valid_data['T'], self.valid_data['failures'], \
            self.valid_data['atrisk'], self.valid_data['ties'] = utils.parse_data(valid_X, valid_label) 
        '''
        self.train_data={}
        self.train_data['X'], self.train_data['E'], \
                self.train_data['T'] = utils.prepare_data(X, label)
        self.train_data['ties']='noties'
        self.valid_data={}
        self.valid_data['X'], self.valid_data['E'], \
                self.valid_data['T'] = utils.prepare_data(valid_X, valid_label)
        self.valid_data['ties']='noties'
        
        # New Graph
        G = tf.Graph()
        with G.as_default():
            # Data input
            X = tf.placeholder(tf.float32, [None, input_node], name = 'x-Input')
            y_ = tf.placeholder(tf.float32, [None, output_node], name = 'label-Input')
            train_gates = tf.placeholder(tf.float32, [1], name='train_gates')
            # hidden layers
            self.nnweights = [] # collect weights of network
            prev_node = input_node
            prev_x = X
            with tf.variable_scope('gates', reuse=tf.AUTO_REUSE): 
                self.alpha = tf.get_variable('alpha', [prev_node,], 
                                          initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                prev_x = self.feature_selector(prev_x, train_gates) 
                
            for i in range(len(hidden_layers_node)):
                layer_name = 'layer' + str(i+1)
                with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
                    weights = tf.get_variable('weights', [prev_node, hidden_layers_node[i]], 
                                              initializer=tf.truncated_normal_initializer(stddev=0.1))
                    self.nnweights.append(weights)

                    biases = tf.get_variable('biases', [hidden_layers_node[i]],
                                             initializer=tf.constant_initializer(0.0))

                    layer_out = tf.nn.dropout(tf.matmul(prev_x, weights) + biases, dropout_keep_prob)

                    if activation == 'relu':
                        layer_out = tf.nn.relu(layer_out)
                    elif activation == 'selu':
                        layer_out = tf.nn.selu(layer_out)
                    elif activation == 'sigmoid':
                        layer_out = tf.nn.sigmoid(layer_out)
                    elif activation == 'tanh':
                        layer_out = tf.nn.tanh(layer_out)
                    else:
                        raise NotImplementedError('activation not recognized')

                    prev_node = hidden_layers_node[i]
                    prev_x = layer_out
            # output layers 
            layer_name = 'layer_last'
            with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
                weights = tf.get_variable('weights', [prev_node, output_node], 
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
                self.nnweights.append(weights)

                biases = tf.get_variable('biases', [output_node],
                                         initializer=tf.constant_initializer(0.0))

                layer_out = tf.matmul(prev_x, weights) + biases
            # Output of Network
            y = layer_out
            # Global step
            with tf.variable_scope('training_step', reuse=tf.AUTO_REUSE):
                global_step = tf.get_variable("global_step", [], 
                                              dtype=tf.int32,
                                              initializer=tf.constant_initializer(0), 
                                              trainable=False)
            # Loss value
            ## L1 - L2 Regularization
            reg_item = tf.contrib.layers.l1_l2_regularizer(L1_reg,
                                                           L2_reg)
            reg_term = tf.contrib.layers.apply_regularization(reg_item, self.nnweights)
            if feature_selection:
                ## Regularization
                reg = 0.5 - 0.5*tf.erf((-1/(2) - self.alpha)/(self.sigma*np.sqrt(2)))
                reg_gates = tf.reduce_mean(reg) * self.lam
            ## Negative log likelihood
            loss_fun = self._negative_log_likelihood(y_, y)
            if feature_selection:
                loss = loss_fun + reg_term + reg_gates
            else:
                loss = loss_fun + reg_term
            # SGD Optimizer
            if optimizer == 'sgd':
                lr = tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    1,
                    learning_rate_decay
                )
                train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step=global_step)
            elif optimizer == 'adam':
                train_step = tf.train.GradientDescentOptimizer(learning_rate).\
                                                               minimize(loss, global_step=global_step)
            else:
                raise NotImplementedError('activation not recognized')
            # init op
            init_op = tf.global_variables_initializer()
        
            # Create a saver
            self.saver = tf.train.Saver()

        # Save into class members
        self.X = X
        self.y_ = y_
        self.y = y
        self.train_gates = train_gates
        self.global_step = global_step 
        self.loss = loss
        self.train_step = train_step
        self.configuration = {
            'input_node': input_node,
            'hidden_layers_node': hidden_layers_node,
            'output_node': output_node,
            'learning_rate': learning_rate,
            'learning_rate_decay': learning_rate_decay,
            'activation': activation,
            'L1_reg': L1_reg,
            'L2_reg': L2_reg,
            'optimizer': optimizer,
            'dropout': dropout_keep_prob
        }


        # Set random state
        tf.set_random_seed(seed)
        # create new Session for the DeepSurv Class
        self.sess = tf.Session(graph=G)
        # Initialize all global variables
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

    def hard_sigmoid(self, x):
        """Segment-wise linear approximation of sigmoid.
        Faster than sigmoid.
        Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
        In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.
        # Arguments
            x: A tensor or variable.
        # Returns
            A tensor.
        """
        x = x + 0.5
        zero = self._to_tensor(0., x.dtype.base_dtype)
        one = self._to_tensor(1., x.dtype.base_dtype)
        x = tf.clip_by_value(x, zero, one)
        return x

    def feature_selector(self, prev_x, train_gates):
        '''
        :param x: input
        :return:  gated input
        '''
        # gaussian reparametrization
        base_noise = tf.random_normal(shape=tf.shape(prev_x), mean=0., stddev=1.)
        z = tf.expand_dims(self.alpha, axis=0) + self.sigma * base_noise * train_gates
        stochastic_gate = self.hard_sigmoid(z)
        new_x = prev_x * stochastic_gate
        return new_x

    def train(self, output_dir, num_epoch=5000, iteration=-1, 
              plot_loss=False, plot_CI=False, plot_gate=False):
        """
        Training DeepSurv network.
        Parameters:
            num_epoch: times of iterating whole train set.
            iteration: print information on train set every iteration train steps.
                       default -1, means keep silence.
            plot_train_loss: plot curve of loss value during training.
            plot_train_CI: plot curve of CI on train set during training.
        Returns:
        """
        # Record training steps
        loss_list, val_loss_list = [], []
        CI_list, val_CI_list = [], []
        gate_list = []
        gate_avg_list = []
        N = self.train_data['E'].shape[0]
        valid_N = self.valid_data['E'].shape[0]
        # Train steps 
        max_val_CI = 0
        for i in range(num_epoch):
            _, output_y, loss_value, step = self.sess.run([self.train_step, self.y, self.loss, self.global_step],
                                                          feed_dict = {self.X:  self.train_data['X'],
                                                                       self.y_: self.train_data['E'].reshape((N, 1)),
                                                                       self.train_gates: [1.0]})
            val_output_y, val_loss = self.sess.run([self.y, self.loss], feed_dict= {self.X: self.valid_data['X'],
                                                                     self.y_: self.valid_data['E'].reshape((valid_N, 1)),
                                                                     self.train_gates: [0.0]})
            # Record information
            loss_list.append(loss_value)
            label = {'t': self.train_data['T'],
                     'e': self.train_data['E']}
            #print('ok')
            #print(label['t'].shape)
            #print(output_y.shape)
            #print(output_y)
            CI = self._Metrics_CI(label, output_y) 
            CI_list.append(CI)  
            val_loss_list.append(val_loss)
            val_label = {'t': self.valid_data['T'],
                     'e': self.valid_data['E']}
            val_CI = self._Metrics_CI(val_label, val_output_y) 
            val_CI_list.append(val_CI)
            if val_CI > max_val_CI: 
                print(val_CI)
                self.save(step=1, model_dir=output_dir)
                max_val_CI = val_CI

            if plot_gate:
                #print("Inside plot_gate...")
                alpha = self.sess.run(self.alpha) # (49581,)
                gate_list.append(alpha.reshape(-1,1)) 
                gate_avg_list.append(np.mean(self.hard_sigmoid_np(alpha.reshape(-1,1))))
                gate_list_np = np.concatenate(gate_list, axis=1)
                with open(output_dir+'/stochastic_gates.pkl', 'wb') as f:
                    pickle.dump(gate_list_np, f) 
                #import ipdb; ipdb.set_trace()
                vision.save_heatmap(gate_list_np, output_dir, 'stg_heatmap')
                vision.save_plot(gate_avg_list, gate_avg_list, output_dir, 'gate')

                #if gate_avg_list[-1] < 0.0001:
                #    self.save(step=i, model_dir=output_dir) 
                if gate_avg_list[-1] == 0:
                    print("gates are all zero..")
                    self.save(step=i, model_dir=output_dir)
                    break


            if plot_loss:
                #vision.plot_train_curve(loss_list, title="Loss(train)")
                #vision.plot_curve(loss_list, val_loss_list, title="Loss(train and valid)")
                vision.save_plot(loss_list, val_loss_list, output_dir,
                                                    'loss', title="Loss(train and valid)")

            if plot_CI:
                #vision.plot_train_curve(CI_list, title="CI(train)")
                #vision.plot_curve(CI_list, val_CI_list, title="CI(train and valid)")
                vision.save_plot(CI_list, val_CI_list, output_dir,
                                                    'CI', title="CI(train and valid)")

            # Print evaluation on test set 
            if (iteration != -1) and (i % iteration == 0):
                print("-------------------------------------------------")
                print("training steps %d:\nloss = %g  CI = %g.\n" % (step, loss_value, CI))
                print("val loss %g:  val CI = %g.\n" % (val_loss, val_CI)) 
        
        #self.save(step=1, model_dir=output_dir)

        # Plot curve
        if plot_loss:
            # vision.plot_train_curve(loss_list, title="Loss(train)")
            #vision.plot_curve(loss_list, val_loss_list, title="Loss(train and valid)")
            vision.save_plot(loss_list, val_loss_list, output_dir,
                                                'loss', title="Loss(train and valid)")

        if plot_CI:
            vision.save_plot(CI_list, val_CI_list, output_dir,
                                                'CI', title="CI(train and valid)")

    def ties_type(self):
        """
        return the type of ties in train data.
        """
        return self.train_data['ties']

    def predict(self, X):
        """
        Predict risk of X using trained network.
        Parameters:
            X: np.array, covariate variables.
        Returns:
            np.array, shape(n,), Proportional risk of X.
        """
        risk = self.sess.run([self.y], feed_dict = {self.X: X, self.train_gates: [0.0]})
        return np.squeeze(risk)

    def eval(self, X, label):
        """
        Evaluate test set using CI metrics.
        Parameters:
            X: np.array, covariate variables.
            label: dict, like {'e': event, 't': time}, Observation and Time in survival analyze.
        Returns:
            np.array, shape(n,), Proportional risk of X.
        """
        N = label['e'].shape[0]
        output_y, loss_value = self.sess.run([self.y, self.loss],
                                            feed_dict = {self.X: X,
                                                         self.y_: label['e'].reshape((N, 1)),
                                                         self.train_gates: [0.0]})
        CI = self._Metrics_CI(label, output_y)

        return (CI, loss_value)

    def close(self):
        """
        close session of tensorflow.
        """
        self.sess.close()
        print("Current session closed!")
    
    def _negative_log_likelihood(self, y_true, y_pred):
        """
        Callable loss function for DeepSurv network.
        the negative average log-likelihood of the prediction
        of this model under a given target distribution. 
        Parameters: 
            y_true: tensor, observations. 
            y_pred: tensor, output of network.
        Returns:
            loss value, means negative log-likelihood. 
        """
        logL = 0
        # pre-calculate cumsum
        cumsum_y_pred = tf.cumsum(y_pred)
        hazard_ratio = tf.exp(y_pred)
        cumsum_hazard_ratio = tf.cumsum(hazard_ratio)
        if self.train_data['ties'] == 'noties':
            log_risk = tf.log(cumsum_hazard_ratio)
            likelihood = y_pred - log_risk
            # dimension for E: np.array -> [None, 1]
            uncensored_likelihood = likelihood * y_true
            logL = -tf.reduce_sum(uncensored_likelihood)
        else:
            # Loop for death times
            for t in self.train_data['failures']:                                                                       
                tfail = self.train_data['failures'][t]
                trisk = self.train_data['atrisk'][t]
                d = len(tfail)
                dr = len(trisk)

                logL += -cumsum_y_pred[tfail[-1]] + (0 if tfail[0] == 0 else cumsum_y_pred[tfail[0]-1])

                if self.train_data['ties'] == 'breslow':
                    s = cumsum_hazard_ratio[trisk[-1]]
                    logL += tf.log(s) * d
                elif self.train_data['ties'] == 'efron':
                    s = cumsum_hazard_ratio[trisk[-1]]
                    r = cumsum_hazard_ratio[tfail[-1]] - (0 if tfail[0] == 0 else cumsum_hazard_ratio[tfail[0]-1])
                    for j in range(d):
                        logL += tf.log(s - j * r / d)
                else:
                    raise NotImplementedError('tie breaking method not recognized')
        # negative average log-likelihood
        observations = tf.reduce_sum(y_true)
        return logL / observations
    
    def _Metrics_CI(self, label_true, y_pred):
        """
        Compute the concordance-index value.
        Parameters:
            label_true: dict, like {'e': event, 't': time}, Observation and Time in survival analyze.
            y_pred: np.array, predictive proportional risk of network.
        Returns:
            concordance index.
        """
        hr_pred = -y_pred
        ci = concordance_index(label_true['t'],
                               hr_pred,
                               label_true['e'])
        return ci
    
    def get_raw_alpha(self):
        """
        evaluate the learned dropout rate
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
 
    def hard_sigmoid_np(self, x):
        return np.minimum(1, np.maximum(0,x+0.5))

    def compute_learned_prob(self, alpha):
        z = alpha
        stochastic_gate = self.hard_sigmoid_np(z)
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

    def evaluate_var_byWeights(self):
        """
        evaluate feature importance by weights of NN.
        """
        # fetch weights of network
        W = [self.sess.run(w) for w in self.nnweights]
        n_w = len(W)
        # matrix multiplication for all hidden layers except last output layer
        hiddenMM = W[- 2].T
        for i in range(n_w - 3, -1, -1):
            hiddenMM = np.dot(hiddenMM, W[i].T)
        # multiply last layer matrix and compute the sum of each variable for VIP
        last_layer = W[-1]
        s = np.dot(np.diag(last_layer[:, 0]), hiddenMM)

        sumr = s / s.sum(axis=1).reshape(s.shape[0] ,1)
        score = sumr.sum(axis=0)
        VIP = score / score.max()
        for i, v in enumerate(VIP):
            print("%dth feature score : %g." % (i, v))
        return VIP

    def survivalRate(self, X, algo="wwe", base_X=None, base_label=None, smoothed=False):
        """
        Estimator of survival function for data X.
        Parameters:
            X: np.array, covariate variables of patients.
            algo: algorithm for estimating survival function.
            base_X: X of patients for estimating survival function.
            base_label: label of patients for estimating survival function.
            smoothed: smooth survival function or not.
        Returns:
            T0: time points of survival function.
            ST: survival rate of survival function.
        """
        risk = self.predict(X)
        hazard_ratio = np.exp(risk.reshape((risk.shape[0], 1)))
        # Estimate S0(t) using data(base_X, base_label)
        T0, S0 = self.basesurv(algo=algo, X=base_X, label=base_label, smoothed=smoothed)
        ST = S0**(hazard_ratio)

        vision.plt_surLines(T0, ST)

        return T0, ST

    def basesurv(self, algo="wwe", X=None, label=None, smoothed=False):
        """
        Estimate base survival function S0(t) based on data(X, label).
        Parameters:
            algo: algorithm for estimating survival function.
            X: X of patients for estimating survival function.
            label: label of patients for estimating survival function.
            smoothed: smooth survival function or not.
        Returns:
            T0: time points of base survival function.
            ST: survival rate of base survival function.
        See:
            Algorithm for estimating basel survival function:
            (1). wwe: WWE(with ties)
            (2). kp: Kalbfleisch & Prentice Estimator(without ties)
            (3). bsl: breslow(with ties, but exists negative value)
        """
        # Get data for estimating S0(t)
        if X is None or label is None:
            X = self.train_data['X']
            label = {'t': self.train_data['T'],
                     'e': self.train_data['E']}
        X, E, T, failures, atrisk, ties = utils.parse_data(X, label)

        s0 = [1]
        risk = self.predict(X)
        hz_ratio = np.exp(risk)
        if algo == 'wwe':        
            for t in T[::-1]:
                if t in atrisk:
                    # R(t_i) - D_i
                    trisk = [j for j in atrisk[t] if j not in failures[t]]
                    dt = len(failures[t]) * 1.0
                    s = np.sum(hz_ratio[trisk])
                    cj = 1 - dt / (dt + s)
                    s0.append(cj)
                else:
                    s0.append(1)
        elif algo == 'kp':
            for t in T[::-1]:
                if t in atrisk:
                    # R(t_i)
                    trisk = atrisk[t]
                    s = np.sum(hz_ratio[trisk])
                    si = hz_ratio[failures[t][0]]
                    cj = (1 - si / s) ** (1 / si)
                    s0.append(cj)
                else:
                    s0.append(1)
        elif algo == 'bsl':
            for t in T[::-1]:
                if t in atrisk:
                    # R(t_i)
                    trisk = atrisk[t]
                    dt = len(failures[t]) * 1.0
                    s = np.sum(hz_ratio[trisk])
                    cj = 1 - dt / s
                    s0.append(cj)
                else:
                    s0.append(1)
        else:
            raise NotImplementedError('tie breaking method not recognized')
        # base survival function
        S0 = np.cumprod(s0, axis=0)
        T0 = np.insert(T[::-1], 0, 0, axis=0)

        if smoothed:
            # smooth the baseline hazard
            ss = SuperSmoother()
            #Check duplication points
            ss.fit(T0, S0, dy=100)
            S0 = ss.predict(T0)

        return T0, S0
        
