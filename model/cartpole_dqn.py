
import numpy as np, gym, sys, copy, argparse
from gym import wrappers
import ffmpeg
import random
import os
import matplotlib.pyplot as plt
import math
import tensorflow as tf 
from new_start_models import *

class QNetwork( object ):

    # This class essentially defines the network architecture. 
    # The network should take in state of the world as an input, 
    # and output Q values of the actions available to the agent as the output. 

    def __init__(self ):
        # Define your network architecture here. It is also a good idea to define any training operations 
        # and optimizers here, initialize your variables, or alternately compile your model here.  

        self.sess = tf.Session() 
        self.init()
        self.saver = tf.train.Saver( max_to_keep = 100 )

    def init( self ):
        self.add_input_placeholder()
        self.construct_clf()
        self.construct_loss()
        self.sess.run( [ tf.global_variables_initializer() ] )
        os.mkdir( self.opts.cpt_path ) if not os.path.exists( self.opts.cpt_path ) else print()

    def add_input_placeholder( self ):
        self.in_state = tf.placeholder( tf.float32, [ None, 4 ] )
        self.in_action = tf.placeholder( tf.int32, [ None , 1 ] )
        self.in_reward = tf.placeholder( tf.float32, [ None, 1 ] )
        self.in_lr = tf.placeholder( tf.float32, 1 )
        self.one_hot_action = tf.one_hot( tf.reshape( self.in_action, [ -1 ] ), 2 )

    def construct_clf( self ):
        
        with tf.variable_scope( "nn" ) as scope:
            self.l0_obj = Group_L0( 4, 30, activation = tf.nn.relu, weight_decay = 1.0 , name = "l0" )
            self.net_l0_train, self.feat_l0_train, self.net_l0_test, self.feat_l0_test, self.l0_regu = self.l0_obj.build( self.in_state, self.in_state )
            
            self.l1_obj = Group_L0( 30, 30, activation = tf.nn.relu, weight_decay = 1.0, name = "l1" )
            self.net_l1_train, self.feat_l1_train, self.net_l1_test, self.feat_l1_test, self.l1_regu = self.l1_obj.build( self.net_l0_train, self.net_l0_test )

            self.concat_train = tf.concat( [ self.feat_l0_train, self.feat_l1_train, self.net_l1_train ], axis = 1 ) 
            self.concat_test = tf.concat( [ self.feat_l0_test, self.feat_l1_test, self.net_l1_test ], axis = 1 )
            # self.logit_obj = Group_L0( 784 + 300 + 100, 10, weight_decay = 5e-4, name = "logit" )
            self.q_train, _, self.final_w, _ =  dense( self.concat_train, 2, initializer = tf.random_normal_initializer( stddev = 1./tf.sqrt( 2./2. ) ), name = "logit" )
            self.q_test, _, _, _ =  dense( self.concat_test, 2, initializer = tf.random_normal_initializer( stddev = 1./tf.sqrt( 2./2. ) ), name = "logit", reuse = True )
            self.prediction = self.q_test
    
    def construct_loss( self ):
        self.loss = tf.reduce_mean( tf.square( tf.reduce_sum( self.q_train * self.one_hot_action, axis = 1 ) - self.in_reward ) )
        self.loss = self.loss \
                        -10.*( tf.reduce_mean( self.l0_regu ) + tf.reduce_mean( self.l1_regu ) )
        #self.loss = tf.reduce_mean( tf.square( self.in_label - self.prediction ) )
        self.optim_nn = tf.train.AdamOptimizer( self.in_lr, beta1 = 0.9, beta2 = 0.99 ).minimize( loss = self.loss, var_list =  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='nn') )

    def calc_sparsity( self, in_sample, in_label, thresh = 0.95 ):
        l0_feat_gate_test, l1_feat_gate_test = self.sess.run( [ self.l0_obj.feature_mask_test, self.l1_obj.feature_mask_test ], 
                                                              feed_dict = { self.in_sample : in_sample, self.in_label: in_label } )
        l0_feat_gate_test = l0_feat_gate_test.ravel()
        l1_feat_gate_test = l1_feat_gate_test.ravel()
        print( l0_feat_gate_test[ :10 ] )
        l0 = np.sum( l0_feat_gate_test >= thresh )
        l1 = np.sum( l1_feat_gate_test >= thresh )

        print( "l0 gate", l0, "out of", l0_feat_gate_test.shape[0], "l1_gate", l1, "out of", l1_feat_gate_test.shape )

    def get_q_val( self, state ):
        return self.sess.run( self.q_test, feed_dict = { self.in_state: state } )

    def optimize( self, sess, state_batch, action_batch, reward_batch, learning_rate ):
        loss, _ = self.sess.run( [ self.loss, self.optim_nn ], feed_dict = { self.in_state: state_batch, self.in_action: action_batch,
                                                                             self.in_reward: reward_batch, self.in_lr: learning_rate } )
        return loss

# memory storage: ( st, a, r, st+1, is_term )
class Replay_Memory():

    def __init__(self, max_memory_size=50000):

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the 
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
        # A simple (if not the most efficient) was to implement the memory his as a list of transitions. 
        self.max_memory_size = max_memory_size
        self.memory = [ [] for i in range( max_memory_size ) ]
        self.cur_index = 0

    def get_memory_size( self ):
        return len( self.memory )

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
        # You will feed this to your model to train.
        assert( len( self.memory ) >= batch_size, "memory does not have enough to sample" )
        # return random.sample( self.memory, batch_size )
        if self.cur_index < self.max_memory_size:
        	return random.sample( self.memory[ :self.cur_index ], batch_size )
        else:
        	return random.sample( self.memory, batch_size )

    def append(self, transition):
        # Appends transition to the memory.     
        assert( len( transition ) == 5, "invalid transition format" )
        self.memory[ self.cur_index % self.max_memory_size ] = transition
        self.cur_index += 1



class DQN_Agent():
    
    def __init__(self, environment_name, hidden_size, render=False):

        # Create an instance of the network itself, as well as the memory. 
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc. 
         # Setting the session to allow growth, so it doesn't allocate all GPU memory. 
        
        self.sess = None

        self.environment_name = environment_name
        

        self.base_dir = "single_work_dir"
        # make working directories
        if not os.path.isdir( self.base_dir ): os.mkdir( self.base_dir )
        self.env_dir = os.path.join( self.base_dir, self.environment_name )
        if not os.path.isdir( self.env_dir ): os.mkdir( self.env_dir )
        self.tmp_dir = os.path.join( self.base_dir, self.environment_name + "_tmp" )
        if not os.path.isdir( self.tmp_dir ): os.mkdir( self.tmp_dir )
        
        self.env =  gym.make( environment_name )
        # if render:
        #     self.env = wrappers.Monitor(self.env, self.tmp_dir + "/monitor_1", force = True )
        if environment_name == "MountainCar-v0":
            self.action_size = 3
            self.state_size = 2
            self.discount_factor = 1.0
            self.learning_rate = 0.0001
            self.start_epi = 0.5
            self.end_epi = 0.05
            self.sample_batch = 32
            self.burn_in_amount = 1000
            self.memory_size = 5000
            self.lr_decay = 0.99
        elif environment_name == "CartPole-v0":
            self.action_size = 2
            self.state_size = 4
            self.discount_factor = 0.99
            self.learning_rate = 0.001
            self.start_epi = 0.5
            self.sample_batch = 64
            self.end_epi = 0.05
            self.burn_in_amount = 5000
            self.memory_size = 50000
            self.lr_decay = 0.99

        self.replay_memory = Replay_Memory( max_memory_size = self.memory_size )
        self.burn_in_memory( self.burn_in_amount )
        self.source_action_value = QNetwork( )
        
        # update target paeam per C episodes
        self.C = 100
        # epi greedy
        self.max_episode = 10000
        self.evaluate_every = 100 # evaluate reward for every * episode

    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.     

        prob = np.ones( self.action_size )
        # reduce all prob to epi
        prob = prob * self.epi / self.action_size
        remainder_prob = 1 - np.sum( prob )
        optim_action = np.argmax( np.array( q_values ) )
        prob[ optim_action ] += remainder_prob
        return prob

        # Creating greedy policy for test time. 
    def greedy_policy(self, q_values):
        
        prob = np.zeros( self.action_size )
        optim_action = np.argmax( np.array( q_values ) )
        prob[ optim_action ] = 1
        return prob
    
    # epi-greedy
    def train(self):
        loss_list = []
        # In this function, we will train our network. 
        # If training without experience replay_memory, then you will interact with the environment 
        # in this function, while also updating your network parameters. 
        # If you are using a replay memory, you should interact with environment here, and store these 
        # transitions to memory, while also updating your model.
        cur_state = self.env.reset()
        epi = np.linspace( self.start_epi, self.end_epi, num = 100000 )
        _iter = 0
        is_term = False
        cur_episode = 1
        print( "Enter training" )
        # sample action and record state, action reward until termination
        while cur_episode < self.max_episode + 1:
            # take action according to epi-greedy
            # with episilon select random action

            if _iter < 100000:
                self.epi = epi[ _iter ]
            else:
                self.epi = 0.01

            cur_state = cur_state.reshape( ( 1, -1 ) )
            if random.random() < self.epi:
                action = self.env.action_space.sample()
            # else select action from q Net
            else:
                
                q_values = self.source_action_value.get_q_val( self.sess, cur_state )
                action = np.argmax( q_values.flatten() )

            next_state, reward, is_term, _ = self.env.step( action )
            # store transition into memory
            self.replay_memory.append( ( cur_state, action, reward, next_state, is_term ) )
            # sample random minibatch drom D
            sample_list = self.replay_memory.sample_batch( batch_size = self.sample_batch )
            # compile states into batch
            state_batch = []
            action_batch = []
            is_term_batch = []
            reward_batch = []
            next_state_batch = []

            for transitions in sample_list:
                [ cur, a, r, _next, _is_term ] = transitions
                state_batch.append( cur.flatten() )
                action_batch.append( a )
                reward_batch.append( r )
                is_term_batch.append( _is_term )
                next_state_batch.append( _next )

            # convert everything to np array
            state_batch = np.array( state_batch )
            action_batch = np.array( action_batch )
            reward_batch = np.array( reward_batch )
            is_term_batch = np.array( is_term_batch )
            next_state_batch = np.array( next_state_batch )

            q_val_next = self.source_action_value.get_q_val( self.sess, next_state_batch )
            q_next_max = np.amax( q_val_next, axis = 1 )
            # print("q next max", q_next_max.shape)
            # invert the is terminate to mask out terminated from loss being added
            is_term_mask = np.invert( is_term_batch ).astype( np.float )
            reward_batch = reward_batch + self.discount_factor * is_term_mask * q_next_max
            # print("reward_batch", reward_batch.shape )
            reward_batch = reward_batch.reshape( ( -1, 1 ) )
            # print("reward_batch", reward_batch.shape )
            # print( "state_batch", state_batch.shape, state_batch )
            action_batch = action_batch.reshape( ( -1, 1 ) )
            # gradient update
            loss = self.source_action_value.optimize( self.sess, state_batch, action_batch, reward_batch, self.learning_rate )
            loss_list.append( loss )
            cur_state = next_state

            _iter += 1

            # check if an episode ends according to env criteria
            if is_term:
                # print( "Position: ", cur_state[ 0 ] )
                # loss_list.append( cur_state[ 0 ] )
                # print( "Finish training episode {}".format( cur_episode ) )
                # save for every evaluate, evaluate if necessary
                if ( cur_episode ) % self.evaluate_every == 0:
                    self.save_model( episode_num = cur_episode )

                    self.learning_rate *= self.lr_decay

                # save loss and plot
                # loss = self.source_action_value.get_loss( self.sess, state_batch, action_batch, reward_batch ) 
                # loss_list.append( loss )
                # if cur_episode % 100 == 0:
                #     print( "Episode {}, loss {}, epi {}".format( cur_episode, loss, self.epi ) )
                    one, two = self.evaluate_total_reward()
                    print( " Episide {} one_step {} two_step {} ".format( cur_episode, one, two ) )
                    print( "----------------------------" )
                    print(" ")
                # reset environment and retrain for next episode
                cur_state = self.env.reset()
                # print( "Env reset" )
                is_term = False
                cur_episode += 1
                
        x = list( range( len( loss_list ) ) )
        plt.plot( x, loss_list )
        plt.title( "Average Loss of {}".format( self.environment_name ) )
        plt.show()


    # evaluate total rewards given current q network
    # evaluate in two different fashion: one_step look ahead( regular estimate ) and two step look ahead
    # use epi greedy
    def evaluate_total_reward( self, epi = 0.05 ):
        # evaluate based on value function ( one step look ahead )
        cur_state = self.env.reset()
        is_term = False
        reward_list = []
        # simulate the environment till is_term
        while not is_term:
            cur_state = np.array( cur_state ).reshape( ( 1, -1 ) )
            # epi greedy
            q_val_list = self.source_action_value.get_q_val( self.sess, cur_state )
            action = np.argmax( q_val_list.flatten() )
            [ next_state, reward, is_term, _ ] = self.env.step( action )
            reward_list.append( reward )
            cur_state = next_state
        one_step_total = np.sum( np.array( reward_list ) )

        two_step_total = 0
        # if cartpole, evaluate based on two steps look ahead
        if self.environment_name == "CartPole-v0":

            cur_state = self.env.reset()
            is_term = False
            reward_list = []
            # simulate the environment till is_term
            while not is_term:
                cur_state = np.array( cur_state ).reshape( ( 1, -1 ) )
                # choose action based on two steps away
                # pretending choosing 0, 1 as next step record average q val for each:
                qvals = np.zeros( 2, dtype = np.float )
                for choice in range( 2 ):
                    next_state, reward, is_term, _ = self.fake_step( choice )
                    next_state = next_state.reshape( ( 1, -1 ) )
                    qvals_res = self.source_action_value.get_q_val( self.sess, next_state )
                    qvals[ choice ] = reward + ( not is_term ) * self.discount_factor * np.amax( qvals_res )
                action = np.argmax( qvals )

                [ next_state, reward, is_term, _ ] = self.env.step( action )
                reward_list.append( reward )
                cur_state = next_state
            two_step_total = np.sum( np.array( reward_list ) )

        return one_step_total, two_step_total

    # call evaluate total award with all the given checkpoints
    def evaluate_wrapper( self, times_avg = 1 ):
        one_step_list, two_step_list = [], []
        for episode in range( self.evaluate_every, self.max_episode, self.evaluate_every ):
            self.load_model( os.path.join( self.env_dir, str( episode ) ) + "/" + "model.bin" )
            total_one, total_two = 0,0
            for _ in range( times_avg ):
            	one, two = self.evaluate_total_reward()
            	total_one += one
            	total_two += two
            one_step_list.append( total_one / times_avg )
            two_step_list.append( total_two  / times_avg )

        return one_step_list, two_step_list

    def test(self, model_file=None):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory. 
        self.load_model( model_file )
        reward = []
        for i in range( 100 ):
            reward_accu = 0
            cur_state = self.env.reset()
            is_term = False
            while not is_term:
                cur_state = cur_state.reshape( 1, -1 )
                q_val = self.source_action_value.get_q_val( None, cur_state )
                action = np.argmax( q_val )
                next_state, r, is_term, _ = self.env.step( action )
                reward_accu += r
                cur_state = next_state
            reward.append( reward_accu )
        reward = np.array( reward )
        print("reward", reward)
        mean, std =  np.mean( reward ), np.std( reward )
        print( "mean: {} std {}".format( mean, std ) )
        return mean, std

    def draw( self, model_file = None ):
        # self.load_model( model_file )
        print(model_file)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        
        # self.env.render()
        self.env = wrappers.Monitor(self.env, self.tmp_dir + "/monitor_1", force = True )
        cur_state = self.env.reset()
        is_term = False
        while not is_term:

            q_val = self.source_action_value.get_q_val( None, cur_state )
            action = np.argmax( q_val )
            next_state, r, is_term, _ = self.env.step( action )
            # reward_accu += reward
            cur_state = next_state
        # while not is_term:
        #     cur_state = np.array( cur_state ).reshape( ( 1, -1 ) )
        #     # choose action based on two steps away
        #     # pretending choosing 0, 1 as next step record average q val for each:
        #     qvals = np.zeros( 2, dtype = np.float )
        #     for choice in range( 2 ):
        #         next_state, reward, is_term, _ = self.fake_step( choice )
        #         next_state = next_state.reshape( ( 1, -1 ) )
        #         qvals_res = self.source_action_value.get_q_val( self.sess, next_state )
        #         qvals[ choice ] = reward + ( not is_term ) * self.discount_factor * np.amax( qvals_res )
        #     action = np.argmax( qvals )

        #     [ next_state, reward, is_term, _ ] = self.env.step( action )
        #     reward_list.append( reward )
        #     cur_state = next_state

    def burn_in_memory( self, burn_in_size ):
        # Initialize your replay memory with a burn_in number of episodes / transitions. 
        burn_in_num = 0
        cur_state = self.env.reset()
        is_term = False
        while burn_in_num < burn_in_size:
            if is_term:
                cur_state = self.env.reset()
                is_term = False
                continue
            action = self.env.action_space.sample()
            next_state, reward, is_term, _ = self.env.step( action )
            self.replay_memory.append( ( cur_state, action, reward, next_state, is_term ) )
            cur_state = next_state
            burn_in_num += 1
        self.env.reset()
        print( "Done with burn in, memory size: {}".format( burn_in_size ) )

    # return the new state after the action without altering the environment
    # used for two-step look ahead
    def fake_step( self, action ):
        if self.environment_name == "MountainCar-v0":
            assert self.env.env.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

            position, velocity = self.env.env.state
            velocity += (action-1)*0.001 + math.cos(3*position)*(-0.0025)
            velocity = np.clip(velocity, -self.env.env.max_speed, self.env.env.max_speed)
            position += velocity
            position = np.clip(position, self.env.env.min_position, self.env.env.max_position)
            if (position==self.env.env.min_position and velocity<0): velocity = 0

            done = bool(position >= self.env.env.goal_position)
            reward = -1.0

            new_state = [ position, velocity ]
            return np.array( new_state ), reward, done, {}
        elif self.environment_name == "CartPole-v0":
            assert self.env.env.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
            state = self.env.env.state
            x, x_dot, theta, theta_dot = state
            force = self.env.env.force_mag if action==1 else -self.env.env.force_mag
            costheta = math.cos(theta)
            sintheta = math.sin(theta)
            temp = (force + self.env.env.polemass_length * theta_dot * theta_dot * sintheta) / self.env.env.total_mass
            thetaacc = (self.env.env.gravity * sintheta - costheta* temp) / (self.env.env.length * (4.0/3.0 - self.env.env.masspole * costheta * costheta / self.env.env.total_mass))
            xacc  = temp - self.env.env.polemass_length * thetaacc * costheta / self.env.env.total_mass
            if self.env.env.kinematics_integrator == 'euler':
                x  = x + self.env.env.tau * x_dot
                x_dot = x_dot + self.env.env.tau * xacc
                theta = theta + self.env.env.tau * theta_dot
                theta_dot = theta_dot + self.env.env.tau * thetaacc
            else: # semi-implicit euler
                x_dot = x_dot + self.env.env.tau * xacc
                x  = x + self.env.env.tau * x_dot
                theta_dot = theta_dot + self.env.env.tau * thetaacc
                theta = theta + self.env.env.tau * theta_dot
            # self.env.env.state = (x,x_dot,theta,theta_dot)
            new_state = [ x, x_dot, theta, theta_dot ]
            done =  x < -self.env.env.x_threshold \
                    or x > self.env.env.x_threshold \
                    or theta < -self.env.env.theta_threshold_radians \
                    or theta > self.env.env.theta_threshold_radians
            done = bool(done)

            if not done:
                reward = 1.0
            elif self.env.env.steps_beyond_done is None:
                # Pole just fell!
                # self.env.env.steps_beyond_done = 0
                reward = 1.0
            else:
                if self.env.env.steps_beyond_done == 0:
                    logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
                # self.env.steps_beyond_done += 1
                reward = 0.0

            return np.array( new_state ), reward, done, {}

    def save_model(self, episode_num = 100 ):
        # Helper function to save your model / weights. 
        path = os.path.join( self.env_dir, str( episode_num ) ) + "/"
        os.mkdir( path )
        # self.saver.save( self.sess, path )
        # print( "Save model for {} at episode {}".format( self.environment_name, episode_num ) )
        cpt = dict()
        cpt[ "q_net" ] = self.source_action_value.network.state_dict()
        cpt[ "q_optim" ] = self.source_action_value.optim.state_dict()
        torch.save( cpt, path + "model.bin" )
        print( "Save model for {} at episode {}".format( self.environment_name, episode_num ) )


    def load_model(self, model_file):
        # Helper function to load an existing model.
        # self.saver.restore( self.sess, model_file)
        cpt = torch.load( model_file )
        self.source_action_value.network.load_state_dict( cpt[ "q_net" ] )
        self.source_action_value.optim.load_state_dict( cpt[ "q_optim" ] )

    def load_model_weights(self,weight_file):
        # Helper funciton to load model weights. 
        pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str,default = "CartPole-v0")
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model',dest='model_file',type=str, default='')
    return parser.parse_args()

def main(args):

    args = parse_arguments()
    environment_name = args.env

    # You want to create an instance of the DQN_Agent class here, and then train / test it. 
    if environment_name == "CartPole-v0":
        agent = DQN_Agent( args.env , 100, render = args.render )
    else:
        agent = DQN_Agent( args.env , 64, render = args.render )
    if args.train:
        agent.train()
    elif args.model_file != "":
        if args.render:
            agent = DQN_Agent( args.env , 64, render = args.render )
            print("before draw\n\n\n\n\n\n\n")
            agent.draw( args.model_file )
        else:
            agent = DQN_Agent( args.env , 64, render = args.render )
            agent.test( args.model_file )
    else:

        one_step_list, two_step_list = agent.evaluate_wrapper()
        x = np.arange( len( one_step_list ) )
        fig1 = plt.figure()
        
        p1 = plt.plot( x, np.array( one_step_list ), label = "one step" )
        if args.env == "CartPole-v0":
            p2 = plt.plot( x, np.array( two_step_list ), label = "two step" )
            plt.legend((p1[0], p2[0]), ('one_step', 'two_step'))
        plt.xlabel( "number of episodes in terms of {}".format( agent.evaluate_every ) )
        plt.ylabel( "reward" )
        plt.title( "Training Reward {}".format( args.env ) )
        plt.show()

if __name__ == '__main__':
    main(sys.argv)
