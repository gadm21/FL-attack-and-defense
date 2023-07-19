

import argparse

from utils import *
from Fed import *
from main import update_args_with_dict

# Create an instance of argparse.Namespace
args = argparse.Namespace()

# Assign values to the attributes
args.dataset = 'cifar10'
args.learning_algorithm = 'fedavg'
args.proxy_data_size = 1000
args.num_clients = 5
args.local_size = 500
args.batch_size = 32
args.rounds = 30
args.local_epochs = 1
args.lr = 0.001
args.early_stop_patience = 10
args.lr_reduction_patience = 10
args.target_model = 'nn'
args.use_dp = True
args.dp_epsilon = 0.5
args.dp_delta = 1e-5
args.dp_norm_clip = 1.0
args.dp_type = 'dp'


experiment_id = args.dataset + '_' + args.learning_algorithm + '_DP' + str(args.use_dp) + '_' + datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
experiment_dir = join(RESULTS_PATH, experiment_id)
if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)

train_data, test_data, metadata = get_data(args.dataset)
args = update_args_with_dict(args, metadata)

train_data = (np.array(train_data[0] / 255, dtype=np.float32), tf.keras.utils.to_categorical(train_data[1]))
test_data = (np.array(test_data[0] / 255, dtype=np.float32), tf.keras.utils.to_categorical(test_data[1]))


centralized_data, clients_data, external_data = split_data(train_data, args.num_clients, args.local_size)

initial_model = create_model_based_on_data(args)
learning_algorithm = FedAvg(exp_path = experiment_dir,
                                clients_data = clients_data,
                                test_data = test_data, 
                                initial_model = initial_model, 
                                args = args)


learning_algorithm.run(rounds = 1) 