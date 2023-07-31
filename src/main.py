




from utils import * 
from attack_utils import *
from Fed import * 

import json 
import time 

from itertools import product


def run_experiment(id, args) : 

    experiment_dir = join(RESULTS_PATH, id)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    train_data, test_data, metadata = get_data(args.dataset)
    args = update_args_with_dict(args, metadata)

    train_data = (np.array(train_data[0] / 255, dtype=np.float32), tf.keras.utils.to_categorical(train_data[1]))
    if args.learning_algorithm == 'fedakd' :
        proxy_limit = args.proxy_data_size
        proxy_data = train_data[0][:proxy_limit] 
        train_data = (train_data[0][proxy_limit:], train_data[1][proxy_limit:])
        
    test_data = (np.array(test_data[0] / 255, dtype=np.float32), tf.keras.utils.to_categorical(test_data[1]))


    # ___________________________________________________________________________________________________
    if args.learning_algorithm == 'central' :
        print("Running centralized training")

        centralized_model = create_model_based_on_data(args, compile_model = False) 
        centralized_model = compile_model(centralized_model, args)
        callbacks = {
            'early_stop_patience' : args.early_stop_patience,
            'lr_reduction_patience' : args.lr_reduction_patience,
            'csv_logger_path' : join(experiment_dir, 'centralized.csv')
        }
        t0 = time.time()
        history = train_keras_model(centralized_model, train_data, test_data, epochs=args.rounds, batch_size = args.batch_size, verbose=1, **callbacks)
        t1 = time.time()
        print("Central time taken: " + str(t1-t0) + " seconds")

        loss_fnn = tf.keras.losses.CategoricalCrossentropy(reduction = 'none')
        train_preds, train_losses = model_stats(centralized_model, train_data[0], train_data[1], loss_fnn)
        test_preds, test_losses = model_stats(centralized_model, test_data[0], test_data[1], loss_fnn)

        # save model 
        model_path = join(experiment_dir, 'model.h5')
        centralized_model.save(model_path)

        # save stats
        train_preds_path = join(experiment_dir, 'train_preds.npy')
        train_losses_path = join(experiment_dir, 'train_losses.npy')
        test_preds_path = join(experiment_dir, 'test_preds.npy')
        test_losses_path = join(experiment_dir, 'test_losses.npy')
        np.save(train_preds_path, train_preds)
        np.save(train_losses_path, train_losses)
        np.save(test_preds_path, test_preds)
        np.save(test_losses_path, test_losses)
        
    # ___________________________________________________________________________________________________
    elif args.learning_algorithm == 'local' :
        print("Running local training")

        centralized_data, clients_data, external_data, p = split_data(train_data, args.num_clients, args.local_size)
        for client_id in range(args.num_clients) :
            client_model = create_model_based_on_data(args, compile_model = False) 
            client_model = compile_model(client_model, args)
            callbacks = {
                'early_stop_patience' : args.early_stop_patience,
                'lr_reduction_patience' : args.lr_reduction_patience,
                'csv_logger_path' : join(experiment_dir, f'client_{client_id}.csv')
            }
            history = train_keras_model(client_model, clients_data[client_id], test_data, epochs=args.rounds, batch_size = args.batch_size, verbose=0, **callbacks)

            loss_fnn = tf.keras.losses.CategoricalCrossentropy(reduction = 'none')
            train_preds, train_losses = model_stats(centralized_model, train_data[0], train_data[1], loss_fnn)
            test_preds, test_losses = model_stats(centralized_model, test_data[0], test_data[1], loss_fnn)
            client_path = join(experiment_dir, f'client_{client_id}.h5')
            # save model 
            model_path = join(client_path, 'model.h5')
            centralized_model.save(model_path)

            # save stats
            train_preds_path = join(client_path, 'train_preds.npy')
            train_losses_path = join(client_path, 'train_losses.npy')
            test_preds_path = join(client_path, 'test_preds.npy')
            test_losses_path = join(client_path, 'test_losses.npy')
            np.save(train_preds_path, train_preds)
            np.save(train_losses_path, train_losses)
            np.save(test_preds_path, test_preds)
            np.save(test_losses_path, test_losses)
        perm_path = join(experiment_dir, 'perm.npy')
        np.save(perm_path, p)
        
    # ___________________________________________________________________________________________________
    elif 'fed' in args.learning_algorithm :

        centralized_data, clients_data, external_data, p = split_data(train_data, args.num_clients, args.local_size)
        args.permutation = p

        if args.learning_algorithm == 'fedavg' :
            initial_model = create_model_based_on_data(args, compile_model=False)
            learning_algorithm = FedAvg(exp_path = experiment_dir,
                                         clients_data = clients_data,
                                          test_data = test_data, 
                                          initial_model = initial_model, 
                                          args = args)

        elif args.learning_algorithm == 'fedprox' :
            initial_model = create_model_based_on_data(args, compile_model = False)
            params = {'mu': 0.2}
            args = update_args_with_dict(args, params)
            learning_algorithm = FedProx(exp_path = experiment_dir,
                                         clients_data = clients_data,
                                          test_data = test_data,
                                           initial_model = initial_model,
                                            args = args)

        elif args.learning_algorithm == 'fedsgd' : 
            initial_model = create_model_based_on_data(args, compile_model = False)
            learning_algorithm = FedSGD(exp_path = experiment_dir,
                                        clients_data = clients_data,
                                        test_data = test_data,
                                        initial_model = initial_model, 
                                        args = args)

        elif args.learning_algorithm == 'fedakd' : 
            params = {
                'temperature' : 0.7,
                'aalpha' : 1000, 
                'bbeta' : 1000
            }
            args = update_args_with_dict(args, params)
            learning_algorithm = FedAKD(exp_path = experiment_dir, 
                                        clients_data = clients_data,
                                        test_data = test_data,
                                        proxy_data = proxy_data,
                                        clients_model_fn = create_model_based_on_data,
                                        args = args)
            
            t0 = time.time() 
            learning_algorithm.run(args.rounds, args.local_epochs )
            t1 = time.time()
            print("{args.learning_algorithm} Time taken: " + str(t1-t0) + " seconds")
            print("Average time: " + str((t1-t0)/args.rounds) + " seconds")
            learning_algorithm.save_scores() 

        else : 
            raise ValueError('Invalid learning algorithm')

    
    # ___________________________________________________________________________________________________
    else :
        raise ValueError('Invalid learning algorithm')

    with open(join(experiment_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f)



def run_path1(args) : 

    learning_algorithms = [] 
    if args.learning_algorithm == 'central' :
        learning_algorithms = ['central', 'local']
    else : 
        learning_algorithms = ['fedavg', 'fedakd', 'fedsgd']

    if not args.use_dp :
        for learning_algorithm in learning_algorithms : 
            args.learning_algorithm = learning_algorithm
            experiment_id =  args.dataset + '_' + args.learning_algorithm + '_' + str(args.use_dp) + '_' + datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            
            print("Running experiment " + experiment_id) 
            print("Arguments: " + str(args)) 
            run_experiment(experiment_id, args) 

            print("Done experiment " + experiment_id )
    else : 
        dp_types = ['dp', 'adv_cmp', 'rdp']
        dp_epsilons = [0.1, 1, 10, 100, 1000]
        for dp_type in dp_types :
            for ep in dp_epsilons :
                for learning_algorithm in learning_algorithms : 
                    args.learning_algorithm = learning_algorithm
                    experiment_id =  args.dataset + '_' + args.learning_algorithm + '_' + str(args.use_dp) + '_' + datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
                    
                    args.dp_epsilon = ep
                    args.dp_type = dp_type
                    print("Running experiment " + experiment_id) 
                    print("Arguments: " + str(args)) 
                    run_experiment(experiment_id, args) 

                    print("Done experiment " + experiment_id )


def run_path2(args) :

    learning_algorithms = ['central', 'local', 'fedavg', 'fedakd', 'fedsgd']
    datasets = ['cifar10']
    local_sizes = [14, 200, 2000]
    local_epochss = [1, 2, 5] 
    
    epsilons = [1000, 100, 10, 1, 0.1]
    dp_norm_clips = [1, 2, 3]
    dp_types = ['dp', 'adv_cmp', 'rdp', 'zcdp']
    lrs = [0.1, 0.01]

    combinations = product(epsilons, learning_algorithms, datasets, local_sizes, local_epochss, dp_norm_clips, dp_types, lrs) 
    for combination in combinations :
        ep, la, ds, ls, le, dnc, dt, lr = combination
        args.learning_algorithm = la
        args.dataset = ds
        args.local_size = ls
        args.local_epochs = le
        args.dp_epsilon = ep
        args.dp_norm_clip = dnc
        args.dp_type = dt
        args.lr = lr

        experiment_id =  args.dataset + '_' + args.learning_algorithm + '_' + str(args.use_dp) + '_' + datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        print("Running experiment " + experiment_id)
        print("Arguments: " + str(args))
        run_experiment(experiment_id, args)

        print("Done experiment " + experiment_id)







if __name__ == "__main__" : 

    parser = argparse.ArgumentParser()

    parser.add_argument('--id', default = None, help='Experiment ID')  # Optional id argument
    parser.add_argument('--dataset', metavar='DATASET', help='Specify the dataset')  # Mandatory dataset argument
    parser.add_argument('--shadow_dataset', default = 'same', help='Specify the shadow dataset')  # Optional shadow_dataset argument

    parser.add_argument('--learning_algorithm', default='local', help='central, local, fedavg, fedmd, fedakd')  # Optional learning_algorithm argument
    parser.add_argument('--proxy_data_size', type = int, default=1000, help='Number of epochs') # Optional epochs argument
    parser.add_argument('--num_clients', type = int, default=10, help='Number of clients participating in FL')  # Optional num_clients argument

    parser.add_argument('--local_size', type = int, default=2000, help='size of data for each client')  # Optional num_clients argument
    parser.add_argument('--batch_size', type = int, default=128, help='Batch size')  # Optional num_clients argument
    parser.add_argument('--rounds', type = int, default=150, help='Number of global') # Optional rounds argument
    parser.add_argument('--local_epochs', type = int, default=1, help='Number of epochs') # Optional epochs argument
    parser.add_argument('--lr', type = float, default=0.1, help='Learning rate') # Optional learning rate argument

    parser.add_argument('--early_stop_patience', type = int, default=-1, help='Patience of Early stopping callback') # early stopping patience
    parser.add_argument('--lr_reduction_patience', type = int, default=-1, help='Patience of lr reduction callback') # lr reduction patience
    
    parser.add_argument('--target_model', default='nn', help='Specify the target model')  # Optional target_model argument
    parser.add_argument('--n_shadow', type = int, default=10, help='Number of shadow models')  # Optional num_clients argument
    parser.add_argument('--gamma', type = float, default=0.5, help='Gamma for MIA attack data split')  # works as target_train:shadow_train ratio and target_train:shadow_test ratio
    parser.add_argument('--target_size', type = int, default=20_000, help='Target model data size')  # Data size for target model

    parser.add_argument('--use_dp', dest='use_dp', action='store_true')
    parser.add_argument('--dp_epsilon', type = float, default=0.5, help='Privacy budget')  # Optional target_model argument
    parser.add_argument('--dp_delta', type = float, default=1e-5, help='Privacy budget')  # Optional target_model argument
    parser.add_argument('--dp_norm_clip', type = float, default=1.5, help='Privacy budget')  # Optional target_model argument
    parser.add_argument('--dp_type', type = str, default='dp', help='DP variation')  # Optional target_model argument
    

    args = parser.parse_args()

    # Shokri_MIA(args) 
    # train_attack_model(args) 
    run_path2(args)
