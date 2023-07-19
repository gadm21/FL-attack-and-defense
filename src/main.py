




from utils import * 
from attack_utils import *
from Fed import * 

import json 




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
        history = train_keras_model(centralized_model, train_data, test_data, epochs=args.rounds, batch_size = args.batch_size, verbose=1, **callbacks)


    # ___________________________________________________________________________________________________
    elif args.learning_algorithm == 'local' :
        print("Running local training")

        centralized_data, clients_data, external_data = split_data(train_data, args.num_clients, args.local_size)
        for client_id in range(args.num_clients) :
            client_model = create_model_based_on_data(args, compile_model = False) 
            client_model = compile_model(client_model, args)
            callbacks = {
                'early_stop_patience' : args.early_stop_patience,
                'lr_reduction_patience' : args.lr_reduction_patience,
                'csv_logger_path' : join(experiment_dir, f'client_{client_id}.csv')
            }
            history = train_keras_model(client_model, clients_data[client_id], test_data, epochs=args.rounds, batch_size = args.batch_size, verbose=0, **callbacks)
        
    # ___________________________________________________________________________________________________
    elif 'fed' in args.learning_algorithm :

        centralized_data, clients_data, external_data = split_data(train_data, args.num_clients, args.local_size)
        if args.learning_algorithm == 'fedavg' :
            initial_model = create_model_based_on_data(args)
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

        else : 
            raise ValueError('Invalid learning algorithm')
        
        learning_algorithm.run(args.rounds, args.local_epochs )
        learning_algorithm.save_scores() 

    
    # ___________________________________________________________________________________________________
    else :
        raise ValueError('Invalid learning algorithm')

    with open(join(experiment_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f)



if __name__ == "__main__" : 

    parser = argparse.ArgumentParser()

    parser.add_argument('--id', default = None, help='Experiment ID')  # Optional id argument
    parser.add_argument('dataset', metavar='DATASET', help='Specify the dataset')  # Mandatory dataset argument
    parser.add_argument('--shadow_dataset', default = 'same', help='Specify the shadow dataset')  # Optional shadow_dataset argument

    parser.add_argument('--learning_algorithm', default='local', help='central, local, fedavg, fedmd, fedakd')  # Optional learning_algorithm argument
    parser.add_argument('--proxy_data_size', type = int, default=1000, help='Number of epochs') # Optional epochs argument
    parser.add_argument('--num_clients', type = int, default=5, help='Number of clients participating in FL')  # Optional num_clients argument

    parser.add_argument('--local_size', type = int, default=500, help='size of data for each client')  # Optional num_clients argument
    parser.add_argument('--batch_size', type = int, default=250, help='Batch size')  # Optional num_clients argument
    parser.add_argument('--rounds', type = int, default=30, help='Number of global') # Optional rounds argument
    parser.add_argument('--local_epochs', type = int, default=15, help='Number of epochs') # Optional epochs argument
    parser.add_argument('--lr', type = float, default=0.001, help='Learning rate') # Optional learning rate argument

    parser.add_argument('--early_stop_patience', type = int, default=-1, help='Patience of Early stopping callback') # early stopping patience
    parser.add_argument('--lr_reduction_patience', type = int, default=-1, help='Patience of lr reduction callback') # lr reduction patience
    
    parser.add_argument('--target_model', default='nn', help='Specify the target model')  # Optional target_model argument
    parser.add_argument('--n_shadow', type = int, default=10, help='Number of shadow models')  # Optional num_clients argument
    parser.add_argument('--gamma', type = float, default=0.5, help='Gamma for MIA attack data split')  # works as target_train:shadow_train ratio and target_train:shadow_test ratio
    parser.add_argument('--target_size', type = int, default=20_000, help='Target model data size')  # Data size for target model

    parser.add_argument('--use_dp', dest='use_dp', action='store_true')
    parser.add_argument('--dp_epsilon', type = float, default=0.5, help='Privacy budget')  # Optional target_model argument
    parser.add_argument('--dp_delta', type = float, default=1e-5, help='Privacy budget')  # Optional target_model argument
    parser.add_argument('--dp_norm_clip', type = float, default=3, help='Privacy budget')  # Optional target_model argument
    parser.add_argument('--dp_type', type = str, default='dp', help='DP variation')  # Optional target_model argument
    

    args = parser.parse_args()

    # Shokri_MIA(args) 
    # train_attack_model(args) 

    experiment_id =  args.dataset + '_' + args.learning_algorithm + '_' + str(args.use_dp) + '_' + datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    
    print("Running experiment " + experiment_id) 
    print("Arguments: " + str(args)) 
    
    run_experiment(experiment_id, args) 

    print("Done experiment " + experiment_id )



