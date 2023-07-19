


import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

from collections import Counter


from utils import * 

# To avoid numerical inconsistency in calculating log
SMALL_VALUE = 1e-6

def log_loss(a, b):
	return [-np.log(max(b[i,int(a[i])], SMALL_VALUE)) for i in range(len(a))]



def generate_target_and_shadow_data(args, save_to = None) : 

    gamma = args.gamma
    target_size = args.target_size

    (x, y), (x_test, y_test), metadata = get_data(args.dataset)   

    x, y = np.array(x / 255, dtype=np.float32), tf.keras.utils.to_categorical(y)
    x_test, y_test = np.array(x_test / 255, dtype=np.float32), tf.keras.utils.to_categorical(y_test)
    

    print("x shape: ", x.shape)
    # assert if data is enough for sampling target data
    assert(len(x) >= (1 + gamma) * target_size)
    # the size of (train_x, train_y) is target_size and (x, y) is the remaining size
    x, train_x, y, train_y = train_test_split(x, y, test_size=target_size, stratify=y)
    print("Training set size:  X: {}, y: {}".format(train_x.shape, train_y.shape))
    # The size of (test_x, test_y) is gamma * target_size and (x, y) is the remaining size
    x, test_x, y, test_y = train_test_split(x, y, test_size=int(gamma*target_size), stratify=y)
    print("Test set size:  X: {}, y: {}".format(test_x.shape, test_y.shape))

    # save target data
    if save_to is not None:
        np.savez(join(save_to, 'target_data.npz'), train_x, train_y, test_x, test_y)
        target_data = (train_x, test_x, train_y, test_y)

    if args.shadow_dataset == 'same':
        # assert if remaining data is enough for sampling shadow data
        assert(len(x) >= (1 + gamma) * target_size)
    else :
        x, y, metadata = get_data(args.shadow_dataset)
         

    # generate shadow data
    shadow_data = []
    for i in range(args.n_shadow):
        print('Generating data for shadow model {}'.format(i))
        strain_x, stest_x, strain_y, stest_y = train_test_split(x, y, train_size=target_size, test_size=int(gamma*target_size), stratify=y)
        print("Training set size:  X: {}, y: {}".format(train_x.shape, train_y.shape))
        print("Test set size:  X: {}, y: {}".format(test_x.shape, test_y.shape))
        shadow_data.append((strain_x, stest_x, strain_y, stest_y))
        if save_to is not None:
            np.savez(join(save_to, 'shadow{}_data.npz'.format(i)), strain_x, stest_x, strain_y, stest_y)
    return target_data, shadow_data, metadata


def generate_attack_data(model, dataset, save_to = None, test_data = False) : 

    train_x, test_x, train_y, test_y = dataset 
    attack_x, attack_y = [], []

    pred_scores = model(train_x)
    attack_x.append(pred_scores)
    attack_y.append(np.ones(train_y.shape[0]))    
    
    print("train x shape:", train_x.shape)
    print("test x shape:", test_x.shape)
    print("train y shape:", train_y.shape)
    print("test y shape:", test_y.shape)

    pred_scores = model(test_x)
    attack_x.append(pred_scores)
    attack_y.append(np.zeros(test_y.shape[0]))

    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')

    
    print("train y shape:", train_y.shape)
    print("test y shape:", test_y.shape)
    print("argmax train y shape:", np.argmax(train_y).shape)
    print("argmax test y shape:", np.argmax(test_y).shape)
    classes = np.concatenate([np.argmax(train_y, axis = -1), np.argmax(test_y, axis = -1)])
    classes = classes.astype('int32')
    

    if save_to is not None:
        if test_data : 
            data_name = 'attack_data_test.npz'
            classes_name = 'classes_test.npz'
        else :
            num_files = len(os.listdir(save_to))
            data_name = 'attack_data_{}.npz'.format(num_files)
            classes_name = 'classes_{}.npz'.format(num_files)

        np.savez(join(save_to, data_name), attack_x, attack_y)
        np.savez(join(save_to, classes_name), classes)

    return attack_x, attack_y, classes


def load_attack_data(data_dir, args):
    
    data_files = [join(data_dir, f) for f in os.listdir(data_dir)]
    # find the index of and remove the element that name contains 'test'
    classes_data_test = join(data_dir, 'classes_test.npz')
    attack_data_test = join(data_dir, 'attack_data_test.npz')
    

    train_x, train_y, train_c = [], [], []
    for i in range(0, (len(data_files) - 2), 2):
        data_file = join(data_dir, 'attack_data_{}.npz'.format(i))
        class_file = join(data_dir, 'classes_{}.npz'.format(i))
        if not os.path.exists(data_file) or not os.path.exists(class_file):
            continue

        d = np.load(data_file)
        train_classes = np.load(class_file) 
        
        x, y = d['arr_0'], d['arr_1']
        train_c.append(train_classes['arr_0']) 
        train_x.append(x)
        train_y.append(y) 


    train_x = np.vstack(train_x)
    train_y = np.concatenate(train_y)
    train_c = np.concatenate(train_c)

    test_data = np.load(attack_data_test)
    test_x, test_y = test_data['arr_0'], test_data['arr_1']

    test_classes = np.load(classes_data_test) 
    test_c = test_classes['arr_0']
    
    print("train_x shape:", train_x.shape)
    print("train_y shape:", train_y.shape)
    print("train_c shape:", train_c.shape)

    print("test_x shape:", test_x.shape)
    print("test_y shape:", test_y.shape)
    print("test_c shape:", test_c.shape)

    return train_x, train_y, test_x, test_y, (train_c, test_c)


def get_attack_model(args) : 
    n_in = args.n_classes 
    n_out = 2 # membership or not
    # input_layer = tf.reshape(features['x'], [-1, n_in])
    # logits = tf.keras.layers.Dense(n_out, activation=tf.nn.softmax)(input_layer)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(n_out, activation=tf.nn.softmax, input_shape=(n_in,))
    ])
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model


def train_attack_model(args):
    """
    Wrapper function to train the meta-model over the shadow models' output.
    During inference time, the meta-model takes the target model's output and 
    predicts if a query record is part of the target model's training set.
    """
    
    exp_id = 'exp_{}'.format(args.id)
    EXP_PATH = join(DATA_PATH, exp_id)
    attack_data_dir = join(EXP_PATH, 'attack_data')
    train_x, train_y, test_x, test_y, classes = load_attack_data(attack_data_dir, update_args_with_dict)

    train_classes, test_classes = classes
    train_indices = np.arange(len(train_x))
    test_indices = np.arange(len(test_x))
    unique_classes = np.unique(train_classes)
    args.n_classes = len(unique_classes)

    pred_y = []
    shadow_membership, target_membership = [], []
    shadow_pred_scores, target_pred_scores = [], []
    shadow_class_labels, target_class_labels = [], []
    attack_models = [] 
    for c in unique_classes:
        print('Training attack model for class {}...'.format(c))
        model = get_attack_model(args) 
        c_train_indices = train_indices[train_classes == c]
        c_train_x, c_train_y = train_x[c_train_indices], train_y[c_train_indices]
        c_test_indices = test_indices[test_classes == c]
        c_test_x, c_test_y = test_x[c_test_indices], test_y[c_test_indices]

        
        c_train = (c_train_x, tf.keras.utils.to_categorical(c_train_y))
        c_test = (c_test_x, tf.keras.utils.to_categorical(c_test_y))

        history = train_keras_model(model, train_data = c_train, test_data = c_test, epochs = args.local_epochs)
        
        c_pred_scores = model(c_train_x)
        c_pred_scores_pos = c_pred_scores[:, 1]
        shadow_membership.append(c_train_y)
        shadow_pred_scores.append(c_pred_scores_pos)
        shadow_class_labels.append([c]*len(c_train_indices))

        c_pred_scores = model(c_test_x)
        c_pred_scores_pos = c_pred_scores[:, 1]
        c_pred_y = tf.argmax(c_pred_scores, axis = 1)

        pred_y.append(c_pred_y)
        target_membership.append(c_test_y)
        target_pred_scores.append(c_pred_scores_pos)
        target_class_labels.append([c]*len(c_test_indices))

    print('-' * 10 + 'FINAL EVALUATION' + '-' * 10 + '\n')
    pred_y = np.concatenate(pred_y)
    shadow_membership = np.concatenate(shadow_membership)
    target_membership = np.concatenate(target_membership)
    shadow_pred_scores = np.concatenate(shadow_pred_scores)
    target_pred_scores = np.concatenate(target_pred_scores)
    shadow_class_labels = np.concatenate(shadow_class_labels)
    target_class_labels = np.concatenate(target_class_labels)
    # prety_print_result(target_membership, pred_y)
    print("target membership stats: ", Counter(target_membership))
    print("pred_y stats: ", Counter(pred_y))

    # fpr, tpr, thresholds = roc_curve(target_membership, pred_y, pos_label=1)
    fpr, tpr, thresholds = roc_curve(target_membership, target_pred_scores, pos_label=1)

    # save the results 
    results = {
        'fpr' : fpr,
        'tpr' : tpr,
        'thresholds' : thresholds,
    }
    results_dir = join(EXP_PATH, 'attack_results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        results_name = 'attack_results.npy'
        np.save(join(results_dir, results_name), results)
    

def Shokri_MIA(args) : 

    exp_id = 'exp_{}'.format(args.id)
    EXP_PATH = join(DATA_PATH, exp_id)
    target_shadow_data_dir = join(EXP_PATH, 'target_shadow_data')
    if not os.path.exists(target_shadow_data_dir) :
        os.makedirs(target_shadow_data_dir)

    target_data, shadow_data, metadata = generate_target_and_shadow_data(args, save_to = target_shadow_data_dir)
    t_train_x, t_test_x, t_train_y, t_test_y =  target_data

    args = update_args_with_dict(args, metadata)

    # train the target model
    target_model = create_model_based_on_data(args, compile_model = False)
    target_model = compile_model(target_model, args) 
    history = train_keras_model(target_model, train_data = (t_train_x, t_train_y), test_data = (t_test_x, t_test_y), epochs = args.local_epochs)

    # train the shadow models
    shadow_models = []
    for i in range(args.n_shadow) :
        shadow_model = create_model_based_on_data(args, compile_model = True) 
        
        s_train_x, s_test_x, s_train_y, s_test_y = shadow_data[i]
        print("shadow train x shape: ", s_train_x.shape)
        print("shadow train y shape: ", s_train_y.shape)
        print("shadow test x shape: ", s_test_x.shape)
        print("shadow test y shape: ", s_test_y.shape)

        history = train_keras_model(shadow_model, train_data = (s_train_x, s_train_y), test_data = (s_test_x, s_test_y), epochs = args.local_epochs)
        shadow_models.append(shadow_model)
    
    # generate the attack data
    # generate the attack training data 
    attack_data_dir = join(EXP_PATH, 'attack_data')
    if not os.path.exists(attack_data_dir):
        os.makedirs(attack_data_dir)
    for i, s_model in enumerate(shadow_models) : 
        attack_x, attack_y, classes = generate_attack_data(s_model, shadow_data[i], save_to = attack_data_dir, test_data = False) 
    generate_attack_data(target_model, target_data, save_to = attack_data_dir, test_data = True)




