import os
import numpy as np
from data_loader import read_data_sets
from convnet import ConvNet
from utils import Trainer

# working directory
workingdir = os.getcwd()

# optimization parameters
nepochs = 11  # 31
name_opt = 'adam'  # adam
momentum = 0.9
lr = 1e-3  # learning rate
decay_rate = 0.1  # decay learning rate by x
decay_after_epoch = 10  # decay learning rate after x epochs
batch_size = 64  # batch size
dropout = 0.9  # dropout rate

# configuration
cost_name = 'cross_entropy'  # default
subsets = False  # default
use_mask = False  # default
compute_uncertainty = False  # default
np_seed = 0  # numpy seed for the sampling
anti_curriculum = True  # default
# limited data
perc = 100  # set percentage of training data to be used
# noisy labels
corrupt_labels = False  # set corrupt_labels to True for experiments with noisy labels, by default is 30% - this can be modified in "data_loader" through "percentage_corrupted_labels"
# class imbalance
unbalance = False  # set unbalance to True for experiments with class-imbalance
unbalance_dict = None
# unbalance_dict = {'percentage': 30, 'label1': 1, 'label2': 7}  # training data for 'label1' and 'label2' is reduced to 'percentage'

strategy = 'baseline'  # ['baseline', 'reorder', 'subsets', 'weights']
curriculum_type = 'prior_knowledge'  # ['uncertainty', 'uncertainty']

if strategy == 'baseline':
    use_curriculum = False
    compute_uncertainty = False
    modeldir = os.path.join('./models/', str(perc), strategy)

elif strategy == 'subsets':
    subsets = True
    if anti_curriculum:
        modeldir = os.path.join('./models/', str(perc), strategy, 'anti-'+curriculum_type)
    else:
        modeldir = os.path.join('./models/', str(perc), strategy, curriculum_type)

elif strategy == 'weights':
    cost_name = 'weights'
    if anti_curriculum:
        modeldir = os.path.join('./models/', str(perc), strategy, 'anti-'+curriculum_type)
    else:
        modeldir = os.path.join('./models/', str(perc), strategy, curriculum_type)

else:
    if anti_curriculum:
        modeldir = os.path.join('./models/', str(perc), strategy, 'anti-'+curriculum_type)
    else:
        modeldir = os.path.join('./models/', str(perc), strategy, curriculum_type)

if curriculum_type == 'uncertainty':
    compute_uncertainty = True
    init_probs = []
else:
    init_probs = [7, 10, 5, 4, 9, 1, 8, 6, 2, 3]  # ranking to assign initial probabilities for each class
    if anti_curriculum:
        init_probs = (11 - np.array(init_probs)).tolist()

if corrupt_labels:
    modeldir = os.path.join('./models/', 'noise', str(perc), strategy, curriculum_type)
elif unbalance:
    modeldir = os.path.join('./models/', 'unbalance', str(unbalance_dict['percentage']), strategy, curriculum_type)

print(modeldir)

# load data
datadir = os.path.join(os.getcwd(), './data/mnist')  # data directory
data_provider = read_data_sets(datadir,  init_probs=init_probs, subsets=subsets,
                               corrupt_labels=corrupt_labels,
                               unbalance=unbalance, unbalance_dict=unbalance_dict,
                               percentage_train=perc/100.0)
n_train = data_provider.train.num_examples
print('Number of training images {:d}'.format(n_train))
# more training parameters
iters_per_epoch = np.ceil(1.0 * n_train / batch_size).astype(np.int32)
decay_steps = decay_after_epoch * iters_per_epoch
opt_kwargs = dict(learning_rate=lr, decay_steps=decay_steps, decay_rate=decay_rate)

# definition of the network
net = ConvNet(channels=1, n_class=10, is_training=True, use_mask=use_mask, cost_name=cost_name)

# definition of the trainer
trainer = Trainer(net, optimizer=name_opt, batch_size=batch_size, opt_kwargs=opt_kwargs)

# start training
path = trainer.train(data_provider, modeldir, training_iters=iters_per_epoch, epochs=nepochs, np_seed=np_seed,
                     dropout=dropout, compute_uncertainty=compute_uncertainty, anti_curriculum=anti_curriculum,
                     strategy=strategy)

print('Optimization Finished!')
