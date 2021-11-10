import numpy as np
import numpy
import gzip
from sklearn.model_selection import train_test_split
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.platform import gfile
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

DEFAULT_SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

    :param f: A file object that can be passed into a gzip reader.
    :returns data: A 4D uint8 numpy array [index, y, x, depth].
    :raises ValueError: If the bytestream does not start with 2051.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
          raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                           (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
    return data


def dense_to_one_hot(labels_dense, n_class):
    """Convert class labels from scalars to one-hot vectors.

    :param labels_dense: dense labels
    :param n_class: number of classes
    """
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * n_class
    labels_one_hot = numpy.zeros((num_labels, n_class))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(f, one_hot=False, num_classes=10):
    """Extract the labels into a 1D uint8 numpy array [index].

    :param f: a file object that can be passed into a gzip reader.
    :param one_hot: Does one hot encoding for the result.
    :param: num_classes: Number of classes for the one hot encoding.
    :returns labels: a 1D uint8 numpy array.
    :raises ValueError: If the bystream doesn't start with 2049.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %(magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        if one_hot:
            return dense_to_one_hot(labels, num_classes)
        return labels


class DataSet(object):
    def __init__(self, images, labels, probs, indices, subsets, fake_data=False, one_hot=False, reshape=True):
        """Construct a DataSet. one_hot arg is used only if fake_data is true.
        :param images: images
        :param labels: labels
        :param probs: curriculum probabilities
        :param fake_data (optional):
        :param: one_hot (optional):
        :param reshape (optional):
        """

        if fake_data:
            self._num_examples = images.shape[0]
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
            self._num_examples = images.shape[0]

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            if reshape:
                assert images.shape[3] == 1
                images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])

            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)

        self._images = images
        self._labels = labels
        self._probs = probs
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._counter = np.zeros(self._num_examples)
        self._init_probs = probs
        self._indices = indices

        if subsets:
            self._subset_size = int(0.25*self._num_examples)
            self._subset_size_no = int(0.25*self._num_examples)
            ii = np.arange(self._num_examples)  # indices [0, 1, ..., num_examples]
            # copy & normalize
            pp = np.squeeze(normalize(np.expand_dims(np.copy(self._probs), 1), axis=0, norm='l1'))
            # clip values
            pp = np.clip(pp, np.finfo(np.float32).eps, np.finfo(np.float32).max)
            # normalize again
            pp = np.squeeze(normalize(np.expand_dims(pp, 1), axis=0, norm='l1'))
            self._probs = pp
            # random choice (according to prob.) instead of random shuffle
            self._subset_ids = np.random.choice(ii, self._subset_size, False, pp)  ## selected subset ids
            # pool ids contain the indices of the samples that were not yet selected
            self._pool_ids = np.asarray(list((set(ii.tolist()) - set(self._subset_ids.tolist()))))  ## pool ids = all training set - subset set
            #self._probs = self._init_probs[self._pool_ids]
        else:
            self._subset_size = self._num_examples
            self._subset_size_no = int(0.25*self._num_examples)
            ii = np.arange(self._num_examples)  # indices [0, 1, ..., num_examples]
            # copy & normalize
            pp = np.squeeze(normalize(np.expand_dims(np.copy(self._probs), 1), axis=0, norm='l1'))
            # clip values
            pp = np.clip(pp, np.finfo(np.float32).eps, np.finfo(np.float32).max)
            # normalize again
            pp = np.squeeze(normalize(np.expand_dims(pp, 1), axis=0, norm='l1'))
            self._probs = pp
            # random choice (according to prob.) instead of random shuffle
            self._subset_ids = np.random.choice(ii, self._subset_size, False, pp)  ## selected subset ids
            # pool ids contain the indices of the samples that were not yet selected
            self._pool_ids = np.asarray(
                list((set(ii.tolist()) - set(self._subset_ids.tolist()))))  ## pool ids = all training set - subset set

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def probs(self):
        return self._probs

    @property
    def indices(self):
        return self._indices

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def counter(self):
        return self._counter

    @property
    def subset_size(self):
        return self._subset_size

    @property
    def subset_ids(self):
        return self._subset_ids

    @property
    def subset_size_no(self):
        return self._subset_size_no

    def next_batch(self, batch_size, fake_data=False):
        """
        Returns the next `batch_size` examples from this data set.

        :param batch_size: batch size
        :param fake_data (optional): flag to indicate whether data should be reshaped
        """
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in range(batch_size)], [fake_label for _ in range(batch_size)]

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1

            # Shuffle the data
            np.random.seed(0)
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            self._probs = self._probs[perm]
            self._indices = self._indices[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end], self._probs[start:end], self._indices[start:end]

    # for experiments with REORDER all training set
    def next_batch_probs(self, batch_size, np_seed, decay=True, replacement=False, fake_data=False):
        """
        At the beginning of each epoch, the training set is ordered according to the curriculum probabilities.
        Returns the next `batch_size` examples from the ordered data set.

        :param batch_size: batch size
        :param np_seed: numpy seed for the sampling
        :param decay: flag to decay probabilities towards a uniform distribution
        :param fake_data: flag to indicate whether data should be reshaped
        """
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in range(batch_size)], [fake_label for _ in range(batch_size)]

        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        # 1st sampling of the dataset when the optimization starts, epoch_completed == 0
        if start == 0 and self._epochs_completed == 0:

            # seed randomness of numpy
            np.random.seed(np_seed)

            # copy & normalize
            pp = np.squeeze(normalize(np.expand_dims(np.copy(self._probs), 1), axis=0, norm='l1'))
            # clip values
            pp = np.clip(pp, np.finfo(np.float32).eps, np.finfo(np.float32).max)
            # normalize again
            pp = np.squeeze(normalize(np.expand_dims(pp, 1), axis=0, norm='l1'))
            self._probs = pp

            ii = np.arange(self._num_examples)  # indices [0, 1, ..., num_examples]
            self._subset_ids = np.random.choice(ii, self._subset_size, replacement, pp)

            # increase counter of selected images
            if replacement:
                bbins = np.bincount(self._subset_ids)   # when sampling with replacement
                self._counter[:len(bbins)] += bbins
            else:
                self._counter[self._subset_ids] += 1   # when sampling without replacement

            # update probabilities according to counter
            if decay:
                pp = self._probs * np.exp(-self._counter ** 2 / 10)
            else:
                pp = self._probs
            # normalize so probs add up to 1
            self._probs = np.squeeze(normalize(np.expand_dims(pp, 1), axis=0, norm='l1'))

        if self._index_in_epoch > self._subset_size:

            # Finished epoch (training round)
            self._epochs_completed += 1

            # copy & normalize
            pp = np.squeeze(normalize(np.expand_dims(np.copy(self._probs), 1), axis=0, norm='l1'))
            # clip values
            pp = np.clip(pp, np.finfo(np.float32).eps, np.finfo(np.float32).max)
            # normalize again
            pp = np.squeeze(normalize(np.expand_dims(pp, 1), axis=0, norm='l1'))
            self._probs = pp

            # indices [0, 1, ..., num_examples]
            ii = np.arange(self._num_examples)  # indices [0, 1, ..., num_examples]
            self._subset_ids = np.random.choice(ii, self._subset_size, replacement, pp)

            # increase counter of selected images
            bbins = np.bincount(self._subset_ids)  # when sampling with replacement
            self._counter[:len(bbins)] += bbins
            # self._counter[self._subset_ids] += 1   # when sampling without replacement

            # update probabilities according to counter
            if decay:
                pp = self._probs * np.exp(-self._counter ** 2 / 10)
            else:
                pp = self._probs

            pp = np.squeeze(normalize(np.expand_dims(pp, 1), axis=0, norm='l1'))
            self._probs = pp

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._subset_size
        end = self._index_in_epoch

        return self._images[self._subset_ids][start:end], \
               self._labels[self._subset_ids][start:end], \
               self._probs[self._subset_ids][start:end],\
               self._indices[self._subset_ids][start:end]

    # for experiments with SUBSETS
    # next_subset_batch, pick an initial subset (25%) of the training set and grow it linearly,
    # the subset is chosen according to the probabilities given by the curriculum (easy -> hard)
    # and update gradually the size of the subset (k <- k+delta) # after 10 epochs, 100% data
    def next_subset_batch(self, batch_size, np_seed, grow=True, replacement=False, random=False, decay=False,
                          fake_data=False):
        """Return the next `batch_size` examples from this data set.
        At the beginning of each epoch, the training set is ordered according to a certain criteria
        given by the variable "probs" that contains probability per sample"""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in range(batch_size)], [
                fake_label for _ in range(batch_size)]

        # in this function, since we work with subsets (instead of all training set)
        # there are no epochs per se, more like training rounds
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if not grow:
            self._subset_size = self._num_examples
            delta = 0
        else:
            # delta = int(0.08*self._num_examples)  # increment gradually (linear) size of the subset
            Es = 10  # warm-up epochs
            delta = int((self._num_examples - self._subset_size_no) / Es)  # + self._subset_size_no

        # 1st sampling of the dataset when the optimization starts, epoch_completed == 0
        if start == 0 and self._epochs_completed == 0:
            # seed randomness of numpy
            print('numpy seeded')
            np.random.seed(np_seed)
            # pp = np.random.rand(self._num_examples)

        if self._index_in_epoch > self._subset_size:
            # Finished epoch (training round)
            self._epochs_completed += 1

            # increase subset_size
            if self._subset_size + delta > self._num_examples:
                self._subset_size = self._num_examples
            else:
                self._subset_size += delta
            print('subset: {}'.format(self._subset_size))

            # if self._pool_ids.shape[0] != 0:
            ii = np.arange(self._num_examples)  # indices [0, 1, ..., num_examples]

            # increase counter of selected images
            # increase counter of selected images
            if replacement:
                bbins = np.bincount(self._subset_ids)  # when sampling with replacement
                self._counter[:len(bbins)] += bbins
            else:
                self._counter[self._subset_ids] += 1  # when sampling without replacement

            if random:
                pp = np.random.rand(self._num_examples)
            elif decay:
                pp = self._probs * np.exp(-self._counter ** 2 / 10)  # decay
            else:
                pp = self._probs

            # copy & normalize
            pp = np.squeeze(normalize(np.expand_dims(np.copy(pp), 1), axis=0, norm='l1'))
            # clip values
            pp = np.clip(pp, np.finfo(np.float32).eps, np.finfo(np.float32).max)
            # normalize again
            pp = np.squeeze(normalize(np.expand_dims(pp, 1), axis=0, norm='l1'))
            self._probs = pp
            self._subset_ids = np.random.choice(ii, self._subset_size, replacement, pp)

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._subset_size
        end = self._index_in_epoch

        return self._images[self._subset_ids][start:end], \
               self._labels[self._subset_ids][start:end], \
               self._probs[self._subset_ids][start:end], \
               self._indices[self._subset_ids][start:end]

    # for experiments with WEIGHTS
    # next batch using probs as weights for WCE, decay after each epoch towards uniform
    def next_batch_weights_only_decay(self, batch_size, decay=True, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in range(batch_size)], [
                fake_label for _ in range(batch_size)]

        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1

            # when epoch is finished
            # increase counter of selected images
            self._counter += 1

            # update probabilities
            pp = self._probs * np.exp(-self._counter ** 2 / 10)
            self._probs = np.squeeze(normalize(np.expand_dims(pp, 1), axis=0, norm='l1'))

            # Shuffle the data
            # np.random.seed(0)
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            self._probs = self._probs[perm]
            self._indices = self._indices[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], \
               self._labels[start:end], \
               self._probs[start:end], \
               self._indices[start:end]



    def remove_epoch(self):
        self._index_in_epoch = 0
        self._epochs_completed -= 1

    def change_probs(self, new_values):
        self._probs = np.squeeze(normalize(np.expand_dims(new_values, 1), axis=0, norm='l1'))


def read_data_sets(data_path, fake_data=False, one_hot=True, subsets=False,
                   init_probs=[],
                   percentage_train=1.,
                   corrupt_labels=False,
                   unbalance=False, unbalance_dict=None,
                   validation_size=5000,
                   source_url=DEFAULT_SOURCE_URL):
    """
    Returns a data provider for a dataset

    :param data_path: local directory to store data
    :param fake_data (optional): flag to indicate whether data should be reshaped
    :param one_hot (optional): flag to indicate whether data is one-hot encoded
    :param init_probs (optional): initial per-class probabilities
    :param percentage_train (optional): percentage of training data
    :param validation_size (optional): validation size
    :param source_url (optional): url where data can be found
    """

    if unbalance_dict is None:
        unbalance_dict = {"percentage": 20, "label1": 0, "label2": 8}
    train_dir = data_path

    class DataSets(object):
        pass

    data_sets = DataSets()

    if fake_data:
        data_sets.train = DataSet([], [], fake_data=True, one_hot=True)
        data_sets.val = DataSet([], [], fake_data=True, one_hot=True)
        data_sets.test = DataSet([], [], fake_data=True, one_hot=True)
        return data_sets

    if not source_url:  # empty string check
        source_url = DEFAULT_SOURCE_URL

    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

    local_file = base.maybe_download(TRAIN_IMAGES, train_dir, source_url + TRAIN_IMAGES)
    with gfile.Open(local_file, 'rb') as f:
        train_images = extract_images(f)

    local_file = base.maybe_download(TRAIN_LABELS, train_dir, source_url + TRAIN_LABELS)
    with gfile.Open(local_file, 'rb') as f:
        train_labels = extract_labels(f, one_hot=one_hot)

    local_file = base.maybe_download(TEST_IMAGES, train_dir, source_url + TEST_IMAGES)
    with gfile.Open(local_file, 'rb') as f:
        test_images = extract_images(f)

    local_file = base.maybe_download(TEST_LABELS, train_dir, source_url + TEST_LABELS)
    with gfile.Open(local_file, 'rb') as f:
        test_labels = extract_labels(f, one_hot=one_hot)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'.format(len(train_images), validation_size))

    val_images = train_images[:validation_size]
    val_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    n_test = test_images.shape[0]
    n_val = val_images.shape[0]
    n_train = train_images.shape[0]

    if not init_probs:
        print('RANDOM INIT PROBABILITIES')
        probs = np.random.rand(n_train)
    else:
        init_probs = np.asarray(init_probs)
        probs_class = np.asarray(1.0 * init_probs / np.sum(init_probs), np.float32)
        dense_train_labels = np.argmax(train_labels, axis=1)
        probs = np.zeros_like(dense_train_labels, np.float32)
        for k in range(0, np.unique(dense_train_labels).max()+1):
            i = np.where(dense_train_labels == k)[0]
            probs[i] = probs_class[k]

    train_probs = np.squeeze(normalize(np.expand_dims(probs, 1), axis=0, norm='l1'))
    val_probs = np.squeeze(normalize(np.expand_dims(np.ones(n_val, np.float32), 1), axis=0, norm='l1'))
    test_probs = np.squeeze(normalize(np.expand_dims(np.ones(n_test, np.float32), 1), axis=0, norm='l1'))

    # For experiments with limited amount of data
    if percentage_train != 1.:
        train_size = int(percentage_train*train_images.shape[0])
        Xtrain_images, Xval_images, ytrain, yval, ptrain, probs_val = train_test_split(train_images,
                                                                                       train_labels,
                                                                                       train_probs,
                                                                                       train_size=train_size,
                                                                                       random_state=0)
        train_images = Xtrain_images
        train_labels = ytrain
        train_probs = ptrain

    # For experiments with class-imbalance distribution
    if unbalance:
        print('CLASS-IMBALANCE')
        n_classes = len(np.unique(np.argmax(train_labels, 1)))
        reduceto = 0.01 * unbalance_dict[0]['percentage']
        label1 = unbalance_dict[0]['label1']
        label2 = unbalance_dict[0]['label2']

        pick_ids = []
        newsize = 0
        all_classes = np.arange(0, n_classes)
        all_classes = np.delete(all_classes, np.where(all_classes == label1)[0])
        all_classes = np.delete(all_classes, np.where(all_classes == label2)[0])

        for lab in [label1, label2]:
            allids = np.where(np.argmax(train_labels, 1) == lab)[0]
            selectedids = np.random.choice(allids, int(reduceto * allids.shape[0]), replace=False)
            pick_ids.append(selectedids)
            newsize += len(selectedids)

        new_ids = convert_list_to_array(pick_ids, newsize)

        other_ids = []
        othersize = 0
        for lab in all_classes.tolist():
            selectedids = np.where(np.argmax(train_labels, 1) == lab)[0]
            other_ids.append(selectedids)
            othersize += len(selectedids)

        keep_ids = convert_list_to_array(other_ids, othersize)

        # new_ids: contains the indices of the reduced (imbalance) classes
        # keep_ids: contains the indices of the rest (keep the same class distribution)
        resulting_ids = np.concatenate((new_ids, keep_ids))
        np.random.shuffle(resulting_ids)

        train_images = train_images[resulting_ids, ...]
        train_labels = train_labels[resulting_ids, ...]
        train_probs = train_probs[resulting_ids]

    train_indices = np.zeros(train_labels.shape[0])
    val_indices = np.zeros(val_labels.shape[0])
    test_indices = np.zeros(test_labels.shape[0])

    if corrupt_labels:
        print('NOISE / CORRUPT LABELS')
        percentage_corrupted_labels = 30
        number_corrupted_labels = int(1.0*percentage_corrupted_labels/100 * train_labels.shape[0])
        dense_train_labels = np.argmax(train_labels, 1)
        old_train_labels = np.copy(dense_train_labels)
        idx_train_labels = np.arange(train_labels.shape[0])
        idx_to_be_corrupted = np.random.choice(idx_train_labels, number_corrupted_labels, replace=False)
        train_indices[idx_to_be_corrupted] = 1
        dense_train_labels[idx_to_be_corrupted] += 1
        dense_train_labels[np.where(dense_train_labels == 10)[0]] = 0
        train_labels = dense_to_one_hot(dense_train_labels, n_class=10)

    data_sets.train = DataSet(train_images, train_labels, train_probs, train_indices, fake_data=True, one_hot=True, subsets=subsets)
    data_sets.val = DataSet(val_images, val_labels, val_probs, val_indices, fake_data=True, one_hot=True, subsets=False)
    data_sets.test = DataSet(test_images, test_labels, test_probs, test_indices, fake_data=True, one_hot=True, subsets=False)

    return data_sets


def convert_list_to_array(elements, size):
    array = np.zeros(size, np.int32)
    for kk, ii in enumerate(elements):
        if kk == 0:
            start = 0
            end = len(ii)
        else:
            end += len(ii)
        array[start:end] = ii
        start = end
    return array
