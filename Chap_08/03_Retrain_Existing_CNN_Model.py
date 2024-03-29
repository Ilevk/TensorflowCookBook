import os
import tarfile
import _pickle as cPickle
import numpy as np
import urllib.request
import scipy.misc

#Download CIFAR-10 DATASET
cifar_link = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
data_dir = 'temp'
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)
#Target Labels
objects = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

target_file = os.path.join(data_dir, 'cifar-10-python.tar.gz')
if not os.path.isfile(target_file):
    print("CIFAR-10 file not found. Downloading CIFAR data (Size = 163MB)")
    print('This may take a few minutes, please wait.')
    filename, headers = urllib.request.urlretrieve(cifar_link, target_file)
#Extract to Memory
tar = tarfile.open(target_file)
tar.extractall(path=data_dir)
tar.close()
#Make train&test_dir for training
train_folder = 'train_dir'
if not os.path.isdir(os.path.join(data_dir, train_folder)):
    for i in range(10):
        folder = os.path.join(data_dir, train_folder, objects[i])
        os.makedirs(folder)

test_folder = 'validation_dir'
if not os.path.isdir(os.path.join(data_dir, test_folder)):
    for i in range(10):
        folder = os.path.join(data_dir, test_folder, objects[i])
        os.makedirs(folder)

#Loading image, make image_dictionary
def load_batch_from_file(file):
    file_conn = open(file, 'rb')
    image_dictionary = cPickle.load(file_conn, encoding='latin1')
    file_conn.close()
    return image_dictionary

#
def save_images_from_dict(image_dict, folder='data_dir'):
    for ix, label in enumerate(image_dict['labels']):
        folder_path = os.path.join(data_dir, folder, objects[label])
        filename = image_dict['filenames'][ix]
        #Transform Image data
        image_array = image_dict['data'][ix]
        image_array.resize([3, 32, 32])
        #Save Images
        output_location = os.path.join(folder_path, filename)

    scipy.misc.imsave(output_location, image_array.transpose())

#Applying Above functions to downloaded data files,
#Assure to save image to appropriate place
data_location = os.path.join(data_dir, 'cifar-10-batches-py')
train_names = ['data_batch_' + str(x) for x in range(1, 6)]
test_names = ['test_batch']

for file in train_names:
    print('Saving images from file: {}'.format(file))
    file_location = os.path.join(data_dir, 'cifar-10-batches-py', file)
    image_dict = load_batch_from_file(file_location)
    save_images_from_dict(image_dict, folder=train_folder)

for file in test_names:
    print('Saving images from file:{}'.format(file))
    file_location = os.path.join(data_dir, 'cifar-10-batches-py', file)
    image_dict = load_batch_from_file(file_location)
    save_images_from_dict(image_dict, folder=test_folder)

cifar_labels_file = os.path.join(data_dir,'cifar10_labels.txt')
print('Writting labels file, {}'.format(cifar_labels_file))
with open(cifar_labels_file, 'w') as labels_file:
    for item in objects:
        labels_file.write("{}\n".format(item))
