import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import roc_auc_score
from keras.datasets import cifar10
import data


def plot_images(data, n_x, n_y, name, text=None):
    (height, width, channel) = data.shape[1:]
    height_inc = height + 1
    width_inc = width + 1
    n = len(data)
    if n > n_x*n_y: n = n_x * n_y

    if channel == 1:
        mode = "L"
        data = data[:,:,:,0]
        image_data = 50 * np.ones((height_inc * n_y + 1, width_inc * n_x - 1), dtype='uint8')
    else:
        mode = "RGB"
        image_data = 50 * np.ones((height_inc * n_y + 1, width_inc * n_x - 1, channel), dtype='uint8')
    for idx in range(n):
        x = idx % n_x
        y = idx // n_x
        sample = data[idx]
        image_data[height_inc*y:height_inc*y+height, width_inc*x:width_inc*x+width] = 255*sample.clip(0, 0.99999)
    img = Image.fromarray(image_data, mode=mode)
    fileName = name + ".png"

    print("Creating file " + fileName)
    if text is not None:
        img.text(10, 10, text)

    img.save(fileName)
    return img


def save_output(session, prefix, epoch, global_iters, batch_size, input, output, limit, augment_avg_at_test=False, original_shape=None):
    result_dict = {}
    for key in output.keys():
        result_dict[key] = []

    nb_batches = limit // batch_size

    if augment_avg_at_test:
        print('Augment and avg at test.')
        augment_placeholder = tf.placeholder(tf.float32, shape=original_shape)
        x_augmented = data.augment_transforms(augment_placeholder)
        for i in range(nb_batches):
            test_batch = np.array(session.run(list(input.values()))[0])
            for idx in range(batch_size):
                img_np = test_batch[idx]
                #print('shape: ', img_np.shape)
                #print('input keys: ', input.keys())
                augmented_batch = session.run([x_augmented]*batch_size, feed_dict={augment_placeholder: img_np})
                #print('augmented batch shape: ', np.array(augmented_batch).shape)
                #print('feed: ', dict(zip(input.keys(), augmented_batch)))
                augmented_test_mean, augmented_test_log_var, augmented_test_reconstloss = \
                    session.run(list(output.values()), feed_dict=dict(zip(input.keys(), [augmented_batch])))
                result_dict["test_mean"].append(np.mean(augmented_test_mean, axis=0))
                result_dict["test_log_var"].append(np.mean(augmented_test_log_var, axis=0))
                result_dict["test_reconstloss"].append(np.mean(augmented_test_reconstloss, axis=-1))
        result_dict = dict(map(lambda kv: (kv[0], np.array(kv[1])), result_dict.items()))
        for k in result_dict.keys():
            print('result_dict[{}] shape: {}'.format(k, result_dict[k].shape))
    else:
        for i in range(nb_batches):
            inp = session.run(list(input.values()))
            res = session.run(list(output.values()), feed_dict=dict(zip(input.keys(), inp)))
            for k, r in enumerate(res):
                result_dict[list(output.keys())[k]].append(r)
        result_dict = dict(map(lambda kv: (kv[0], np.concatenate(kv[1], axis=0)), result_dict.items()))

    for k in output.keys():
        filename = "{}_{}_epoch{}_iter{}.npy".format(prefix, k, epoch+1, global_iters)
        print("Saving {} pointcloud mean to {}".format(k, filename))
        np.save(filename, result_dict[k])
    return result_dict


def save_kldiv(session, prefix, epoch, global_iters, batch_size, input, output, limit):
    result_dict = {}
    for key in output.keys():
        result_dict[key] = []

    for i in range(limit // batch_size):
        inp = session.run(list(input.values()))
        res = session.run(list(output.values()), feed_dict=dict(zip(input.keys(), inp)))
        for k, r in enumerate(res):
            result_dict[list(output.keys())[k]].append(r)

    mean = np.concatenate(result_dict['mean'], axis=0)
    log_var = np.concatenate(result_dict['log_var'], axis=0)
    kldiv = 0.5 * np.sum(- 1 - log_var + np.square(mean) + np.exp(log_var), axis=-1)
    filename = "{}_{}_epoch{}_iter{}.npy".format(prefix, 'kldiv', epoch, global_iters)
    print("Saving kldiv to {}".format(filename))
    np.save(filename, kldiv)
    return kldiv

def oneclass_eval(normal_class, kldiv_file, margin):
    if normal_class == -1:
        pass
    else:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_true = [1 if item == normal_class else 0 for item in y_test]
        kldiv = np.load(kldiv_file)
        y_scores = margin - kldiv
        auc = roc_auc_score(y_true, y_scores)
        print('AUC: ', auc)
