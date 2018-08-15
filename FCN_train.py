from __future__ import print_function, absolute_import, division

import tensorflow as tf
import numpy as np
import time
import os
import scipy.misc as misc

from FCN.FCN_DatasetReader import DatasetReader, ImageReader
import FCN.FCN_model

WEIGHTS = np.load('D:/cain_ha/vgg19_weights.npy', encoding='bytes').item()

FLAGS = tf.flags.FLAGS
# FCN parameters
tf.flags.DEFINE_string('mode',              'predict',         "Mode of FCN: finetune / predict")
tf.flags.DEFINE_float('learning_rate',      1e-4,               "Learning rate initial value")
tf.flags.DEFINE_float('keep_prob',          0.5,                "Keep probability")
tf.flags.DEFINE_integer('num_of_epoch',     80,                 "Number of epoch")
tf.flags.DEFINE_integer('batch_size',       20,                  "Batch size")

# FCN data parameters
tf.flags.DEFINE_integer('num_of_class',     2,                  "Number of classes")
tf.flags.DEFINE_integer('image_height',     224,                "Heighfinetunet of image")
tf.flags.DEFINE_integer('image_width',      224,                "Width of image")

# FCN storage parameters
tf.flags.DEFINE_string('train_dir',         'data/train',       "Train dataset dir")
tf.flags.DEFINE_string('valid_dir',         'data/valid',       "Valid dataset dir")
tf.flags.DEFINE_string('log_dir',           'D:/presto/Python/Lib/site-packages/FCN/logs/',             "Logs dir")
tf.flags.DEFINE_string('checkpoint_dir',    'D:/presto/Python/Lib/site-packages/FCN/checkpoints/',      "Checkpoints dir")

# FCN test parameters
tf.flags.DEFINE_string('test_dir',          'test',             "Test dataset dir")

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
def main(argv=None):

    print(">>> Setting up FCN model ...")

    # model - input
    img_holder, ant_holder = FCN.FCN_model.input(FLAGS.image_height, FLAGS.image_width)  #(2, 224 ,224, 3), (2, 224, 224, 1)

    # model - inference
    logits, predictions = FCN.FCN_model.inference(img_holder, FLAGS.num_of_class, WEIGHTS, FLAGS.keep_prob)

    # model - loss
    loss_op = FCN.FCN_model.loss(logits, ant_holder) #   (224,224,3,2), (2,224,224,1), 

    # model - evaluate
    accuracy = FCN.FCN_model.evaluate(predictions, ant_holder) #(2,224,224,1),(2,224,224,1) 
    precision, recall, f_score, matthews_cc = FCN.FCN_model.statistics(predictions, ant_holder)

    # model - train var list
    var_list = tf.trainable_variables()

    # model - train
    train_op = FCN.FCN_model.train(loss_op, FLAGS.learning_rate, var_list)

    print(">>> Setting up FCN summary ...")

    # summary - input and predictions
    input_img_sum = tf.summary.image('input_images', img_holder, max_outputs=8)
    input_tru_sum = tf.summary.image('ground_truth', tf.cast(ant_holder * 255, tf.uint8), max_outputs=8)
    input_pre_sum = tf.summary.image('predictions', tf.cast(predictions * 255, tf.uint8), max_outputs=8)

    # summary - train loss
    train_loss = tf.summary.scalar('train_loss', loss_op)

    # summary - merge
    train_summary = tf.summary.merge([input_img_sum, input_tru_sum, input_pre_sum, train_loss])

    print(">>> Setting up FCN writer and saver ...")

    # process - summary writer and model saver
    writer = tf.summary.FileWriter(FLAGS.log_dir)
    saver = tf.train.Saver()

    # save train and valid statistics
    train_statistics = []

    if FLAGS.mode == 'finetune':

        # feed
        train_dataset = DatasetReader(FLAGS.train_dir, [FLAGS.image_height, FLAGS.image_width], True)

        print(">>> Finish loading train dataset and valid dataset ")

        with tf.Session(config=config) as sess:
            # initilize model
            init = tf.global_variables_initializer()
            sess.run(init)

            writer.add_graph(sess.graph)

            # if trained, restore the model
            #if tf.train.latest_checkpoint(FLAGS.checkpoint_dir):
                #print("Load model from {}".format(tf.train.latest_checkpoint(FLAGS.checkpoint_dir)))
                #saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))

            # train - parameters
            num_of_epoch = FLAGS.num_of_epoch
            batch_size = FLAGS.batch_size
            num_of_batch = int(train_dataset.num // batch_size)
            step = 0
            # train - main process
            print("============>>>> Begin to train ... <<<<============")
            for epoch in range(num_of_epoch):

                for batch in range(num_of_batch):
                    start_time = time.time()
                    # train batch
                    batch_img, batch_ant = train_dataset.next_batch(batch_size)
                    
                    _, loss, acc, pre, rec, fsc, mcc, summary_str = sess.run([
                        train_op, loss_op, accuracy, precision, recall, f_score, matthews_cc, train_summary],
                        feed_dict={img_holder: batch_img, ant_holder: batch_ant})

                    batch_time = time.time() - start_time

                    # save accuracy and loss
                    train_statistics.append([loss, acc, pre, rec, fsc, mcc, batch_time])

                    print("Epoch: [%d / %d] Batch: [%d / %d] Loss: %.6f, Time: %.3f sec" %
                          (epoch, num_of_epoch, batch, num_of_batch, loss, batch_time))

                    # write train summary
                    writer.add_summary(summary_str, global_step=step)

                    step += 1

                if (epoch + 1) % 20 == 0:
                    checkpoint_file = os.path.join(FLAGS.checkpoint_dir, 'FCN_epoch_' + str(epoch) + '.ckpt')
                    saver.save(sess, checkpoint_file)
                    print("FCN training file {} is saving ... ".format(checkpoint_file))

            print("============>>>> Finish train ... <<<<============")
            save_statistics('train_statistics.npy', train_statistics)
            print("============>>>> Result save ... <<<<============")

    elif FLAGS.mode == 'predict':
        # images
        image_set = ImageReader(FLAGS.test_dir)

        print(">>> Loading images from test directory ...")

        if tf.train.latest_checkpoint(FLAGS.checkpoint_dir):
            print("Load model from {}".format(tf.train.latest_checkpoint(FLAGS.checkpoint_dir)))
        else:
            print("Please train model first !!!")

        #with tf.Session(config=config) as sess:
        with tf.Session(config=config) as sess:
            
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
            # predict - process
            print("============>>>> Begin to predict ... <<<<============")

            for i in range(image_set.num):

                start_time = time.time()

                image, save_name, save_shape = image_set.next_image() #(666,1000)

                resized_image = misc.imresize(image, size=[FLAGS.image_height, FLAGS.image_width]) #(224,224,3)
                resized_image = np.expand_dims(resized_image, axis=0) #(1,224,224,3)
                prediction = sess.run(predictions, feed_dict={img_holder: resized_image}) #(1,224,224,1)
                save_png(save_name, prediction, save_shape)
                predict_time = time.time() - start_time
                print('Image: [%d / %d] Time: %.3f sec' % (i, image_set.num, predict_time))

            print("============>>>> Finish predict ... <<<<============")
    elif FLAGS.mode == 'valid':
        # images
        image_set = ImageReader(FLAGS.test_dir)

        print(">>> Loading images from test directory ...")

        if tf.train.latest_checkpoint(FLAGS.checkpoint_dir):
            print("Load model from {}".format(tf.train.latest_checkpoint(FLAGS.checkpoint_dir)))
        else:
            print("Please train model first !!!")

        #with tf.Session(config=config) as sess:
        with tf.Session(config=config) as sess:
            
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
            # predict - process
            print("============>>>> Begin to predict ... <<<<============")

            for i in range(image_set.num):

                start_time = time.time()

                image, save_name, save_shape = image_set.next_image() #(666,1000)

                resized_image = misc.imresize(image, size=[FLAGS.image_height, FLAGS.image_width]) #(224,224,3)
                resized_image = np.expand_dims(resized_image, axis=0) #(1,224,224,3)
                prediction = sess.run(predictions, feed_dict={img_holder: resized_image}) #(1,224,224,1)
                save_png(save_name, prediction, save_shape)
                predict_time = time.time() - start_time
                print('Image: [%d / %d] Time: %.3f sec' % (i, image_set.num, predict_time))

            print("============>>>> Finish predict ... <<<<============")
        
def save_statistics(file_name, list):
    list_ndarray = np.array(list)
    np.save(file_name, list_ndarray)

def save_png(file_name, ndarray, new_size):
    image = np.squeeze(ndarray)  #(224,224)
    print("image----->", image.shape)
    new_image = misc.imresize(image, size=new_size) #(666,1000)
    print("new_image----->", new_image.shape)
    misc.imsave(file_name, new_image)

if __name__ == '__main__':
    tf.app.run()
