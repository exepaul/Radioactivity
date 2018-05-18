import tensorflow as tf
from file_lstm import encode_query


user_query = encode_query(input('Ask the query'))

with tf.Session() as sess:

    saver = tf.train.import_meta_graph('/Users/exepaul/Downloads/wei/model.meta')
    restore = saver.restore(sess,tf.train.latest_checkpoint('/Users/exepaul/Downloads/wei/'))

    graph=tf.get_default_graph()

    print(user_query)

    query= graph.get_tensor_by_name("input:0")
    result=graph.get_tensor_by_name("netout:0")

    feed_dict = {query:[user_query]}

    prediction = result.eval(feed_dict=feed_dict)

    print(prediction)
#
#
# with tf.Session() as sess:
#     saver = tf.train.import_meta_graph('/Users/exepaul/Desktop/.meta')
#     new=saver.restore(sess, tf.train.latest_checkpoint('/Users/exepaul/Desktop/'))
#
#     graph = tf.get_default_graph()
#     input_x = graph.get_tensor_by_name("input:0")
#     result = graph.get_tensor_by_name("result:0")
#
#     feed_dict = {input_x: x_data,}
#
#     predictions = result.eval(feed_dict=feed_dict)
#     print(predictions)


