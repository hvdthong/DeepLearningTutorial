import tensorflow as tf

a = tf.constant([2])
b = tf.constant([3])
c = tf.add(a, b)

# sess = tf.Session()
# result = sess.run(c)
# print result
# sess.close()

with tf.Session() as session:
    result = session.run(c)
    print result

state = tf.Variable(1)
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)
init_op = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init_op)
    print session.run(state)
    for _ in range(3):
        session.run(update)
        print session.run(state), state.eval()

a = tf.placeholder(tf.float32)
b = a * 2
with tf.Session() as session:
    result = session.run(b, feed_dict={a: 3.5})
    print result, b.eval(feed_dict={a: 3.5})
