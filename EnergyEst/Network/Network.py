import tensorflow as tf


def qm9net(atoms_, inv_distances_, u0_, cm_):
    with tf.variable_scope('qm9_atom_invdis_u0_cm'):

        with tf.variable_scope('qm9_atoms'):
            output = tf.layers.conv2d_transpose(atoms_, 2, 3, strides=(1, 2), padding='same')
            # output = tf.layers.batch_normalization(output, training=True)
            output = tf.nn.relu(output)

            output = tf.layers.conv2d_transpose(output, 2, 3, strides=(1, 4), padding='same')
            # output = tf.layers.batch_normalization(output, training=True)
            output = tf.nn.relu(output)

            # batchsize * 30 * 30 * 2
            output = tf.layers.conv2d_transpose(output, 2, 3, strides=(1, 4), padding='same')
            # output = tf.layers.batch_normalization(output, training=True)
            atoms_output = tf.nn.relu(output)
        with tf.variable_scope('qm9_concat'):

            # bs * 32 * 32 * 4
            concat_map = tf.concat([atoms_output, inv_distances_, cm_], 3)

            # bs * 16 * 16 * 16
            output = tf.layers.conv2d(concat_map, 16, 3, strides=(1,1), padding='same')
            # output = tf.layers.batch_normalization(output, training=True)
            output = tf.nn.relu(output)
            output = tf.nn.max_pool(output, [1,2,2,1],[1,2,2,1], padding='VALID')

            # bs * 8 * 8 * 32
            output = tf.layers.conv2d(output, 32, 3, strides=(1, 1), padding='same')
            # output = tf.layers.batch_normalization(output, training=True)
            output = tf.nn.relu(output)
            output = tf.nn.max_pool(output, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')

            # bs * 4 * 4 * 64
            output = tf.layers.conv2d(output, 64, 3, strides=(1, 1), padding='same')
            # output = tf.layers.batch_normalization(output, training=True)
            output = tf.nn.relu(output)
            output = tf.nn.max_pool(output, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')

        with tf.variable_scope('qm9_fc'):
            fc_input = tf.reshape(output, [-1, 4*4*64])
            fc_out = tf.layers.dense(fc_input, 32, activation=None)
            energy_est = tf.layers.dense(fc_out, 1, activation=None)
            loss = tf.reduce_mean(tf.nn.l2_loss(u0_ - energy_est))
            print(energy_est, loss)
    return energy_est, loss