from DataLoader.PreProcess import *
from Network.Network import *
import os
energy_est_root = os.path.abspath(os.path.join(os.path.dirname(__file__), r'..'))
training_proportion = 0.8  # 训练样本比例
shuffle_seed = 0  # 将样本打乱顺序的随机种子
max_atom_size = 32  # 原子最大数目，使得网络结构可固定化
batch_size = 16
training_from_scratch = True
model_path = None
pre_processer = PreProcesser(training_proportion, shuffle_seed, max_atom_size)
pre_processer.load_data(os.path.join(energy_est_root, r'Resources', r'QM9_nano.npz'))
train_set = pre_processer.training_set
all_data = pre_processer.all_data

# max_atom_num = 0
# max_dis = -1
# min_dis = 13
# max_u0 = -20000
# min_u0 = -100
# for i in range(0, all_data.length):
#     if max_u0 < all_data.data_list[i].u0:
#         max_u0 = all_data.data_list[i].u0
#     if min_u0 > all_data.data_list[i].u0:
#         min_u0 = all_data.data_list[i].u0
#     # for x in range(all_data.data_list[i].atoms_num):
#     #     for y in range(all_data.data_list[i].atoms_num):
#     #         if x != y:
#     #             if max_dis < all_data.data_list[i].inv_distance[x][y]:
#     #                 max_dis = all_data.data_list[i].inv_distance[x][y]
#     #             if min_dis > all_data.data_list[i].inv_distance[x][y]:
#     #                 min_dis = all_data.data_list[i].inv_distance[x][y]
# print(max_u0, min_u0)


with tf.Session() as sess:
    atoms_ = tf.placeholder(tf.float32, [batch_size, max_atom_size, 1, 1])
    inv_distances_ = tf.placeholder(tf.float32, [batch_size, max_atom_size, max_atom_size, 1])
    u0_ = tf.placeholder(tf.float32, [batch_size, 1])
    output, loss = qm9net(atoms_, inv_distances_, u0_)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    saver = tf.train.Saver()
    if training_from_scratch:
        sess.run(tf.global_variables_initializer())
    else:
        pass

    tf.summary.scalar('Loss', loss)
    merged = tf.summary.merge_all()
    TrainWriter = tf.summary.FileWriter(os.path.join(energy_est_root, 'Resources', 'Training_log'), sess.graph)

    for epoch in range(0, 20):
        net_id = epoch
        os.mkdir(os.path.join(energy_est_root, 'Resources', 'Model', 'Net') + str(net_id))
        saver.save(sess, os.path.join(energy_est_root, 'Resources', 'Model', 'Net' + str(net_id), 'model.ckpt'))

        for itr in range(0, int(train_set.length / batch_size)):
            atoms_list = []
            inv_distances_list = []
            u0_list = []
            for record in range(0, batch_size):
                atoms_input = train_set.data_list[itr * batch_size + record].atoms
                atoms_list.append(atoms_input[:, np.newaxis, np.newaxis])
                inv_distances_input = train_set.data_list[itr * batch_size + record].inv_distance
                inv_distances_list.append(inv_distances_input[:,:,np.newaxis])
                u0_input = train_set.data_list[itr * batch_size + record].u0
                u0_list.append(-(u0_input + 1000) / 1800)
            summary, train_loss, output_u0, _ = \
                sess.run([merged, loss, output, train_step], feed_dict={
                     atoms_: atoms_list,
                     inv_distances_: inv_distances_list,
                     u0_: u0_list
                 })

            TrainWriter.add_summary(summary, itr + epoch * int(train_set.length / batch_size))
            print('epoch: ', epoch, 'itr: ',itr, 'loss: ', train_loss)
            print('label:',u0_list)
            print('output:', output_u0)