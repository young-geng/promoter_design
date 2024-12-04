import numpy as np
import os
import mlxu


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    designed_sequences_dir='',
    file_name='optimized_seqs.pkl',
    output_file='',
)


def main(argv):
    assert FLAGS.designed_sequences_dir != ''
    assert FLAGS.output_file != ''

    all_data = []
    for d in os.listdir(FLAGS.designed_sequences_dir):
        if os.path.exists(os.path.join(FLAGS.designed_sequences_dir, d, FLAGS.file_name)):
            data = mlxu.load_pickle(os.path.join(FLAGS.designed_sequences_dir, d, FLAGS.file_name))
            all_data.append(data)

    new_data = {}
    for key in all_data[0].keys():
        new_data[key] = np.concatenate([data[key] for data in all_data], axis=0)
    
    mlxu.save_pickle(new_data, FLAGS.output_file)
    

if __name__ == '__main__':
    mlxu.run(main)