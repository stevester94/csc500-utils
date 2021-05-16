#! /usr/bin/python3
import io
import pprint
import re
import numpy as np
import os
import sys
import multiprocessing as mp
import random
import time
import tensorflow as tf
import functools

tf.random.set_seed(1337)


def interleaved_IQ_binary_file_to_dataset(
    binary_path: str,
    num_samples_per_chunk: int,
    I_or_Q_datatype: tf.DType, 
    num_parallel_calls=tf.data.AUTOTUNE,
    num_parallel_reads=1,
    buffer_size=None,
):
    """Parse a binary file of arbitrary IQ into a tensorflow dataset

    Args:
        binary_path: Path to the binary file
        num_samples_per_chunk: The number of IQ pairs per tensor in the output dataset
        I_or_Q_datatype: The datatype of each I or Q (IE if you have complex128, use float64)
        num_parallel_calls: Passed directly to tf
        num_parallel_reads: Passed directly to tf
        buffer_size: Passed directly to tf

    Returns:
        A tensorflow dataset with element_spec of 
        (TensorSpec(shape=(), dtype=tf.int64, name=None), TensorSpec(shape=(2, <num_samples_per_chunk>), dtype=tf.float64, name=None))
        And the resulting dataset's cardinality, which is calculated by the file size
        
        With elements being <index of chunk from original file> <IQ>

        Note the shape of the IQ is 2,<num_samples_per_chunk>
    """
    chunk_size_bytes = I_or_Q_datatype.size * num_samples_per_chunk * 2
    dataset = tf.data.FixedLengthRecordDataset(
        binary_path, record_bytes=chunk_size_bytes, header_bytes=None, footer_bytes=None, buffer_size=buffer_size,
        compression_type=None, num_parallel_reads=num_parallel_reads
    )

    dataset = dataset.map(
        lambda IQ: (
            tf.io.decode_raw(IQ, I_or_Q_datatype),
        ),
        num_parallel_calls=num_parallel_calls,
        deterministic=True
    )

    dataset = dataset.map(
        lambda IQ: (
            # IQ is interleaved. This is known as fortran order, and is easy to do in numpy
            # but is not directly supported in TF, so we do this BS
            tf.transpose(tf.reshape(IQ, (num_samples_per_chunk,2)))
        ),
        num_parallel_calls=num_parallel_calls,
        deterministic=True
    )

    dataset = dataset.enumerate()

    return dataset, int(get_file_size(binary_path) / chunk_size_bytes)


def get_datasets_base_path():
    return os.environ["DATASETS_ROOT_PATH"]

def file_ds_to_record_ds(file_ds, record_batch_size):
    ds = file_ds
    ds = ds.interleave(
        lambda path: symbol_dataset_from_file(path, record_batch_size),
        cycle_length=10, 
        block_length=1,
        deterministic=True
    )

    return ds

# Train ds is reshuffled on each epoch
# An 'all_ds' is also returned which will use the entire directory
def shuffled_dataset_accessor( 
    path, # Path to a dir of shuffled datasets
    record_batch_size, # The batch size of each record in the ds
    desired_batch_size=None, # If we should change the batch size. If none we leave as is
    train_val_test_split=(0.6, 0.2, 0.2),
    ):

    symbol_size = 384
    record_size = symbol_size + 1 + 1 + 1 + 8

    # Our data is thoroughly shuffled already, and is split into many-ish files. So we can take paths in order to build our train-val-test subsets.
    formatted_path = os.path.normpath("{}/*batch-{}*ds".format(path, record_batch_size))
    ds_files = tf.data.Dataset.list_files(formatted_path, shuffle=True)

    # Calc number of records based on file sizes
    total_records = int(sum(
            [get_file_size(str(f.numpy().decode('utf8'))) for f in ds_files]
        )/record_size
    )


    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = train_val_test_split
    num_train_files = TRAIN_SPLIT * ds_files.cardinality().numpy()
    num_val_files   = VAL_SPLIT   * ds_files.cardinality().numpy()
    num_test_files  = TEST_SPLIT  * ds_files.cardinality().numpy()

    train_files = ds_files.take(num_train_files)
    val_files   = ds_files.skip(num_train_files).take(num_val_files)
    test_files  = ds_files.skip(num_train_files).skip(num_val_files).take(num_test_files)
    
    train_files = [f.decode('utf8') for f in train_files.as_numpy_iterator()]
    val_files   = [f.decode('utf8') for f in val_files.as_numpy_iterator()]
    test_files  = [f.decode('utf8') for f in test_files.as_numpy_iterator()]

    train_ds  = tf.data.Dataset.from_tensor_slices(train_files)
    val_ds  = tf.data.Dataset.from_tensor_slices(val_files)
    test_ds  = tf.data.Dataset.from_tensor_slices(test_files)


    train_ds = train_ds.shuffle(train_ds.cardinality(), reshuffle_each_iteration=True)

    all_ds   = file_ds_to_record_ds(ds_files, record_batch_size)
    train_ds = file_ds_to_record_ds(train_ds, record_batch_size)
    val_ds = file_ds_to_record_ds(val_ds, record_batch_size)
    test_ds = file_ds_to_record_ds(test_ds, record_batch_size)
    

    if desired_batch_size != None:
        train_ds = train_ds.unbatch().batch(desired_batch_size)
        val_ds = val_ds.unbatch().batch(desired_batch_size)
        test_ds = test_ds.unbatch().batch(desired_batch_size)

    return {
        "all_ds": all_ds,
        "train_ds":train_ds,
        "val_ds":val_ds,
        "test_ds":test_ds,
        "total_records": total_records,
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files,
    }

def get_iterator_cardinality(it):
    total = 0
    for e in it:
        total += 1
    
    return total

def get_file_size(path):
    size = 0
    with open(path, "rb") as handle:
        handle.seek(0, io.SEEK_END)
        size = handle.tell()
    
    return size


def symbol_tuple_from_bytes(bytes):
    symbol_size = 384
    record_size = symbol_size + 1 + 1 + 1 + 8

    assert( len(bytes) == record_size )

    # frequency_domain_IQ,  tf.float32 (2,48)
    # day,                  tf.uint8 ()
    # transmitter_id,       tf.uint8 ()
    # transmission_id,      tf.uint8 ()
    # symbol_index_in_file, tf.int64 ()

    frequency_domain_IQ = np.frombuffer(bytes[:symbol_size], dtype=np.float32)
    # frequency_domain_IQ = frequency_domain_IQ.reshape((2,int(len(frequency_domain_IQ)/2)), order="F")
    frequency_domain_IQ = frequency_domain_IQ.reshape((2,int(len(frequency_domain_IQ)/2)))

    day                  = np.frombuffer(bytes[384:385],  dtype=np.uint8)[0]
    transmitter_id       = np.frombuffer(bytes[385:386],  dtype=np.uint8)[0]
    transmission_id      = np.frombuffer(bytes[386:387],  dtype=np.uint8)[0]
    symbol_index_in_file = np.frombuffer(bytes[387:], dtype=np.int64)[0]

    return (
        frequency_domain_IQ,
        day,
        transmitter_id,
        transmission_id,
        symbol_index_in_file,
    )

    # tf.strings.reduce_join(tf.slice(x, [0],                [symbol_size*batch_size])),
    # tf.strings.reduce_join(tf.slice(x, [(symbol_size+0)*batch_size], [1*batch_size])), # Yes it's 0 because we are 0 indexed
    # tf.strings.reduce_join(tf.slice(x, [(symbol_size+1)*batch_size], [1*batch_size])),
    # tf.strings.reduce_join(tf.slice(x, [(symbol_size+2)*batch_size], [1*batch_size])),
    # tf.strings.reduce_join(tf.slice(x, [(symbol_size+3)*batch_size], [8*batch_size])),

def get_files_with_suffix_in_dir(path, suffix):
    """Returns full path"""
    (_, _, filenames) = next(os.walk(path))
    return [os.path.join(path,f) for f in filenames if f.endswith(suffix)]

def get_files_in_dir(path):
    """Returns full path"""
    (_, _, filenames) = next(os.walk(path))
    return [os.path.join(path,f) for f in filenames]

def filter_paths(
        paths, 
        day_to_get='All', 
        transmitter_id_to_get='All',
        transmission_id_to_get='All'):

    def is_any_pattern_in_string(list_of_patterns, string):
        for p in list_of_patterns:
            if re.search(p, string) != None:
                return True
        return False

    filtered_paths = paths
    if day_to_get != "All":
        assert(isinstance(day_to_get, list))
        assert(len(day_to_get) > 0)
        
        filt = ["day-{}(?![0-9])".format(f) for f in day_to_get]
        filtered_paths = [p for p in filtered_paths if is_any_pattern_in_string(filt, p)]

    if transmitter_id_to_get != "All":
        assert(isinstance(transmitter_id_to_get, list))
        assert(len(transmitter_id_to_get) > 0)

        filt = ["transmitter-{}(?![0-9])".format(f) for f in transmitter_id_to_get]
        filtered_paths = [p for p in filtered_paths if is_any_pattern_in_string(filt, p)]

    if transmission_id_to_get != "All":
        assert(isinstance(transmission_id_to_get, list))
        assert(len(transmission_id_to_get) > 0)

        filt = ["transmission-{}(?![0-9])".format(f) for f in transmission_id_to_get]
        filtered_paths = [p for p in filtered_paths if is_any_pattern_in_string(filt, p)]

    return filtered_paths

def metadata_from_path(path):
    match  = re.search("day-([0-9]+)_transmitter-([0-9]+)_transmission-([0-9]+)", path)
    (day, transmitter_id, transmission_id) = match.groups()

    return {
        "day": int(day),
        "transmitter_id": int(transmitter_id),
        "transmission_id": int(transmission_id)
    }



def vanilla_binary_file_to_symbol_dataset(
    binary_path
):
    symbol_size=384

    metadata = metadata_from_path(binary_path)

    dataset = tf.data.FixedLengthRecordDataset(
        binary_path, record_bytes=symbol_size, header_bytes=None, footer_bytes=None, buffer_size=None,
        compression_type=None, num_parallel_reads=1
    )

    dataset = dataset.enumerate()

    # frequency_domain_IQ,  tf.float32 (2,48)
    # day,                  tf.uint8 ()
    # transmitter_id,       tf.uint8 ()
    # transmission_id,      tf.uint8 ()
    # symbol_index_in_file, tf.int64 ()

    dataset = dataset.map(
        lambda index,frequency_domain_IQ: (
            tf.io.decode_raw(frequency_domain_IQ, tf.float32),
            tf.constant(metadata["day"], dtype=tf.uint8),
            tf.constant(metadata["transmitter_id"], dtype=tf.uint8),
            tf.constant(metadata["transmission_id"], dtype=tf.uint8),
            tf.cast(index, dtype=tf.int64)
        ),
        num_parallel_calls=10,
        deterministic=True
    )

    dataset = dataset.map(
        lambda frequency_domain_IQ,day,transmitter_id,transmission_id,symbol_index_in_file: (
            # IQ is interleaved. This is known as fortran order, and is easy to do in numpy
            # but is not directly supported in TF, so we do this BS
            tf.transpose(tf.reshape(frequency_domain_IQ, (48,2))),
            tf.reshape(day, () ),
            tf.reshape(transmitter_id, () ),
            tf.reshape(transmission_id, () ),
            tf.reshape(symbol_index_in_file, () ),
        ),
        num_parallel_calls=10,
        deterministic=True
    )


    return dataset

def tensor_to_np_bytes(symbol):
    members_as_bytes = [m.numpy().tobytes() for m in symbol]

    return b''.join(members_as_bytes)
    # return sum(members_as_bytes) # Nope

def symbol_dataset_to_file(dataset, out_path):
    with open(out_path, "wb") as f:
        for e in dataset:
            frequency_domain_IQ, day, transmitter_id, transmission_id, symbol_index_in_file = e

            members = [frequency_domain_IQ, day, transmitter_id, transmission_id, symbol_index_in_file]

            members_as_numpy = [m.numpy() for m in members]
            members_as_buffer = [m_np.tobytes() for m_np in members_as_numpy]
            
            for buffer in members_as_buffer:
                f.write(buffer)

# Files use format string. Ex: ="shuffled_batch-100_part-{part}.ds"
def symbol_dataset_to_split_files(dataset, out_dir, max_file_size_Bytes, file_format_str):
    current_file_index = 0
    current_file_size = 0
    current_file = open(out_dir + "/" + file_format_str.format(part=current_file_index), "wb")

    for batch in dataset:
        b = tensor_to_np_bytes(batch)


        if current_file_size + len(b) > max_file_size_Bytes:
            current_file.close()
            current_file_index += 1
            current_file_size = 0
            current_file = open(out_dir + "/" + file_format_str.format(part=current_file_index), "wb")
            print("Swapping to", file_format_str.format(part=current_file_index))
        current_file.write(b)
        current_file_size += len(b)

# If batch_size is 1, then will not batch at all, else will parse as batches.
# NB: Batch size must match the batch setting that the file was created with
def symbol_dataset_from_file(path, batch_size):
    symbol_size=384
    record_size=symbol_size + 1 + 1 + 1 + 8
    
    dataset = tf.data.FixedLengthRecordDataset(
        path, record_bytes=record_size*batch_size, header_bytes=None, footer_bytes=None, buffer_size=None,
        compression_type=None, num_parallel_reads=1
    )

    # return dataset

    dataset = dataset.map(
        lambda x: tf.strings.bytes_split(
            x, name=None
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    # print(dataset.element_spec)
    # sys.exit(1)

    # dataset = dataset.map(
    #     lambda x: tf.io.decode_raw(
    #         x, tf.uint8, little_endian=True, fixed_length=None, name=None
    #     )
    # )

    # dataset = dataset.map(
    #     lambda x: tf.reshape(x, (symbol_size,))
    # )

    # frequency_domain_IQ,  tf.float32 (2,48)
    # day,                  tf.uint8 ()
    # transmitter_id,       tf.uint8 ()
    # transmission_id,      tf.uint8 ()
    # symbol_index_in_file, tf.int64 ()



    # for e in dataset.take(1):
    #     print(tf.rank(e))
    
    # c = tf.constant([1,2,3,4,5,6,7,8])
    # o = tf.slice(c, [0], [1])

    # print(o)


    dataset = dataset.map(
        lambda x: (
            tf.strings.reduce_join(tf.slice(x, [0],                [symbol_size*batch_size])),
            tf.strings.reduce_join(tf.slice(x, [(symbol_size+0)*batch_size], [1*batch_size])), # Yes it's 0 because we are 0 indexed
            tf.strings.reduce_join(tf.slice(x, [(symbol_size+1)*batch_size], [1*batch_size])),
            tf.strings.reduce_join(tf.slice(x, [(symbol_size+2)*batch_size], [1*batch_size])),
            tf.strings.reduce_join(tf.slice(x, [(symbol_size+3)*batch_size], [8*batch_size])),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    # return dataset

    # dataset = dataset.map(
    #     lambda x: (
    #         tf.slice(x, [0],                [symbol_size*batch_size]),
    #         tf.slice(x, [(symbol_size+0)*batch_size], [1*batch_size]), # Yes it's 0 because we are 0 indexed
    #         tf.slice(x, [(symbol_size+1)*batch_size], [1*batch_size]),
    #         tf.slice(x, [(symbol_size+2)*batch_size], [1*batch_size]),
    #         tf.slice(x, [(symbol_size+3)*batch_size], [8*batch_size]),
    #     ),
    #     num_parallel_calls=10,
    #     deterministic=True
    # )


    # dataset = dataset.map(
    #     lambda frequency_domain_IQ, day, transmitter_id, transmission_id, symbol_index_in_file: (
    #         tf.strings.join(frequency_domain_IQ),
    #         tf.strings.join(day),
    #         tf.strings.join(transmitter_id),
    #         tf.strings.join(transmission_id),
    #         tf.strings.join(symbol_index_in_file)
    #     ),
    #     num_parallel_calls=10,
    #     deterministic=True
    # )

    dataset = dataset.map(
        lambda frequency_domain_IQ, day, transmitter_id, transmission_id, symbol_index_in_file: (
            tf.strings.reduce_join(frequency_domain_IQ),
            tf.strings.reduce_join(day),
            tf.strings.reduce_join(transmitter_id),
            tf.strings.reduce_join(transmission_id),
            tf.strings.reduce_join(symbol_index_in_file)
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    # dataset = dataset.map(
    #     lambda x: (
    #          tf.split(x, (symbol_size*batch_size, 1*batch_size, 1*batch_size, 1*batch_size, 8*batch_size))
    #     ),
    #     num_parallel_calls=10,
    #     deterministic=True
    # )

    dataset = dataset.map(
        lambda frequency_domain_IQ,day,transmitter_id,transmission_id,symbol_index_in_file: (
            tf.io.decode_raw(frequency_domain_IQ, tf.float32),
            tf.io.decode_raw(day, tf.uint8),
            tf.io.decode_raw(transmitter_id, tf.uint8),
            tf.io.decode_raw(transmission_id, tf.uint8),
            tf.io.decode_raw(symbol_index_in_file, tf.int64),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True
    )

    if batch_size == 1:
        dataset = dataset.map(
            lambda frequency_domain_IQ,day,transmitter_id,transmission_id,symbol_index_in_file: (
                tf.reshape(frequency_domain_IQ, (2,48)),
                tf.reshape(day, () ),
                tf.reshape(transmitter_id, () ),
                tf.reshape(transmission_id, () ),
                tf.reshape(symbol_index_in_file, () ),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True
        )
    else:
        dataset = dataset.map(
            lambda frequency_domain_IQ,day,transmitter_id,transmission_id,symbol_index_in_file: (
                tf.reshape(frequency_domain_IQ, (batch_size,2,48)),
                tf.reshape(day, (batch_size,) ),
                tf.reshape(transmitter_id, (batch_size,) ),
                tf.reshape(transmission_id, (batch_size,) ),
                tf.reshape(symbol_index_in_file, (batch_size,) ),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True
        )

    return dataset

def speed_test(iterable, batch_size=1):
    import time

    last_time = time.time()
    count = 0
    
    for i in iterable:
        count += 1

        if count % int(10000/batch_size) == 0:
            items_per_sec = count / (time.time() - last_time)
            print("Items per second:", items_per_sec*batch_size)
            last_time = time.time()
            count = 0

def check_if_symbol_datasets_are_equivalent(ds1, ds2):
    ds = tf.data.Dataset.zip((ds1, ds2))

    ds = ds.map(
        lambda one, two: (
                tf.math.reduce_all(
                    tf.reshape(
                        tf.math.equal(one[0], two[0]), (96,)
                    )
                ),
                # tf.math.equal(one[0], two[0]), 
                tf.math.equal(one[1], two[1]),
                tf.math.equal(one[2], two[2]),
                tf.math.equal(one[3], two[3]),
                tf.math.equal(one[4], two[4]),
        ),
        num_parallel_calls=10,
        deterministic=True
    )

    ds = ds.map(
        lambda a,b,c,d,e: 
            tf.math.reduce_all(tf.convert_to_tensor((a,b,c,d,e))),
        num_parallel_calls=10,
        deterministic=True
    )

    for e in ds.enumerate():
        if not e[1]:
            print("Datasets not equivalent, differ at index:", e[0])
            return False

    print("Datasets are equivalent")
    return True
    


if __name__ == "__main__":
    ds = symbol_dataset_from_file("t1", batch_size=1)
    print(ds.element_spec)

    for e in ds:
        print(e[4])