import json
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import sentencepiece as spm

seq_length = 256
max_predictions_per_seq = 20

name_to_features = {
    'input_ids':
        tf.io.FixedLenFeature([seq_length], tf.int64),
    'input_mask':
        tf.io.FixedLenFeature([seq_length], tf.int64),
    'segment_ids':
        tf.io.FixedLenFeature([seq_length], tf.int64),
    'masked_lm_positions':
        tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
    'masked_lm_ids':
        tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
    'masked_lm_weights':
        tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
    'next_sentence_labels':
        tf.io.FixedLenFeature([1], tf.int64),
}

def decode_record(record, name_to_features=name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.io.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.cast(t, tf.int32)
    example[name] = t

  return example

spm_name = sys.argv[2]
sp = spm.SentencePieceProcessor()
sp.Load(spm_name)

def splice(ids, masked, sp):
    iit = iter(ids)
    mit = iter(masked)
    
    out = []
    for i in ids:
        if i == 0:
            try:
                out.append(f"[{sp.id_to_piece(next(mit))}]")
            except StopIteration:
                break
        else:
            out.append(sp.id_to_piece(i))
            
    return "".join(out)


def rsplice(record, sp):
    ids = record["input_ids"]
    ids = list(map(int, ids.numpy()))
    #print(ids)
    masked = record["masked_lm_ids"]
    masked = list(map(int, masked.numpy()))
    mit = iter(masked)
    poss = list(record["masked_lm_positions"].numpy()) + [0]
    posit = iter(poss)
    nextpos = next(posit)
    
    out = []
    for i, id_ in enumerate(ids):
        if i == nextpos:
            nextpos = next(posit)
            out.append(f"[{sp.id_to_piece(next(mit))}]")
        else:
            out.append(sp.id_to_piece(id_))
            
    return "".join(out).replace("▁", " ")


def decode(record, sp):
    ids = record["input_ids"]
    ids = list(map(int, ids.numpy()))
    print(ids)
    masked = record["masked_lm_ids"]
    masked = list(map(int, masked.numpy()))
    
    return splice(ids, masked, sp)
    #print(sp.decode_ids(masked))
    #print(" ".join(map(sp.id_to_piece, masked)))

    #return sp.decode_ids(ids)
    #return "".join(map(sp.id_to_piece, ids))

    pass


name = sys.argv[1]

print("Opening", name)
d =  tf.data.TFRecordDataset([name])

count = 20
for s in d:
    sample = decode_record(s)
    print(rsplice(sample, sp))
    print()
    count -= 1
    if count <= 0:
        break
