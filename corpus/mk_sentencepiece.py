import sentencepiece as spm

files = [f"tenten_proc_{i}.txt" for i in range(101, 136 + 1)]
prefix = "preproc/tenten/"

files = ",".join(prefix + f for f in files)
spm.SentencePieceTrainer.Train(
    f"--input {files} --model_prefix=spm_tneten_30K --vocab_size=30000 --logtostderr"
    f"--pad_id=0 --unk_id=1 --eos_id=-1 --bos_id=-1"
    f"--control_symbols=[CLS],[SEP],[MASK]"
    f"--shuffle_input_sentence=true --input_sentence_size=10000000"
    f"--character_coverage=1 --model_type=unigram"
)
