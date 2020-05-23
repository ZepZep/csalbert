# Create SentencePiece model from text file. Each sentence on its own line.
# The recomanded file size is 10 000 000 sentences (approx. 1 GB)

# this configuration adds special tokens (control symbols) used in ALBERT

import sentencepiece as spm

files="spm/tta.txt"

spm.SentencePieceTrainer.Train(
    f"--input={files} --model_prefix=spm_tta_30K --vocab_size=30000 "
    f"--pad_id=0 --unk_id=1 --eos_id=-1 --bos_id=-1 "
    f"--control_symbols=[CLS],[SEP],[MASK] "
    f'--user_defined_symbols=(,),\',- '
    f"--shuffle_input_sentence=true --input_sentence_size=10000000 "
    f"--character_coverage=1 --model_type=unigram"
)
