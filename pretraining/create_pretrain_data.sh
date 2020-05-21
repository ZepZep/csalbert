# script for running pretraining data creation

# usage: ./create_pretrain_data.sh corpus.txt

# multiple in parallel:
# find datasets/preproc/tenten/*.txt | parallel --ungroup -j 6 ./create_pretrain_data.sh {}

outdir="datasets/tta256"           # output directory
spm="../corpus/spm/spm_tta_30K"    # SentencePiece model

infile="$1"                        # input corpus txt file
bn=`basename "$infile"`

python3 albert/create_pretraining_data.py \
	--input_file "$1" \
	--output_file "$outdir/$bn.pre" \
	--meta_data_file_path "$bn.meta" \
	--vocab_file "$spm.vocab" \
	--spm_model_file "$spm.model" \
	--max_seq_length 256 \
	--do_lower_case=True \
	--dupe_factor 1 \
	--do_whole_word_mask=False
