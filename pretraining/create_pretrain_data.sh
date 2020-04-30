
base="albert"
#infiles="datasets/tenten/preproc_sentences/tenten_proc_101.txt,datasets/tenten/preproc_sentences/tenten_proc_102.txt"
#infiles="datasets/tenten/preproc_sentences/test1.txt"
bn=`basename "$1"`
outdir="datasets/tta512"


python3 $base/create_pretraining_data.py \
	--input_file "$1" \
	--output_file "$outdir/$bn.pre" \
	--vocab_file ../corpus/spm/spm_tta_30K.vocab \
	--spm_model_file ../corpus/spm/spm_tta_30K.model \
	--max_seq_length 512 \
	--meta_data_file_path "$bn.meta" \
	--do_lower_case=True \
	--dupe_factor 1 \
	--do_whole_word_mask=False
