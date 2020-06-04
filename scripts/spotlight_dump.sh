
lang=$1
port=$2
util_dir=data/AMR/en_es_it_de_utils
if [[ "$lang" = "en" ]]; then
  test_data=data/AMR/amr_2.0/test.txt
else
  test_data=data/AMR/amr_2.0/test_${lang}.txt
fi
mkdir ${util_dir}/spotlight
spotlight_path=spotlight/${lang}_test_spotlight_wiki
printf "Wikification...`date`\n"
python -u -m xlamr_stog.data.dataset_readers.amr_parsing.postprocess.wikification \
    --amr_files ${test_data} \
    --util_dir ${util_dir}\
    --spotlight_wiki $spotlight_path\
    --spotlight_port $port\
    --lang ${lang}\
    --dump_spotlight_wiki
printf "Done.`date`\n\n"

