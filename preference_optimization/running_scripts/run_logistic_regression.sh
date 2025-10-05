# randomly generate annotators of size 5 from 0 - 35

# iterate 1000 
for i in {1..1000}
do
python logistic_regression.py --annotator_ids $(shuf -i 0-35 -n 5 | tr '\n' ',') --save_name imdb_qwen_dpo
done
