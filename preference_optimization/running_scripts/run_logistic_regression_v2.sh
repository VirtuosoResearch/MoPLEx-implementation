# randomly generate annotators of size 5 from 0 - 35

# sleep one hour
sleep 3600

# iterate 1000 
for i in {1..1000}
do
CUDA_VISIBLE_DEVICES=1 python logistic_regression.py --annotator_ids $(shuf -i 0-35 -n 5 | tr '\n' ',') --save_name imdb_qwen
done
