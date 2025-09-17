```
conda create -n handbook python=3.10.14 && conda activate handbook

git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
rm -r .git
python -m pip install .

conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia

python -m pip install flash-attn --no-build-isolation
```