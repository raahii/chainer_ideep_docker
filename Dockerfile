FROM continuumio/miniconda3:latest

WORKDIR chainer_ideep

# apt
RUN apt-get -y update && \
    apt-get install -y git mercurial build-essential libssl-dev libbz2-dev libreadline-dev libsqlite3-dev curl

# install ideep4py, chainer
RUN pip install -U pip && \
    pip install ideep4py chainer pillow imageio && \
    conda install -y opencv

# try image-net example
RUN git clone https://github.com/terasakisatoshi/chainer-imagenet && \
    cd chainer-imagenet && \
    curl -L -O http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz && \
    tar xvzf 101_ObjectCategories.tar.gz && \
    python reshape.py --source_dir 101_ObjectCategories --target_dir reshaped && \
    python create_dataset.py reshaped && \
    python compute_mean.py train.txt

WORKDIR /chainer_ideep/chainer-imagenet

RUN curl -L -O https://gist.github.com/ikeyasu/5e834ba10527b4eec22c16857ded45e7/raw/fa5edcfd5b9b6a1d8f50fed9dbfadada52853794/model_epoch_46.npz && \
    mkdir -p result && \
    mv model_epoch_46.npz result/

COPY predict.py .
