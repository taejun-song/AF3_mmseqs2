# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

FROM nvidia/cuda:12.6.3-base-ubuntu24.04

RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update --quiet \
    && apt-get install --yes --quiet python3.12 python3-pip python3.12-venv python3.12-dev \
    && apt-get install --yes --quiet git wget gcc g++ make zlib1g-dev zstd

RUN python3 -m venv /alphafold3_venv
ENV PATH="/mmseqs/bin:/hmmer/bin:/alphafold3_venv/bin:$PATH"
RUN pip3 install --no-cache-dir --upgrade pip

RUN mkdir -p /mmseqs_build /mmseqs/bin && \
    cd /mmseqs_build && \
    wget https://github.com/soedinglab/MMseqs2/releases/download/15-6f452/mmseqs-linux-avx2.tar.gz && \
    tar xzf mmseqs-linux-avx2.tar.gz && \
    mv mmseqs/bin/* /mmseqs/bin/ && \
    chmod +x /mmseqs/bin/mmseqs && \
    rm -rf /mmseqs_build

RUN mkdir /hmmer_build /hmmer && \
    wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz --directory-prefix /hmmer_build && \
    (cd /hmmer_build && echo "ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3 hmmer-3.4.tar.gz" | sha256sum --check) && \
    (cd /hmmer_build && tar zxf hmmer-3.4.tar.gz && rm hmmer-3.4.tar.gz)

COPY docker/jackhmmer_seq_limit.patch /hmmer_build/
RUN (cd /hmmer_build && patch -p0 < jackhmmer_seq_limit.patch)

RUN (cd /hmmer_build/hmmer-3.4 && ./configure --prefix /hmmer) && \
    (cd /hmmer_build/hmmer-3.4 && make -j) && \
    (cd /hmmer_build/hmmer-3.4 && make install) && \
    (cd /hmmer_build/hmmer-3.4/easel && make install) && \
    rm -R /hmmer_build

COPY . /app/alphafold
WORKDIR /app/alphafold

RUN pip3 install --no-cache-dir setuptools \
    && pip3 install --no-cache-dir -r dev-requirements.txt \
    && pip3 install --no-cache-dir --no-deps .

RUN build_data

ENV XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
ENV XLA_PYTHON_CLIENT_PREALLOCATE=true
ENV XLA_CLIENT_MEM_FRACTION=0.95

CMD ["python3", "run_alphafold.py"]
