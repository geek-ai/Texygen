FROM ubuntu:14.04
ENV HOME /root
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN apt-get update
RUN apt-get install -y --force-yes --no-install-recommends apt-utils apt-transport-https software-properties-common iputils-ping wget
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && bash miniconda.sh -b -p $HOME/miniconda 
ENV PATH $HOME/miniconda/bin:$PATH
RUN conda install numpy=1.12.1 tensorflow==1.3.0 nltk==3.2.3 -y
RUN conda install -c rdkit rdkit -y
RUN mkdir $HOME/DiversityNet/
COPY . $HOME/DiversityNet/
WORKDIR $HOME/DiversityNet/