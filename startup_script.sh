# Download Miniconda Installer
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Source the bashrc file to set up correct paths
source ~/.bashrc

# Create conda environment
conda create -n pytorch0.3 python=3.6

# Source activate environment
source activate pytorch0.3

# Pip install pytorch
pip install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp36-cp36m-linux_x86_64.whl

# Clone the repo
git clone https://github.com/DevSinghSachan/ssl_text_classification

# Install the requirements
pip install -r requirements.txt