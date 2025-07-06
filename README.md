# ViTacFormer: Learning Cross-Modal Representation for Visuo-Tactile Dexterous Manipulation

#### Project Website: https://roboverseorg.github.io/ViTacFormerPage/

### Installation

    conda create -n vitacformer python=3.8.10
    conda activate vitacformer
    pip install torchvision
    pip install torch
    pip install opencv-python
    pip install matplotlib
    pip install tqdm
    pip install einops
    pip install h5py
    pip install ipython
    pip install transforms3d
    pip install zarr
    pip install transformers
    cd dataset/ha_data && pip install -e .
    cd detr && pip install -e .

### Example Usages

Please download and unzip the example data [here](https://drive.google.com/file/d/1GzQSymfzw2YDY0VtCyutV5LnmFRbu0nm/view?usp=sharing). To train ViTacFormer, run:

    conda activate vitacformer
    bash train.sh
