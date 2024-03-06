## LEAD: Learning Decomposition for Source-free Universal Domain Adaptation

*The code repo of our anonymous paper submitted to CVPR-2024, "LEAD: Learning Decomposition for Source-free Universal Domain Adaptation".*

### Prerequisites
1. python3, pytorch, numpy, PIL, scipy, sklearn, tqdm
2. RTX-3090 with CUDA 11.3

To reproduce the results, we present the conda environment file.

### Step
1. Please prepare the environment first.
2. Please download the datasets from the corresponding official websites, and then unzip them to the `./data` folder.
3. Preparing the source model.
4. Performing the target model adaptation.

### Demo
Here, we present a demo in the OPDA scenario on VisDA dataset.
> `bash scripts/train_visda_opda.sh`

