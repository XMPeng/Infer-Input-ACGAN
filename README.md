# Infer-Input-ACGAN
PyTorch implementation of the methods described in ICIP 2020 submission of "INFER THE INPUT TO THE GENERATOR OF AUXILIARY CLASSIFIER GENERATIVE ADVERSARIAL NETWORKS"

Take the following steps to test the code for the Intel Image Classification Dataset.
1. Download the dataset from https://www.kaggle.com/puneet6060/intel-image-classification.

2. Run main_Xiaoming_intel.py to train an ACGAN.

3. Run invert_all_intel.py for Invert-ACGAN-approx.

4. Run modify_gen.py to decompose the ACGAN into a series of regular GANs.

5. Run invert_class_specific_intel.py for Invert-ACGAN-decomposed.

6. Run main_infer_intel3.py to train an ACGAN with an encoder.

7. Run invert_infer_intel.py for Infer-ACGAN-encoder.
