Part 1: Data download, integration and preprocess.

In order to download the data follow the steps 1 to 6 of Part 1 on https://github.com/yuesOctober/GDCproject
Please choose all disease types instead of a single one.

After downloading the data, please check the integrity of it by executing:
python3 check.py
(Please edit the path accordingly)

If some files fail download, use the following command:
./<path-to-gdc-client>/gdc-client download <id>

Once we get the biomarker files. We also need get the case ids related to the files:
python3 parse_file_case_id.py
(Please edit the path accordingly)

Get the meta data for the files and corresponding cases:
python3 request_meta.py
(Please edit the path accordingly)

Now we can generate the miRNA matrix for all the files with labeled normal or specific tumor:
python3 gen_miRNA_matrix.py
(Please edit the path accordingly)


Part 2: Apply Machine Learning Package (sklearn) to the above data.

In order to execute the Linear learner model, the following has to be done:

