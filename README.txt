I. STUDENT INFO
-------------------------
Deepa Borkar
dborkar@uchicago.edu

II. FILES INCLUDED
-------------------------
README.txt
read_data.py
final_from_scratch.py 
final_from_pkg.py
data/<data files>
plot/<plots>
ADA_Final_Project.pdf

III. DESCRIPTION
-------------------------
This project explores dimensionality reduction through the Singular Value Decomposition and Non-Negative Matrix Factorization methods. I have written these methods using Stochastic Gradient Descent in the final_from_scratch.py file and have used the sklearn packages for SVD and NMF in the final_from_pkg.py file. The dataset used is from the Netflix Kaggle Competition.

IV. HOW TO RUN PROGRAM
-------------------------
I was not able to put the Netflix Prize dataset in the github repo due to size restrictions. Before running the python files, please download the file combined_data_1.txt (and other combined data files if you would like) and put the file in the data directory of this repo. The dataset can be found here: https://www.kaggle.com/netflix-inc/netflix-prize-data.

The following are how to run the python files:
    python3 read_data.py
    
    python3 final_from_scratch.py <path to data file>
    Example: 
    python3 final_from_scratch.py data/sample_data.csv

    python3 final_from_pkg.py <path to data file>
    Example: 
    python3 final_from_pkg.py data/data.csv


V. QUESTIONS
-------------------------
For any questions or issues, please e-mail dborkar@uchicago.edu.

