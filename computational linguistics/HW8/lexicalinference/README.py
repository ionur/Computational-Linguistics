#MODEL1 :
#./diy_datasets/hearst_diy_combined.txt is the file used to generate results for model 1

#MODEL2:

#run diy_models2.py
#This will generate files in bless2011 folder with diy_0.txt, diy_1.txt and so on till diy_9.text

#For the convenience these generated are also submitted in the diy_datasets folder.

#The diy_0..diy_9.txt files generated in the bless2011 folder are manually added to hearst_diy_combined.txt
#and labelled hearst_diy_combined_0.txt to hearst_diy_combined_9.text
#For your convenience, these files are also included in the submission
#So these files can be directly used to generate the results for train/val/test sets
#Each of this hearst_diy_combined_<no>.text is then run using run_pred.sh to generate the train/validation results.
#pred.sh is just a simple script that wraps the extractDatasetPredictions.py and computePRF.py
#pred.sh is also included in the submission
