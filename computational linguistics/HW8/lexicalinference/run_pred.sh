#/bin/sh

#python hearst/extractHearstHyponyms.py --inputwikifile ../wikipedia_sentences.txt --outputfile ../hearst.txt

#python3 hearst_post_processing.py
#python3 extractDatasetPredictions.py --extractionsfile ../hearst_p.txt --trdata ../bless2011/data_lex_train.tsv --valdata ../bless2011/data_lex_val.tsv --testdata ../bless2011/data_lex_test.tsv --trpredfile ../bless2011/train_pred.txt --valpredfile ../bless2011/val_pred.txt --testpredfile ../bless2011/hearst.txt
#echo "Train Results"
#python3 computePRF.py --goldfile ../bless2011/data_lex_train.tsv --predfile ../bless2011/train_pred.txt
#echo "Validation results"
#python3 computePRF.py --goldfile ../bless2011/data_lex_val.tsv --predfile ../bless2011/val_pred.txt

for i in 0 1 2 3 4 5 6 7 8 9
do

  echo 'In Interation:'$i
  python3 extractDatasetPredictions.py --extractionsfile ../diy_$i.txt --trdata ../bless2011/data_lex_train.tsv --valdata ../bless2011/data_lex_val.tsv --testdata ../bless2011/data_lex_test.tsv --trpredfile ../bless2011/train_pred.txt --valpredfile ../bless2011/val_pred.txt --testpredfile ../bless2011/hearst.txt
  echo ' '
  echo "Train Results"
  python3 computePRF.py --goldfile ../bless2011/data_lex_train.tsv --predfile ../bless2011/train_pred.txt
  echo ' '
  echo "Validation results"
  python3 computePRF.py --goldfile ../bless2011/data_lex_val.tsv --predfile ../bless2011/val_pred.txt

done
