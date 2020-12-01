import csv
import pandas as pd

sub1=pd.read_csv('sub1.csv')
sub2=pd.read_csv('sub2.csv')


sub1=sub1.sort_values(by=['Id']).reset_index(drop=True)
sub2=sub2.sort_values(by=['Id']).reset_index(drop=True)



sub1["Label"]=sub1["Label"]*0.5+sub2["Label"]*0.5



sub1.to_csv('merge_sub.csv')