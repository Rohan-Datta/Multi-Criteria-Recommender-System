import pandas as pd

def post_process(recommendations):
    title_lookup = pd.read_csv('title_lookup.tsv',sep='\t')
    
    for r in recommendations:
        row = title_lookup.loc[title_lookup['tconst']==r]
        row.drop(['tconst'],axis=1)
        print('\n')
        print(row)
