import pandas as pd
import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directories', nargs='+', default=[])
    return parser.parse_args()

def create_text_files(dir,df,filename):
    genes = df.columns
    data = df.to_numpy().T
    with open(dir+'/genes.txt', 'w') as f:
        for g in genes:
            f.write(g)
            f.write('\n')
    with open('{}/{}.txt'.format(dir, filename), 'w') as f:
        f.write('\t'.join(['gene']+["Sample{}".format(i+1) for i in range(data.shape[1])]))
        f.write('\n')
        for line in np.arange(data.shape[0]):
            f.write('\t'.join([genes[line]]+ ["{:.2f}".format(a) for a in data[line]]) + '\n')

def convert_to_csv(dir):
    df_network = pd.read_csv("{}/{}_goldstandard.tsv".format(dir,dir), sep='\t', names=['start','end', 'edge'],header=0)
    df_network.to_csv("{}/{}_goldstandard.csv".format(dir,dir), index=False)
    df_wild = pd.read_csv("{}/{}_wildtype.tsv".format(dir,dir), sep='\t', header=0)

    df_time = pd.read_csv("{}/{}_timeseries.tsv".format(dir,dir), sep='\t', header=0)
    df_mult = pd.read_csv("{}/{}_multifactorial.tsv".format(dir,dir),sep='\t', header=0)

    df_time_unperturb = df_time[(df_time["Time"] > 500) & (df_time["Time"] <= 1000)].drop(columns="Time")
    df_obs = pd.concat([df_wild,df_mult,df_time_unperturb])

    # NORMALIZE
    mean = df_obs.mean(axis=0)
    stddev = df_obs.std(axis=0)
    df_norm = (df_obs - mean) / stddev
    df_obs = df_norm
    create_text_files(dir,df_obs, "data_obs")

    df_obs['target'] = np.zeros(df_obs.shape[0])
    df_obs.to_csv("{}/{}_obs.csv".format(dir,dir), index=False)

    # Normalize interventional data separately
    df_ko = pd.read_csv("{}/{}_knockouts.tsv".format(dir,dir), sep='\t', header=0)
    df_kd = pd.read_csv("{}/{}_knockdowns.tsv".format(dir,dir), sep='\t', header=0)
    df_inter = pd.concat([df_ko, df_kd])
    mean = df_inter.mean(axis=0)
    stddev = df_inter.std(axis=0)
    df_norm = (df_inter - mean) / stddev
    df_inter = df_norm
    targets = np.arange(1,df_ko.shape[0]+1)
    df_inter['target'] = np.concatenate([targets, targets])
    df_combine = pd.concat([df_obs, df_inter])
    df_combine.to_csv("{}/{}_combine.csv".format(dir,dir), index=False)
    create_text_files(dir,df_combine, "data_combine")

args = get_args()
for d in args.directories:
    print(d)
    convert_to_csv(d)
