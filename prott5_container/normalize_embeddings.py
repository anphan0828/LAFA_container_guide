import pandas as pd
import sys

input_file=sys.argv[1]
df = pd.read_csv(input_file, sep="\t")
x_max = df["Similarity"].max()
def norm_sim(x):
    x = (x_max - x)/x_max
    sim_per = x*100
    return round(sim_per, 2)
# TODO: current similarity is Euclidean distance, try using something else instead
df["Similarity"] = df["Similarity"].map(norm_sim)

# Format output using 
output_norm = input_file.replace(".tsv","_norm.tsv")

df2 = df.copy()
df2 = df2.iloc[:,1:]
# Remove 'sp|' prefixes
df2 = df2.replace(r'sp\|', '', regex=True)
# Remove 'tr|' prefixes  
df2 = df2.replace(r'tr\|', '', regex=True)
# Remove everything from '|' to the next tab (keeping the tab)
df2 = df2.replace(r'\|.*', '', regex=True)


df2.to_csv(output_norm, sep="\t", index=False)