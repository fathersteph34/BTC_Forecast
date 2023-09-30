import subprocess
import pandas as pd
from tqdm.auto import tqdm
root = "./btc_data"
py   = "python" # if linux or mac,then use `python3`
model_names = ["tcn","cnn","bilstm","ann",'transformer']
result = []

for i in tqdm( range(1,4) ):
    bar = tqdm( model_names )
    for name in bar:
        command = f'{py} main.py --csv="{root}/reg_interval{i}.csv"  --model_name="{name}" --epochs=2000'
        print(command)
        ret = subprocess.getstatusoutput(f'{py} main.py --csv="{root}/reg_interval{i}.csv"  --model_name="{name}" --epochs=2000')
        metrics = ret[1].split("\n")[-1]
        metrics = metrics.strip()
        print(metrics)
        metrics = eval(metrics)
        metrics["interval"] = f"interval{i}"
        metrics["model_name"] = name
        result.append(metrics)
        # print(metrics)
        bar.set_postfix(metrics)

pd.DataFrame(result).to_csv("./result_interval_regression.csv",index=False)

print(result)