import subprocess
import pandas as pd
from tqdm.auto import tqdm
root = "./btc_data"
py   = "python" # if linux or mac,then use `python3`
days     = ["seven","thirty","ninety"]
result   = []
model_names = ["tcn","cnn","bilstm","ann",'transformer']

for day in tqdm( days ):
    bar = tqdm( model_names )
    for name in bar:
        ret = subprocess.getstatusoutput(f'{py} main.py --csv="{root}/reg_{day}.csv"  --model_name="{name}" --epochs=2000')
        metrics = ret[1].split("\n")[-1]
        metrics = metrics.strip()
        metrics = eval(metrics)
        metrics["window_size"] = day
        metrics["model_name"] = name
        result.append(metrics)
        # print(metrics)
        bar.set_postfix(metrics)

pd.DataFrame(result).to_csv("./result_future_regression.csv",index=False)

print(result)