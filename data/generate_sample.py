from pathlib import Path
import pandas as pd 
import argparse


items = ["macbook", "book", "iphone"]
prices = [1000, 200, 500]

def create_sample(k):
    k = int(k)
    print("Create a multiple of", k)
    def f(x):
        return "type" + str(x % 3 + 1) + "_" + str(x)
    df = pd.DataFrame({"id": range(1000*k)})
    df["name"] = df["id"].apply(f)
    df["money"] = df["id"].apply(lambda x: prices[x % 3])
    folder_name = "sample"
    dbdir = Path.cwd() / folder_name
    print(dbdir, df.shape)
    dbdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(dbdir / "base.csv", index=False)
    
    price = pd.DataFrame({"item": items, "price": prices})
    price.to_csv(dbdir / "price.csv", index=False)
    
    df2 = pd.DataFrame({"name": df["name"], "item": df["id"].apply(lambda x: items[x % 3])})
    df2.to_csv(dbdir / "trans.csv", index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--times',
                        type=str,
                        default=1,
                        help='times to run exp')
    args = parser.parse_args()
    create_sample(args.times)