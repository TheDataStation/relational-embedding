import pandas as pd
from pathlib import Path
import argparse
import sqlalchemy
import pymysql


def download_csv(args):
    db = args.task
    url = 'mysql+pymysql://guest:relational@relational.fit.cvut.cz:3306/{}'
    url = url.format(db)
    engine = sqlalchemy.create_engine(url)
    metadata = sqlalchemy.MetaData(bind=engine.connect(), reflect=True)

    with engine.connect() as conn:
        dbdir = Path.cwd() / "data" / db
        dbdir.mkdir(parents=True, exist_ok=True)
        tables = list(metadata.tables.keys())
        print(tables)
        for t in tables:
            df = pd.read_sql_table(t, conn, index_col=0)
            df.to_csv(dbdir / f'{t}.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True,
                        help="which task to download")
    args = parser.parse_args()
    download_csv(args)
