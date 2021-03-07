import pandas as pd
from pathlib import Path
import sqlalchemy 
import pymysql

db = 'restbase'
dbdir = Path.cwd() / 'data' / db
dbdir.mkdir(parents=True, exist_ok=True)
url = 'mysql+pymysql://guest:relational@relational.fit.cvut.cz:3306/{}'
url = url.format(db)
engine = sqlalchemy.create_engine(url)
metadata = sqlalchemy.MetaData(bind=engine.connect(), reflect=True)

with engine.connect() as conn:
    tables = list(metadata.tables.keys())
    for t in tables:
        df = pd.read_sql_table(t, conn, index_col=0)
        df.to_csv(dbdir / f'{t}.csv', index = False)
  