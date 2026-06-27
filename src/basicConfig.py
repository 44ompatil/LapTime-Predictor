import pathlib
import fastf1 as f1
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

class config:
	def __init__(self) -> None:
		pass
		
	def enableCache(self, cachePath : Path =  Path("cache")):
		try:
			if not pathlib.Path.is_dir(cachePath):
				pathlib.Path.mkdir(cachePath, exist_ok=True, parents=True)
			f1.Cache.enable_cache(cache_dir=str(cachePath))
			return True
		except Exception as e:
			print(f"ERROR : {e}")
		return False
	
	def exportDatatoParquet(self, data : pd.DataFrame, parquetPath : Path = Path(r"./data/raw/rawData.parquet")):
		try:
			table = pa.Table.from_pandas(data)
			pq.write_table(table=table, where=parquetPath)
		except Exception as e:
			print(f"ERROR : {e}")
	
	def importDataFromParquet(self, parquetPath : Path = Path("./data/raw/rawData.parquet")):
		try:
			table = pq.read_table(parquetPath)
			return table.to_pandas() if table else None
		except Exception as e:
			print(f"ERROR : {e}")
