import polars as pl
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# 祝日データ
holidays = [
    datetime(2019, 1, 1),  # 元日
    datetime(2019, 1, 14), # 成人の日
    datetime(2019, 2, 11), # 建国記念の日
    # 他の祝日も追加
]

def is_holiday(date):
    return date.weekday() >= 5 or date in holidays

def generate_test_data(num_apis: int, start_date: datetime, end_date: datetime) -> list[pl.DataFrame]:
    test_data = []
    for api_id in range(num_apis):
        rows = []
        current_date = start_date
        while current_date <= end_date:
            if is_holiday(current_date):
                if current_date.month <= 6:
                    access_count = np.random.normal(15000, 800)
                else:
                    access_count = np.random.normal(20000, 1000)
            else:
                if current_date.month <= 6:
                    access_count = np.random.normal(10000, 500)
                else:
                    access_count = np.random.normal(12000, 600)
            
            # 約2%の確率で異常なアクセス数を発生させる
            if np.random.rand() < 0.02:
                access_count *= np.random.uniform(5, 10)
            
            rows.append({
                'date': current_date.strftime("%Y-%m-%d"),
                'access_count': int(max(0, access_count))
            })
            current_date += timedelta(days=1)
        
        test_data.append(pl.DataFrame(rows))
    
    return test_data

# テストデータの生成
start_date = datetime(2019, 1, 1)
end_date = datetime(2023, 12, 31)
test_data: list[pl.DataFrame] = generate_test_data(5000, start_date, end_date)

# テストデータの保存
output_path = Path("./sample_data/")
output_path.mkdir(parents=True, exist_ok=True)
for i, df in enumerate(test_data):
    file_path = output_path.joinpath(f'api_{i}.csv')
    df.write_csv(file_path)