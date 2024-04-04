import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import pickle

# 1. データの読み込み
api_data = []
for i in range(1000):
    df = pd.read_csv(f'./sample_data/api_{i}.csv')
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    # 前日のアクセス数が存在しない場合は、同じ月の平均で補完
    df['prev_count'] = df.groupby(pd.to_datetime(df['date']).dt.month)['access_count'].shift(1).fillna(df.groupby(pd.to_datetime(df['date']).dt.month)['access_count'].transform('mean'))
    
    # 1週間前のアクセス数が存在しない場合は、同じ週の平均で補完
    df['week_ago_count'] = df.groupby(pd.to_datetime(df['date']).dt.isocalendar().week)['access_count'].shift(1).fillna(df.groupby(pd.to_datetime(df['date']).dt.isocalendar().week)['access_count'].transform('mean'))
    
    # 1ヶ月前のアクセス数が存在しない場合は、同じ月の平均で補完
    df['month_ago_count'] = df.groupby(pd.to_datetime(df['date']).dt.month)['access_count'].shift(1).fillna(df.groupby(pd.to_datetime(df['date']).dt.month)['access_count'].transform('mean'))
    
    # 1年前のアクセス数が存在しない場合は、同じ月の平均で補完
    df['year_ago_count'] = df.groupby(pd.to_datetime(df['date']).dt.month)['access_count'].shift(12).fillna(df.groupby(pd.to_datetime(df['date']).dt.month)['access_count'].transform('mean'))
    
    api_data.append(df)

# 2. 特徴量の抽出
features = ['month', 'day_of_week', 'prev_count', 'week_ago_count', 'month_ago_count', 'year_ago_count']
X = pd.concat(api_data, ignore_index=True)[features].values

# 3. データの標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. OneClassSVMの学習
ocsvm = OneClassSVM(kernel='rbf', gamma='auto')
ocsvm.fit(X_scaled)

#学習モデルの保存
with open('model.pickle', mode='wb') as f:
    pickle.dump(ocsvm, f, protocol=2)

# 5. 異常スコアの計算
y_pred = ocsvm.decision_function(X_scaled)

# 6. 異常検知
threshold = -0.5  # 適切な閾値を設定
anomalies = X_scaled[y_pred < threshold]
print(f"異常データの数: {len(anomalies)}")