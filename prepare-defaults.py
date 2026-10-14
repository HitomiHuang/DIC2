import sys, json, pathlib, hashlib
sys.path.insert(0, '.qa-deps')
sys.path.insert(0, 'dist')
from engine import experiment
import pandas as pd
path = pathlib.Path('dist/50_Startups.csv')
text = path.read_text(encoding='utf-8-sig')
df = pd.read_csv(path)
assert df.shape == (50, 5)
assert df['Profit'].iloc[0] == 192261.83
result = json.loads(experiment(text, 'Profit', 0.2, 42))
assert len(result['rows']) == 25 and result['p'] == 5
payload = dict(csv=text, result=result, target='Profit', test=0.2, seed=42,
               source='https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv',
               sha256=hashlib.sha256(path.read_bytes()).hexdigest())
pathlib.Path('dist/startup-data.js').write_text('window.STARTUP_DATA = '+json.dumps(payload, ensure_ascii=False)+';\n', encoding='utf-8')
print('Verified 50 rows, 5 raw columns, 5 encoded predictors, 25 calculated model results.')
