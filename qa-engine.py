import sys, json
sys.path.insert(0, '.qa-deps')
sys.path.insert(0, 'dist')
from engine import experiment
import numpy as np
import pandas as pd
rng = np.random.default_rng(42)
x = rng.normal(size=(90, 3))
df = pd.DataFrame(x, columns=['a', 'b', 'c'])
df['region'] = ['North', 'South', 'West'] * 30
df['profit'] = 100 + 8*x[:, 0] - 2*x[:, 1] + rng.normal(size=90)
df.loc[0, 'a'] = np.nan
csv = df.to_csv(index=False)
a = json.loads(experiment(csv, 'profit'))
b = json.loads(experiment(csv, 'profit'))
assert a == b, 'Seed must reproduce results'
assert len(a['rows']) == 5*a['maxk']
assert {r['method'] for r in a['rows']} == {'elastic', 'rfe', 'forward', 'lasso', 'exhaustive'}
assert all(len(r['features']) == r['k'] for r in a['rows'])
assert all(np.isfinite(r['rmse']) and np.isfinite(r['r2']) for r in a['rows'])
assert a['train'] == 72 and a['test'] == 18
for r in a['rows']:
    point = r['residuals'][0]
    source = df.iloc[point['sample']-1]
    values = dict(source)
    values.update(region_South=float(source['region']=='South'), region_West=float(source['region']=='West'))
    reconstructed = r['intercept'] + sum(c*values[f] for f,c in zip(r['features'],r['coefficients']))
    assert np.isclose(reconstructed, point['predicted']), 'Original-unit coefficients must reproduce predictions'
try:
    experiment(df.head(5).to_csv(index=False), 'profit')
    raise AssertionError('Short CSV must be rejected')
except ValueError:
    pass
print('PASS: five methods, missing data, categorical encoding, deterministic split, finite metrics, invalid data rejection')
