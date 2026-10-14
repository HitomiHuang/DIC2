"""Independent NumPy audit of all 25 default chart points.

Uses neither engine.experiment nor sklearn estimators/metrics/selection.
The separate engine run checks that the cached results are up to date.
"""
import csv
import hashlib
import io
import json
import pathlib
import sys
from itertools import combinations

sys.path.insert(0, '.qa-deps')
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
payload_text = (ROOT / 'dist/startup-data.js').read_text(encoding='utf-8')
payload = json.loads(payload_text.removeprefix('window.STARTUP_DATA = ').strip().removesuffix(';'))
raw_bytes = (ROOT / 'dist/50_Startups.csv').read_bytes()
assert hashlib.sha256(raw_bytes).hexdigest() == payload['sha256']
raw_text = raw_bytes.decode('utf-8-sig')
assert list(csv.reader(io.StringIO(raw_text))) == list(csv.reader(io.StringIO(payload['csv'])))
upstream = ROOT / 'upstream-50_Startups.csv'
if upstream.exists():
    assert list(csv.reader(io.StringIO(upstream.read_text(encoding='utf-8-sig')))) == list(csv.reader(io.StringIO(raw_text)))
records = list(csv.DictReader(io.StringIO(raw_text)))
assert len(records) == 50 and len(records[0]) == 5
assert all(all(v != '' for v in r.values()) for r in records)
assert len({tuple(r.values()) for r in records}) == 50
assert sorted({r['State'] for r in records}) == ['California', 'Florida', 'New York']
names = ['R&D Spend', 'Administration', 'Marketing Spend', 'State_Florida', 'State_New York']
raw_x = np.array([[float(r['R&D Spend']), float(r['Administration']), float(r['Marketing Spend']),
                   float(r['State'] == 'Florida'), float(r['State'] == 'New York')] for r in records])
y = np.array([float(r['Profit']) for r in records])
assert np.isfinite(raw_x).all() and np.isfinite(y).all()
permutation = np.random.RandomState(42).permutation(50)
test, train = permutation[:10], permutation[10:]
assert not set(test) & set(train)
assert len(set(test) | set(train)) == 50
a, b, yt, yv = raw_x[train].copy(), raw_x[test].copy(), y[train], y[test]
mu, sigma = a[:, :3].mean(axis=0), a[:, :3].std(axis=0)
a[:, :3] = (a[:, :3] - mu) / sigma
b[:, :3] = (b[:, :3] - mu) / sigma

def ols(x, target, prediction_x):
    # Centering matches a fitted intercept and avoids scale/conditioning issues.
    mean_x, mean_y = x.mean(axis=0), target.mean()
    coef = np.linalg.lstsq(x - mean_x, target - mean_y, rcond=None)[0]
    return (prediction_x - mean_x) @ coef + mean_y, coef

def elastic_coefficients(l1_ratio):
    # Coordinate descent for 1/(2n)||y-Xw||² + alpha*l1*|w| + alpha*(1-l1)/2*||w||².
    x = a - a.mean(axis=0)
    z = (yt - yt.mean()) / yt.std()
    w = np.zeros(5)
    for iteration in range(100000):
        previous = w.copy()
        for j in range(5):
            residual = z - x @ w + x[:, j] * w[j]
            rho = np.mean(x[:, j] * residual)
            w[j] = np.sign(rho) * max(abs(rho) - .01*l1_ratio, 0) / (np.mean(x[:, j]**2) + .01*(1-l1_ratio))
        if np.max(np.abs(w-previous)) < 1e-11:
            break
    else:
        raise AssertionError('Independent coordinate descent did not converge')
    return w

fold_permutation = np.random.RandomState(42).permutation(40)
folds = [np.sort(x) for x in np.array_split(fold_permutation, 3)]
def cv_mse(ids):
    errors = []
    for validation in folds:
        fitting = np.array(sorted(set(range(40))-set(validation)))
        prediction, _ = ols(a[fitting][:, ids], yt[fitting], a[validation][:, ids])
        errors.append(float(np.mean((yt[validation]-prediction)**2)))
    return float(np.mean(errors))  # equal weight per fold, matching cross_val_score().mean()

rankings = {m: np.argsort(-np.abs(elastic_coefficients(ratio)), kind='stable').tolist()
            for m, ratio in [('lasso', 1), ('elastic', .5)]}
selections = {}
forward = []
for k in range(1, 6):
    forward.append(min((j for j in range(5) if j not in forward), key=lambda j: cv_mse(forward+[j])))
    rfe = list(range(5))
    while len(rfe) > k:
        _, coef = ols(a[:, rfe], yt, a[:, rfe])
        del rfe[int(np.argsort(coef**2, kind='stable')[0])]
    exhaustive = min(combinations(range(5), k), key=lambda ids: cv_mse(list(ids)))
    for method, ids in dict(lasso=rankings['lasso'][:k], elastic=rankings['elastic'][:k],
                           forward=forward.copy(), rfe=rfe, exhaustive=list(exhaustive)).items():
        selections[method, k] = ids

assert len(payload['result']['rows']) == 25
verified = []
max_metric_error, max_prediction_error = 0.0, 0.0
for row in payload['result']['rows']:
    ids = selections[row['method'], row['k']]
    assert set(row['features']) == {names[j] for j in ids}, (row['method'], row['k'], row['features'], ids)
    pred, _ = ols(a[:, ids], yt, b[:, ids])
    residual = yv - pred
    rmse = float(np.sqrt(np.mean(residual**2)))
    r2 = float(1-np.sum(residual**2)/np.sum((yv-yv.mean())**2))
    mae = float(np.mean(np.abs(residual)))
    for key, value in [('rmse', rmse), ('r2', r2), ('mae', mae)]:
        error = abs(value-row[key])
        max_metric_error = max(max_metric_error, error)
        assert np.isclose(value, row[key], rtol=1e-10, atol=1e-7), (row['method'], row['k'], key, value, row[key])
    original_ids = [names.index(f) for f in row['features']]
    restored = row['intercept'] + raw_x[test][:, original_ids] @ np.array(row['coefficients'])
    max_prediction_error = max(max_prediction_error, float(np.max(np.abs(pred-restored))))
    assert np.allclose(pred, restored, rtol=1e-10, atol=1e-6)
    assert [r['sample']-1 for r in row['residuals']] == test.tolist()
    for i, residual_row in enumerate(row['residuals']):
        assert np.isclose(residual_row['actual'], yv[i])
        assert np.isclose(residual_row['predicted'], pred[i], rtol=1e-10, atol=1e-6)
    verified.append(dict(method=row['method'], k=row['k'], rmse=rmse, r2=r2, mae=mae,
                         features=row['features']))

sys.path.insert(0, str(ROOT / 'dist'))
from engine import experiment
fresh = json.loads(experiment(raw_text, 'Profit', .2, 42))
for row in fresh['rows']:
    cached = next(r for r in payload['result']['rows'] if r['method']==row['method'] and r['k']==row['k'])
    assert row['features'] == cached['features']
    for key in ['rmse','r2','mae']:
        assert np.isclose(row[key], cached[key], rtol=1e-10, atol=1e-7)
report = dict(status='passed', verified_points=25, independent_method='NumPy OLS, coordinate descent, RFE elimination, exhaustive 3-fold CV',
              source_sha256=payload['sha256'], training_samples=(train+1).tolist(), test_samples=(test+1).tolist(),
              max_metric_absolute_error=max_metric_error, max_prediction_absolute_error=max_prediction_error,
              rows=verified)
chart_path = ROOT / 'docs/browser-chart-coordinates.json'
if chart_path.exists():
    browser_charts = json.loads(chart_path.read_text(encoding='utf-8'))
    palette = dict(lasso='#4952b8', forward='#c87838', rfe='#53795f', elastic='#b06486', exhaustive='#8b7845')
    for key, chart in browser_charts.items():
        _, _, width, height = map(float, chart['viewBox'].split())
        assert len(chart['series']) == 5 and len(chart['markers']) == 25
        # Audited at the desktop viewport; verify the mapping of values to SVG coordinates.
        assert width == 860 and height == 315
        low, high = min(r[key] for r in verified), max(r[key] for r in verified)
        padding = (high-low)*.18
        low, high = low-padding, high+padding
        for method, color in palette.items():
            series = next(s for s in chart['series'] if s['stroke']==color)
            points = np.array([[float(n) for n in pair.split(',')] for pair in series['points'].split()])
            method_rows = sorted((r for r in verified if r['method']==method), key=lambda r:r['k'])
            expected = np.array([[78+(r['k']-1)/4*(860-78-27), 22+(high-r[key])/(high-low)*(315-22-45)] for r in method_rows])
            assert np.allclose(points, expected, rtol=1e-10, atol=1e-7), (key,method)
    report['browser_chart_mapping'] = 'passed: 25 RMSE and 25 R2 coordinates, five series each'
browser_results_path = ROOT / 'docs/browser-recalculated-results.csv'
if browser_results_path.exists():
    labels = dict(lasso='Lasso', forward='Forward SFS', rfe='RFE', elastic='ElasticNet', exhaustive='Exhaustive Search')
    with browser_results_path.open(encoding='utf-8-sig', newline='') as stream:
        browser_results = list(csv.DictReader(stream))
    assert len(browser_results) == 25
    browser_max_error = 0.0
    for row in verified:
        actual = next(r for r in browser_results if r['Method']==labels[row['method']] and int(r['Number of Features'])==row['k'])
        assert actual['Selected Features'].split(', ') == next(r['features'] for r in payload['result']['rows'] if r['method']==row['method'] and r['k']==row['k'])
        for csv_key, metric_key in [('RMSE','rmse'), ('R-squared','r2')]:
            error = abs(float(actual[csv_key])-row[metric_key])
            browser_max_error = max(browser_max_error,error)
            assert np.isclose(float(actual[csv_key]), row[metric_key], rtol=1e-9, atol=1e-6)
    report['browser_recalculation'] = 'passed: all 25 feature sets, RMSE and R2 values'
    report['browser_max_metric_absolute_error'] = browser_max_error
(ROOT / 'docs').mkdir(exist_ok=True)
(ROOT / 'docs/calculation-audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
with (ROOT / 'docs/verified-results.csv').open('w', encoding='utf-8-sig', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['Method','k','RMSE','R2','MAE','Features'])
    for r in verified:
        writer.writerow([r['method'],r['k'],r['rmse'],r['r2'],r['mae'],', '.join(r['features'])])
print(json.dumps({k:v for k,v in report.items() if k!='rows'}, ensure_ascii=False))
