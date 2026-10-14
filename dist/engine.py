import io, json, warnings
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

def experiment(csv_text, target, test_size=0.2, seed=42):
    warnings.filterwarnings('ignore', category=UserWarning)
    df = pd.read_csv(io.StringIO(csv_text))
    if not df.columns.is_unique or target not in df:
        raise ValueError('CSV 欄位必須唯一，且必須包含目標欄位。')
    y = pd.to_numeric(df[target], errors='coerce')
    valid = y.notna() & np.isfinite(y)
    dropped = int((~valid).sum())
    X, y = df.loc[valid].drop(columns=[target]), y[valid]
    if len(X) < 20 or len(X) > 3000 or X.shape[1] < 1 or X.shape[1] > 30:
        raise ValueError('請提供 20–3,000 筆有效資料及 1–30 個輸入欄位。')
    if y.nunique() < 2:
        raise ValueError('目標欄位必須是有變化的數值。')
    X = X.replace([np.inf, -np.inf], np.nan)
    train, test, yt, yv = train_test_split(X, y, test_size=test_size, random_state=seed)
    nums = X.select_dtypes(include='number').columns.tolist()
    cats = [c for c in X if c not in nums]
    if cats and sum(train[c].nunique() for c in cats) > 30:
        raise ValueError('類別值總數超過 30；請移除 ID 或高基數欄位。')
    prep = ColumnTransformer([
        ('num', make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler()), nums),
        ('cat', make_pipeline(SimpleImputer(strategy='most_frequent', keep_empty_features=True), OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)), cats)
    ], verbose_feature_names_out=False)
    a, b = prep.fit_transform(train), prep.transform(test)
    names = prep.get_feature_names_out()
    p = a.shape[1]
    if p > 10:
        raise ValueError('完整搜尋最多支援 10 個編碼後特徵；請減少欄位或類別。')
    maxk = min(p, 10)
    if p < 1:
        raise ValueError('資料編碼後沒有可用的特徵。')
    # Lasso is a ranking method here: fixed alpha on standardized X and y.
    z = (np.asarray(yt)-np.mean(yt))/np.std(yt)
    lasso = Lasso(alpha=0.01, max_iter=10000).fit(a, z)
    elastic = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000).fit(a, z)
    rankings = {
        'lasso': np.argsort(-np.abs(lasso.coef_), kind='stable'),
        'elastic': np.argsort(-np.abs(elastic.coef_), kind='stable')
    }
    out = []
    forward = []
    cv = KFold(n_splits=3, shuffle=True, random_state=seed)
    for k in range(1, maxk+1):
        candidates = [j for j in range(p) if j not in forward]
        nextj = max(candidates, key=lambda j: cross_val_score(LinearRegression(), a[:, forward+[j]], yt, cv=cv, scoring='neg_mean_squared_error', n_jobs=1).mean())
        forward.append(nextj)
        rfe = np.flatnonzero(RFE(LinearRegression(), n_features_to_select=k).fit(a, yt).support_) if k < p else np.arange(p)
        selections = {m: ranks[:k] for m, ranks in rankings.items()}
        best_subset = max(combinations(range(p), k), key=lambda ids: cross_val_score(LinearRegression(), a[:, list(ids)], yt, cv=cv, scoring='neg_mean_squared_error', n_jobs=1).mean())
        selections.update(rfe=rfe, forward=list(forward), exhaustive=list(best_subset))
        for method, ids in selections.items():
            model = LinearRegression().fit(a[:, ids], yt)
            pred = model.predict(b[:, ids])
            rmse, r2 = float(np.sqrt(mean_squared_error(yv, pred))), float(r2_score(yv, pred))
            if not np.isfinite(rmse) or not np.isfinite(r2):
                raise ValueError('評估結果無法計算；請檢查資料。')
            # Store coefficients in original spending units for the live predictor.
            coef = model.coef_.copy()
            intercept = float(model.intercept_)
            scaler = prep.named_transformers_['num'].named_steps.get('standardscaler') if nums else None
            for pos, idx in enumerate(ids):
                if idx < len(nums):
                    coef[pos] /= scaler.scale_[idx]
                    intercept -= float(coef[pos]*scaler.mean_[idx])
            residuals = [dict(sample=int(i)+1, actual=float(actual), predicted=float(estimated), residual=float(actual-estimated)) for i, actual, estimated in zip(test.index, yv, pred)]
            out.append(dict(method=method, k=k, features=names[ids].tolist(), rmse=rmse, r2=r2, mae=float(np.mean(np.abs(np.asarray(yv)-pred))), intercept=intercept, coefficients=coef.tolist(), residuals=residuals))
    return json.dumps(dict(rows=out, n=len(X), train=len(train), test=len(test), dropped=dropped, p=p, maxk=maxk), ensure_ascii=False)
