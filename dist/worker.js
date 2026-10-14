importScripts('https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js');
let runtime;
onmessage = async ({data}) => {
  try {
    if (!runtime) {
      postMessage({status:'首次載入 Python 與 scikit-learn，可能需要 1–2 分鐘…'});
      runtime = await loadPyodide();
      await runtime.loadPackage(['numpy','pandas','scikit-learn']);
      const response = await fetch('./engine.py');
      if (!response.ok) throw new Error('無法載入分析程式');
      await runtime.runPythonAsync(await response.text());
    }
    postMessage({status:'正在比較五種方法，請稍候…'});
    runtime.globals.set('csv_text', data.csv);
    runtime.globals.set('target_col', data.target);
    runtime.globals.set('test_fraction', data.test);
    runtime.globals.set('random_seed', data.seed);
    const result = await runtime.runPythonAsync('experiment(csv_text, target_col, test_fraction, random_seed)');
    postMessage({result:JSON.parse(result)});
  } catch(error) { runtime = undefined; postMessage({error:String(error)}); }
};
