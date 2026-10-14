const $ = id => document.getElementById(id);
const methods={lasso:'Lasso',forward:'Forward SFS',rfe:'RFE',elastic:'ElasticNet',exhaustive:'Exhaustive Search'};
const colors={lasso:'#4952b8',forward:'#c87838',rfe:'#53795f',elastic:'#b06486',exhaustive:'#8b7845'};
const dashes={lasso:'',forward:'8 4',rfe:'3 3',elastic:'10 3 2 3',exhaustive:'6 3'};
const symbols={lasso:'●',forward:'◆',rfe:'■',elastic:'▲',exhaustive:'✕'};
const defaults=window.STARTUP_DATA;
let results=defaults.result.rows,csv=defaults.csv,worker=null,busy=false;
let mode='compare',metric='rmse',selected='lasso',selectedK=2;
let activeMethods=new Set(Object.keys(methods)),isDefault=true,currentModel=null,evaluation='',resultTarget='Profit';
const fmt=(n,d=3)=>Number(n).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
function activeRows(){return results.filter(r=>mode==='single'?r.method===selected:activeMethods.has(r.method))}
function button(text,pressed,action){const b=document.createElement('button');b.type='button';b.textContent=text;b.setAttribute('aria-pressed',String(pressed));b.onclick=action;return b}
function textCell(tag,text){const el=document.createElement(tag);el.textContent=text;return el}
function setContext(a,target,test,seed,name){resultTarget=target;evaluation=name+' · 目標 '+target+' · 訓練 '+a.train+' / 測試 '+a.test+' 筆 · 測試比例 '+test*100+'% · 種子 '+seed+' · '+a.p+' 個編碼後特徵'+(a.dropped?' · 排除 '+a.dropped+' 筆無效目標':'');$('context').textContent=evaluation;$('encodednote').textContent='編碼後共有 '+a.p+' 個特徵';}
function chart(id,key,data,wide=false){
 const small=window.innerWidth<=720,W=wide&&!small?860:430,H=wide&&!small?315:280,L=small?61:(wide?78:68),R=wide&&!small?27:20,T=22,B=45;
 let lo=Math.min(...data.map(r=>r[key])),hi=Math.max(...data.map(r=>r[key]));const pad=(hi-lo)*.18||Math.max(Math.abs(lo)*.05,.01);lo-=pad;hi+=pad;
 const maxK=Math.max(...results.map(r=>r.k)),x=k=>maxK===1?(L+W-R)/2:L+(k-1)/(maxK-1)*(W-L-R),y=v=>T+(hi-v)/(hi-lo)*(H-T-B);
 const NS='http://www.w3.org/2000/svg',svg=document.createElementNS(NS,'svg');svg.setAttribute('viewBox','0 0 '+W+' '+H);svg.setAttribute('role','group');svg.setAttribute('aria-label',(key==='rmse'?'RMSE':'R²')+' 隨特徵數變化折線圖');
 const add=(tag,attrs,text)=>{const e=document.createElementNS(NS,tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;svg.append(e);return e};
 for(let i=0;i<5;i++){const v=lo+(hi-lo)*i/4,yy=y(v);add('line',{x1:L,y1:yy,x2:W-R,y2:yy,stroke:'#e6eaf1','stroke-dasharray':'3 4'});add('text',{x:L-10,y:yy+4,'text-anchor':'end'},key==='rmse'?fmt(v,0):fmt(v,3))}
 add('line',{x1:L,y1:T,x2:L,y2:H-B,stroke:'#cbd4e0'});add('line',{x1:L,y1:H-B,x2:W-R,y2:H-B,stroke:'#cbd4e0'});
 add('line',{x1:x(selectedK),x2:x(selectedK),y1:T,y2:H-B,stroke:'#b9c0df','stroke-dasharray':'4 4'});
 for(let k=1;k<=maxK;k++)add('text',{x:x(k),y:H-B+22,'text-anchor':'middle','font-weight':k===selectedK?'700':'400'},k);
 add('text',{x:(L+W-R)/2,y:H-2,'text-anchor':'middle'},'特徵數 k');
 for(const m of Object.keys(methods)){const ds=data.filter(r=>r.method===m).sort((a,b)=>a.k-b.k);if(!ds.length)continue;add('polyline',{points:ds.map(r=>x(r.k)+','+y(r[key])).join(' '),fill:'none',stroke:colors[m],'stroke-width':2,'stroke-dasharray':dashes[m]});
  for(const r of ds){const cx=x(r.k),cy=y(r[key]),size=4.5,attrs={stroke:colors[m],'stroke-width':2,fill:m==='lasso'?colors[m]:'white',tabindex:0,role:'button','aria-label':methods[m]+' '+r.k+' 特徵 '+key+' '+fmt(r[key],6),class:'point'};let c;
   if(m==='rfe')c=add('rect',{...attrs,x:cx-size,y:cy-size,width:size*2,height:size*2});
   else if(m==='forward'||m==='elastic')c=add('polygon',{...attrs,points:m==='forward'?[cx+','+(cy-size*1.4),(cx+size)+','+cy,cx+','+(cy+size*1.4),(cx-size)+','+cy].join(' '):[cx+','+(cy-size*1.3),(cx+size*1.2)+','+(cy+size),(cx-size*1.2)+','+(cy+size)].join(' ')});
   else if(m==='exhaustive')c=add('path',{...attrs,fill:'none',d:'M '+(cx-size)+' '+(cy-size)+' L '+(cx+size)+' '+(cy+size)+' M '+(cx-size)+' '+(cy+size)+' L '+(cx+size)+' '+(cy-size)});
   else c=add('circle',{...attrs,cx,cy,r:size});
   const title=document.createElementNS(NS,'title');title.textContent=methods[m]+' · k='+r.k+' · '+fmt(r[key],6);c.append(title);
   c.onclick=()=>selectModel(r,true);c.onfocus=()=>{$('pointdetail').textContent=describe(r)};c.onmouseenter=()=>{$('pointdetail').textContent=describe(r)};c.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();selectModel(r,true)}};
  }
 }
 $(id).replaceChildren(svg);
}
function describe(r){return methods[r.method]+' · k='+r.k+' · RMSE '+fmt(r.rmse,3)+' · R² '+fmt(r.r2,4)+' ｜ '+r.features.join('、')}
function selectModel(r,redraw){selected=r.method;selectedK=r.k;if(redraw)render();else renderModel(r);$('pointdetail').textContent=describe(r)}
function renderModel(r){
 currentModel=r;$('modeltitle').textContent=methods[r.method]+' · k='+r.k+' 的模型細節';$('featurechips').replaceChildren(...r.features.map(f=>textCell('span',f)));
 let formula=resultTarget+' = '+fmt(r.intercept,2);r.features.forEach((f,i)=>{const c=r.coefficients[i];formula+=' '+(c>=0?'+':'−')+' '+fmt(Math.abs(c),4)+' × '+f});$('equation').textContent=formula;
 $('predictor').hidden=!isDefault;
 for(const [id,feature] of [['predrd','R&D Spend'],['predmarketing','Marketing Spend'],['predadmin','Administration']])$(id).disabled=!r.features.includes(feature);
 $('predstate').disabled=!r.features.some(f=>f.startsWith('State_'));
 predict();
}
function predict(){
 if(!currentModel||!isDefault)return;
 const values={'R&D Spend':Number($('predrd').value),'Marketing Spend':Number($('predmarketing').value),Administration:Number($('predadmin').value),'State_Florida':$('predstate').value==='Florida'?1:0,'State_New York':$('predstate').value==='New York'?1:0};
 if(['predrd','predmarketing','predadmin'].some(id=>$(id).value===''||!Number.isFinite(Number($(id).value))||Number($(id).value)<0)){$('predprofit').textContent='請輸入有效支出';return}
 let pred=currentModel.intercept;currentModel.features.forEach((f,i)=>pred+=currentModel.coefficients[i]*(values[f]??0));$('predprofit').textContent=fmt(pred,2);
}
function render(){
 const data=activeRows(),atK=data.filter(r=>r.k===selectedK).sort((a,b)=>a.rmse-b.rmse),best=atK[0];
 $('rmselabel').textContent=mode==='compare'?'k='+selectedK+' · 已顯示方法最低 RMSE':'目前模型 RMSE';
 $('bestRMSE').textContent=fmt(best.rmse,2);$('bestR2').textContent=fmt(best.r2,4);$('bestK').textContent=selectedK+' 個';$('bestMethod').textContent=methods[best.method];
 $('comparemode').setAttribute('aria-pressed',String(mode==='compare'));$('singlemode').setAttribute('aria-pressed',String(mode==='single'));
 $('combinedchart').hidden=mode!=='compare';$('singlecharts').hidden=mode!=='single';$('metric-controls').hidden=mode!=='compare';$('methodtabs').hidden=mode!=='single';
 $('metricrmse').setAttribute('aria-pressed',String(metric==='rmse'));$('metricr2').setAttribute('aria-pressed',String(metric==='r2'));
 $('plottitle').textContent=mode==='compare'?'五種方法的 '+(metric==='rmse'?'RMSE':'R²')+' 比較':methods[selected]+' 的特徵選取表現';
 $('plotsubtitle').textContent=mode==='compare'?'顯示 '+activeMethods.size+' / 5 種方法 · '+(metric==='rmse'?'RMSE 越低越好；單位與目標相同':'R² 越高越好；無單位'):'同時檢視誤差與解釋程度 · 相同測試資料';
 $('legend').replaceChildren();
 if(mode==='compare')for(const m of Object.keys(methods)){const b=button(symbols[m]+' '+methods[m],activeMethods.has(m),()=>{if(activeMethods.has(m)){if(activeMethods.size===1)return;activeMethods.delete(m)}else activeMethods.add(m);render()});b.title='顯示或隱藏 '+methods[m];b.style.borderColor=activeMethods.has(m)?colors[m]:'';$('legend').append(b)}
 $('methodtabs').replaceChildren(...Object.keys(methods).map(m=>button(methods[m],m===selected,()=>{selected=m;render()})));
 if(mode==='compare')chart('combinedchart',metric,data,true);else{chart('rmsechart','rmse',data);chart('r2chart','r2',data)}
 const maxK=Math.max(...results.map(r=>r.k));$('kbuttons').replaceChildren(...Array.from({length:maxK},(_,i)=>button('k = '+(i+1),selectedK===i+1,()=>{selectedK=i+1;render()})));
 $('tabletitle').textContent=mode==='compare'?'k='+selectedK+'：相同特徵數下的方法比較':methods[selected]+'：各特徵數結果';
 const tableRows=mode==='compare'?atK:data.slice().sort((a,b)=>a.k-b.k);$('rows').replaceChildren();
 for(const r of tableRows){const tr=document.createElement('tr');if(r.rmse===best.rmse&&r.k===best.k)tr.className='best';const td=document.createElement('td'),dot=document.createElement('span');dot.className='method-dot';dot.style.background=colors[r.method];const link=document.createElement('button');link.textContent=methods[r.method];link.setAttribute('aria-label','檢視 '+methods[r.method]+' k='+r.k+' 模型');link.onclick=()=>selectModel(r,true);td.append(dot,link);tr.append(td);for(const v of [r.k,r.features.join(', '),fmt(r.rmse,3),fmt(r.r2,4)])tr.append(textCell('td',v));$('rows').append(tr)}
 const shown=atK.find(r=>r.method===selected)||best;selected=shown.method;renderModel(shown);$('pointdetail').textContent=describe(shown);
}
function parseCSV(text){const rows=[];let row=[],value='',quoted=false;text=text.replace(/^\uFEFF/,'');for(let i=0;i<text.length;i++){const c=text[i];if(c==='"'){if(quoted&&text[i+1]==='"'){value+='"';i++}else quoted=!quoted}else if(c===','&&!quoted){row.push(value);value=''}else if((c==='\n'||c==='\r')&&!quoted){if(c==='\r'&&text[i+1]==='\n')i++;row.push(value);if(row.some(v=>v!==''))rows.push(row);row=[];value=''}else value+=c}if(quoted)throw Error('CSV 引號未正確關閉。');row.push(value);if(row.some(v=>v!==''))rows.push(row);return rows}
function preview(rows){$('datahead').replaceChildren();$('databody').replaceChildren();const tr=document.createElement('tr');rows[0].forEach(c=>tr.append(textCell('th',c)));$('datahead').append(tr);rows.slice(1,51).forEach(row=>{const tr=document.createElement('tr');row.forEach(c=>tr.append(textCell('td',c)));$('databody').append(tr)});$('datainfo').textContent=(isDefault?'50 Startups':'自訂 CSV')+' · '+(rows.length-1)+' 筆 · 最多預覽前 50 筆';}
function setCSV(text,name,useDefault=false){
 const rows=parseCSV(text),cols=rows[0]||[];if(cols.length<2||cols.some(c=>!c.trim())||new Set(cols).size!==cols.length)throw Error('CSV 至少需兩個欄位，欄名不得空白或重複。');if(rows.slice(1).some(r=>r.length!==cols.length))throw Error('CSV 各列欄位數不一致。');
 csv=text;isDefault=useDefault;$('filename').textContent=name;$('datasetname').textContent=useDefault?'50 Startups':'自訂資料集';$('datasetmeta').textContent=(rows.length-1)+' 筆資料 · '+(cols.length-1)+' 個輸入欄位';$('target').replaceChildren(...cols.map(c=>new Option(c,c)));$('target').value=cols.find(c=>/profit|target|收益|利潤/i.test(c))||cols.at(-1);$('targetname').textContent=$('target').value;preview(rows);$('predictor').hidden=!useDefault;$('source').textContent='上次運算結果';markDirty();
}
function markDirty(){$('dirty').hidden=false;$('status').textContent='資料或設定已變更，請重新執行以更新結果。'}
function reset(){
 setCSV(defaults.csv,'50_Startups.csv',true);results=defaults.result.rows;selectedK=2;selected='lasso';mode='compare';metric='rmse';activeMethods=new Set(Object.keys(methods));$('split').value='20';$('splitval').textContent='20%';$('seed').value='42';$('dirty').hidden=true;$('source').textContent='50 Startups · 已計算結果';setContext(defaults.result,'Profit',.2,42,'50 Startups');$('status').textContent='已載入預先計算的真實資料結果，可直接探索圖表。';render();
}
$('file').onchange=async e=>{const f=e.target.files[0];if(!f)return;try{if(f.size>2*1024*1024)throw Error('檔案超過 2 MB。');setCSV(await f.text(),f.name)}catch(err){$('status').textContent=err.message}e.target.value=''};
$('sample').onclick=reset;
$('previewdata').onclick=()=>{$('datapanel').open=true;$('datapanel').scrollIntoView({behavior:'smooth',block:'start'})};
$('split').oninput=()=>{$('splitval').textContent=$('split').value+'%';markDirty()};
$('seed').oninput=markDirty;$('target').onchange=()=>{$('targetname').textContent=$('target').value;markDirty()};
$('comparemode').onclick=()=>{mode='compare';render()};
$('singlemode').onclick=()=>{mode='single';render()};
$('metricrmse').onclick=()=>{metric='rmse';render()};
$('metricr2').onclick=()=>{metric='r2';render()};
for(const id of ['predrd','predmarketing','predadmin'])$(id).oninput=predict;$('predstate').onchange=predict;
function lock(on){busy=on;for(const id of ['file','sample','target','split','seed','run'])$(id).disabled=on;$('cancel').hidden=!on;$('run').firstChild.textContent=on?'運算中…':'重新執行五種方法 '}
$('run').onclick=()=>{
 if(busy||!csv)return;const seed=Number($('seed').value);if($('seed').value===''||!Number.isInteger(seed)||seed<0||seed>2147483647){$('status').textContent='隨機種子必須為 0–2147483647 的整數。';return}
 lock(true);$('status').textContent='準備分析環境…';const target=$('target').value,test=Number($('split').value)/100,name=isDefault?'50 Startups':'自訂 CSV';
 try{
  if(!worker)worker=new Worker('./worker.js');
  worker.onmessage=({data})=>{
   if(data.status)$('status').textContent=data.status;
   if(data.error){$('status').textContent='運算失敗：'+data.error;worker.terminate();worker=null;lock(false)}
   if(data.result){results=data.result.rows;selectedK=Math.min(selectedK,data.result.maxk);$('source').textContent=name+' · 新運算結果';setContext(data.result,target,test,seed,name);$('dirty').hidden=true;$('status').textContent='五種方法已完成。可切換比較指標、單一方法與特徵數。';render();lock(false)}
  };
  worker.onerror=()=>{$('status').textContent='Python 環境載入失敗，請檢查網路後重試。';worker.terminate();worker=null;lock(false)};
  worker.postMessage({csv,target,test,seed});
 }catch(e){$('status').textContent=e.message;lock(false)}
};
$('cancel').onclick=()=>{worker?.terminate();worker=null;lock(false);$('status').textContent='已取消，保留上次結果。'};
$('export').onclick=()=>{const escape=v=>'"'+String(v).replaceAll('"','""')+'"',rows=[['Method','Number of Features','Selected Features','RMSE','R-squared','Evaluation Context'],...results.map(r=>[methods[r.method],r.k,r.features.join(', '),r.rmse,r.r2,evaluation])];const blob=new Blob(['\uFEFF'+rows.map(r=>r.map(escape).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='50-startups-method-comparison.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
$('settingspanel').open=window.innerWidth>720;
window.addEventListener('resize',()=>{if(window.innerWidth>720)$('settingspanel').open=true;render()});
reset();
