import vm from 'node:vm';
import fs from 'node:fs';
import assert from 'node:assert/strict';
class Element {
 constructor(){this.children=[];this.style={};this.value='';this.textContent='';this.firstChild={textContent:''}}
 setAttribute(k,v){this[k]=v}
 append(...v){this.children.push(...v)}
 replaceChildren(...v){this.children=v;if(v[0]?.value!==undefined)this.value=v[0].value}
 click(){}
 scrollIntoView(){}
}
const html=fs.readFileSync('dist/index.html','utf8');
const elements=Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],new Element()]));
elements.predrd.value='100000';elements.predmarketing.value='150000';elements.predadmin.value='120000';elements.predstate.value='California';
let worker;
const context=vm.createContext({window:{innerWidth:1200,addEventListener(){}},document:{getElementById:id=>{assert.ok(elements[id],id);return elements[id]},createElement:()=>new Element(),createElementNS:()=>new Element(),createTextNode:text=>text},Option:class{constructor(text,value){this.text=text;this.value=value}},Worker:class{constructor(){worker=this}postMessage(data){this.data=data}terminate(){}},Blob,URL,setTimeout,console});
vm.runInContext(fs.readFileSync('dist/startup-data.js','utf8'),context);
vm.runInContext(fs.readFileSync('dist/app.js','utf8'),context);
assert.equal(elements.rows.children.length,5);
assert.equal(elements.legend.children.length,5);
assert.equal(elements.bestRMSE.textContent,'8,206.33');
assert.equal(elements.predadmin.disabled,true);
elements.metricr2.onclick();assert.match(elements.plottitle.textContent,/R²/);
elements.singlemode.onclick();assert.equal(elements.rows.children.length,5);assert.equal(elements.singlecharts.hidden,false);
elements.methodtabs.children[3].onclick();assert.match(elements.modeltitle.textContent,/ElasticNet/);
elements.kbuttons.children[4].onclick();assert.match(elements.modeltitle.textContent,/k=5/);
elements.comparemode.onclick();elements.legend.children[0].onclick();assert.equal(elements.rows.children.length,4);
for(let i=0;i<4;i++)elements.legend.children[i].onclick();
assert.ok(elements.rows.children.length>=1,'Cannot hide all methods');
elements.sample.onclick();assert.equal(elements.rows.children.length,5);assert.equal(elements.dirty.hidden,true);
elements.seed.value='19';elements.seed.oninput();assert.equal(elements.dirty.hidden,false);
elements.run.onclick();assert.equal(worker.data.seed,19);assert.equal(worker.data.test,.2);
worker.onmessage({data:{result:context.window.STARTUP_DATA.result}});assert.equal(elements.dirty.hidden,true);
elements.run.onclick();elements.cancel.onclick();assert.equal(elements.cancel.hidden,true);
elements.export.onclick();
console.log('PASS: 50 Startups defaults, combined and individual charts, metric switch, feature count, legend filters, predictor state, stale settings, worker settings, cancellation, export.');
