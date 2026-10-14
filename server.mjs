import http from 'node:http';
import {readFile} from 'node:fs/promises';
import path from 'node:path';
const base=path.resolve('dist');
http.createServer(async(req,res)=>{try{const pathname=decodeURIComponent(new URL(req.url,'http://localhost').pathname);const target=path.resolve(base,'.'+(pathname==='/'?'/index.html':pathname));if(!target.startsWith(base+path.sep)){res.writeHead(403).end();return}const body=await readFile(target);res.setHeader('Content-Type',({'.html':'text/html; charset=utf-8','.css':'text/css','.js':'text/javascript','.py':'text/plain'})[path.extname(target)]||'application/octet-stream');res.end(body)}catch{res.writeHead(404).end('Not found')}}).listen(4173,'127.0.0.1',()=>console.log('Local: http://127.0.0.1:4173'));
