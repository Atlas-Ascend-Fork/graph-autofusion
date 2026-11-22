/* One-model-at-a-time, offline payload loading. Raw evidence is preserved. */
(() => {
'use strict';
const header=JSON.parse(document.getElementById('report-data').textContent);
const compressed=header.payload_encoding==='gzip-base64-models-v1';
let cachedIndex=-1,cachedModel=null,queue=Promise.resolve();
const pending=new Map();
async function decode(index){
 if(cachedIndex===index)return cachedModel;
 if(typeof DecompressionStream!=='function')throw new Error('当前浏览器不支持离线压缩报告解压，请使用支持 DecompressionStream 的新版浏览器。');
 const element=document.getElementById('report-model-'+index);
 if(!element)throw new Error('报告缺少所选模型的数据块，请重新生成报告。');
 // Release the loader reference before materializing another full model.
 cachedModel=null;cachedIndex=-1;
 try {
  const encoded=element.textContent.trim(),binary=atob(encoded),bytes=new Uint8Array(binary.length);
  for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);
  const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  const decoded=await new Response(stream).text();
  const model=JSON.parse(decoded);
  cachedModel=model;cachedIndex=index;
  return model;
 } catch(error){throw new Error('所选模型数据解压失败，压缩内容可能不完整或已损坏。请重新生成报告。',{cause:error});}
}
function loadModel(index){
 index=Number(index);
 if(!Number.isInteger(index)||index<0||index>=(header.models||[]).length)return Promise.reject(new Error('所选模型不存在，请重新选择。'));
 if(!compressed)return Promise.resolve(header.models[index]);
 if(pending.has(index))return pending.get(index);
 const job=queue.then(()=>decode(index));
 // Serialize different selections so large models never decompress concurrently.
 queue=job.then(()=>undefined,()=>undefined);
 pending.set(index,job);
 job.then(()=>pending.delete(index),()=>pending.delete(index));
 return job;
}
window.ReportPayload={header,loadModel,compressed};
})();
