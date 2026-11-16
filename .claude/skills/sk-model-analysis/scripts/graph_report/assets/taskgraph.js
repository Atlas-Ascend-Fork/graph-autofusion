/* Graph-only internal task renderer; filters never change dependency evidence. */
(() => {
'use strict';
let serial=0;
const ns='http://www.w3.org/2000/svg',el=(tag,attrs={},text)=>{const n=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,String(v));if(text!=null)n.textContent=text;return n;};
const type=t=>t.type?.startsWith('EVENT')?'EVENT':t.type||'OTHER';
const direction=t=>({'AIC_TO_AIV':'C→V','AIV_TO_AIC':'V→C','AIC_TO_AIC':'C→C','AIV_TO_AIV':'V→V','ALL':'ALL'}[t.sync_direction]||'');
function render(host,model,sid,callbacks={}){
 const errorCodes=new Set(['sync_endpoint_missing','sync_queue_invalid','sync_count_mismatch','sync_dependency_cycle','sync_barrier_mismatch']);
 const data=model.exploration?.sync_by_scope?.[sid]||{tasks:[],links:[]},tasks=data.tasks,byId=new Map(tasks.map(t=>[String(t.id),t])),relations=window.GraphRelations?.create(model),queues=new Map();
 for(const t of tasks){if(!queues.has(t.queue_id))queues.set(t.queue_id,[]);queues.get(t.queue_id).push(t);}for(const q of queues.values())q.sort((a,b)=>a.index-b.index);
 const lanes=[...queues],raw=Object.fromEntries(lanes.flatMap(([,q])=>q.map((t,i)=>[t.id,i]))),signals=[...(data.links||[]).filter(e=>e.kind==='sync'),...(data.barrier_pairs||[])];
 const root=document.createElement('section');root.className='sk-dependencies dep-graph-only task-graph';const viewport=document.createElement('div');viewport.className='dep-viewport';root.append(viewport);host.append(root);
 let chosen=null,seeds=[],activeSeeds=[],aligned=true,hidden=new Set(),only=false,reveals=new Set(),width=400,lastHidden=0,lastVisible=0;
 const marker='task-arrow-'+(++serial);
 const columns=()=>aligned?(data.layout?.aligned||raw):raw;
 function relationSet(){if(chosen&&byId.get(chosen)?.type==='SYNC')return new Set([chosen]);return relations?relations.cohort(sid,chosen?[chosen]:seeds):new Set(chosen?[chosen]:seeds);}
 function highlights(){if(!chosen)return new Set(seeds);const set=relationSet(),starts=new Set(set);for(const e of signals)if(starts.has(e.source)||starts.has(e.target)){set.add(e.source);set.add(e.target);}return set;}
 const visible=(t,related)=>!hidden.has(type(t))&&(!only||!chosen&&!seeds.length||related.has(String(t.id)))||reveals.has(String(t.id));
 function paint(){
  const cols=columns(),related=highlights(),hasSelection=!!chosen||seeds.length>0,positions=new Map();

  width=120+Math.max(1,...Object.values(cols).map(c=>c+1))*136;const h=Math.max(145,lanes.length*150+30);
  const svg=el('svg',{viewBox:`0 0 ${width} ${h}`,width,height:h,role:'group','aria-label':'AIC/AIV 任务与同步关系'}),defs=el('defs'),mark=el('marker',{id:marker,viewBox:'0 0 10 10',refX:10,refY:5,markerWidth:5,markerHeight:5,orient:'auto'});mark.append(el('path',{d:'M0 0 L10 5 L0 10 Z',fill:'context-stroke'}));defs.append(mark);svg.append(defs);
  lanes.forEach(([qid,q],i)=>{const y=i*150+40;svg.append(el('rect',{x:0,y:y-22,width,height:130,rx:8,class:'dep-lane'}),el('text',{x:12,y:y+25,class:'dep-lane-title'},q[0]?.queue_kind||qid),el('line',{x1:108,x2:width,y1:y+31,y2:y+31,class:'queue-guide'}));
   for(const t of q)if(visible(t,related)){const w=({FUNC:124,PRELOAD:104,SYNC:96,EVENT:112})[type(t)]||112;positions.set(String(t.id),{x:120+(cols[t.id]||0)*136+(112-w)/2,y,w,t});}
  });
  const drawn=signals;
  for(const e of drawn){const a=positions.get(String(e.source)),b=positions.get(String(e.target));if(!a||!b)continue;const all=e.kind==='all_sync',active=related.has(e.source)&&related.has(e.target),down=a.y<b.y;
   const x1=a.x+a.w/2,x2=b.x+b.w/2,y1=down?a.y+64:a.y-2,y2=down?b.y-2:b.y+64,middle=(y1+y2)/2;
   const d=a.y===b.y?`M${a.x+a.w} ${a.y+31} L${b.x} ${b.y+31}`:`M${x1} ${y1} L${x1} ${middle} L${x2} ${middle} L${x2} ${y2}`;
   const attrs={d,class:'dep-edge '+(all?'sync-all':'sync-flow')+(active?' dep-active':hasSelection?' dep-dim':''),'data-source':e.source,'data-target':e.target,'data-evidence':e.evidence||'unknown'};if(!all)attrs['marker-end']=`url(#${marker})`;
   const path=el('path',attrs);path.append(el('title',{},e.reason||'同步依赖'));svg.append(path);
   const label=(all?'ALL':direction(a.t)||direction(b.t));svg.append(el('text',{x:(x1+x2)/2+5,y:middle-5,class:'sync-label'},label));
  }
  for(const [id,p] of positions){const t=p.t,kind=type(t),error=(data.unresolved||[]).find(u=>u.task_id===id&&errorCodes.has(u.code)),selected=chosen===id||activeSeeds.includes(id),linked=related.has(id),g=el('g',{'data-task-id':id,'data-task-type':kind,tabindex:0,role:'button','aria-label':`${t.type} Task ${t.index}`,'aria-pressed':selected,class:`dep-task task-${kind.toLowerCase()}${error?' sync-error':''}${selected?' dep-selected':linked?' dep-related':hasSelection?' dep-dim':''}`});
   const attrs={class:'task-outline',x:p.x,y:p.y,width:p.w,height:62,rx:kind==='EVENT'?28:7};
   if(kind==='PRELOAD'||kind==='SYNC'){g.append(el('rect',{...attrs,class:'task-hit'}));g.append(el('path',{class:'task-outline',d:`M${p.x+12} ${p.y} H${p.x+p.w-12} L${p.x+p.w} ${p.y+31} L${p.x+p.w-12} ${p.y+62} H${p.x+12} L${p.x} ${p.y+31} Z`}));}else g.append(el('rect',attrs));
   const name=kind==='SYNC'?(t.sync_kind||'SYNC')+' '+direction(t):t.type;
   const issue=(data.unresolved||[]).find(u=>u.task_id===id && u.code==='sync_pair_ambiguous');
   g.append(el('text',{x:p.x+13,y:p.y+18},name),el('text',{x:p.x+13,y:p.y+35},'Task '+t.index),el('text',{x:p.x+13,y:p.y+51,class:'dep-task-type'},error?'同步异常':t.sync_peer_id?'对应 Task '+byId.get(t.sync_peer_id)?.index:issue?'同步信息异常':t.origin_node_id!=null?'来源 #'+t.origin_node_id:t.companion_node_id!=null?'配套 #'+t.companion_node_id:''),el('title',{},`${name} · Task ${t.index}`+(t.origin_label?' · '+t.origin_label:'')));
   g.onclick=()=>{if(chosen!==id)choose(id);};g.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();e.stopPropagation();if(e.key==='Enter')callbacks.onTaskDetails?.(t);else choose(id);}};svg.append(g);
  }
  lastVisible=positions.size;
  lastHidden=[...related].filter(id=>byId.has(id)&&!positions.has(id)).length;
  if(!tasks.length)svg.append(el('text',{x:20,y:50},'缺少内部任务证据'));
  viewport.replaceChildren(svg);
 }
 function choose(id,notify=true){if(!notify&&chosen===id&&!seeds.length)return;chosen=id;seeds=[];activeSeeds=[];if(id)reveals.add(id);paint();if(notify)callbacks.onSelect?.(id?[id]:[]);}
 viewport.addEventListener('dblclick',e=>{const node=document.elementFromPoint(e.clientX,e.clientY)?.closest('[data-task-id]'),t=byId.get(node?.getAttribute('data-task-id'));if(t){e.preventDefault();callbacks.onTaskDetails?.(t);}});
 paint();return {revealTask:id=>{reveals.add(String(id));paint();},get displayState(){return {hidden:[...hidden],only,reveals:[...reveals]};},restoreDisplay:s=>{if(!s)return;hidden=new Set(s.hidden);only=s.only;reveals=new Set(s.reveals);paint();},selectTask:choose,highlightTasks:(ids,activated=[])=>{chosen=null;seeds=ids;activeSeeds=activated;paint();},setAligned:value=>{aligned=value;paint();},get width(){return width;},setFilters:(types,onlyRelated)=>{hidden=new Set(types);only=onlyRelated;reveals.clear();paint();},get filters(){return {types:[...hidden],only};},get hiddenCount(){return tasks.length-lastVisible;},get hiddenRelated(){return lastHidden;},revealRelated:()=>{for(const id of highlights())reveals.add(id);paint();}};
}
window.SKTaskGraph={render};
})();
