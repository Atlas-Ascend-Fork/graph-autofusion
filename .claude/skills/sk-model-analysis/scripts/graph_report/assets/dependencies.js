/* Evidence-only dependency views. No dependencies, network calls, or guessed links. */
(() => {
'use strict';
const ns='http://www.w3.org/2000/svg';
let serial=0;
const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=String(text);if(cls)n.className=cls;return n;};
const svgEl=(tag,attrs={},text)=>{const n=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,String(v));if(text!==undefined)n.textContent=String(text);return n;};
const key=v=>String(v??'');
const seq=e=>/^queue$|queue[-_ ]?order|sequence|stream[-_ ]?order|sequential/.test(e.kind||'');
const taskType=t=>t.sync_kind?({SET:'发信',WAIT:'等待',ALL:'全核屏障',CROSS:'同类同步',NONE:'无同步',UNKNOWN:'未知同步'}[t.sync_kind]||'同步'):(t.type||'任务');
const taskDirection=t=>({ALL:'全部核心',NONE:'无同步方向',UNKNOWN:'方向未知'}[t.sync_direction]||key(t.sync_direction).replace(/_TO_/g,' → '));
const label=t=>'Task '+(t.index??t.id)+' · '+taskType(t)+(t.sync_direction?' · '+taskDirection(t):'');
const typeLabel=e=>seq(e)?'队列顺序':e.evidence==='inferred'?'证据推导同步':/memory/.test(e.kind||'')?'内存同步':/event/.test(e.kind||'')?'Event 同步':'同步依赖';
const evidence=(host,data,title='展开原始证据')=>{const d=el('details');d.append(el('summary',title));d.addEventListener('toggle',()=>{if(d.open&&d.childNodes.length===1)d.append(el('pre',JSON.stringify(data,null,2)));});host.append(d);};
const button=(text,fn)=>{const n=el('button',text);n.type='button';n.onclick=fn;return n;};
function walk(id,edges,direction,depth){const result=new Set([id]);let frontier=[id];for(let i=0;i<depth&&frontier.length;i++){const next=[];for(const e of edges){const a=key(direction==='up'?e.target:e.source),b=key(direction==='up'?e.source:e.target);if(frontier.includes(a)&&!result.has(b)){result.add(b);next.push(b);}}frontier=next;}return result;}
function render(body,model,scopeId,callbacks={}){
 if(callbacks.graphOnly&&window.SKTaskGraph)return SKTaskGraph.render(body,model,key(scopeId),callbacks);
 const graphOnly=!!callbacks.graphOnly;
 const data=model.exploration?.sync_by_scope?.[key(scopeId)];
 const root=el('section',undefined,'sk-dependencies');body.append(root);
 if(!graphOnly)root.append(el('h3','SK 内部同步与边界依赖'),el('p','横向表示各队列的 Task 顺序，不表示执行时间。实线表示明确同步证据，虚线表示证据推断，浅色箭头表示队列顺序。','dep-note'));
 if(!data){root.append(el('p','未提供内部依赖数据；无法确认跨队列同步。','dep-warning'));return;}
 const tasks=(data.tasks||[]),byId=new Map(tasks.map(t=>[key(t.id),t])),links=(data.links||[]).filter(e=>byId.has(key(e.source))&&byId.has(key(e.target))),unresolved=data.unresolved||[];
 const externalPorts=graphOnly?[...new Map((data.relations||[]).filter(r=>r.kind==='address').map(r=>[r.stage+':'+r.node_id,r])).values()]:[];
 const groups=new Map();for(const t of tasks){const id=key(t.queue_id);if(!groups.has(id))groups.set(id,{id,kind:t.queue_kind||'未标明',tasks:[],offset:0});groups.get(id).tasks.push(t);}
 const lanes=[...groups.values()];for(const lane of lanes)lane.tasks.sort((a,b)=>(a.index??0)-(b.index??0));
 const associations=window.GraphRelations?.create(model);let externalSeeds=[],aligned=true;
 let chosen=null,depth=1,direction='both',includeOrder=false,pageSize=graphOnly?Math.max(1,...lanes.map(l=>l.tasks.length)):8,zoom=1;
 const controls=el('div',undefined,'dep-controls');if(!graphOnly)root.append(controls);
 const select=(name,values,current,change)=>{const l=el('label',name+' '),s=el('select');for(const [value,title] of values){const o=el('option',title);o.value=value;s.append(o);}s.value=String(current);s.onchange=()=>change(s.value);l.append(s);controls.append(l);return s;};
 select('追踪方向',[['both','上下游'],['up','上游'],['down','下游']],direction,v=>{direction=v;paint();});
 select('跳数',[[1,'1 跳'],[2,'2 跳'],[3,'3 跳'],[5,'5 跳']],depth,v=>{depth=Number(v);paint();});
 select('每队列窗口',[[8,'8 项'],[20,'20 项'],[50,'50 项'],[100,'100 项']],pageSize,v=>{pageSize=Number(v);paint();});
 const orderLabel=el('label'),order=el('input');order.type='checkbox';order.onchange=()=>{includeOrder=order.checked;paint();};orderLabel.append(order,document.createTextNode('追踪包含队列顺序'));controls.append(orderLabel);
 controls.append(button('缩小',()=>{zoom=Math.max(.5,zoom-.15);paint();}),button('放大',()=>{zoom=Math.min(2,zoom+.15);paint();}),button('恢复比例',()=>{zoom=1;paint();}));
 const search=el('input');search.type='search';search.placeholder='查找 Task 序号、类型或标识';search.setAttribute('aria-label','查找内部 Task');const results=el('div',undefined,'dep-search-results');search.oninput=()=>{results.replaceChildren();const q=search.value.trim().toLowerCase();if(!q)return;const found=tasks.filter(t=>[t.id,t.index,t.type,t.node_index].some(x=>key(x).toLowerCase().includes(q)));results.append(el('span',found.length+' 项匹配'+(found.length>30?'，显示前 30 项':''),'dep-note'));for(const t of found.slice(0,30))results.append(button(t.queue_kind+' · '+label(t),()=>choose(key(t.id))));};controls.append(search);if(!graphOnly)root.append(results);
 const navigation=el('div',undefined,'dep-lane-nav'),viewport=el('div',undefined,'dep-viewport'),selection=el('div',undefined,'dep-selection');viewport.tabIndex=0;viewport.setAttribute('aria-label','内部 Task 依赖图，可横向滚动');if(graphOnly){root.classList.add('dep-graph-only');root.append(viewport);}else root.append(navigation,viewport,selection);
 viewport.addEventListener('dblclick',e=>{const target=document.elementFromPoint(e.clientX,e.clientY)?.closest('[data-task-id]'),task=byId.get(target?.getAttribute('data-task-id'));if(task&&callbacks.onTaskDetails){e.preventDefault();callbacks.onTaskDetails(task);}});
 const marker='dep-arrow-'+(++serial);
 function choose(id,notify=true){if(!notify&&chosen===id&&!externalSeeds.length)return;chosen=id;externalSeeds=[];const task=byId.get(id);if(task){const lane=groups.get(key(task.queue_id)),index=lane.tasks.indexOf(task);if(index<lane.offset||index>=lane.offset+pageSize)lane.offset=Math.floor(index/pageSize)*pageSize;}paint();if(notify)callbacks.onSelect?.(id?[...associations?.cohort(key(scopeId),[id])||[id]]:[]);}
 function boundaryColumns(columns){let end=0;return externalPorts.map(r=>{const ids=(data.relations||[]).filter(a=>a.kind==='address'&&a.node_id===r.node_id).map(a=>a.task_id),anchor=Math.min(...ids.map(id=>columns?.[id]??byId.get(id)?.index??0));const column=Math.max(end,Number.isFinite(anchor)?anchor:0);end=column+1;return column;});}
 function paint(){
  navigation.replaceChildren();viewport.replaceChildren();selection.replaceChildren();
  const seeds=chosen?[chosen]:externalSeeds;let highlighted=associations?associations.cohort(key(scopeId),seeds):new Set(seeds);const follow=includeOrder?links:links.filter(e=>!seq(e));for(const seed of [...highlighted]){if(direction!=='down')for(const id of walk(seed,follow,'up',depth))highlighted.add(id);if(direction!=='up')for(const id of walk(seed,follow,'down',depth))highlighted.add(id);}
  const columns=graphOnly?(aligned?data.layout?.aligned:data.layout?.raw):null;const columnCount=columns?Math.max(0,...Object.values(columns))+1:pageSize,portColumns=boundaryColumns(columns);
  const w=120+Math.max(columnCount,...portColumns.map(c=>c+1))*136,h=Math.max(145,lanes.length*134+38)+(externalPorts.length?110:0),svg=svgEl('svg',{viewBox:`0 0 ${w} ${h}`,width:w*zoom,height:h*zoom,role:'group','aria-label':'AIC 与 AIV 队列任务及已确认同步依赖'}),defs=svgEl('defs'),mark=svgEl('marker',{id:marker,viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:7,markerHeight:7,orient:'auto-start-reverse'});mark.append(svgEl('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:'context-stroke'}));defs.append(mark);svg.append(defs);const positions=new Map();
  lanes.forEach((lane,i)=>{const nav=el('div');nav.append(el('strong',lane.kind+' · 队列 '+lane.id+' '));const prev=button('上一窗',()=>{lane.offset=Math.max(0,lane.offset-pageSize);paint();}),next=button('下一窗',()=>{lane.offset+=pageSize;paint();});prev.disabled=lane.offset===0;next.disabled=lane.offset+pageSize>=lane.tasks.length;nav.append(prev,el('span',`${Math.min(lane.offset+1,lane.tasks.length)}–${Math.min(lane.offset+pageSize,lane.tasks.length)} / ${lane.tasks.length}`),next);navigation.append(nav);
   const y=i*134+28;svg.append(svgEl('rect',{x:0,y:y-10,width:w,height:120,rx:8,class:'dep-lane'}),svgEl('text',{x:12,y:y+30,class:'dep-lane-title'},lane.kind),svgEl('text',{x:12,y:y+50,class:'dep-lane-id'},'队列 '+lane.id));lane.tasks.slice(lane.offset,lane.offset+pageSize).forEach((t,j)=>positions.set(key(t.id),{x:120+(columns?.[key(t.id)]??j)*136,y:y+20,t}));
  });
  if(graphOnly){for(const lane of lanes){let run=[];const flush=()=>{if(!run.length)return;const a=positions.get(key(run[0].id)),b=positions.get(key(run.at(-1).id));if(a&&b)svg.append(svgEl('rect',{x:a.x-5,y:a.y-9,width:b.x-a.x+122,height:85,rx:6,fill:'#e1f0e5',stroke:'#c4ddcc','pointer-events':'none',class:'dep-origin-group'}));run=[];};for(const t of lane.tasks){const origin=t.origin_node_id??t.companion_node_id;if(origin==null){flush();continue;}if(run.length&&(run[0].origin_node_id??run[0].companion_node_id)!==origin)flush();run.push(t);}flush();}}
  for(const e of links){const a=positions.get(key(e.source)),b=positions.get(key(e.target));if(!a||!b)continue;const active=seeds.length&&highlighted.has(key(e.source))&&highlighted.has(key(e.target));const x1=a.x+112,y1=a.y+22,x2=b.x,y2=b.y+22;const d=a.y===b.y?`M${x1},${y1} L${x2},${y2}`:`M${x1},${y1} C${x1+32},${y1+55} ${x2-32},${y2-55} ${x2},${y2}`;const path=svgEl('path',{d,class:'dep-edge '+(seq(e)?'dep-order':'dep-signal')+(e.evidence==='inferred'?' dep-inferred':'')+(active?' dep-active':seeds.length?' dep-dim':''),'marker-end':`url(#${marker})`});path.append(svgEl('title',{},typeLabel(e)+' · '+(e.reason||'有来源证据')));svg.append(path);}
  for(const [id,p] of positions){const warning=unresolved.some(u=>key(u.task_id)===id);const g=svgEl('g',{'data-task-id':id,class:'dep-task'+(id===chosen?' dep-selected':highlighted.has(id)?' dep-related':seeds.length?' dep-dim':'')+(warning?' dep-unresolved':''),tabindex:0,role:'button','aria-label':label(p.t)+(warning?'，同步未确认':''),'aria-pressed':id===chosen});g.append(svgEl('rect',{x:p.x,y:p.y,width:112,height:62,rx:7}),svgEl('text',{x:p.x+8,y:p.y+18},'Task '+(p.t.index??'?')),svgEl('text',{x:p.x+8,y:p.y+35,class:'dep-task-type'},taskType(p.t).slice(0,15)),svgEl('text',{x:p.x+8,y:p.y+51,class:'dep-task-type'},taskDirection(p.t)||(p.t.origin_node_id!=null?'来源 #'+p.t.origin_node_id:p.t.companion_node_id!=null?'配套 #'+p.t.companion_node_id:'')),svgEl('title',{},label(p.t)+(warning?' · 未确认':'')));g.onclick=()=>{if(chosen!==id)choose(id);};g.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();e.stopPropagation();if(e.key==='Enter'&&callbacks.onTaskDetails)callbacks.onTaskDetails(p.t);else choose(id);}};svg.append(g);}
  if(externalPorts.length){const y=lanes.length*134+48;svg.append(svgEl('text',{x:120+(portColumns[0]||0)*136,y:y-8,class:'dep-lane-title'},'更新后外部节点 · 地址关联，轮次未确认'));externalPorts.forEach((r,i)=>{const active=(data.relations||[]).some(a=>a.kind==='address'&&a.node_id===r.node_id&&highlighted.has(a.task_id)),g=svgEl('g',{'data-external-id':r.node_id,class:'dep-external'+(active?' dep-related':seeds.length?' dep-dim':''),tabindex:0,role:'button','aria-label':'外部 '+(r.role==='wait'?'VALUE_WAIT':'VALUE_WRITE')+' '+r.node_id});const x=120+portColumns[i]*136;g.append(svgEl('rect',{x,y,width:122,height:62,rx:6,fill:active?'#dbeafe':'#fff7e6',stroke:active?'#2563eb':'#c5b28a'}),svgEl('text',{x:x+6,y:y+17},(r.role==='wait'?'VALUE_WAIT':'VALUE_WRITE')+' #'+r.node_id),svgEl('text',{x:x+6,y:y+35},'值 '+r.value),svgEl('title',{},r.address+' · '+r.reason));g.onclick=()=>callbacks.onExternal?.(r);g.ondblclick=()=>callbacks.onExternalDetails?.(r);g.onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();e.stopPropagation();callbacks.onExternalDetails?.(r);}};svg.append(g);});}
  viewport.append(svg);
  if(graphOnly)return;
  if(!tasks.length)selection.append(el('p','该融合区域未提供内部队列 Task。','dep-warning'));
  if(!chosen){selection.append(el('p',`共 ${tasks.length} 个 Task，${links.filter(e=>!seq(e)&&e.evidence!=='inferred').length} 条明确同步，${links.filter(e=>!seq(e)&&e.evidence==='inferred').length} 条推断同步，${unresolved.length} 项待确认。点击 Task 查看生产者、消费者和原始证据。`,'dep-note'));return;}
  const t=byId.get(chosen);selection.append(el('h4',t.queue_kind+' · '+label(t)));
  if(unresolved.some(u=>key(u.task_id)===chosen)&&['SET','WAIT'].includes(t.sync_kind)){const candidates=tasks.filter(other=>other.sync_direction===t.sync_direction&&other.sync_kind===(t.sync_kind==='SET'?'WAIT':'SET'));if(candidates.length){const d=el('details',undefined,'dep-candidates');d.append(el('summary','同方向候选 · '+candidates.length+' 项（配对未确认）'),el('p','以下仅用于人工核对；缺少轮次证据，不作为依赖边。','dep-note'));for(const other of candidates.slice(0,30))d.append(button(other.queue_kind+' · '+label(other),()=>choose(key(other.id))));if(candidates.length>30)d.append(el('p','仅列出前 30 项，可使用内部搜索继续定位。','dep-note'));selection.append(d);}}
  const connected=links.filter(e=>key(e.source)===chosen||key(e.target)===chosen);const hidden=[...highlighted].filter(id=>!positions.has(id));if(hidden.length)selection.append(el('p',`${hidden.length} 个关联 Task 位于其他窗口，可点击下方关联项定位。`,'dep-note'));
  for(const [dir,title] of [['up','上游 · 生产者'],['down','下游 · 消费者']]){const list=el('div',undefined,'dep-related-list');list.append(el('strong',title));const related=connected.filter(e=>key(dir==='up'?e.target:e.source)===chosen);if(!related.length)list.append(el('span','无可追踪依赖','dep-note'));for(const e of related){const id=key(dir==='up'?e.source:e.target);const row=el('div');row.append(button(label(byId.get(id))+' · '+typeLabel(e),()=>choose(id)));evidence(row,e);list.append(row);}selection.append(list);}
  const indirect=[...highlighted].filter(id=>id!==chosen&&!connected.some(e=>key(e.source)===id||key(e.target)===id));if(indirect.length){const d=el('details');d.append(el('summary','更多跳关联 · '+indirect.length+' 项'));for(const id of indirect)d.append(button(label(byId.get(id)),()=>choose(id)));selection.append(d);}
  for(const u of unresolved.filter(u=>key(u.task_id)===chosen)){const box=el('div',undefined,'dep-warning');box.append(el('p',u.message||'同步元数据不足，不能确认配对。'));evidence(box,u,'展开未确认依据');selection.append(box);}if(callbacks.onTaskDetails)selection.append(button('查看选中 Task 详情',()=>callbacks.onTaskDetails(t)));else evidence(selection,t,'展开选中 Task 原始字段与来源');
 }
 paint();
 if(graphOnly)return {selectTask:choose,highlightTasks:ids=>{chosen=null;externalSeeds=ids;paint();},setAligned:value=>{aligned=value;paint();},get width(){return 120+Math.max(0,...boundaryColumns(aligned?data.layout?.aligned:data.layout?.raw).map(c=>c+1),data.layout?Math.max(0,...Object.values(aligned?data.layout.aligned:data.layout.raw))+1:pageSize)*136;},height:Math.max(145,lanes.length*134+38)};
 const boundary=el('section',undefined,'dep-boundary');root.append(boundary);boundary.append(el('h4','边界端口 · 内部与外部'));
 boundary.append(el('p','边界端口表示融合区域与外部图节点的依赖。只有明确的 Task 映射才会连入内部；区域边界不能替代内部同步证据。','dep-note'));
 const ports=data.boundary||[];if(!ports.length)boundary.append(el('p','当前证据未提供可确认的外部边界连接。','dep-warning'));
 let boundaryOffset=0;const more=button('显示更多边界端口',appendPorts);boundary.append(more);
 function appendPorts(){for(const e of ports.slice(boundaryOffset,boundaryOffset+30)){const row=el('article',undefined,'dep-port');const stage=e.stage||'before';const nodeMap=new Map((model[stage]?.nodes||[]).map(n=>[key(n.id),n]));const scope=(model.scopes||[]).find(s=>key(s.id)===key(scopeId));const members=new Set((scope?.node_ids||[]).map(key));const source=nodeMap.get(key(e.source)),target=nodeMap.get(key(e.target));const sourceInside=members.has(key(e.source))||key(source?.scope_id)===key(scopeId),targetInside=members.has(key(e.target))||key(target?.scope_id)===key(scopeId);const incoming=!sourceInside&&targetInside,outgoing=sourceInside&&!targetInside;row.append(el('strong',(incoming?'输入端口':outgoing?'输出端口':'边界连接')+' · '+(stage==='after'?'更新后':'更新前')));
 for(const [id,node,prefix] of [[e.source,source,'来源'],[e.target,target,'去向']]){if(id==null)continue;const isTask=byId.has(key(id));row.append(button(prefix+' · '+(node?.label||node?.type||'节点')+' '+id,()=>isTask?choose(key(id)):callbacks.onGraphNode?.(stage,key(id))));}
 const taskId=e.task_id??e.internal_task_id;if(taskId!=null&&byId.has(key(taskId)))row.append(button('定位内部 '+label(byId.get(key(taskId))),()=>choose(key(taskId))));else row.append(el('span','内部 Task 映射未确认','dep-unknown'));evidence(row,e,'展开边界来源证据');boundary.insertBefore(row,more);}boundaryOffset+=30;more.hidden=boundaryOffset>=ports.length;}
 appendPorts();
 return {selectTask:choose};
}
function renderNodeDependencies(body,model,stage,nodeId,callbacks={}){
 const graph=model[stage]||{},nodes=new Map((graph.nodes||[]).map(n=>[key(n.id),n]));const root=el('section',undefined,'sk-dependencies dep-node-links');body.append(root);root.append(el('h3','节点上下游依赖'));
 for(const [direction,title] of [['up','上游节点'],['down','下游节点']]){const edges=(graph.edges||[]).filter(e=>key(direction==='up'?e.target:e.source)===key(nodeId));root.append(el('h4',title+' · '+edges.length));if(!edges.length)root.append(el('p','没有已确认的关联节点。','dep-note'));let offset=0;const more=button('显示更多关联',append);root.append(more);function append(){for(const e of edges.slice(offset,offset+30)){const id=key(direction==='up'?e.source:e.target),n=nodes.get(id),row=el('div',undefined,'dep-port');row.append(button((n?.label||n?.type||'节点')+' · '+id,()=>callbacks.onGraphNode?.(stage,id)));if(n?.scope_id!=null)row.append(button('查看融合区域 '+n.scope_id,()=>callbacks.onScope?.(key(n.scope_id))));evidence(row,e,'展开依赖证据');root.insertBefore(row,more);}offset+=30;more.hidden=offset>=edges.length;}append();}
}
function renderTaskDetails(body,model,scopeId,task,callbacks={}){
 const data=model.exploration?.sync_by_scope?.[key(scopeId)];
 if(!data){body.append(el('p','缺少内部依赖证据。','dep-note'));return;}
 const byId=new Map((data.tasks||[]).map(t=>[key(t.id),t]));
 const root=el('section',undefined,'sk-dependencies dep-task-details');body.append(root);
 const related=window.GraphRelations?.create(model).cohort(key(scopeId),[key(task.id)])||new Set([key(task.id)]);
 const companions=[...related].filter(id=>id!==key(task.id)&&byId.has(id));if(companions.length){root.append(el('h4','同来源与配套任务（非依赖边）'));for(const id of companions)root.append(button(byId.get(id).queue_kind+' · '+label(byId.get(id)),()=>callbacks.onTask?.(byId.get(id))));}
 const cross=window.GraphRelations?.create(model).relatedTasks(key(scopeId),[key(task.id)]).filter(r=>r.scopeId!==key(scopeId))||[];if(cross.length){root.append(el('h4','其他 SK 内的同地址任务'));for(const r of cross)root.append(button('SK '+r.scopeId+' · '+r.taskId,()=>callbacks.onRelatedTask?.(r.scopeId,r.taskId)));}
 const mappings=(data.relations||[]).filter(r=>r.task_id===key(task.id)&&r.node_id!=null);if(mappings.length)root.append(el('h4','融合前后及外部关联'));
 for(const r of mappings){const row=el('div',undefined,'dep-port');row.append(button((r.kind==='origin'?'融合前 Kernel':r.kind==='signal_bridge'?'更新前 '+r.role:r.role==='wait'?'外部 VALUE_WAIT':'外部 VALUE_WRITE')+' #'+r.node_id+(r.value!=null?' · 值 '+r.value:''),()=>callbacks.onGraphNode?.(r.stage,r.node_id)));if(r.kind==='address'||r.kind==='signal_bridge')row.append(el('span',r.address+' · 地址关联，轮次未确认','dep-note'));evidence(row,r,'关联依据');root.append(row);}

 for(const [dir,title] of [['up','上游 · 生产者'],['down','下游 · 消费者']]){
  const links=(data.links||[]).filter(e=>key(dir==='up'?e.target:e.source)===key(task.id));
  root.append(el('h4',title+' · '+links.length));
  if(!links.length)root.append(el('p','没有已确认的关联任务。','dep-note'));
  for(const e of links){const id=key(dir==='up'?e.source:e.target),other=byId.get(id),row=el('div',undefined,'dep-port');
   if(other)row.append(button(other.queue_kind+' · '+label(other)+' · '+typeLabel(e),()=>callbacks.onTask?.(other)));
   else row.append(el('span','任务 '+id+' · '+typeLabel(e)));
   evidence(row,e,'依赖来源');root.append(row);
  }
 }
 const ports=(data.boundary||[]).filter(e=>[e.task_id,e.internal_task_id,e.source,e.target].some(id=>id!=null&&key(id)===key(task.id)));
 if(ports.length){root.append(el('h4','与外部节点的关联'));for(const e of ports){const row=el('div',undefined,'dep-port');for(const id of [e.source,e.target])if(id!=null&&!byId.has(key(id)))row.append(button('定位外部节点 '+id,()=>callbacks.onGraphNode?.(e.stage||'before',key(id))));evidence(row,e,'边界来源');root.append(row);}}
 else if((data.boundary||[]).length)root.append(el('p','SK 存在外部依赖，但没有证据确认它们属于当前 Task。','dep-note'));
 for(const u of (data.unresolved||[]).filter(u=>key(u.task_id)===key(task.id))){root.append(el('p',u.message||'依赖关系未确认','dep-warning'));evidence(root,u,'未确认依据');}
}
window.SKDependencies={render,renderNodeDependencies,renderTaskDetails};
})();
