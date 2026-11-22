/* Stable selected-object actions and navigation independent of the relation source. */
let trackingSource=null,relationCursor=-1,contextActionStamp='',temporaryView=null;
function traceSpec(){const local=selection?.kind==='local'?scopeLocalNodes.find(n=>n.id===selection.id):null,raw=local?.raw||(selection?.kind==='node'?(model?.[stage]?.nodes||[]).find(n=>String(n.id)===String(selection.id)):null);return relationFocus?{scopeId:relationFocus.scopeId,taskIds:[...relationFocus.taskIds]}:selection?.kind==='scope'?{scopeId:String(selection.id)}:displaySK(raw)?{scopeId:String(raw.scope_id)}:{stage:local?.stage||stage,nodeId:local?.rawId||(selection?.kind==='node'?String(selection.id):undefined)};}
function setPanelAlignment(p,value){p.aligned=value;p.view.setAligned(value);p.w=p.view.width+28;p.root.style.width=p.w+'px';const box=p.header.querySelector('.sk-alignment input');if(box)box.checked=value;positionSKPanels();}
function internalTaskCenter(p,id){
 // Read the actual rendered offsets, including borders, toolbar, legend and SVG padding.
 // The measured transform may still be from the previous animation frame.
 renderInternalControls();for(const [key,entry] of skPanels)if(openSKs.has(key)&&!entry.root.hidden)entry.h=entry.root.offsetHeight;positionSKPanels();
 const task=[...p.root.querySelectorAll('[data-task-id]')].find(e=>e.dataset.taskId===id);if(!task)return null;
 const rootBox=p.root.getBoundingClientRect(),taskBox=task.getBoundingClientRect(),renderScale=rootBox.width/p.root.offsetWidth;if(!renderScale)return null;
 return {x:p.x+(taskBox.x+taskBox.width/2-rootBox.x)/renderScale,y:p.y+(taskBox.y+taskBox.height/2-rootBox.y)/renderScale};
}
function focusInternal(p,id){const point=internalTaskCenter(p,id);if(!point)return;scale=Math.max(.6,Math.min(1.2,scale));ox=width*.5-point.x*scale;oy=height*.45-point.y*scale;}
function focusScopeOnly(id){const r=scopeRects.find(r=>String(r.scope.id)===String(id));if(!r)return;history=[...history,captureContext()].slice(-32);scale=Math.max(.4,Math.min(1,(width-160)/r.w));ox=120-r.x*scale;oy=60-r.y*scale;schedule();}
function relationTargetState(t){if(t.kind==='task'){const p=skPanels.get(skKey(t.sk));if(!openSKs.has(skKey(t.sk)))return '未展开';if(!p?.root.querySelector('[data-task-id="'+t.taskId+'"]'))return '已过滤';const e=p.root.querySelector('[data-task-id="'+t.taskId+'"]'),b=e.getBoundingClientRect(),v=canvas.getBoundingClientRect();return b.right<v.left||b.left>v.right||b.bottom<v.top||b.top>v.bottom?'视野外':'';}const local=scopeLocalNodes.find(n=>n.rawId===t.id&&n.stage===t.stage),n=local||byId.get(t.id);if(!n)return '未展开';if(local?!visibleLocal(local):!visibleNode(n))return '已过滤';return ''; }
function relationTargetScope(id,targetStage){const node=model?.[targetStage]?.nodes.find(n=>String(n.id)===String(id));if(node?.scope_id!=null)return node.scope_id;return targetStage==='after'?afterDisplayOwners.get(String(id)):(model?.scopes||[]).find(s=>(s.node_ids||[]).map(String).includes(String(id)))?.id;}
function relationTargetKey(t){return t.kind==='task'?'task:'+t.scopeId+':'+t.taskId:'node:'+t.stage+':'+t.id;}
function relationTargets(){
 const trace=selectionTrace(),spec=trackingSource||traceSpec(),out=[],seen=new Set();
 for(const ref of trace.tasks){
  if(spec.scopeId===ref.scopeId&&spec.taskIds?.includes(ref.taskId))continue;
  const t=relations().view(ref.scopeId)?.tasks.find(t=>t.id===ref.taskId),sk=(model?.after?.nodes||[]).find(n=>displaySK(n)&&String(n.scope_id)===ref.scopeId);if(!t||!sk)continue;
  const target={kind:'task',scopeId:ref.scopeId,taskId:ref.taskId,sk,label:'SK '+ref.scopeId+' / '+t.queue_kind+' / Task '+t.index+' / '+t.type},key=relationTargetKey(target);if(seen.has(key))continue;seen.add(key);out.push(target);
 }
 for(const st of ['before','after'])for(const id of trace.nodes[st]){
  if(spec.nodeId!=null&&String(spec.nodeId)===String(id))continue;
  const n=model?.[st]?.nodes.find(n=>String(n.id)===String(id));if(!n||displaySK(n))continue;
  const sid=relationTargetScope(id,st),shownStage=sid==null?stage:scopeViews.get(String(sid))?.stage||stage;if(st!==shownStage)continue;
  const target={kind:'node',id:String(id),stage:st,label:(st==='before'?'更新前':'更新后')+' / #'+id+' / '+n.type},key=relationTargetKey(target);if(seen.has(key))continue;seen.add(key);out.push(target);
 }
 return out;
}
function stepRelation(direction){const targets=relationTargets();if(!targets.length)return;const index=relationCursor<0?(direction<0?targets.length-1:0):(relationCursor+direction+targets.length)%targets.length;visitRelation(targets[index]);}
function visitRelation(target){
 if(!target)return;const previous=captureContext(),trail=history,source=trackingSource||traceSpec();if(!temporaryView)temporaryView=previous;
 // Explicit "view original task" can inspect another snapshot, but navigation
 // must never toggle either the whole graph or a region's Update state.
 if(target.kind==='node'){
  const sid=relationTargetScope(target.id,target.stage),shownStage=sid==null?stage:scopeViews.get(String(sid))?.stage||stage;
  if(target.stage!==shownStage){renderInspector('node',String(target.id),target.stage,model?.[target.stage]?.nodes.find(n=>String(n.id)===String(target.id)));history=[...trail,previous].slice(-32);contextActionStamp='';schedule();return;}
 }
 const cursor=relationTargets().findIndex(t=>relationTargetKey(t)===relationTargetKey(target));if(cursor>=0)relationCursor=cursor;
 const oldSelection=selection,oldSelected=selected,oldFocus=relationFocus;
 trackingSource=source;
 if(target.kind==='task'){
  const n=target.sk,key=skKey(n);ensureSKPanel(n);openSKs.add(key);activeInternal=key;
  positionSKPanels();const p=skPanels.get(key);p.view.revealTask(target.taskId);focusInternal(p,target.taskId);
 }else{
  const sid=relationTargetScope(target.id,target.stage);
  if(sid!=null)collapsedScopes.delete(String(sid));
  filterReveals.add(String(target.id));const run=GraphNetwork.resetGroups(model?.[target.stage]||{},model?.scopes||[],target.stage).find(g=>g.node_ids.map(String).includes(String(target.id)));if(run)expandedResets.add(run.id);
  build(true);selection=oldSelection;selected=oldSelected;relationFocus=oldFocus;trackingSource=source;
  const n=scopeLocalNodes.find(n=>n.rawId===String(target.id)&&n.stage===target.stage)||byId.get(String(target.id));if(n)focusNode(n);
 }
 history=[...trail,previous].slice(-32);relationStamp='';contextActionStamp='';schedule();
}
function focusCurrent(){if(relationFocus){const sid=relationFocus.scopeId,sk=model.after.nodes.find(n=>displaySK(n)&&String(n.scope_id)===sid);if(sk)visitRelation({kind:'task',sk,scopeId:sid,taskId:relationFocus.taskIds[0]});return;}if(selection?.kind==='scope'){focusScopeOnly(selection.id);return;}const spec=trackingSource||traceSpec();if(spec.scopeId!=null){focusScopeOnly(spec.scopeId);return;}if(spec.nodeId!=null)visitRelation({kind:'node',id:spec.nodeId,stage:spec.stage});}
function renderContextActions(){
 const host=$('scope-actions'),spec=trackingSource||traceSpec(),sid=relationFocus?.scopeId??(selection?.kind==='scope'?String(selection.id):null),local=selection?.kind==='local'?scopeLocalNodes.find(n=>n.id===selection.id):null;
 const raw=local?.raw||(model?.[stage]?.nodes||[]).find(n=>String(n.id)===String(selected)),sk=relationFocus?(model?.after?.nodes||[]).find(n=>displaySK(n)&&String(n.scope_id)===relationFocus.scopeId):displaySK(raw)?raw:sid&&(scopeViews.get(sid)?.updated.filter(displaySK).length===1)?(model?.after?.nodes||[]).find(n=>displaySK(n)&&String(n.scope_id)===sid):null;
 const targets=relationTargets(),stamp=JSON.stringify([selection,relationFocus,spec,tracePrefs,[...openSKs],[...scopeStages],stage,[...collapsedScopes],targets.map(t=>t.label),sk&&skPanels.get(skKey(sk))?.view?.filters,sk&&skPanels.get(skKey(sk))?.aligned,!!temporaryView]);
 if(relationFocus){const t=relations().view(relationFocus.scopeId)?.tasks.find(t=>t.id===relationFocus.taskIds[0]);$('selection-label').textContent='SK '+relationFocus.scopeId+' / 节点 #'+(sk?.id||'—')+' / '+(t?.queue_kind||'')+' / Task '+(t?.index??'—');}
 if(sid&&!relationFocus)$('selection-label').textContent='SK '+sid+' · '+(scopeViews.get(sid)?.stage==='after'?'更新后':'更新前');
 if(stamp===contextActionStamp)return;contextActionStamp=stamp;const opened=[...host.querySelectorAll('details[open]')].map(d=>d.dataset.menu),q=host.querySelector('.relation-search')?.value||'';host.replaceChildren();
 const button=(text,action)=>{const b=el('button',text,'toolbar-btn');b.onclick=action;host.append(b);return b;};
 if(sid&&!relationFocus){$('selection-label').textContent='SK '+sid+' · '+(scopeViews.get(sid)?.stage==='after'?'更新后':'更新前');button('定位 SK',()=>focusScopeOnly(sid));scopeControls(host,sid);const view=scopeViews.get(sid);for(const n of view?.updated.filter(displaySK)||[])host.append(skToggleButton(n,view.updated.filter(displaySK).length>1));}
 else if(sk){button('定位 '+(relationFocus?'Task':'SK'),focusCurrent);if(!relationFocus)host.append(skToggleButton(sk));}
 else if(selection)button('定位 Task',focusCurrent);
 if(sk&&openSKs.has(skKey(sk))&&skPanels.get(skKey(sk))?.view){const p=skPanels.get(skKey(sk)),menu=el('details');menu.dataset.menu='internal';menu.append(el('summary','内部显示设置'));const body=el('div',undefined,'context-menu');menu.append(body);for(const typ of ['FUNC','PRELOAD','SYNC','EVENT']){const label=el('label'),check=el('input');check.type='checkbox';check.checked=!p.view.filters.types.includes(typ);check.setAttribute('aria-label','当前 SK 显示 '+typ);check.onchange=()=>{history=[...history,captureContext()].slice(-32);const types=new Set(p.view.filters.types);check.checked?types.delete(typ):types.add(typ);p.view.setFilters([...types],p.view.filters.only);schedule();};label.append(check,document.createTextNode(typ));body.append(label);}for(const name of ['仅看关联','关联对齐']){const l=el('label'),c=el('input');c.type='checkbox';c.checked=name==='关联对齐'?p.aligned:p.view.filters.only;c.onchange=()=>{history=[...history,captureContext()].slice(-32);if(name==='关联对齐')setPanelAlignment(p,c.checked);else p.view.setFilters(p.view.filters.types,c.checked);schedule();};l.append(c,document.createTextNode(name));body.append(l);}host.append(menu);}
 const direct=spec.nodeId!=null?relations().fromNode(spec.stage,String(spec.nodeId)):[];
 if(direct.length){const ref=direct.find(r=>r.relation.kind==='origin')||direct[0],target=targets.find(t=>t.kind==='task'&&t.scopeId===ref.scopeId&&t.taskId===ref.taskId);button('查看对应内部任务',()=>visitRelation(target));}
 if(!direct.length&&raw&&!displaySK(raw)&&/KERNEL/.test(raw.type))host.append(el('span','未记录对应内部任务','muted'));
 if(relationFocus){const origin=(relations().view(relationFocus.scopeId)?.relations||[]).find(r=>r.kind==='origin'&&r.task_id===relationFocus.taskIds[0]);if(origin)button('查看原始任务',()=>visitRelation({kind:'node',id:origin.node_id,stage:'before'}));}
 if(targets.length){const menu=el('details');menu.dataset.menu='relations';menu.append(el('summary','关联 · '+targets.length));const body=el('div',undefined,'context-menu relation-menu'),search=el('input');search.className='relation-search';search.placeholder='搜索关联对象';search.value=q;body.append(search);const list=el('div',undefined,'relation-list');body.append(list);const fill=()=>{list.replaceChildren();const matches=targets.filter(t=>t.label.toLowerCase().includes(search.value.toLowerCase()));for(const t of matches.slice(0,100)){const status=relationTargetState(t),b=el('button',t.label+(status?' · '+status:''),'object-link');b.onclick=()=>{menu.open=false;visitRelation(t);};list.append(b);}if(matches.length>100)list.append(el('p','显示前 100 项，请搜索缩小范围。'));};search.oninput=fill;menu.addEventListener('toggle',()=>{if(menu.open)fill();});fill();menu.append(body);host.append(menu);button('上一关联',()=>stepRelation(-1));button('下一关联',()=>stepRelation(1));}
 if(temporaryView)button('恢复查看前状态',()=>{const snapshot=temporaryView;temporaryView=null;history=[...history,snapshot];goBack();});
 for(const d of host.querySelectorAll('details')){d.open=opened.includes(d.dataset.menu);d.querySelector('summary').onclick=e=>{e.preventDefault();const next=!d.open;for(const other of host.querySelectorAll('details'))other.open=false;d.open=next;};}
}
