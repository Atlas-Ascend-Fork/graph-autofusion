/* Cross-level relations are associations, never implicit dependency edges. */
(() => {
'use strict';
const cache=new WeakMap();
function create(model){
 if(model&&cache.has(model))return cache.get(model);
 const views=model?.exploration?.sync_by_scope||{},external=new Map();
 const breakByScope=new Map(),breakByNode=new Map();for(const b of model?.graph_fusion?.breaks||[]){const sid=String(b.scope_id),id=String(b.node_id);if(!breakByScope.has(sid))breakByScope.set(sid,[]);if(!breakByNode.has(id))breakByNode.set(id,[]);breakByScope.get(sid).push(b);breakByNode.get(id).push(b);}
 for(const [sid,v] of Object.entries(views))for(const r of v.relations||[]){if(r.node_id==null)continue;const k=r.stage+':'+r.node_id;if(!external.has(k))external.set(k,[]);external.get(k).push({scopeId:sid,taskId:r.task_id,relation:r});}
 function cohort(sid,ids){const v=views[sid]||{},set=new Set(ids),origins=new Set();for(const r of v.relations||[])if(r.kind==='origin'&&set.has(r.task_id))origins.add(r.node_id);
  for(const r of v.relations||[])if(r.kind==='companion'&&(set.has(r.source)||set.has(r.target))){set.add(r.source);set.add(r.target);}
  for(const r of v.relations||[])if(r.kind==='origin'&&set.has(r.task_id))origins.add(r.node_id);
  for(const r of v.relations||[])if(r.kind==='origin'&&origins.has(r.node_id))set.add(r.task_id);
  for(const r of v.relations||[])if(r.kind==='companion'&&set.has(r.target))set.add(r.source);
  return set;
 }
 const signalByTask=new Map();for(const g of model?.exploration?.signal_groups||[])for(const t of g.internal){const k=t.scope_id+':'+t.task_id;if(!signalByTask.has(k))signalByTask.set(k,[]);signalByTask.get(k).push(g);}
 // A fused member no longer has a visible one-to-one node after Update. Keep a
 // deterministic display representative so an old event association can still
 // be drawn against the updated SK boundary.
 const beforeNodes=new Map((model?.before?.nodes||[]).map(n=>[String(n.id),n])),afterNodes=new Map((model?.after?.nodes||[]).map(n=>[String(n.id),n])),afterNodeIds=new Set(afterNodes.keys()),afterForBefore=new Map(),fusionTargets=new Map();
 for(const row of model?.transformations||[])if(row.kind==='fusion')for(const id of row.before_ids||[]){const key=String(id);if(!fusionTargets.has(key))fusionTargets.set(key,new Set());for(const target of row.after_ids||[])if(afterNodeIds.has(String(target)))fusionTargets.get(key).add(String(target));}
 for(const [id,targets] of fusionTargets)if(targets.size===1&&afterNodes.get([...targets][0])?.is_sk)afterForBefore.set(id,[...targets]);
 const compatible=(id,target)=>{const old=beforeNodes.get(String(id)),updated=afterNodes.get(String(target));if(updated?.update_status==='not_recorded')return false;return old?.type==='NOTIFY'&&['NOTIFY','MEMORY_WRITE'].includes(updated?.type)||old?.type==='WAIT'&&['WAIT','MEMORY_WAIT'].includes(updated?.type);};
 for(const row of model?.transformations||[])if(row.kind!=='fusion')for(const id of row.before_ids||[]){const key=String(id),targets=[...new Set((row.after_ids||[]).map(String).filter(target=>compatible(key,target)))];if(!fusionTargets.has(key)&&targets.length===1)afterForBefore.set(key,targets);}
 for(const n of model?.before?.nodes||[])if(!fusionTargets.has(String(n.id))&&!afterForBefore.has(String(n.id))&&compatible(n.id,n.id))afterForBefore.set(String(n.id),[String(n.id)]);
 const events={before:new Map(),after:new Map()},eventByNode={before:new Map(),after:new Map()};
 for(const st of ['before','after'])for(const n of model?.[st]?.nodes||[]){if(n.update_status==='not_recorded')continue;if(n.type!=='NOTIFY'&&n.type!=='WAIT')continue;const raw=n.params?.eventParams?.eventId??n.params?.event_id;let id;try{id=BigInt(raw);if(id<=0n)continue;}catch{continue;}const k=id.toString();if(!events[st].has(k))events[st].set(k,[]);events[st].get(k).push(n);eventByNode[st].set(String(n.id),k);}
 // Whole-graph and selected lines share exactly the same Event identity and
 // explicit Update mapping. These associations never create dependency edges.
 function eventIdentityReasons(st,sourceId,targetId){
  const result=[{kind:'event_identity',stage:st,source:sourceId,target:targetId}];
  if(st==='before'&&(fusionTargets.has(sourceId)||fusionTargets.has(targetId)))for(const source of afterForBefore.get(sourceId)||[])for(const target of afterForBefore.get(targetId)||[])if(source!==target)result.push({kind:'event_identity',stage:'after',source,target,cross_stage:true,before_source:sourceId,before_target:targetId});
  return result;
 }
 let allWaitingRelations=null;
 function waitingRelations(){
  if(allWaitingRelations)return allWaitingRelations;
  allWaitingRelations=[];const seen=new Set();
  for(const st of ['before','after'])for(const group of events[st].values())for(const source of group.filter(n=>n.type==='NOTIFY'))for(const target of group.filter(n=>n.type==='WAIT'))for(const reason of eventIdentityReasons(st,String(source.id),String(target.id))){const key=JSON.stringify([reason.stage,reason.source,reason.target,reason.before_source,reason.before_target]);if(!seen.has(key)){seen.add(key);allWaitingRelations.push(reason);}}
  return allWaitingRelations;
 }
 function relatedTasks(sid,ids){const result=new Map(),add=(scopeId,taskId)=>result.set(scopeId+':'+taskId,{scopeId,taskId});for(const id of cohort(sid,ids)){add(sid,id);for(const g of signalByTask.get(sid+':'+id)||[])for(const t of g.internal)add(t.scope_id,t.task_id);}return [...result.values()];}
 // Logical endpoints stay independent of their current collapsed representation.
 function trace({scopeId,taskIds=[],stage='before',nodeId,direction='both',depth=1,kind='sync'}={}){
  const taskRefs=new Map(),nodes={before:new Set(),after:new Set()},reasons=[];
  const selectedScopeNodes=scopeId!=null&&!taskIds.length?new Set([...(model?.scopes?.find(s=>String(s.id)===String(scopeId))?.node_ids||[]),(model?.after?.nodes||[]).filter(n=>String(n.scope_id)===String(scopeId)).map(n=>n.id)].flat().map(String)):null;
  const addTask=(sid,id)=>taskRefs.set(String(sid)+':'+String(id),{scopeId:String(sid),taskId:String(id)});
  const addNode=(st,id)=>nodes[st]?.add(String(id));
  if(scopeId!=null){for(const id of taskIds.length?taskIds:(views[scopeId]?.tasks||[]).map(t=>t.id))addTask(scopeId,id);}
  if(nodeId!=null)addNode(stage,nodeId);
  if(scopeId!=null&&!taskIds.length){for(const n of model?.scopes?.find(s=>String(s.id)===String(scopeId))?.node_ids||[])addNode('before',n);for(const n of model?.after?.nodes||[])if(String(n.scope_id)===String(scopeId))addNode('after',n.id);}
  const expandedTasks=new Set(),expandedNodes=new Set();
  const bridgedEvents=new Set();
  function identities(){
   // Identity mappings cost no dependency hop. A selected SYNC stays at its paired endpoints.
   for(const st of ['before','after'])for(const id of nodes[st]){const key=st+':'+id;if(expandedNodes.has(key))continue;expandedNodes.add(key);for(const r of external.get(key)||[])addTask(r.scopeId,r.taskId);}
   for(const r of taskRefs.values()){
    const key=r.scopeId+':'+r.taskId;if(expandedTasks.has(key))continue;expandedTasks.add(key);
    const task=views[r.scopeId]?.tasks.find(t=>t.id===r.taskId);
    if(task?.type!=='SYNC')for(const peer of relatedTasks(r.scopeId,[r.taskId]))addTask(peer.scopeId,peer.taskId);
    for(const mapping of views[r.scopeId]?.relations||[])if(mapping.node_id!=null&&mapping.task_id===r.taskId){addNode(mapping.stage,mapping.node_id);reasons.push({...mapping,scope_id:r.scopeId});}
   }
  }
  identities();
  for(let hop=0;hop<Math.max(0,Math.min(3,Number(depth)||0));hop++){
   const taskStarts=new Set(taskRefs.keys());
   for(const [sid,v] of Object.entries(views))for(const e of [...(v.links||[]),...(v.barrier_pairs||[])]){
    const sequential=e.kind==='queue';if(kind==='sync'&&sequential||kind==='stream'&&!sequential)continue;
    const source=sid+':'+e.source,target=sid+':'+e.target;
    if((direction!=='up'&&taskStarts.has(source))||(direction!=='down'&&taskStarts.has(target))||(e.kind==='all_sync'&&(taskStarts.has(source)||taskStarts.has(target)))){addTask(sid,e.source);addTask(sid,e.target);reasons.push({...e,scope_id:sid});}
   }
   for(const st of ['before','after']){const starts=new Set(nodes[st]);for(const e of model?.[st]?.edges||[]){if(kind==='sync'&&e.kind==='stream'||kind==='stream'&&e.kind!=='stream')continue;if((direction!=='up'&&starts.has(String(e.source)))||(direction!=='down'&&starts.has(String(e.target)))){addNode(st,e.source);addNode(st,e.target);reasons.push({...e,stage:st});}}
    if(kind!=='stream'){
     const groups=new Set();for(const id of starts){const eventId=eventByNode[st].get(id);if(eventId==null||groups.has(eventId))continue;groups.add(eventId);
      const group=events[st].get(eventId)||[],current=group.filter(n=>starts.has(String(n.id))),allowNotify=direction!=='down'||current.some(n=>n.type==='NOTIFY'),allowWait=direction!=='up'||current.some(n=>n.type==='WAIT');
      for(const peer of group)if(peer.type==='NOTIFY'?allowNotify:allowWait)addNode(st,peer.id);
      const addEventIdentity=(source,target)=>{
       const sourceId=String(source.id),targetId=String(target.id);
       if(selectedScopeNodes&&!selectedScopeNodes.has(sourceId)&&!selectedScopeNodes.has(targetId))return;
       for(const reason of eventIdentityReasons(st,sourceId,targetId)){
        if(!reason.cross_stage){reasons.push(reason);continue;}
        const key=reason.source+'>'+reason.target;
        if(bridgedEvents.has(key))continue;
        bridgedEvents.add(key);addNode('after',reason.source);addNode('after',reason.target);reasons.push(reason);
       }
      };
      if(direction==='both'){for(const a of group.filter(n=>n.type==='NOTIFY'))for(const b of group.filter(n=>n.type==='WAIT'))addEventIdentity(a,b);}
      else for(const a of current)for(const b of group){if(a.type===b.type||a.type==='NOTIFY'&&direction==='up'||a.type==='WAIT'&&direction==='down')continue;addEventIdentity(a,b);}
     }
    }
   }
   identities();
  }
  const direct=scopeId!=null&&!taskIds.length?(breakByScope.get(String(scopeId))||[]):nodeId!=null?(breakByNode.get(String(nodeId))||[]):[];
  for(const b of direct){const sid=String(b.scope_id),id=String(b.node_id);addNode('before',id);if((model?.after?.nodes||[]).some(n=>String(n.id)===id&&n.type!=='INVALID'))addNode('after',id);for(const n of model?.after?.nodes||[])if(n.is_sk&&String(n.scope_id)===sid)addNode('after',n.id);reasons.push({kind:'fusion_break',scope_id:sid,node_id:id,description:b.reason});}
  return {tasks:[...taskRefs.values()],nodes,reasons};
 }
 const index={trace,waitingRelations,fromNode:(stage,id)=>external.get(stage+':'+id)||[],cohort,relatedTasks,
  toNodes:(sid,ids,stage)=>{const out=new Map();for(const t of relatedTasks(sid,ids))for(const r of views[t.scopeId]?.relations||[])if(r.stage===stage&&r.task_id===t.taskId&&r.node_id!=null)out.set(r.node_id,r);return [...out.values()];},
  view:sid=>views[sid]};if(model)cache.set(model,index);return index;
}
window.GraphRelations={create};
})();
