/* Stream-preserving projection and horizontal swimlane layout. */
(() => {
'use strict';
const W=146,H=50,GAP=44,LANE=190;
const streamOf=n=>String(n.stream??n.stream_index??'?');
function waitingEdges(edges,reasons=null,stage=null){
 const matches=reasons&&new Set(reasons.filter(r=>r.stage===stage&&(r.kind==='event'||r.kind==='memory')).map(r=>JSON.stringify([r.kind,String(r.source),String(r.target)])));
 return edges.filter(e=>(e.kind==='event'||e.kind==='memory')&&(!matches||matches.has(JSON.stringify([e.kind,String(e.sourceMember??e.source),String(e.targetMember??e.target)]))));
}
const orderOf=(n,i)=>Number.isFinite(Number(n.display_order??n.order??n.params?.original_order))?Number(n.display_order??n.order??n.params?.original_order):i;
function streamsOf(snapshot){const streams=new Map();(snapshot.nodes||[]).forEach((n,i)=>{const stream=streamOf(n);if(!streams.has(stream))streams.set(stream,[]);streams.get(stream).push({n,order:orderOf(n,i),index:i});});for(const list of streams.values())list.sort((a,b)=>a.order-b.order||a.index-b.index);return streams;}
function scopeOwners(snapshot,scopes,stage){const owners=new Map();if(stage==='before')for(const s of scopes)for(const id of s.node_ids||[])if(!owners.has(String(id)))owners.set(String(id),String(s.id));for(const n of snapshot.nodes||[])if(n.scope_id!=null&&!owners.has(String(n.id)))owners.set(String(n.id),String(n.scope_id));return owners;}
function resetGroups(snapshot,scopes=[],stage='before'){
 const owners=scopeOwners(snapshot,scopes,stage),groups=[];
 for(const [stream,list] of streamsOf(snapshot)){let run=[],ownership;const flush=()=>{if(run.length>1)groups.push({id:'__reset:'+stream+':'+run[0],node_ids:run,stream,scope_id:ownership??null});run=[];};
  for(const {n} of list){const scope=owners.get(String(n.id));if(String(n.type).toUpperCase()!=='RESET'){flush();continue;}if(run.length&&scope!==ownership)flush();ownership=scope;run.push(String(n.id));}flush();
 }return groups;
}
function project(snapshot,groups=[],expanded=new Set(),stage='before',expandedIterations=new Set(),opts={}){
 const owner=new Map();for(const g of groups)if(!expanded.has(g.id)){const parts=g.iterations||[];if(parts.length){for(const part of parts)if(!expandedIterations.has(g.id+':'+part.index))for(const id of (stage==='before'?part.node_ids:part.after_ids)||[])if(!owner.has(String(id)))owner.set(String(id),g);}else for(const id of (stage==='before'?g.node_ids:g.after_ids)||[])if(!owner.has(String(id)))owner.set(String(id),g);}
 const memberships=scopeOwners(snapshot,opts.scopes||[],stage),scopeFoldOwner=new Map();for(const raw of snapshot.nodes||[]){const sid=memberships.get(String(raw.id)),view=opts.scopeViews?.get(sid);if(view&&(view.collapsed||view.stage!==stage))scopeFoldOwner.set(String(raw.id),view);}
 const resets=resetGroups(snapshot,opts.scopes||[],stage),resetOwner=new Map();if(opts.foldResets!==false)for(const g of resets)if(!opts.expandedResets?.has(g.id)&&!g.node_ids.some(id=>owner.has(id)||scopeFoldOwner.has(id)))for(const id of g.node_ids)resetOwner.set(id,g);
 const nodes=[],proxies=new Map(),mapping=new Map();
 for(const raw of snapshot.nodes||[]){const id=String(raw.id),g=owner.get(id),reset=resetOwner.get(id),scopeView=scopeFoldOwner.get(id);if(!g&&!reset&&!scopeView){nodes.push({...raw,id,streamPositions:[]});mapping.set(id,id);continue;}
  const key=scopeView?'__scope:'+scopeView.id:g?'__fold:'+g.id:reset.id;let n=proxies.get(key);if(!n){n=scopeView?{id:key,label:'SK '+scopeView.id,type:'SCOPE',scopeProxy:true,scopeView,scopeWidth:scopeView.collapsed?320:scopeView.width,scope_id:scopeView.id,memberCount:0,streamPositions:[]}:g?{id:key,label:g.topology_status==='candidate'?'重复结构候选':'重复结构',type:'重复结构',fold:g,is_sk:false,scope_id:null,memberCount:0,streams:new Set(),streamPositions:[],remainingCount:g.iterations?.length?g.iterations.filter(i=>!expandedIterations.has(g.id+':'+i.index)).length:g.repeat_count}:{...raw,id:key,label:'RESET',type:'RESET',resetGroup:{...reset,node_ids:[]},scope_id:reset.scope_id,memberCount:0,streamPositions:[]};n._layoutOnly=true;proxies.set(key,n);nodes.push(n);}if(!raw._layoutOnly){n._layoutOnly=false;n.memberCount++;if(n.resetGroup){n.resetGroup.node_ids.push(id);n.label='RESET ×'+n.memberCount;}}if(g)n.streams.add(streamOf(raw));mapping.set(id,key);
 }
 const byId=new Map(nodes.map(n=>[n.id,n]));for(const [stream,list] of streamsOf(snapshot)){let previous=null;for(const {n:raw,order,index} of list){const id=String(raw.id),node=byId.get(mapping.get(id));if(previous?.node===node){previous.port.node_ids.push(id);previous.port.lastOrder=order;}else{const port={stream,order,lastOrder:order,index,node_ids:[id]};node.streamPositions.push(port);previous={node,port};}}}
 const merged=new Map();for(const e of snapshot.edges||[]){const sourceMember=String(e.source),targetMember=String(e.target),source=mapping.get(sourceMember),target=mapping.get(targetMember);if(!source||!target||(source===target&&(owner.has(sourceMember)||resetOwner.has(sourceMember)||scopeFoldOwner.has(sourceMember))))continue;
  // Keep boundary member identity: collapsed edges attach to their actual stream port.
  const key=JSON.stringify([source,target,e.kind,e.evidence,e.event_id,sourceMember,targetMember]);const prev=merged.get(key);if(prev)prev.count++;else merged.set(key,{...e,source,target,sourceMember,targetMember,count:1});
 }return {nodes,edges:[...merged.values()],resetGroups:resets};
}
function bounds(boxes){let x=Infinity,y=Infinity,right=-Infinity,bottom=-Infinity;for(const b of boxes){x=Math.min(x,b.x);y=Math.min(y,b.y);right=Math.max(right,b.x+b.w);bottom=Math.max(bottom,b.y+b.h);}return {x,y,w:right-x,h:bottom-y};}
// Exact member boxes joined through lane gutters. The bbox is only a navigation extent.
function shape(boxes,spine=36){const connectors=[];if(boxes.length>1){let top=Infinity,bottom=-Infinity;for(const b of boxes){const y=b.y-10;top=Math.min(top,y);bottom=Math.max(bottom,y);connectors.push([[b.x+12,b.y],[b.x+12,y],[spine,y]]);}connectors.push([[spine,top],[spine,bottom]]);}return {...bounds(boxes),boxes,connectors};}
function layout(nodes,edges,scopes=[],stage='before',extraStreams=[],alignmentEdges=null){
 const byId=new Map(nodes.map(n=>[String(n.id),n])),streams=new Map();nodes.forEach((n,i)=>{if(!n.streamPositions?.length)n.streamPositions=[{stream:streamOf(n),order:orderOf(n,i),index:i,node_ids:[String(n.id)]}];for(const port of n.streamPositions){port.node=n;if(!streams.has(port.stream))streams.set(port.stream,[]);streams.get(port.stream).push(port);}});
 for(const id of extraStreams)if(!streams.has(String(id)))streams.set(String(id),[]);
 const sorted=[...streams].sort((a,b)=>a[0].localeCompare(b[0],undefined,{numeric:true})),lanes=[];let worldW=500;
 sorted.forEach(([id,ports],row)=>{const y=80+row*LANE;lanes.push({id,y:y-34,h:LANE});ports.sort((a,b)=>a.order-b.order||a.index-b.index);let x=120;for(const p of ports){p.x=x;p.y=y;p.w=p.node.scopeWidth||(p.node.fold?200:W);p.h=p.node.fold?100:H;x+=p.w+GAP;}worldW=Math.max(worldW,x+80);});
 // Treat each SK as one alignment block, constrained by every Stream's order.
 // Unscoped tasks remain in their own lane; SK blocks reserve full columns.
 const owners=scopeOwners({nodes},scopes,stage),units=new Map(),unitOf=new Map();
 for(const [,ports] of sorted)for(const p of ports){const sid=p.node.fold?null:p.node.resetGroup?.scope_id??owners.get(p.node.id),key=sid==null?'task:'+p.node.id+':'+p.stream+':'+p.index:'scope:'+sid;
  if(!units.has(key))units.set(key,{key,sid,ports:[],next:new Set(),degree:0});const u=units.get(key);u.ports.push(p);unitOf.set(p,u);}
 for(const [,ports] of sorted){let previous=null;for(const p of ports){const u=unitOf.get(p);if(previous&&previous!==u&&!previous.next.has(u)){previous.next.add(u);u.degree++;}previous=u;}}
 const membersById=new Map();for(const [,ports] of sorted)for(const p of ports)for(const id of p.node_ids||[])membersById.set(String(id),p);
 const localEvents=new Map();if(stage==='before'||alignmentEdges)for(const e of alignmentEdges||edges){if(!['event','memory'].includes(e.kind))continue;const a=membersById.get(String(e.sourceMember??e.source)),b=membersById.get(String(e.targetMember??e.target));if(!a||!b||a.stream===b.stream)continue;const u=unitOf.get(a);if(!u||u.sid==null||u!==unitOf.get(b))continue;if(!localEvents.has(u))localEvents.set(u,[]);localEvents.get(u).push([a,b]);}
 function alignLocal(rows,start,links){
  if(!links?.length)return null;const ports=[...rows.values()].flat(),next=new Map(ports.map(p=>[p,[]])),degree=new Map(ports.map(p=>[p,0]));
  const add=(a,b,d)=>{next.get(a).push([b,d]);degree.set(b,degree.get(b)+1);};
  for(const q of rows.values())for(let i=1;i<q.length;i++)add(q[i-1],q[i],q[i-1].w+GAP);
  for(const [a,b] of links)add(a,b,0);
  const ready=ports.filter(p=>!degree.get(p)),x=new Map(ports.map(p=>[p,start]));let processed=0;
  for(let i=0;i<ready.length;i++){const p=ready[i];processed++;for(const [q,d] of next.get(p)){x.set(q,Math.max(x.get(q),x.get(p)+d));degree.set(q,degree.get(q)-1);if(!degree.get(q))ready.push(q);}}
  if(processed!==ports.length)return null;
  const producers=new Set(links.map(([a])=>a));for(let i=ready.length-1;i>=0;i--){const p=ready[i];if(!producers.has(p))continue;let limit=Infinity;for(const [q,d] of next.get(p))limit=Math.min(limit,x.get(q)-d);if(Number.isFinite(limit))x.set(p,Math.max(x.get(p),limit));}
  return x;
 }
 const ready=[...units.values()].filter(u=>!u.degree),ordered=[];for(let i=0;i<ready.length;i++){const u=ready[i];ordered.push(u);for(const next of u.next)if(!--next.degree)ready.push(next);}
 const alignmentConflict=ordered.length!==units.size;
 if(!alignmentConflict){const laneEnd=new Map(sorted.map(([id])=>[id,120]));for(const u of ordered){const rows=new Map();for(const p of u.ports){if(!rows.has(p.stream))rows.set(p.stream,[]);rows.get(p.stream).push(p);}let start=120;for(const stream of rows.keys())start=Math.max(start,laneEnd.get(stream));if(u.sid!=null)for(const end of laneEnd.values())start=Math.max(start,end);let end=start;
   const aligned=alignLocal(rows,start,localEvents.get(u));for(const [stream,ports] of rows){let x=start;for(const p of ports){p.x=aligned?.get(p)??x;x=p.x+p.w+GAP;}laneEnd.set(stream,x);end=Math.max(end,x);}if(u.sid!=null){end=Math.max(end,start+360+GAP);for(const stream of laneEnd.keys())laneEnd.set(stream,end);}
  }worldW=Math.max(500,...laneEnd.values())+80;}
 const portByMember=new Map();for(const n of nodes){const boxes=n.streamPositions.map(p=>({stream:p.stream,x:p.x,y:p.y,w:p.w,h:p.h,node_ids:[...p.node_ids]}));for(const p of n.streamPositions){for(const id of p.node_ids)portByMember.set(String(id),p);delete p.node;}Object.assign(n,shape(boxes));}
 const scopeById=new Map(scopes.map(s=>[String(s.id),s])),owner=scopeOwners({nodes},scopes,stage),members=new Map();
 for(const n of nodes){if(n.fold)continue;for(const p of n.streamPositions){const ownership=n.resetGroup?.scope_id??owner.get(n.id);const sid=ownership==null?null:String(ownership);if(sid===null||!scopeById.has(sid))continue;if(!members.has(sid))members.set(sid,[]);members.get(sid).push({n,p});}}
 const regions=[];for(const [sid,list] of members){const rows=new Map();for(const {n,p} of list){if(!rows.has(p.stream))rows.set(p.stream,[]);rows.get(p.stream).push({n,p});}const boxes=[];for(const [stream,items] of rows){items.sort((a,b)=>a.p.x-b.p.x);let box=null;for(const {n,p} of items){if(box&&p.x-(box.x+box.w)<=GAP+12){box.w=p.x+p.w+12-box.x;box.node_ids.push(n.id);}else{box={stream,x:p.x-12,y:p.y-18,w:p.w+24,h:p.h+36,node_ids:[n.id]};boxes.push(box);}}}const extent=bounds(boxes);if(!alignmentConflict)extent.w=Math.max(extent.w,360);const geometry=alignmentConflict?shape(boxes,24):{...extent,y:extent.y-10,h:extent.h+10,boxes:[{...extent,y:extent.y-10,h:extent.h+10,node_ids:[...new Set(list.map(({n})=>n.id))]}],connectors:[]};regions.push({scope:scopeById.get(sid),...geometry,node_ids:[...new Set(list.map(({n})=>n.id))],rows:boxes.map(b=>({...b}))});}
 // All cross-lane paths use the left gutter; horizontal segments use free lane space.
 for(const e of edges){const a=portByMember.get(e.sourceMember)||byId.get(String(e.source))?.streamPositions[0],b=portByMember.get(e.targetMember)||byId.get(String(e.target))?.streamPositions[0];if(!a||!b)continue;e.sourcePort={x:a.x,y:a.y,w:a.w,h:a.h,stream:a.stream};e.targetPort={x:b.x,y:b.y,w:b.w,h:b.h,stream:b.stream};const start=[a.x+a.w,a.y+a.h/2],end=[b.x,b.y+b.h/2];if(a.stream===b.stream&&(e.kind==='stream'||b.x>a.x&&b.x-a.x-a.w<=GAP)){e.points=[start,end];continue;}const exitX=a.x+a.w+10,entryX=b.x-10,exitY=a.y+116,entryY=b.y+116;if(a.stream===b.stream)e.points=[start,[exitX,start[1]],[exitX,exitY],[entryX,entryY],[entryX,end[1]],end];else e.points=[start,[exitX,start[1]],[exitX,exitY],[70,exitY],[70,entryY],[entryX,entryY],[entryX,end[1]],end];}
 return {regions,lanes,alignmentConflict,worldW,worldH:Math.max(240,80+sorted.length*LANE)};
}
window.GraphNetwork={project,layout,resetGroups,waitingEdges};
})();
