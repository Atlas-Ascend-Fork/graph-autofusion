/* Auxiliary annotation layer: no layout ownership, hit targets or node colours. */
let userScopeGeometry=[],userScopeGeometryKey='',userScopeLayoutNodes=null;
function userScopeRegions(){
 const key=JSON.stringify([minimapRevision,stage,[...filters.types],[...filters.streams],[...filterReveals],tracePrefs.only,selection,relationFocus,[...scopeStages]]);
 if(userScopeLayoutNodes===nodes&&userScopeGeometryKey===key)return userScopeGeometry;
 userScopeLayoutNodes=nodes;userScopeGeometryKey=key;userScopeGeometry=[];
 if(!(model?.user_scopes||[]).length)return userScopeGeometry;
 const regions=new Map(scopeRects.map(r=>[String(r.scope.id),r]));
 for(const annotation of model.user_scopes){
  const membership={before:new Set((annotation.node_ids||[]).map(String)),after:new Set((annotation.after_node_ids||[]).map(String))},boxes=[];
  const accepts=(raw,st)=>membership[st]?.has(String(raw.id))&&!filters.types.has(category(raw))&&!filters.streams.has(String(raw.stream??raw.stream_index??'?'));
  for(const n of nodes){
   if(!visibleNode(n))continue;
   if(n.scopeProxy){const view=scopeViews.get(String(n.scope_id));if(!view||!view.collapsed)continue;const r=regions.get(String(n.scope_id));if(r&&view.nodes.some(raw=>accepts(raw,view.stage)))boxes.push(r);continue;}
   for(const box of n.streamPositions||shapeBoxes(n)){const ids=box.node_ids||[n.id];if(!filters.streams.has(String(box.stream??n.stream??n.stream_index))&&ids.some(id=>membership[stage].has(String(id))))boxes.push(box);}
  }
  for(const n of scopeLocalNodes)if(visibleLocal(n)&&accepts(n.raw,n.stage))boxes.push(n);
  if(!boxes.length)continue;
  let left=Infinity,top=Infinity,right=-Infinity,bottom=-Infinity;for(const b of boxes){left=Math.min(left,b.x);top=Math.min(top,b.y);right=Math.max(right,b.x+b.w);bottom=Math.max(bottom,b.y+b.h);}const x=left-16,y=top-48;right+=16;bottom+=34;
  userScopeGeometry.push({id:annotation.id,name:annotation.name,x,y,w:right-x,h:bottom-y,memberCount:annotation.node_ids.length});
 }
 return userScopeGeometry;
}
function paintUserScopes(intersects){
 if(!$('show-user-scopes').checked)return;
 ctx.save();ctx.globalAlpha=.68;ctx.strokeStyle='#829eae';ctx.lineWidth=1/scale;ctx.setLineDash([7/scale,6/scale]);
 for(const r of userScopeRegions()){
  if(!intersects(r))continue;
  ctx.strokeRect(r.x,r.y,r.w,r.h);
  if(scale>=.3){ctx.font='11px sans-serif';ctx.fillStyle='#a7bdca';ctx.fillText('用户 Scope · '+r.name,r.x+8,r.y+15);}
 }
 ctx.restore();
}
