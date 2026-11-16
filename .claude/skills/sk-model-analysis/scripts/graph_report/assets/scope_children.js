/* Reserve vertical space below each owner; keep original coordinates for reversible folding. */
let scopeLayoutBase=[],worldLeft=0;
function rememberScopeLayout(){
 const seen=new Set();scopeLayoutBase=[];
 const record=o=>{if(!o||seen.has(o))return;seen.add(o);if(typeof o.y==='number')scopeLayoutBase.push({object:o,y:o.y,h:o.h});for(const k of ['boxes','rows','streamPositions'])for(const child of o[k]||[])record(child);for(const points of o.connectors||[])for(const p of points)recordPoint(p);};
 const recordPoint=p=>{if(seen.has(p))return;seen.add(p);scopeLayoutBase.push({point:p,y:p[1]});};
 for(const o of [...nodes,...scopeRects,...scopeLocalNodes,...lanes])record(o);
 for(const e of edges){record(e.sourcePort);record(e.targetPort);for(const p of e.points||[])recordPoint(p);}
}
function positionSKPanels(){
 for(const b of scopeLayoutBase){if(b.point)b.point[1]=b.y;else{b.object.y=b.y;if(b.h!==undefined)b.object.h=b.h;}}
 worldW=graphBaseW;worldH=graphBaseH;worldLeft=0;
 const groups=new Map();
 for(const [key,p] of skPanels){
  p.root.hidden=!openSKs.has(key);if(p.root.hidden)continue;
  const scope=scopeRects.find(r=>String(r.scope.id)===String(p.n.scope_id));
  const owner=scope||scopeLocalNodes.find(n=>n.rawId===String(p.n.id))||byId.get(String(p.n.id));
  if(!owner){p.root.hidden=true;continue;}
  const cut=owner.y+owner.h+32;
  if(!groups.has(cut))groups.set(cut,[]);
  groups.get(cut).push({p,owner,x:owner.x+(owner.w-p.w)/2});
 }
 const bands=[];
 for(const [cut,items] of [...groups].sort((a,b)=>a[0]-b[0])){
  const placed=[];let size=0;
  for(const item of items){const {p,x}=item;let offset=8;for(const q of placed)if(x<q.x+q.p.w+24&&x+p.w+24>q.x)offset=Math.max(offset,q.offset+q.p.h+32);item.offset=offset;placed.push(item);size=Math.max(size,offset+p.h+32);}
  bands.push({cut,size,items});
 }
 // Strict bottom boundaries keep the owner above its own inserted child band.
 const offset=(y,inclusive=true)=>bands.reduce((sum,b)=>sum+((inclusive?b.cut<=y:b.cut<y)?b.size:0),0);
 for(const b of scopeLayoutBase){if(b.point)b.point[1]=b.y+offset(b.y);else{b.object.y=b.y+offset(b.y);if(b.h!==undefined)b.object.h=b.h+offset(b.y+b.h,false)-offset(b.y);}}
 for(const band of bands)for(const {p,owner,x,offset:dy} of band.items){
  p.x=x;p.y=band.cut+offset(band.cut,false)+dy;
  p.anchor={x:owner.x+owner.w/2,y:owner.y+owner.h};
  p.root.style.left=p.x+'px';p.root.style.top=p.y+'px';
  worldW=Math.max(worldW,p.x+p.w+40);worldLeft=Math.min(worldLeft,p.x-40);
 }
 worldH=graphBaseH+bands.reduce((sum,b)=>sum+b.size,0);
 minimapRevision++;
}
