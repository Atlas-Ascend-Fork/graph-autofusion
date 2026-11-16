/* Navigation overview uses world bounds, independently of camera pan/zoom. */
let minimapTransform=null,minimapDrag=null,minimapScene=null,minimapSceneKey='';
function paintMinimapBatch(c,entries){const groups=new Map();for(const entry of entries){const key=entry.color;let group=groups.get(key);if(!group){group={color:key,boxes:[]};groups.set(key,group);}group.boxes.push(entry);}for(const group of groups.values()){c.fillStyle=group.color;c.beginPath();for(const b of group.boxes)c.rect(b.x,b.y,b.w,b.h);c.fill();}}
function drawMinimap(){
 const mini=$('minimap');if(!$('minimap-panel').open)return;
 const mw=240,mh=140,pad=8,dpr=devicePixelRatio||1;
 if(mini.width!==mw*dpr||mini.height!==mh*dpr){mini.width=mw*dpr;mini.height=mh*dpr;}
 const c=mini.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);c.clearRect(0,0,mw,mh);c.fillStyle='#101b22';c.fillRect(0,0,mw,mh);
 const left=Number.isFinite(worldLeft)?worldLeft:0,w=Math.max(1,worldW-left),h=Math.max(1,worldH);
 const sx=(mw-2*pad)/w,sy=(mh-2*pad)/h;minimapTransform={left,sx,sy,pad,mw,mh};
 const sceneKey=[model?.key||'',stage,minimapRevision,[...filters.types].sort().join(','),[...filters.streams].sort().join(','),[...filterReveals].sort().join(','),[...activeBreakNodes].sort().join(',')].join('|');
 if(!minimapScene||minimapSceneKey!==sceneKey){
  const scene=document.createElement('canvas');scene.width=mw;scene.height=mh;const sc=scene.getContext('2d');sc.fillStyle='#101b22';sc.fillRect(0,0,mw,mh);
  const box=(n,color)=>({x:pad+(n.x-left)*sx,y:pad+n.y*sy,w:Math.max(.8,n.w*sx),h:Math.max(1,n.h*sy),color});
  const scopes=[],graphNodes=[],localNodes=[];
  for(const r of scopeRects)if(!filters.types.has('scope'))scopes.push(box(r,r.fusionStatus==='sk'?fusionColor(r):r.fusionStatus==='failed'?'#d96f78':'#526873'));
  for(const n of nodes)if(!n.scopeProxy&&visibleNode(n))graphNodes.push(box(n,nodeFill(n)));
  for(const n of scopeLocalNodes)if(visibleLocal(n))localNodes.push(box(n,nodeFill(n.raw,n.stage)));
  paintMinimapBatch(sc,scopes);paintMinimapBatch(sc,graphNodes);paintMinimapBatch(sc,localNodes);
  for(const [key,p] of skPanels)if(openSKs.has(key)&&!p.root.hidden){const b=box(p,'#284754');sc.fillStyle=b.color;sc.fillRect(b.x,b.y,b.w,b.h);sc.strokeStyle='#80cda9';sc.strokeRect(b.x,b.y,b.w,b.h);}
  minimapScene=scene;minimapSceneKey=sceneKey;
 }
 c.drawImage(minimapScene,0,0,mw,mh);
 const x=pad+(-ox/scale-left)*sx,y=pad+(-oy/scale)*sy,vw=width/scale*sx,vh=height/scale*sy;
 c.save();c.beginPath();c.rect(pad,pad,mw-2*pad,mh-2*pad);c.clip();c.fillStyle='#82caff24';c.strokeStyle='#82caff';c.lineWidth=1.5;c.fillRect(x,y,vw,vh);c.strokeRect(x,y,Math.max(2,vw),Math.max(2,vh));c.restore();
 mini.dataset.viewport=JSON.stringify({x,y,w:vw,h:vh});
}
function minimapPoint(event){const t=minimapTransform,r=$('minimap').getBoundingClientRect();if(!t)return null;return {x:t.left+((event.clientX-r.left)*t.mw/r.width-t.pad)/t.sx,y:((event.clientY-r.top)*t.mh/r.height-t.pad)/t.sy};}
function minimapMove(point){if(!point)return;const cx=Math.max(worldLeft,Math.min(worldW,point.x)),cy=Math.max(0,Math.min(worldH,point.y));ox=width/2-cx*scale;oy=height/2-cy*scale;schedule();}
$('minimap').addEventListener('pointerdown',e=>{if(e.button!==0||!model)return;e.preventDefault();e.stopPropagation();const p=minimapPoint(e);if(!p)return;history=[...history,captureContext()].slice(-32);const left=-ox/scale,top=-oy/scale,inside=p.x>=left&&p.x<=left+width/scale&&p.y>=top&&p.y<=top+height/scale;minimapDrag={dx:inside?(left+width/scale/2-p.x):0,dy:inside?(top+height/scale/2-p.y):0};$('minimap').setPointerCapture(e.pointerId);minimapMove({x:p.x+minimapDrag.dx,y:p.y+minimapDrag.dy});});
$('minimap').addEventListener('pointermove',e=>{if(!minimapDrag)return;const p=minimapPoint(e);if(p)minimapMove({x:p.x+minimapDrag.dx,y:p.y+minimapDrag.dy});});
for(const event of ['pointerup','pointercancel','lostpointercapture'])$('minimap').addEventListener(event,()=>{minimapDrag=null;});
$('minimap').addEventListener('keydown',e=>{const dirs={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]},d=dirs[e.key];if(!d||!model)return;e.preventDefault();e.stopPropagation();history=[...history,captureContext()].slice(-32);minimapMove({x:(width/2-ox)/scale+d[0]*width/scale*.5,y:(height/2-oy)/scale+d[1]*height/scale*.5});});
$('minimap-panel').addEventListener('toggle',schedule);
