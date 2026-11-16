/* Graph configuration and per-SK toolbar presentation; injected in the app closure. */
function renderInternalControls(){
 for(const [key,entry] of skPanels){
  if(!entry.view)continue;
  const state=entry.view?.filters||{types:[],only:false},hidden=entry.view?.hiddenRelated||0;
  const stamp=JSON.stringify([state,hidden,entry.view?.hiddenCount]);if(entry.controlStamp===stamp)continue;entry.controlStamp=stamp;
  const host=entry.controls,wasOpen=host.querySelector('details')?.open;host.replaceChildren();
  const menu=el('details'),summary=el('summary','显示任务'+(entry.view?.hiddenCount?' · 隐藏 '+entry.view.hiddenCount+' 个':''));menu.append(summary);menu.open=!!wasOpen;
  const body=el('div',undefined,'sk-tool-menu');menu.append(body);host.append(menu);
  const apply=(types,only)=>{history=[...history,captureContext()].slice(-32);entry.view.setFilters(types,only);schedule();};
  for(const typ of ['FUNC','PRELOAD','SYNC','EVENT']){const label=el('label',undefined,'legend-'+typ.toLowerCase()),box=el('input');box.type='checkbox';box.checked=!state.types.includes(typ);box.setAttribute('aria-label','内部显示 '+typ);box.onchange=()=>{const types=new Set(state.types);box.checked?types.delete(typ):types.add(typ);apply([...types],state.only);};label.append(box,document.createTextNode(typ));body.append(label);}
  const only=el('label'),check=el('input');check.type='checkbox';check.checked=state.only;check.onchange=()=>apply(state.types,check.checked);only.append(check,document.createTextNode('仅看关联'));host.append(only);
  if(hidden){const show=el('button','显示隐藏关联 · '+hidden);show.onclick=()=>{entry.view.revealRelated();schedule();};host.append(show);}
  const batch=el('button','应用到全部已展开 SK');batch.onclick=()=>{history=[...history,captureContext()].slice(-32);for(const [k,p] of skPanels)if(openSKs.has(k)){if(p.view)p.view.setFilters(state.types,state.only);else p.pendingFilters={types:[...state.types],only:state.only};}schedule();};body.append(batch);
  const labels=entry.legend;labels.replaceChildren();for(const typ of ['FUNC','PRELOAD','SYNC','EVENT']){const item=el('span',undefined,'task-legend-item');item.append(el('i',undefined,'type-key type-'+typ.toLowerCase()),document.createTextNode(typ));labels.append(item);}for(const [kind,text] of [['flow','SET/WAIT'],['all','ALL']]){const item=el('span',undefined,'task-legend-item');item.append(el('i',undefined,'line-key line-'+kind),document.createTextNode(text));labels.append(item);}
 }
}
function positionSKTools(){for(const p of skPanels.values())p.root.classList.toggle('controls-small',scale<.25);}
let optionsModel=null;
function renderGraphOptions(){if(optionsModel===model)return;optionsModel=model;const host=$('graph-options-body'),options=model?.graph_options||{};host.replaceChildren();
 const heading=el('div',undefined,'options-heading');heading.append(el('strong',modelName()+' · Options'),el('span','采集记录 · 只读','muted'));const close=el('button','收起配置','toolbar-btn');close.onclick=()=>{$('graph-options').open=false;};heading.append(close);host.append(heading);
 const search=el('input');search.type='search';search.placeholder='搜索 Option 名称或记录值';search.setAttribute('aria-label','搜索图配置');host.append(search);
 const keys=[...new Set([...Object.keys(options.before||{}),...Object.keys(options.after||{})])].sort();const table=el('table'),head=el('thead'),titles=el('tr'),body=el('tbody');for(const name of ['Option','更新前','更新后','来源'])titles.append(el('th',name));head.append(titles);table.append(head,body);
 for(const key of keys){const before=options.before?.[key],after=options.after?.[key],row=el('tr');row.dataset.option=key;row.append(el('td',key));for(const record of [before,after]){const cell=el('td');cell.append(el('code',record?JSON.stringify(record.value):'未采集'));row.append(cell);}if(before&&after&&JSON.stringify(before.value)!==JSON.stringify(after.value))row.classList.add('option-changed');const evidence=el('td'),d=el('details');d.append(el('summary',[...new Set([before?.parser,after?.parser].filter(Boolean))].join(' / ')||'未采集'),el('pre',JSON.stringify({before:before?.source,after:after?.source},null,2)));evidence.append(d);row.append(evidence);body.append(row);}
 if(!keys.length)host.append(el('p','当前 Model 未采集图级配置。'));else{const wrap=el('div',undefined,'options-table');wrap.append(table);host.append(wrap);}const count=el('span',keys.length+' 项记录','muted');host.append(count);search.oninput=()=>{const q=search.value.toLowerCase();let shown=0;for(const row of table.querySelectorAll('[data-option]')){row.hidden=!row.textContent.toLowerCase().includes(q);if(!row.hidden)shown++;}count.textContent=shown+' / '+keys.length+' 项记录';};$('graph-options-summary').textContent='Options · '+keys.length;}
