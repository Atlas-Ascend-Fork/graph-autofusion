/* DFX_VIEW */

function dfxDiagnosticLabel(code){
 const labels={graph_option_mismatch:'图配置来源不一致',graph_options_fallback:'图配置使用 Log 记录',invalid_graph_options:'图配置字段异常',exploration_sync_unresolved:'内部同步证据不足',ambiguous_synchronization:'同步依赖存在歧义',count_mismatch:'数量不一致',dangling_pointer:'节点引用缺失',duplicate_node_id:'节点 ID 重复',inconsistent_pointer:'前后节点引用不一致',invalid_node:'节点数据无效',invalid_queue:'队列数据无效',invalid_task_info:'Task 信息无效',model_identity_mismatch:'模型标识不一致',unbound_queue:'队列未能关联',unsupported_memory_predicate:'内存等待条件暂不支持',log_after_partial:'Log 更新后数据不完整',log_before_coverage:'Log 更新前覆盖受限',log_instance_partition:'Log Model 分区异常',log_runtime_coverage:'运行证据覆盖不足',log_scope_members_missing:'SK 成员数据缺失',log_fusion_analysis_coverage:'融合断图证据覆盖不足',capability_unavailable:'分析能力不可用',parser_mismatch:'JSON 与 Log 不一致',partial_coverage:'数据覆盖不完整',single_side_objects:'仅单侧来源存在的对象',source_fallback:'来源降级',missing_source:'缺少来源文件',unreadable_source:'来源文件不可读'};
 return labels[code]||(/^log_event_ambiguous/.test(code)?'Event 依赖存在歧义':/^log_memory_ambiguous/.test(code)?'内存依赖存在歧义':code||'诊断信息');
}

function dfxDisplayMessage(value){return String(value||'').replace(/\bScope\b/g,'SK').replace(/实例/g,'Model');}

function dfxIssueElement(issue){
 const article=el('article',undefined,'issue'),severity=issue?.severity||'info';
 article.append(el('span',dfxStateLabel(severity),'status '+severity),el('h3',dfxDisplayMessage(issue?.message)||issue?.code||'诊断信息'));
 if(issue?.code)article.append(el('p',issue.code,'metadata'));
 if(issue?.impact)article.append(el('p','影响 · '+dfxDisplayMessage(issue.impact)));
 if(issue?.next_step)article.append(el('p','建议 · '+dfxDisplayMessage(issue.next_step)));
 if(issue?.source)article.append(el('p',sourceText(issue.source),'source'));
 if(issue?.object&&(issue.object.node_id!=null||issue.object.scope_id!=null)){
  const button=el('button','在图中定位 ↗','object-link');button.onclick=()=>jump(issue.object);article.append(button);
 }
 if(issue?.details&&Object.keys(issue.details).length)detail(article,'证据',issue.details);
 return article;
}

function dfxObjectLink(host,label,target){
 if(!target)return;
 const id=target.node_id??target.scope_id;if(id==null)return;
 const button=el('button',label,'object-link');
 button.onclick=()=>jump({...(target.node_id!=null?{node_id:String(target.node_id)}:{scope_id:String(target.scope_id)}),stage:target.stage==='after'?'after':'before'},true);
 host.append(button);
}

function dfxNumber(value){const n=Number(value);return Number.isFinite(n)?n.toLocaleString():'—';}

function dfxReasonLabel(code){
 const labels={DEADLOCK_DETECTED:'为避免死锁而拆分融合',UNFUSIBLE_NODE:'算子导致融合断开',SYNCALL_OP_DROP:'同步信号导致融合断开',OP_UNSUPPORT:'算子不支持当前融合条件',BINDMAP_INIT_EMPTY:'算子不支持静态编译，或静态编译失败',RESOURCE_INSUFFICIENT:'资源不足，未能完成融合',SCOPE_FUSE_PART:'SK 只完成部分融合',UNKNOWN:'其他已记录原因'};
 const text=String(code||'').trim();return labels[text]||(/^[A-Z0-9_]+$/.test(text)?labels.UNKNOWN:text||labels.UNKNOWN);
}

function dfxUnknownFailureReason(value){const text=String(value||'').trim();return !text||text==='UNKNOWN'||/未确认|未标明|unknown/i.test(text);}

function dfxStateLabel(state){return ({available:'可用',ready:'可用',ok:'可用',partial:'部分可用',complete:'完整',unavailable:'不可用',abnormal:'异常',clean:'未发现明确异常',insufficient:'证据不足',attention:'需处理',confirmed:'已确认',info:'已确认',error:'ERROR',warning:'Warning',normal:'正常'})[state]||state||'未知';}
function dfxStatusClass(state){return state==='abnormal'||state==='unavailable'?'error':state==='partial'||state==='insufficient'||state==='attention'?'warning':'info';}
function dfxResultLabel(kind,state){if(kind==='Graph'&&state==='partial')return '已生成';if(kind==='解析'&&state==='partial')return '部分解析';if(kind==='解析'&&(state==='complete'||state==='ready'))return '解析完成';if(kind==='解析'&&state==='unavailable')return '无法解析';if(kind==='融合'&&state==='partial')return '已有结果';if(kind==='分析'&&state==='partial')return '已整理';if(kind==='融合'&&state==='complete')return '已解析';return dfxStateLabel(state);}
function dfxResultClass(kind,state){return (kind==='Graph'||kind==='融合'||kind==='分析')&&state==='partial'?'info':dfxStatusClass(state);}

function dfxMetric(label,value,note){
 const item=el('div',undefined,'dfx-metric');item.append(el('span',label,'dfx-metric-label'),el('strong',dfxNumber(value),'dfx-metric-value'));if(note)item.append(el('small',note,'dfx-metric-note'));return item;
}

function dfxFusionRing(done,total,remaining){
 const wrap=el('div',undefined,'dfx-ring-layout'),ring=el('div',undefined,'dfx-ring'),copy=el('div',undefined,'dfx-ring-copy'),d=Number(done),t=Number(total),valid=Number.isFinite(d)&&Number.isFinite(t)&&t>0,ratio=valid?Math.max(0,Math.min(100,d/t*100)):0;
 ring.style.setProperty('--dfx-ratio',ratio+'%');ring.setAttribute('role','img');ring.setAttribute('aria-label',valid?'融合完成度 '+ratio.toFixed(1)+'%':'融合完成度不可用');
 ring.append(el('strong',valid?ratio.toFixed(1)+'%':'—','dfx-ring-value'));copy.append(el('strong',valid?dfxNumber(d)+' / '+dfxNumber(t):'无法计算','dfx-ring-number'),el('span',valid?'算子已进入融合 SK':'当前无完整算子统计'));if(remaining!=null)copy.append(el('span','未融合 '+dfxNumber(remaining)+' 个算子','dfx-ring-remaining'));wrap.append(ring,copy);return wrap;
}

function dfxFusionReasonKey(value){
 const text=String(value||'').trim(),aliases={'算子导致融合断开':'UNFUSIBLE_NODE','同步信号导致融合断开':'SYNCALL_OP_DROP','算子不支持当前融合条件':'OP_UNSUPPORT','算子不支持静态编译，或静态编译失败':'BINDMAP_INIT_EMPTY','资源不足，未能完成融合':'RESOURCE_INSUFFICIENT','SK 只完成部分融合':'SCOPE_FUSE_PART'};
 return aliases[text]||text||'UNKNOWN';
}

function dfxFusionReasonBreakdown(fusion){
 const counts=new Map(),add=(value,count)=>{const key=dfxFusionReasonKey(value),amount=Number(count);if(!Number.isFinite(amount)||amount<=0)return;counts.set(key,(counts.get(key)||0)+amount);};
 const failures=Array.isArray(fusion?.node_failures)?fusion.node_failures:[];for(const failure of failures)add(failure.reason||failure.reason_code,1);let source=failures.length?'failures':'none';
 if(!counts.size){for(const item of Array.isArray(fusion?.operator_impacts)?fusion.operator_impacts:[])for(const [reason,count] of Object.entries(item?.direct_breaks?.by_reason||{}))add(reason,count);if(counts.size)source='boundaries';}
 return{source,items:[...counts.entries()].map(([code,count])=>({code,label:dfxReasonLabel(code),count})).sort((a,b)=>b.count-a.count)};
}

function dfxFusionReasonPanel(fusion){
 const data=dfxFusionReasonBreakdown(fusion),panel=el('section',undefined,'dfx-fusion-panel'),title=data.source==='boundaries'?'融合断开原因':'未融合原因分布';panel.append(el('h3',title));
 if(!data.items.length){panel.append(el('p','当前没有可展示的原因记录。','muted'));return panel;}
 const total=data.items.reduce((sum,item)=>sum+item.count,0),rows=el('div',undefined,'dfx-reason-bars');
 for(const item of data.items){const row=el('div',undefined,'dfx-reason-row'),bar=el('div',undefined,'dfx-reason-bar'),fill=el('i');fill.style.width=(total?Math.max(2,item.count/total*100):0)+'%';bar.append(fill);row.append(el('span',item.label),bar,el('strong',dfxNumber(item.count)));rows.append(row);}
 panel.append(rows);return panel;
}

function dfxStatusItem(label,state,note){
 const item=el('div',undefined,'dfx-status-item');item.append(el('span',label,'dfx-status-label'),el('strong',dfxResultLabel(label,state),'status '+dfxResultClass(label,state)));if(note)item.append(el('p',note,'muted'));return item;
}

function dfxObjectLinks(host,objects){
 for(const target of (objects||[]).slice(0,5)){if(target.kind==='scope'&&target.scope_id!=null)dfxObjectLink(host,'定位 SK '+target.scope_id,target);else if(target.kind==='node'&&target.node_id!=null)dfxObjectLink(host,'定位节点 #'+target.node_id,target);}
}

function dfxFindingItem(finding){
 const article=el('article',undefined,'dfx-finding'),head=el('div',undefined,'dfx-finding-head'),severity=finding.severity||(finding.state==='attention'?'warning':'info');
 head.append(el('span',dfxStateLabel(severity),'status '+dfxStatusClass(severity)),el('h3',finding.title||'分析结果'));const count=Number(finding.count)||0;if(finding.show_count!==false&&(count>1||finding.count_label))head.append(el('span',dfxNumber(count)+' '+(finding.count_label||'项'),'dfx-finding-count'));article.append(head);
 if(finding.summary)article.append(el('p',finding.summary));if(finding.location)article.append(el('p','定位 · '+finding.location,'metadata'));if(finding.impact)article.append(el('p','影响 · '+finding.impact,'muted'));if(finding.next_step)article.append(el('p','处理 · '+finding.next_step,'dfx-action'));
 if(Array.isArray(finding.reason_breakdown)&&finding.reason_breakdown.length){const reasons=el('dl',undefined,'dfx-reason-list');for(const item of finding.reason_breakdown.slice(0,3))reasons.append(el('dt',item.label||'其他已记录原因'),el('dd',dfxNumber(item.count)));article.append(reasons);}
 const links=el('div',undefined,'dfx-links');dfxObjectLinks(links,finding.objects);if(links.childNodes.length)article.append(links);return article;
}

function dfxImpactTargetKey(target){return target.kind+':'+(target.stage==='after'?'after':'before')+':'+String(target.node_id??target.scope_id);}

function dfxImpactTargets(item){
 const targets=[],seen=new Set(),add=(target)=>{if(!target||!['scope','node'].includes(target.kind))return;const id=target.node_id??target.scope_id;if(id==null)return;const normalized={kind:target.kind,stage:target.stage==='after'?'after':'before',...(target.kind==='node'?{node_id:String(id)}:{scope_id:String(id)})},key=dfxImpactTargetKey(normalized);if(seen.has(key))return;seen.add(key);targets.push(normalized);};
 for(const target of Array.isArray(item?.targets)?item.targets:[])add(target);
 for(const id of Array.isArray(item?.observed_effect?.affected_scope_ids)?item.observed_effect.affected_scope_ids:[])add({kind:'scope',stage:'before',scope_id:id});
 for(const id of Array.isArray(item?.direct_breaks?.trigger_node_ids)?item.direct_breaks.trigger_node_ids:[])add({kind:'node',stage:'before',node_id:id});
 return targets;
}

function dfxImpactRows(items){
 const list=Array.isArray(items)?items:[],rows=[];
 list.forEach((item,index)=>{
  const operator=item?.operator||{},direct=item?.direct_breaks||{},observed=item?.observed_effect||{},evidence=item?.evidence||{},targets=dfxImpactTargets(item),reasons=[];
  for(const [code,value] of Object.entries(direct.by_reason||{})){const count=Number(value);if(!Number.isFinite(count)||count<=0)continue;reasons.push({code,label:dfxReasonLabel(code),count,kind:String(code)==='SYNCALL_OP_DROP'?'sync':'operator'});}
  const breakSum=reasons.reduce((sum,reason)=>sum+reason.count,0),rawBreaks=Number(direct.total),rawScopes=Number(observed.affected_scope_count??observed.affected_region_count),rawFailures=Number(evidence.failure_count),scopeCount=Number.isFinite(rawScopes)?rawScopes:new Set(targets.filter(target=>target.kind==='scope').map(target=>target.scope_id)).size;
  rows.push({key:(operator.normalized_name||'未命名算子')+'|'+index,name:operator.normalized_name||'未命名算子',breaks:Number.isFinite(rawBreaks)?rawBreaks:breakSum,scopes:scopeCount,failures:Number.isFinite(rawFailures)?rawFailures:0,reasons,targets,breakDetails:item.break_details||[],breakDetailCount:item.break_detail_count||0});
 });
 return rows.sort((a,b)=>b.breaks-a.breaks||b.failures-a.failures||a.name.localeCompare(b.name));
}

function dfxImpactReasonMatches(row,filter){
 if(filter==='all')return true;if(filter==='evidence')return row.breaks===0&&row.failures>0;return row.reasons.some(reason=>reason.kind===filter);
}

function dfxImpactTargetLabel(target){return target.kind==='scope'?'定位 SK '+target.scope_id:'定位节点 #'+target.node_id;}

function dfxImpactDetailRow(row){
 const detailRow=el('tr',undefined,'dfx-impact-detail-row'),cell=el('td');cell.colSpan=6;
 if(row.breakDetails.length){cell.append(el('strong','拆分位置 · '+row.breakDetails.length+' / '+row.breakDetailCount,'dfx-impact-detail-title'));const list=el('ul',undefined,'dfx-impact-explanations');for(const item of row.breakDetails){const entry=el('li'),links=el('div',undefined,'dfx-links');entry.append(el('p',item.explanation));for(const target of item.targets||[])dfxObjectLink(links,dfxImpactTargetLabel(target),target);entry.append(links);list.append(entry);}cell.append(list);}
 cell.append(el('strong','关联位置','dfx-impact-detail-title'));
 const links=el('div',undefined,'dfx-links');for(const target of row.targets)dfxObjectLink(links,dfxImpactTargetLabel(target),target);
 if(!row.targets.length)links.append(el('span','没有可确认的定位对象。','muted'));cell.append(links);detailRow.append(cell);return detailRow;
}

function dfxFusionImpactTable(items){
 const panel=el('section',undefined,'dfx-fusion-panel dfx-impact-table');panel.setAttribute('aria-labelledby','dfx-impact-title');
 const heading=el('div',undefined,'dfx-table-heading'),title=el('h3','影响算子');title.id='dfx-impact-title';const count=el('span',undefined,'dfx-table-count');heading.append(title,count);
 const controls=el('div',undefined,'dfx-table-controls'),search=el('input'),reason=el('select');search.type='search';search.placeholder='搜索算子名称';search.setAttribute('aria-label','搜索算子名称');reason.setAttribute('aria-label','原因筛选');
 for(const [value,label] of [['all','全部原因'],['operator','算子导致融合断开'],['sync','同步信号导致融合断开'],['evidence','仅有未融合记录']]){const option=el('option',label);option.value=value;reason.append(option);}controls.append(search,reason);
 const scroll=el('div',undefined,'dfx-table-scroll'),table=el('table',undefined,'dfx-impact-table-grid'),thead=el('thead'),headRow=el('tr'),tbody=el('tbody');
 const columns=[['name','算子'],['breaks','已确认断开'],['scopes','涉及 SK'],['failures','未融合记录'],['reasons','断开原因'],['actions','定位']],rows=dfxImpactRows(items);let sortKey='breaks',sortDirection=-1,expandedKey=null;
 const renderHeader=()=>{headRow.replaceChildren();for(const [key,label] of columns){const th=el('th');if(['name','breaks','scopes','failures'].includes(key)){const button=el('button',label,'dfx-sort-button');button.type='button';button.dataset.key=key;const active=sortKey===key;button.dataset.dir=active?(sortDirection<0?'desc':'asc'):'none';button.setAttribute('aria-sort',active?(sortDirection<0?'descending':'ascending'):'none');button.onclick=()=>{if(sortKey===key)sortDirection*=-1;else{sortKey=key;sortDirection=key==='name'?1:-1;}renderHeader();renderRows();};th.append(button);}else th.append(el('span',label));headRow.append(th);}};
 const renderRows=()=>{
  const query=search.value.trim().toLocaleLowerCase(),filter=reason.value,visible=rows.filter(row=>(!query||row.name.toLocaleLowerCase().includes(query))&&dfxImpactReasonMatches(row,filter)).sort((a,b)=>{if(sortKey==='name')return sortDirection*a.name.localeCompare(b.name);return sortDirection*((Number(a[sortKey])||0)-(Number(b[sortKey])||0));});
  count.textContent=dfxNumber(visible.length)+' / '+dfxNumber(rows.length)+' 个算子';tbody.replaceChildren();
  if(!visible.length){const empty=el('tr',undefined,'dfx-impact-empty'),cell=el('td','没有匹配的算子');cell.colSpan=columns.length;empty.append(cell);tbody.append(empty);return;}
  for(const row of visible){
   const tableRow=el('tr',undefined,'dfx-impact-row');tableRow.tabIndex=0;tableRow.dataset.rowKey=row.key;tableRow.title='双击查看断开原因与关联位置';
   const nameCell=el('td',undefined,'dfx-impact-name');nameCell.append(el('strong',row.name));
   const numberCell=(value)=>el('td',dfxNumber(value),'dfx-impact-number');
   const reasonCell=el('td'),reasonList=el('div',undefined,'dfx-impact-reasons');
   if(row.reasons.length)for(const item of row.reasons){const chip=el('span',item.label+' · '+dfxNumber(item.count),'dfx-reason-chip '+(item.kind==='sync'?'dfx-reason-chip-sync':'dfx-reason-chip-operator'));reasonList.append(chip);}else reasonList.append(el('span','未记录明确断开类型','muted'));reasonCell.append(reasonList);
   const actionCell=el('td'),actions=el('div',undefined,'dfx-impact-actions'),primary=row.targets.find(target=>target.kind==='scope')||row.targets[0];
   if(primary)dfxObjectLink(actions,'定位',primary);else actions.append(el('span','—','muted'));
   if(row.targets.length>1||row.breakDetails.length){const expand=el('button','查看','dfx-impact-expand');expand.type='button';expand.title='查看断开原因与关联位置';expand.onclick=(event)=>{event.stopPropagation();toggleDetails(row,tableRow,detailRow);};actions.append(expand);}
   actionCell.append(actions);tableRow.append(nameCell,numberCell(row.breaks),numberCell(row.scopes),numberCell(row.failures),reasonCell,actionCell);
   const detailRow=dfxImpactDetailRow(row),toggle=()=>{if(expandedKey===row.key){expandedKey=null;detailRow.remove();tableRow.classList.remove('is-expanded');}else{expandedKey=row.key;for(const sibling of tbody.querySelectorAll('.dfx-impact-detail-row'))sibling.remove();tbody.insertBefore(detailRow,tableRow.nextSibling);tbody.querySelectorAll('.dfx-impact-row.is-expanded').forEach(sibling=>sibling.classList.remove('is-expanded'));tableRow.classList.add('is-expanded');}};
   function toggleDetails(_row,_tableRow,_detailRow){toggle();}
   tableRow.ondblclick=toggle;tableRow.onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();toggle();}};
   tbody.append(tableRow);if(expandedKey===row.key){tableRow.classList.add('is-expanded');tbody.append(detailRow);}
  }
 };
 search.oninput=renderRows;reason.onchange=renderRows;thead.append(headRow);table.append(thead,tbody);scroll.append(table);panel.append(heading,controls,scroll);renderHeader();renderRows();return panel;
}

function dfxFusionBoundaryItem(boundary){
 const trigger=boundary?.trigger||{},operator=trigger.operator||{},article=el('article',undefined,'dfx-evidence-item'),scope=boundary.scope_id==null?'未关联 SK':'SK '+boundary.scope_id;article.append(el('h3',scope+' · 融合中断'));
 const triggerParts=[];if(trigger.node_id!=null)triggerParts.push('触发节点 #'+trigger.node_id);if(operator.normalized_name)triggerParts.push(operator.normalized_name);if(triggerParts.length)article.append(el('p',triggerParts.join(' · '),'metadata'));
 const adjacent=(boundary.adjacent_scope_ids||[]).map(id=>'SK '+id).join('、')||'无';article.append(el('p','前置 SK '+(boundary.previous_scope_id==null?'—':boundary.previous_scope_id)+' · 后置 SK '+(boundary.following_scope_id==null?'—':boundary.following_scope_id)+' · 相邻 '+adjacent,'muted'));
 const links=el('div',undefined,'dfx-links');dfxObjectLinks(links,boundary.targets);if(trigger.node_id!=null)dfxObjectLink(links,'定位触发节点 #'+trigger.node_id,{node_id:String(trigger.node_id),stage:'before'});if(links.childNodes.length)article.append(links);return article;
}

function dfxFusionFailureItem(failure){
 const op=failure?.operator||{},article=el('article',undefined,'dfx-evidence-item');article.append(el('h3',(op.normalized_name||'未命名算子')+' · 未融合节点 #'+(failure.node_id??'—')));if(!dfxUnknownFailureReason(failure.reason))article.append(el('p',failure.reason));if(failure.detail)article.append(el('p',failure.detail,'muted'));
 const links=el('div',undefined,'dfx-links');dfxObjectLinks(links,failure.targets);if(failure.node_id!=null)dfxObjectLink(links,'定位节点 #'+failure.node_id,{node_id:String(failure.node_id),stage:'before'});if(links.childNodes.length)article.append(links);return article;
}

function dfxEvidenceDetails(title,items,renderItem){
 const list=Array.isArray(items)?items:[],d=el('details',undefined,'dfx-evidence'),body=el('div',undefined,'dfx-evidence-body');d.append(el('summary',title),body);let offset=0;const more=el('button','加载更多','toolbar-btn');
 const append=()=>{for(const item of list.slice(offset,offset+30)){const rendered=renderItem(item);if(rendered)body.insertBefore(rendered,more);}offset+=30;more.hidden=offset>=list.length;more.textContent='加载更多 · '+Math.max(0,list.length-offset)+' 项剩余';};
 d.addEventListener('toggle',()=>{if(!d.open||d.dataset.loaded)return;d.dataset.loaded='true';body.append(more);if(list.length)append();else{body.insertBefore(el('p','没有可展示记录。','muted'),more);more.hidden=true;}});return d;
}

function dfxDebugCard(dfx){
 const debug=dfx?.debug;if(!debug)return null;const card=el('section',undefined,'dfx-card dfx-debug-card');card.append(el('span','Debug profile','eyebrow'),el('h2','开发排查信息'));const diagnostics=Array.isArray(debug.diagnostics)?debug.diagnostics:[];card.append(el('p','此产物由显式调试选项生成，包含原始解析证据；用户版不会携带这些内容。','muted'));
 if(diagnostics.length){const groups=new Map();for(const item of diagnostics){const key=(item.code||'其他').replace(/_0x[0-9a-f]+$/i,'');if(!groups.has(key))groups.set(key,[]);groups.get(key).push(item);}for(const [code,items] of groups){const d=el('details');d.append(el('summary',dfxDiagnosticLabel(code)+' · '+items.length+' 条记录'));let offset=0;const more=el('button','加载更多','toolbar-btn');const append=()=>{for(const item of items.slice(offset,offset+30))d.insertBefore(dfxIssueElement(item),more);offset+=30;more.hidden=offset>=items.length;more.textContent='加载更多 · '+Math.max(0,items.length-offset)+' 项剩余';};d.addEventListener('toggle',()=>{if(!d.open||d.dataset.loaded)return;d.dataset.loaded='true';d.append(more);append();});card.append(d);}}
 if(debug.comparison&&Object.keys(debug.comparison).length)detail(card,'解析器交叉校验证据',debug.comparison);if(debug.runtime?.evidence&&Object.keys(debug.runtime.evidence).length)detail(card,'完整运行证据（原始字段）',debug.runtime.evidence);if(debug.fusion_analysis)detail(card,'融合日志覆盖与原始来源',debug.fusion_analysis.sources||{});return card;
}

function renderFusionDFX(data){
 const dfx=data&&data.profile?data:null,card=el('section',undefined,'dfx-card dfx-fusion-card');card.append(el('span','解析状态','eyebrow'),el('h2','融合概览'));
 if(!dfx){card.append(el('span','ERROR','status error'),el('p','没有找到可解析的融合日志，无法判断哪些节点融合成功。','muted'));return card;}
 const fusion=dfx.fusion||{},summary=fusion.summary||{},metricsData=dfx.metrics||{},actual=Number(metricsData.fused_operator_count),expected=Number(metricsData.expected_operator_count),skCount=Number.isFinite(Number(metricsData.fused_sk_count))?Number(metricsData.fused_sk_count):Number(summary.successful_sk_count),hasProgress=Number.isFinite(actual)&&Number.isFinite(expected)&&expected>0,remaining=hasProgress?Math.max(expected-actual,0):Number.isFinite(Number(metricsData.unfused_operator_count))?Number(metricsData.unfused_operator_count):null;
 const metrics=el('div',undefined,'dfx-metrics');for(const [label,value,note] of [['实际融合算子',hasProgress?actual:null,'已进入 SK'],['期望融合算子',hasProgress?expected:null,'参与融合范围内的算子'],['融合 SK',Number.isFinite(skCount)?skCount:null,'已生成的 SK 数量'],['未融合算子',remaining,'保留为原始节点'],['算子导致融合断开',summary.operator_triggered_boundary_count,'断图边界数量'],['同步信号导致融合断开',summary.control_boundary_count,'断图边界数量']])metrics.append(dfxMetric(label,value,note));
 const lead=hasProgress?'已将 '+dfxNumber(actual)+' / '+dfxNumber(expected)+' 个算子融合到 '+(Number.isFinite(skCount)?dfxNumber(skCount):'—')+' 个 SK；'+(remaining?'其余 '+dfxNumber(remaining)+' 个算子保留为原始节点。':'全部参与融合的算子均已进入 SK。'):Number.isFinite(skCount)?'已形成 '+dfxNumber(skCount)+' 个 SK；当前无法统计算子融合完成度。':'当前没有可展示的融合结果。';
 const analysis=el('div',undefined,'dfx-fusion-analysis');analysis.append(dfxFusionReasonPanel(fusion),dfxFusionImpactTable(fusion.operator_impacts||[]));
 card.append(el('span',dfxResultLabel('解析',fusion.status),'status '+dfxResultClass('解析',fusion.status)),el('p',lead,'dfx-fusion-summary'),dfxFusionRing(actual,expected,remaining),metrics,analysis,dfxEvidenceDetails('融合断开位置 · '+dfxNumber((fusion.boundaries||[]).length)+' 条',fusion.boundaries||[],dfxFusionBoundaryItem),dfxEvidenceDetails('未融合算子 · '+dfxNumber((fusion.node_failures||[]).length)+' 个',fusion.node_failures||[],dfxFusionFailureItem));return card;
}

function renderDFX(){
 const target=$('dfx-content');target.replaceChildren();const dfx=model?.dfx||{},grid=el('div',undefined,'dfx-layout'),hero=el('section',undefined,'dfx-card dfx-hero'),conclusion=dfx.conclusion||{},status=dfx.status||{},state=conclusion.state||'normal';
 hero.append(el('span','当前 Model','eyebrow'),el('div',undefined,'dfx-hero-heading'));const heading=hero.lastChild;heading.append(el('h2',conclusion.title||'结果摘要'),el('span',dfxStateLabel(state),'status '+dfxStatusClass(state)));hero.append(el('p',conclusion.summary||dfx.headline||'当前 Model 没有可展示的结论。','dfx-headline'));
 const errorCount=Number(conclusion.errors)||0,warningCount=Number(conclusion.warnings)||0,counts=el('div',undefined,'dfx-conclusion-counts');counts.append(el('span','ERROR '+dfxNumber(errorCount),errorCount?'status error':'status info'),el('span','Warning '+dfxNumber(warningCount),warningCount?'status warning':'status info'));hero.append(counts);
 const statuses=el('div',undefined,'dfx-status-grid');statuses.append(dfxStatusItem('Graph',status.graph,'执行图与节点信息'),dfxStatusItem('融合',status.fusion,'融合结果'));hero.append(statuses);grid.append(hero);
 const attention=(Array.isArray(dfx.findings)?dfx.findings:[]).filter(item=>item&&(item.state==='attention'||item.severity==='error'||item.severity==='warning'));if(attention.length){const alerts=el('section',undefined,'dfx-card dfx-alerts-card'),alertHeading=el('div',undefined,'dfx-card-heading'),alertList=el('div',undefined,'dfx-findings');alertHeading.append(el('h2','日志诊断'),el('p','仅列出解析过程中记录的 ERROR 或 Warning。','muted'));alerts.append(alertHeading);for(const item of attention.slice(0,12))alertList.append(dfxFindingItem(item));if(attention.length>12)alertList.append(el('p','其余问题已按类别聚合。','muted'));alerts.append(alertList);grid.append(alerts);}
 grid.append(renderFusionDFX(dfx));const debug=dfxDebugCard(dfx);if(debug)grid.append(debug);target.append(grid);const badge=[];if(errorCount)badge.push('ERROR '+dfxNumber(errorCount));if(warningCount)badge.push('Warning '+dfxNumber(warningCount));$('issue-count').textContent=badge.length?' · '+badge.join(' · '):'';
}
