const $ = id => document.getElementById(id);
let session = crypto.randomUUID(), count = 0;

function updateEmptyState() {
  const container = $('messages');
  const emptyState = $('empty-state');
  if (container && emptyState) {
    if (container.children.length > 0) {
      emptyState.style.display = 'none';
    } else {
      emptyState.style.display = 'flex';
    }
  }
}

function message(role, text) {
  const container = $('messages');
  const article = document.createElement('article');
  article.className = `msg-card msg-${role}`;
  
  const header = document.createElement('header');
  header.className = 'msg-header';
  
  const roleLabel = document.createElement('span');
  roleLabel.className = `msg-role-badge badge-${role}`;
  roleLabel.innerHTML = role === 'user' 
    ? '<svg viewBox="0 0 16 16" fill="currentColor" width="12" height="12"><path d="M8 8a3 3 0 100-6 3 3 0 000 6zm2 1.4c1.8 1.1 3 3 3 5.6H3c0-2.6 1.2-4.5 3-5.6 1.2.7 2.8.7 4 0z"/></svg> OPERATOR' 
    : '<svg viewBox="0 0 16 16" fill="currentColor" width="12" height="12"><path d="M4 3a2 2 0 00-2 2v6a2 2 0 002 2h8a2 2 0 002-2V5a2 2 0 00-2-2H4zm8 3a1 1 0 110 2 1 1 0 010-2zM4 6a1 1 0 110 2 1 1 0 010-2zm2 5h4v1H6v-1z"/></svg> STOCKPILOT';
  
  const time = document.createElement('time');
  time.className = 'msg-time';
  time.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  
  header.append(roleLabel, time);
  
  const content = document.createElement('div');
  content.className = 'msg-body';
  content.textContent = text;
  
  article.append(header, content);
  container.append(article);
  updateEmptyState();
  article.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

function renderStatus(status, stopReason, steps) {
  const pill = $('status-pill');
  const statusText = $('status-text');
  const statusDetails = $('status-details');

  let pillClass = 'status-idle';
  let label = status ? status.toUpperCase() : 'IDLE';

  if (status === 'Running…' || status === 'running') {
    pillClass = 'status-running';
    label = 'RUNNING…';
  } else if (status === 'completed' || status === 'goal_completed') {
    pillClass = 'status-completed';
    label = '✓ COMPLETED';
  } else if (status === 'blocked' || stopReason === 'action_blocked') {
    pillClass = 'status-blocked';
    label = '⚠ BLOCKED';
  } else if (status === 'failed' || status === 'contract_error') {
    pillClass = 'status-failed';
    label = '✕ FAILED';
  }

  if (pill) pill.className = `status-pill ${pillClass}`;
  if (statusText) statusText.textContent = label;
  if (statusDetails) {
    if (steps !== undefined) {
      statusDetails.textContent = `Stop Reason: ${stopReason || 'N/A'} | Steps: ${steps}`;
    } else {
      statusDetails.textContent = stopReason || 'System ready';
    }
  }
}

function renderTimeline(toolCalls) {
  const timelineView = $('timeline-view');
  const tracePre = $('trace');
  if (tracePre) tracePre.textContent = JSON.stringify(toolCalls, null, 2);
  
  if (!timelineView) return;
  timelineView.innerHTML = '';
  
  if (!Array.isArray(toolCalls) || toolCalls.length === 0) {
    timelineView.innerHTML = '<div class="timeline-empty">No tool executions recorded yet.</div>';
    return;
  }
  
  toolCalls.forEach((tc, idx) => {
    const item = document.createElement('div');
    item.className = 'timeline-item';
    
    const action = tc.action || tc.name || tc.tool || 'unknown_tool';
    const step = tc.step !== undefined ? `Step ${tc.step}` : `#${idx + 1}`;
    const args = tc.arguments ? JSON.stringify(tc.arguments, null, 1) : '{}';
    
    item.innerHTML = `
      <div class="timeline-step-badge">${step}</div>
      <div class="timeline-content">
        <div class="timeline-tool-title">
          <span class="tool-name-tag">🔧 ${action}</span>
        </div>
        <div class="timeline-args-box">
          <span class="args-label">ARGS:</span> <code>${args}</code>
        </div>
      </div>
    `;
    timelineView.append(item);
  });
}

function renderObservations(events) {
  const obsFormatted = $('obs-formatted');
  const obsPre = $('observations');
  if (obsPre) obsPre.textContent = JSON.stringify(events, null, 2);
  
  if (!obsFormatted) return;
  obsFormatted.innerHTML = '';
  
  if (!Array.isArray(events) || events.length === 0) {
    obsFormatted.innerHTML = '<div class="obs-empty">No observations recorded yet.</div>';
    return;
  }
  
  events.forEach((ev, idx) => {
    const card = document.createElement('div');
    card.className = 'obs-event-card';
    
    const evType = ev.type || ev.event || 'observation';
    const evText = typeof ev === 'string' ? ev : JSON.stringify(ev, null, 1);
    
    card.innerHTML = `
      <div class="obs-event-header">
        <span class="obs-tag">EVT #${idx + 1}</span>
        <span class="obs-type">${evType}</span>
      </div>
      <pre class="obs-code">${evText}</pre>
    `;
    obsFormatted.append(card);
  });
}

async function checkHealth() {
  const statusPill = $('system-status-pill');
  if (!statusPill) return;
  
  statusPill.className = 'system-status-pill waking';
  statusPill.innerHTML = '<span class="status-dot waking-dot"></span> WAKING UP';

  try {
    const r = await fetch('/health');
    if (r.ok) {
      const data = await r.json();
      if (data.status === 'ok') {
        statusPill.className = 'system-status-pill online';
        statusPill.innerHTML = '<span class="status-dot"></span> SYSTEM ONLINE';
        return;
      }
    }
    statusPill.className = 'system-status-pill offline';
    statusPill.innerHTML = '<span class="status-dot offline-dot"></span> SYSTEM OFFLINE';
  } catch (e) {
    statusPill.className = 'system-status-pill offline';
    statusPill.innerHTML = '<span class="status-dot offline-dot"></span> SYSTEM OFFLINE';
  }
}

async function init() {
  await checkHealth();
  try {
    const r = await fetch('/models');
    if (!r.ok) throw Error('Models unavailable');
    const data = await r.json();
    const modelSelect = $('model');
    modelSelect.innerHTML = '';
    for (const m of data.models) {
      const o = document.createElement('option');
      o.value = m;
      o.textContent = m;
      modelSelect.append(o);
    }
  } catch (e) {
    renderStatus('failed', e.message, 0);
    $('send').disabled = true;
  }
  updateEmptyState();
}

// Prompt Chip Click Handlers
document.addEventListener('click', e => {
  const chip = e.target.closest('.chip');
  if (chip) {
    const prompt = chip.getAttribute('data-prompt');
    if (prompt) {
      const taskInput = $('task');
      taskInput.value = prompt;
      taskInput.focus();
    }
  }
});

// JSON Raw Toggles
$('toggle-raw-trace')?.addEventListener('click', function() {
  const pre = $('trace');
  const view = $('timeline-view');
  if (pre && view) {
    pre.classList.toggle('hidden');
    view.classList.toggle('hidden');
    this.textContent = pre.classList.contains('hidden') ? 'View Raw' : 'View Visual';
  }
});

$('toggle-raw-obs')?.addEventListener('click', function() {
  const pre = $('observations');
  const view = $('obs-formatted');
  if (pre && view) {
    pre.classList.toggle('hidden');
    view.classList.toggle('hidden');
    this.textContent = pre.classList.contains('hidden') ? 'View Raw' : 'View Visual';
  }
});

$('chat').onsubmit = async e => {
  e.preventDefault();
  const task = $('task').value.trim();
  if (!task) return;
  
  const note = $('external').value.trim();
  message('user', task);
  
  $('send').disabled = true;
  $('reset').disabled = true;
  
  renderStatus('Running…', 'Executing task loop', 0);
  renderTimeline([{ step: '...', action: 'thinking', arguments: { task } }]);
  renderObservations(['Task received. Initializing agent reasoning...']);

  try {
    const r = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: session,
        task,
        model: $('model').value,
        external_context: note ? [{ source: 'browser-note', content: note, trust: 'untrusted' }] : []
      })
    });
    
    const data = await r.json();
    if (!r.ok) throw Error(JSON.stringify(data.detail || data));
    
    message('agent', data.final_response);
    count += 2;
    $('memory').textContent = `${count} messages in session`;
    
    renderStatus(data.status, data.stop_reason, data.steps);
    renderTimeline(data.tool_calls || []);
    renderObservations(data.events || []);
    
    $('task').value = '';
  } catch (e) {
    renderStatus('failed', `Error: ${e.message}`, 0);
    message('agent', `⚠️ System Error: ${e.message}`);
  } finally {
    $('send').disabled = false;
    $('reset').disabled = false;
  }
};

$('reset').onclick = async () => {
  try {
    const r = await fetch('/chat/' + session, { method: 'DELETE' });
    if (!r.ok) throw Error('Reset failed');
    
    session = crypto.randomUUID();
    count = 0;
    
    $('messages').replaceChildren();
    updateEmptyState();
    
    $('memory').textContent = '0 messages in session';
    $('external').value = '';
    $('task').value = '';
    
    renderStatus('idle', 'System ready', 0);
    renderTimeline([]);
    renderObservations([]);
  } catch (e) {
    renderStatus('failed', e.message, 0);
  }
};

init();
