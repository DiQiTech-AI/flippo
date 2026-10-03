let documentState = null;
let session = null;
let currentRun = null;
let isUploading = false;
let isRunning = false;
let isModelsLoading = true;
let modelState = null;
let selectedModelId = (() => {
  try { return localStorage.getItem('doc-rag-model-id') || null; } catch (_) { return null; }
})();

const messages = {
  zh: {
    metaDescription: '基于单份文档进行可验证问答与结构化分析', pageTitle: '翻翻文档 · Flippo', skipLink: '跳到主要内容',
    brandHome: '翻翻文档 · Flippo 首页', brandName: '翻翻文档 · Flippo', languageLabel: '界面语言', healthChecking: '正在检查服务', pleaseWait: '请稍候',
    eyebrow: '可验证的文档智能', heroTitle: '翻翻文档 · Flippo', heroCopy: '上传 PDF 或 Markdown，针对原文提问或生成结构化报告。每条结论都可追溯到具体证据。',
    capabilityNote: '能力说明', traceableEvidence: '证据可追溯', serverValidated: '引用由服务端校验', prepareMaterials: '准备资料', uploadDocument: '上传文档',
    chooseDocument: '选择一份文档', dropDocument: '或将文件拖放到这里', uploadAndIndex: '上传并建立索引', documentInfoHere: '文档信息将在这里显示',
    startAnalysisStep: '开始分析', askDocument: '向文档提问', waitingDocument: '等待文档', analysisMode: '分析方式', preciseQa: '精准问答', structuredReport: '结构化报告', modelLabel: '分析模型', refreshModels: '刷新模型列表',
    messageLabel: '输入基于文档的问题或分析要求', messageBeforeUpload: '请先上传文档，然后输入你想了解的问题', questionExample: '例如：文档中的审批记录需要保留多久？',
    reportExample: '例如：总结文档中的主要结论与行动建议', answerWithEvidence: '答案将附带原文证据', reportHelp: '将整理核心发现、建议与待确认问题', startAnalysis: '开始分析',
    analysisResults: '分析结果', answersConclusions: '回答与结论', resultsHere: '结果会显示在这里', resultsHereDetail: '文档就绪后输入问题，系统将检索原文并给出可验证的回答。',
    sourceBasis: '来源依据', citedEvidence: '引用证据', technicalTrace: '技术运行记录', traceDescription: '以下为本次分析的完整 Trace（JSON），供技术排查和结果审计使用。',
    traceAfterRun: '运行完成后显示', footer: '单文档分析工作台 · 内容仅基于当前上传文档', requestFailed: '请求失败（{status}）',
    pageRange: '第 {start}–{end} 页', pageSingle: '第 {page} 页', sourcePosition: '原文位置 {position}', documentSource: '文档原文',
    uploadedDocument: '已上传文档', indexReady: '索引已就绪 · 可以开始分析', pagesLabel: '页数', charactersLabel: '字符', chunksLabel: '段落', formatLabel: '格式',
    pagesValue: '{count} 页', versionValue: '版本 {id}', documentIncomplete: '文档处理未完成', ocrFallback: '该 PDF 似乎是扫描件，请先进行 OCR 后再上传。', retryFallback: '请检查文件后重试。',
    retrievingDocument: '正在检索文档', generatingReport: '正在生成结构化报告', locatingEvidence: '系统正在定位相关原文并校验证据', analysisIncomplete: '分析未完成', analysisComplete: '已完成分析',
    noContent: '未返回内容', confidenceHigh: '高置信度', confidenceMedium: '中置信度', confidenceLow: '低置信度', supportSupported: '证据充分', supportPartial: '部分支持', supportUnsupported: '证据不足', supportConflicting: '证据冲突', pendingAssessment: '待评估',
    reportTitleFallback: '文档分析报告', findingsCount: '{count} 项发现', noExecutiveSummary: '未提供执行摘要。', coreFindings: '核心发现', findingNumber: '发现 {number}', noFindings: '文档中没有找到足够证据形成结论。', recommendations: '建议事项', openQuestions: '待确认问题',
    searchRounds: '{count} 轮检索', toolCalls: '{count} 次工具调用', evidenceSufficient: '证据充分', searchEnded: '检索已结束',
    viewFullEvidence: '查看完整证据', collapseFullEvidence: '收起完整证据', loadingSource: '正在载入原文…', evidenceNumber: '证据编号', documentVersion: '文档版本', chunkNumber: '段落编号', sourceLocation: '原文位置', fullSource: '完整原文', noSource: '无原文内容', evidenceLoadFailed: '证据载入失败：{message}',
    citationsCount: '{count} 条', sourceEvidence: '原文证据', traceLoadingSummary: '正在载入运行记录…', traceLoading: '正在载入…', traceEvents: '{count} 个事件 · 用于调试与审计', traceLoadFailed: '运行记录载入失败', traceAudit: '用于调试与审计',
    preparingNewDocument: '正在准备新文档', preparingNewDocumentDetail: '索引完成后即可针对这份文档开始新的分析。', demoMode: '演示模式', serviceNormal: '分析服务正常', serviceNeedsConfig: '服务需要配置', serviceConnectionFailed: '服务连接失败',
    demoRuntime: '本地演示分析已就绪', adkRuntime: 'Google ADK 已通过 {provider} 配置', geminiMissing: '缺少 GOOGLE_API_KEY；文档导入和检索仍可使用', unslothMissing: '缺少 UNSLOTH_API_KEY 或 UNSLOTH_BASE_URL；文档导入和检索仍可使用',
    modelsLoading: '正在检查已加载模型…', automaticModel: '自动选择当前模型', automaticResolved: '自动 · 当前为 {model}', automaticNeedsChoice: '自动 · 需要选择模型', modelAutoHelp: '每次运行前都会重新检查，当前将使用 {model}', modelExplicitHelp: '本次运行将使用 {model}；服务端会再次确认它仍已加载', modelReadonlyHelp: '当前服务使用 {model}', modelUnavailable: '先前选择的模型已不可用：{model}。请选择当前已加载模型。', noLoadedModel: 'Unsloth Studio 当前没有已加载的聊天模型。', multipleModels: '当前加载了多个模型，请选择一个具体模型。', modelAuthFailed: 'Unsloth Studio API 密钥验证失败。', modelTimeout: '检查 Unsloth Studio 模型超时。', modelConnectionFailed: '无法连接 Unsloth Studio 以检查模型。', modelStateUnsupported: '服务未标明模型是否已加载，系统不会猜测。', modelDiscoveryFailed: '无法获取已加载模型。', runModel: '模型：{model}',
    parsingIndexing: '正在解析并建立索引', currentStage: '当前阶段：{stage}', stageUploaded: '已上传', stageParsing: '正在解析', stageEmbedding: '正在生成索引', stageIndexing: '正在建立索引', stageReady: '已就绪', stagePreparing: '准备中',
    preparingDocument: '正在准备文档', uploadingDocument: '正在上传文档', documentReady: '文档已就绪', needsOcr: '需要先进行 OCR', documentUnavailable: '文档不可用', documentFailed: '文档处理失败', processingFailed: '处理失败', uploadFailed: '上传失败',
    analyzing: '正在分析', noEnoughEvidence: '未找到充分证据', noAnalysisResult: '未返回分析结果。', analysisErrorFallback: '分析过程中发生错误。',
  },
  en: {
    metaDescription: 'Verifiable Q&A and structured analysis grounded in one document', pageTitle: '翻翻文档 · Flippo', skipLink: 'Skip to main content',
    brandHome: '翻翻文档 · Flippo home', brandName: '翻翻文档 · Flippo', languageLabel: 'Interface language', healthChecking: 'Checking service', pleaseWait: 'Please wait',
    eyebrow: 'Verifiable document intelligence', heroTitle: '翻翻文档 · Flippo', heroCopy: 'Upload a PDF or Markdown file to ask questions or generate a structured report. Every conclusion links back to specific evidence.',
    capabilityNote: 'Capability note', traceableEvidence: 'Traceable evidence', serverValidated: 'Citations are validated by the server', prepareMaterials: 'Prepare source', uploadDocument: 'Upload document',
    chooseDocument: 'Choose a document', dropDocument: 'or drop a file here', uploadAndIndex: 'Upload and build index', documentInfoHere: 'Document information will appear here',
    startAnalysisStep: 'Start analysis', askDocument: 'Ask the document', waitingDocument: 'Waiting for document', analysisMode: 'Analysis mode', preciseQa: 'Focused Q&A', structuredReport: 'Structured report', modelLabel: 'Analysis model', refreshModels: 'Refresh model list',
    messageLabel: 'Enter a question or analysis request based on the document', messageBeforeUpload: 'Upload a document, then enter what you want to know', questionExample: 'Example: How long must approval records be retained?',
    reportExample: 'Example: Summarize the main conclusions and recommended actions', answerWithEvidence: 'Answers will include source evidence', reportHelp: 'Organizes key findings, recommendations, and open questions', startAnalysis: 'Start analysis',
    analysisResults: 'Analysis results', answersConclusions: 'Answer and conclusions', resultsHere: 'Results will appear here', resultsHereDetail: 'Once the document is ready, enter a question to retrieve source text and receive a verifiable answer.',
    sourceBasis: 'Source basis', citedEvidence: 'Cited evidence', technicalTrace: 'Technical run trace', traceDescription: 'Complete JSON trace for this analysis, provided for troubleshooting and auditing.',
    traceAfterRun: 'Shown after a run completes', footer: 'Single-document analysis workspace · Content is based only on the current upload', requestFailed: 'Request failed ({status})',
    pageRange: 'Pages {start}–{end}', pageSingle: 'Page {page}', sourcePosition: 'Source position {position}', documentSource: 'Document source',
    uploadedDocument: 'Uploaded document', indexReady: 'Index ready · You can start analyzing', pagesLabel: 'Pages', charactersLabel: 'Characters', chunksLabel: 'Passages', formatLabel: 'Format',
    pagesValue: '{count} pages', versionValue: 'Version {id}', documentIncomplete: 'Document processing is incomplete', ocrFallback: 'This PDF appears to be scanned. Run OCR and upload it again.', retryFallback: 'Check the file and try again.',
    retrievingDocument: 'Retrieving document', generatingReport: 'Generating structured report', locatingEvidence: 'Locating relevant source text and validating evidence', analysisIncomplete: 'Analysis incomplete', analysisComplete: 'Analysis complete',
    noContent: 'No content returned', confidenceHigh: 'High confidence', confidenceMedium: 'Medium confidence', confidenceLow: 'Low confidence', supportSupported: 'Supported', supportPartial: 'Partially supported', supportUnsupported: 'Unsupported', supportConflicting: 'Conflicting evidence', pendingAssessment: 'Pending assessment',
    reportTitleFallback: 'Document analysis report', findingsCount: '{count} findings', noExecutiveSummary: 'No executive summary provided.', coreFindings: 'Key findings', findingNumber: 'Finding {number}', noFindings: 'The document does not contain enough evidence to form a conclusion.', recommendations: 'Recommendations', openQuestions: 'Open questions',
    searchRounds: '{count} search rounds', toolCalls: '{count} tool calls', evidenceSufficient: 'Evidence sufficient', searchEnded: 'Retrieval ended',
    viewFullEvidence: 'View full evidence', collapseFullEvidence: 'Hide full evidence', loadingSource: 'Loading source text…', evidenceNumber: 'Evidence ID', documentVersion: 'Document version', chunkNumber: 'Passage ID', sourceLocation: 'Source location', fullSource: 'Full source text', noSource: 'No source text', evidenceLoadFailed: 'Could not load evidence: {message}',
    citationsCount: '{count} citations', sourceEvidence: 'Source evidence', traceLoadingSummary: 'Loading run trace…', traceLoading: 'Loading…', traceEvents: '{count} events · For debugging and auditing', traceLoadFailed: 'Could not load run trace', traceAudit: 'For debugging and auditing',
    preparingNewDocument: 'Preparing a new document', preparingNewDocumentDetail: 'You can start a new analysis when indexing finishes.', demoMode: 'Demo mode', serviceNormal: 'Analysis service ready', serviceNeedsConfig: 'Service needs configuration', serviceConnectionFailed: 'Could not connect to service',
    demoRuntime: 'Local demo analysis is ready', adkRuntime: 'Google ADK is configured with {provider}', geminiMissing: 'GOOGLE_API_KEY is missing; document import and retrieval remain available', unslothMissing: 'UNSLOTH_API_KEY or UNSLOTH_BASE_URL is missing; document import and retrieval remain available',
    modelsLoading: 'Checking loaded models…', automaticModel: 'Automatically use current model', automaticResolved: 'Automatic · currently {model}', automaticNeedsChoice: 'Automatic · choose a model', modelAutoHelp: 'The server rechecks before every run; it would currently use {model}', modelExplicitHelp: 'This run will use {model}; the server will verify that it is still loaded', modelReadonlyHelp: 'This service uses {model}', modelUnavailable: 'The previous selection is no longer available: {model}. Select a currently loaded model.', noLoadedModel: 'No chat model is currently loaded in Unsloth Studio.', multipleModels: 'Multiple models are loaded. Select a specific model.', modelAuthFailed: 'The Unsloth Studio API key was rejected.', modelTimeout: 'Checking Unsloth Studio models timed out.', modelConnectionFailed: 'Could not connect to Unsloth Studio to check models.', modelStateUnsupported: 'The service did not identify loaded models, so no model was guessed.', modelDiscoveryFailed: 'Could not retrieve loaded models.', runModel: 'Model: {model}',
    parsingIndexing: 'Parsing and building index', currentStage: 'Current stage: {stage}', stageUploaded: 'Uploaded', stageParsing: 'Parsing', stageEmbedding: 'Generating index', stageIndexing: 'Building index', stageReady: 'Ready', stagePreparing: 'Preparing',
    preparingDocument: 'Preparing document', uploadingDocument: 'Uploading document', documentReady: 'Document ready', needsOcr: 'OCR required', documentUnavailable: 'Document unavailable', documentFailed: 'Document processing failed', processingFailed: 'Processing failed', uploadFailed: 'Upload failed',
    analyzing: 'Analyzing', noEnoughEvidence: 'Insufficient evidence', noAnalysisResult: 'No analysis result returned.', analysisErrorFallback: 'An error occurred during analysis.',
  },
};

let language = (() => {
  try { return localStorage.getItem('doc-rag-language') === 'en' ? 'en' : 'zh'; } catch (_) { return 'zh'; }
})();

const $ = id => document.getElementById(id);
const pretty = value => JSON.stringify(value, null, 2);
const wait = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds));
const t = (key, params = {}) => Object.entries(params).reduce(
  (text, [name, value]) => text.replaceAll(`{${name}}`, String(value)),
  messages[language][key] || messages.zh[key] || key,
);

function setUiText(element, key, params = {}) {
  delete element.dataset.i18n;
  element.dataset.uiKey = key;
  element.dataset.uiParams = JSON.stringify(params);
  element.textContent = t(key, params);
  return element;
}

function uiNode(tag, className, key, params = {}) {
  return setUiText(node(tag, className), key, params);
}

function setLocationText(element, location = {}) {
  element.dataset.uiLocation = JSON.stringify(location);
  element.textContent = formatLocation(location);
  return element;
}

function stageText(stage) {
  const key = {
    uploaded: 'stageUploaded', parsing: 'stageParsing', embedding: 'stageEmbedding',
    indexing: 'stageIndexing', ready: 'stageReady',
  }[stage] || 'stagePreparing';
  return t(key);
}

function setStageText(element, stage) {
  element.dataset.uiStage = stage || '';
  element.textContent = t('currentStage', { stage: stageText(stage) });
  return element;
}

function setNumberText(element, value, unitKey = '') {
  element.dataset.uiNumber = Number.isFinite(value) ? String(value) : '';
  element.dataset.uiNumberUnit = unitKey;
  const formatted = formatNumber(value);
  element.textContent = unitKey && Number.isFinite(value) ? t(unitKey, { count: formatted }) : formatted;
  return element;
}

function applyLanguage() {
  document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
  $('language').value = language;
  document.querySelectorAll('[data-i18n]').forEach(element => { element.textContent = t(element.dataset.i18n); });
  document.querySelectorAll('[data-i18n-aria]').forEach(element => { element.setAttribute('aria-label', t(element.dataset.i18nAria)); });
  document.querySelectorAll('[data-i18n-content]').forEach(element => { element.setAttribute('content', t(element.dataset.i18nContent)); });
  document.querySelectorAll('[data-ui-key]').forEach(element => {
    let params = {};
    try { params = JSON.parse(element.dataset.uiParams || '{}'); } catch (_) { params = {}; }
    element.textContent = t(element.dataset.uiKey, params);
  });
  document.querySelectorAll('[data-ui-location]').forEach(element => {
    try { element.textContent = formatLocation(JSON.parse(element.dataset.uiLocation)); } catch (_) { element.textContent = t('documentSource'); }
  });
  document.querySelectorAll('[data-ui-stage]').forEach(element => {
    element.textContent = t('currentStage', { stage: stageText(element.dataset.uiStage) });
  });
  document.querySelectorAll('[data-ui-number]').forEach(element => {
    const value = element.dataset.uiNumber === '' ? NaN : Number(element.dataset.uiNumber);
    const formatted = formatNumber(value);
    element.textContent = element.dataset.uiNumberUnit && Number.isFinite(value)
      ? t(element.dataset.uiNumberUnit, { count: formatted })
      : formatted;
  });
  document.querySelectorAll('[data-label-key]').forEach(element => {
    if (element.lastChild?.nodeType === Node.TEXT_NODE) element.lastChild.nodeValue = t(element.dataset.labelKey);
  });
  document.querySelectorAll('[data-ui-aria-key]').forEach(element => {
    element.setAttribute('aria-label', t(element.dataset.uiAriaKey));
  });
  updateControls();
  $('prompt-help').textContent = t($('type').value === 'report' ? 'reportHelp' : 'answerWithEvidence');
  const health = $('health');
  health.title = health.querySelector('small').textContent || health.querySelector('strong').textContent;
}

const icons = {
  check: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6.5 12.5 3.5 3.5 7.5-8"></path></svg>',
  file: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 3.75h7.1L18 7.65v12.6H7z"></path><path d="M14 3.75v4h4M9.5 12h5M9.5 15h5"></path></svg>',
  warning: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4 21 20H3zM12 9v5M12 17.25v.25"></path></svg>',
  chevron: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m8 10 4 4 4-4"></path></svg>',
};

async function call(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get('content-type') || '';
  const value = contentType.includes('json') ? await response.json() : await response.text();
  if (!response.ok) {
    const message = typeof value === 'object'
      ? (value.detail || value.error?.message || pretty(value))
      : value;
    throw new Error(message || t('requestFailed', { status: response.status }));
  }
  return value;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes === 0) return '0 KB';
  const units = ['B', 'KB', 'MB', 'GB'];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / (1024 ** index);
  return `${value.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function formatNumber(value) {
  return Number.isFinite(value) ? new Intl.NumberFormat(language === 'zh' ? 'zh-CN' : 'en').format(value) : '—';
}

function formatLocation(location = {}) {
  if (location.page_start) {
    return location.page_end && location.page_end !== location.page_start
      ? t('pageRange', { start: formatNumber(location.page_start), end: formatNumber(location.page_end) })
      : t('pageSingle', { page: formatNumber(location.page_start) });
  }
  if (location.heading_path?.length) return location.heading_path.join(' › ');
  if (Number.isFinite(location.source_start)) return t('sourcePosition', { position: formatNumber(location.source_start) });
  return t('documentSource');
}

function setChildren(element, children) {
  element.replaceChildren(...children.filter(Boolean));
}

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function setRuntimeStatus(kind, titleKey, detailKey, params = {}, rawDetail = null) {
  const health = $('health');
  health.className = `runtime-status ${kind}`;
  setUiText(health.querySelector('strong'), titleKey, params);
  const detail = health.querySelector('small');
  if (rawDetail !== null) {
    delete detail.dataset.i18n;
    delete detail.dataset.uiKey;
    delete detail.dataset.uiParams;
    detail.textContent = rawDetail;
  } else {
    setUiText(detail, detailKey, params);
  }
  health.title = detail.textContent || health.querySelector('strong').textContent;
}

function setSessionStatus(kind, labelKey) {
  const status = $('session-status');
  status.className = `session-badge ${kind}`;
  setUiText(status.querySelector('.session-badge__label'), labelKey);
}

function modelErrorKey(code) {
  return {
    UNSLOTH_NO_LOADED_MODEL: 'noLoadedModel',
    UNSLOTH_MODEL_SELECTION_REQUIRED: 'multipleModels',
    UNSLOTH_AUTH_FAILED: 'modelAuthFailed',
    UNSLOTH_DISCOVERY_TIMEOUT: 'modelTimeout',
    UNSLOTH_UNAVAILABLE: 'modelConnectionFailed',
    UNSLOTH_LOADED_STATE_UNSUPPORTED: 'modelStateUnsupported',
  }[code] || 'modelDiscoveryFailed';
}

function renderModelSelector() {
  const select = $('model-select');
  const refresh = $('model-refresh');
  const help = $('model-help');
  const previous = selectedModelId;
  select.replaceChildren();
  help.classList.remove('is-error');

  if (isModelsLoading || !modelState) {
    select.append(node('option', '', t('modelsLoading')));
    setUiText(help, 'modelsLoading');
    refresh.hidden = false;
    return;
  }

  const models = modelState.models || [];
  if (modelState.provider !== 'unsloth') {
    const current = models[0];
    select.append(node('option', '', current?.label || current?.id || '—'));
    select.value = select.options[0]?.value || '';
    refresh.hidden = true;
    setUiText(help, 'modelReadonlyHelp', { model: current?.label || current?.id || '—' });
    return;
  }

  refresh.hidden = false;
  const automatic = modelState.automatic_selection?.model_id || null;
  const autoOption = node('option', '', automatic
    ? t('automaticResolved', { model: automatic })
    : t(modelState.error?.code === 'UNSLOTH_MODEL_SELECTION_REQUIRED' ? 'automaticNeedsChoice' : 'automaticModel'));
  autoOption.value = '';
  select.append(autoOption);
  models.forEach(model => {
    const option = node('option', '', model.label === model.id ? model.id : `${model.label} · ${model.id}`);
    option.value = model.id;
    select.append(option);
  });

  if (previous && !models.some(model => model.id === previous)) {
    const stale = node('option', '', t('modelUnavailable', { model: previous }));
    stale.value = previous;
    stale.disabled = true;
    select.append(stale);
    select.value = previous;
    help.classList.add('is-error');
    setUiText(help, 'modelUnavailable', { model: previous });
  } else {
    select.value = previous || '';
    if (previous) {
      setUiText(help, 'modelExplicitHelp', { model: previous });
    } else if (automatic) {
      setUiText(help, 'modelAutoHelp', { model: automatic });
    } else {
      help.classList.add('is-error');
      setUiText(help, modelErrorKey(modelState.error?.code));
    }
  }
}

function modelSelectionReady() {
  if (isModelsLoading || !modelState) return false;
  if (modelState.provider !== 'unsloth') return true;
  if (selectedModelId) return modelState.models.some(model => model.id === selectedModelId);
  return Boolean(modelState.automatic_selection?.model_id);
}

function updateControls() {
  const ready = Boolean(session) && !isUploading && !isRunning;
  $('upload-button').disabled = isUploading || isRunning;
  $('file').disabled = isUploading || isRunning;
  $('message').disabled = !ready;
  $('send').disabled = !ready || !modelSelectionReady() || !$('message').value.trim();
  $('type').disabled = isRunning;
  $('model-select').disabled = isRunning || isModelsLoading || modelState?.provider !== 'unsloth';
  $('model-refresh').disabled = isRunning || isModelsLoading;
  $('message').placeholder = ready
    ? t($('type').value === 'report' ? 'reportExample' : 'questionExample')
    : t('messageBeforeUpload');
}

function renderDocumentProgress(state, titleKey, detailKey, detailParams = {}, rawDetail = null) {
  const wrapper = node('div', 'document-progress');
  const line = node('div', 'status-line');
  const icon = node('span', 'status-icon');
  icon.innerHTML = icons.file;
  const copy = node('span', 'status-copy');
  const title = uiNode('strong', '', titleKey);
  let detail;
  if (detailKey === 'currentStage') detail = setStageText(node('small'), detailParams.stage);
  else detail = rawDetail === null ? uiNode('small', '', detailKey, detailParams) : node('small', '', rawDetail);
  copy.append(title, detail);
  line.append(icon, copy);
  const track = node('div', 'progress-track');
  track.dataset.uiAriaKey = titleKey;
  track.setAttribute('aria-label', t(titleKey));
  track.append(node('span'));
  wrapper.append(line, track);
  setChildren($('document'), [wrapper]);
}

function renderDocumentReady(doc) {
  const card = node('div', 'document-card');
  const nameRow = node('div', 'document-card__name');
  const icon = node('span', 'status-icon is-success');
  icon.innerHTML = icons.check;
  const copy = node('div');
  copy.append(doc.filename ? node('strong', '', doc.filename) : uiNode('strong', '', 'uploadedDocument'), uiNode('small', '', 'indexReady'));
  nameRow.append(icon, copy);

  const metrics = node('div', 'document-metrics');
  const values = [
    ['pagesLabel', doc.page_count, 'pagesValue'],
    ['charactersLabel', doc.character_count, ''],
    ['chunksLabel', doc.chunk_count, ''],
    ['formatLabel', doc.media_type?.includes('pdf') ? 'PDF' : 'Markdown', null],
  ];
  values.forEach(([labelKey, value, numberUnit]) => {
    const metric = node('div', 'metric');
    const renderedValue = numberUnit === null ? node('strong', '', value) : setNumberText(node('strong'), value, numberUnit);
    metric.append(uiNode('span', '', labelKey), renderedValue);
    metrics.append(metric);
  });
  card.append(nameRow, metrics, uiNode('div', 'document-id', 'versionValue', { id: doc.version_id }));
  setChildren($('document'), [card]);
}

function renderDocumentProblem(doc, titleKey, kind = 'error') {
  const wrapper = node('div', 'document-progress');
  const line = node('div', 'status-line');
  const icon = node('span', `status-icon is-${kind}`);
  icon.innerHTML = icons.warning;
  const copy = node('span', 'status-copy');
  copy.append(uiNode('strong', '', titleKey), doc.filename ? node('small', '', doc.filename) : uiNode('small', '', 'documentIncomplete'));
  line.append(icon, copy);
  const message = doc.error_message
    ? node('div', 'error-message', doc.error_message)
    : uiNode('div', 'error-message', doc.status === 'needs_ocr' ? 'ocrFallback' : 'retryFallback');
  wrapper.append(line, message);
  setChildren($('document'), [wrapper]);
}

function renderAnswerLoading(labelKey = 'retrievingDocument') {
  const loading = node('div', 'loading-state');
  const inner = node('div');
  inner.append(node('span', 'loading-spinner'), uiNode('strong', '', labelKey), uiNode('span', '', 'locatingEvidence'));
  loading.append(inner);
  setChildren($('answer'), [loading]);
}

function renderAnswerEmpty(titleKey = 'resultsHere', detailKey = 'resultsHereDetail') {
  const wrapper = node('div', 'empty-state');
  const icon = node('span', 'empty-state__icon');
  icon.innerHTML = icons.file;
  wrapper.append(icon, uiNode('h4', '', titleKey), uiNode('p', '', detailKey));
  setChildren($('answer'), [wrapper]);
}

function renderAnswerText(text, options = {}) {
  const classes = `answer-content${options.error ? ' is-error' : ''}${options.empty ? ' is-empty-answer' : ''}`;
  const wrapper = node('div', classes);
  const label = node('div', 'answer-content__label');
  const icon = node('span');
  icon.innerHTML = options.error ? icons.warning : icons.check;
  const labelText = options.labelKey ? t(options.labelKey) : (options.label || t(options.error ? 'analysisIncomplete' : 'analysisComplete'));
  const dynamicLabel = document.createTextNode(labelText);
  label.append(icon, dynamicLabel);
  if (options.labelKey || !options.label) {
    label.dataset.labelKey = options.labelKey || (options.error ? 'analysisIncomplete' : 'analysisComplete');
  }
  wrapper.append(label);
  if (!text && options.textKey) {
    if (options.prefix) {
      const paragraph = node('p');
      paragraph.append(document.createTextNode(options.prefix), uiNode('span', '', options.textKey));
      wrapper.append(paragraph);
    } else {
      wrapper.append(uiNode('p', '', options.textKey));
    }
  } else {
    const paragraphs = String(text || t('noContent')).split(/\n{2,}/).filter(Boolean);
    paragraphs.forEach(paragraph => wrapper.append(node('p', '', paragraph)));
  }
  setChildren($('answer'), [wrapper]);
}

const confidenceKeys = { high: 'confidenceHigh', medium: 'confidenceMedium', low: 'confidenceLow' };
const supportKeys = { supported: 'supportSupported', partially_supported: 'supportPartial', unsupported: 'supportUnsupported', conflicting: 'supportConflicting' };

function semanticBadge(text, kind) {
  return node('span', `semantic-badge is-${kind || ''}`, text);
}

function semanticUiBadge(key, kind) {
  return uiNode('span', `semantic-badge is-${kind || ''}`, key);
}

function renderReport(report) {
  const wrapper = node('div', 'report');
  const hero = node('div', 'report-hero');
  const heroTop = node('div', 'report-hero__top');
  heroTop.append(
    report.title ? node('h4', '', report.title) : uiNode('h4', '', 'reportTitleFallback'),
    semanticUiBadge('findingsCount', 'supported'),
  );
  setUiText(heroTop.lastChild, 'findingsCount', { count: formatNumber(report.findings?.length || 0) });
  hero.append(heroTop, report.executive_summary ? node('p', '', report.executive_summary) : uiNode('p', '', 'noExecutiveSummary'));
  wrapper.append(hero);

  const findingsSection = node('section', 'report-section');
  findingsSection.append(uiNode('h4', '', 'coreFindings'));
  if (report.findings?.length) {
    const list = node('div', 'finding-list');
    report.findings.forEach((finding, index) => {
      const item = node('article', 'finding');
      const top = node('div', 'finding__top');
      const badges = node('div', 'badge-row');
      badges.append(
        confidenceKeys[finding.confidence]
          ? semanticUiBadge(confidenceKeys[finding.confidence], finding.confidence)
          : semanticBadge(finding.confidence || t('pendingAssessment'), finding.confidence),
        supportKeys[finding.support]
          ? semanticUiBadge(supportKeys[finding.support], finding.support === 'partially_supported' ? 'partial' : finding.support)
          : semanticBadge(finding.support || t('supportSupported'), finding.support),
      );
      top.append(finding.finding_id ? node('span', 'finding__id', finding.finding_id) : uiNode('span', 'finding__id', 'findingNumber', { number: formatNumber(index + 1) }), badges);
      item.append(top, finding.claim ? node('p', '', finding.claim) : uiNode('p', '', 'noContent'));
      list.append(item);
    });
    findingsSection.append(list);
  } else {
    findingsSection.append(uiNode('p', 'report-empty', 'noFindings'));
  }
  wrapper.append(findingsSection);

  if (report.recommendations?.length) {
    const recommendationSection = node('section', 'report-section');
    recommendationSection.append(uiNode('h4', '', 'recommendations'));
    const list = node('ol', 'report-list');
    report.recommendations.forEach(item => {
      const value = typeof item === 'string' ? item : item.text;
      list.append(value ? node('li', '', value) : uiNode('li', '', 'noContent'));
    });
    recommendationSection.append(list);
    wrapper.append(recommendationSection);
  }

  if (report.open_questions?.length) {
    const questionsSection = node('section', 'report-section');
    questionsSection.append(uiNode('h4', '', 'openQuestions'));
    const list = node('ul', 'report-list');
    report.open_questions.forEach(item => list.append(item ? node('li', '', item) : uiNode('li', '', 'noContent')));
    questionsSection.append(list);
    wrapper.append(questionsSection);
  }

  setChildren($('answer'), [wrapper]);
}

function renderRunMeta(run) {
  const meta = $('run-meta');
  meta.replaceChildren();
  const items = [];
  if (Number.isFinite(run.search_rounds)) items.push(['searchRounds', { count: formatNumber(run.search_rounds) }]);
  if (Number.isFinite(run.tool_calls)) items.push(['toolCalls', { count: formatNumber(run.tool_calls) }]);
  if (run.stop_reason) items.push([run.stop_reason === 'evidence_sufficient' ? 'evidenceSufficient' : 'searchEnded', {}]);
  if (run.model) items.push(['runModel', { model: run.model }]);
  items.forEach(([key, params]) => meta.append(uiNode('span', '', key, params)));
  meta.hidden = items.length === 0;
}

async function toggleEvidence(button, detail, citation) {
  const isOpen = button.getAttribute('aria-expanded') === 'true';
  if (isOpen) {
    button.setAttribute('aria-expanded', 'false');
    setUiText(button.querySelector('span'), 'viewFullEvidence');
    detail.hidden = true;
    return;
  }

  button.setAttribute('aria-expanded', 'true');
  setUiText(button.querySelector('span'), 'collapseFullEvidence');
  detail.hidden = false;
  if (detail.dataset.loaded === 'true') return;

  detail.replaceChildren(uiNode('p', 'report-empty', 'loadingSource'));
  try {
    const evidence = await call(`/evidence/${citation.evidence_id}`);
    const definitions = node('dl');
    const fields = [
      ['evidenceNumber', evidence.evidence_id, null],
      ['documentVersion', evidence.document_version, null],
      ['chunkNumber', evidence.chunk_id, null],
      ['sourceLocation', null, evidence.location],
    ];
    fields.forEach(([termKey, value, location]) => {
      const definition = location ? setLocationText(node('dd'), location) : node('dd', '', value || '—');
      definitions.append(uiNode('dt', '', termKey), definition);
    });
    detail.replaceChildren(
      definitions,
      uiNode('h5', '', 'fullSource'),
      evidence.context_text || evidence.raw_text ? node('pre', '', evidence.context_text || evidence.raw_text) : uiNode('pre', '', 'noSource'),
    );
    detail.dataset.loaded = 'true';
  } catch (error) {
    const wrapper = uiNode('div', 'error-message', 'evidenceLoadFailed', { message: error.message });
    detail.replaceChildren(wrapper);
  }
}

function renderCitations(citations = []) {
  const section = $('evidence-section');
  const container = $('citations');
  container.replaceChildren();
  setUiText($('citation-count'), 'citationsCount', { count: formatNumber(citations.length) });
  section.hidden = citations.length === 0;

  citations.forEach((citation, index) => {
    const card = node('article', 'citation');
    const body = node('div', 'citation__body');
    const top = node('div', 'citation__top');
    const citationIndex = node('span', 'citation__index');
    citationIndex.append(node('span', '', `E${index + 1}`), document.createTextNode(t('sourceEvidence')));
    citationIndex.dataset.labelKey = 'sourceEvidence';
    top.append(citationIndex, setLocationText(node('span', 'citation__location'), citation.location));
    body.append(top, node('blockquote', '', citation.excerpt || ''));

    const detailId = `evidence-detail-${index}`;
    const button = node('button', 'evidence-toggle');
    button.type = 'button';
    button.setAttribute('aria-expanded', 'false');
    button.setAttribute('aria-controls', detailId);
    button.append(uiNode('span', '', 'viewFullEvidence'));
    const chevron = node('span');
    chevron.innerHTML = icons.chevron;
    button.append(chevron.firstChild);

    const detail = node('div', 'evidence-detail');
    detail.id = detailId;
    detail.hidden = true;
    button.addEventListener('click', () => toggleEvidence(button, detail, citation));
    card.append(body, button, detail);
    container.append(card);
  });
}

async function loadTrace(runId) {
  const disclosure = $('trace-disclosure');
  disclosure.hidden = false;
  setUiText($('trace-summary'), 'traceLoadingSummary');
  setUiText($('trace'), 'traceLoading');
  try {
    const trace = await call(`/runs/${runId}/trace`);
    const count = trace.events?.length || 0;
    setUiText($('trace-summary'), 'traceEvents', { count: formatNumber(count) });
    delete $('trace').dataset.uiKey;
    delete $('trace').dataset.uiParams;
    $('trace').textContent = pretty(trace);
  } catch (error) {
    setUiText($('trace-summary'), 'traceLoadFailed');
    delete $('trace').dataset.uiKey;
    delete $('trace').dataset.uiParams;
    $('trace').textContent = error.message;
  }
}

function resetResults({ resetAnswer = false } = {}) {
  $('run-meta').hidden = true;
  $('evidence-section').hidden = true;
  $('citations').replaceChildren();
  $('trace-disclosure').hidden = true;
  $('trace-disclosure').open = false;
  if (resetAnswer) {
    renderAnswerEmpty('preparingNewDocument', 'preparingNewDocumentDetail');
  }
}

async function checkHealth() {
  try {
    const health = await call('/health');
    if (health.agent_ready) {
      if (health.agent_provider === 'demo') {
        setRuntimeStatus('is-ready', 'demoMode', 'demoRuntime');
      } else {
        setRuntimeStatus('is-ready', 'serviceNormal', 'adkRuntime', { provider: health.agent_provider });
      }
    } else {
      const detailKey = health.agent_provider === 'gemini' ? 'geminiMissing' : health.agent_provider === 'unsloth' ? 'unslothMissing' : null;
      if (detailKey) setRuntimeStatus('is-warning', 'serviceNeedsConfig', detailKey);
      else setRuntimeStatus('is-warning', 'serviceNeedsConfig', '', {}, health.runtime_message || '');
    }
  } catch (error) {
    setRuntimeStatus('is-error', 'serviceConnectionFailed', '', {}, error.message);
  }
}

async function loadModels() {
  isModelsLoading = true;
  renderModelSelector();
  updateControls();
  try {
    modelState = await call('/models');
  } catch (error) {
    modelState = {
      provider: 'unsloth', models: [], automatic_selection: { model_id: null, reason: null },
      error: { code: 'UNSLOTH_UNAVAILABLE', message: error.message },
    };
  } finally {
    isModelsLoading = false;
    renderModelSelector();
    updateControls();
  }
}

async function pollDocument(initial) {
  let state = initial;
  while (!['ready', 'failed', 'needs_ocr'].includes(state.status)) {
    renderDocumentProgress(state.status, 'parsingIndexing', 'currentStage', { stage: state.stage || state.status || '' });
    await wait(700);
    state = await call(`/documents/${state.document_id}/versions/${state.version_id}`);
  }
  return state;
}

$('file').addEventListener('change', () => {
  const file = $('file').files[0];
  if (file) {
    $('file-name').textContent = file.name;
    $('file-size').textContent = formatBytes(file.size);
    $('file-selection').hidden = false;
  } else {
    $('file-selection').hidden = true;
  }
});

const dropzone = document.querySelector('.dropzone');
['dragenter', 'dragover'].forEach(eventName => dropzone.addEventListener(eventName, event => {
  event.preventDefault();
  if (!$('file').disabled) dropzone.classList.add('is-dragging');
}));
dropzone.addEventListener('dragleave', event => {
  if (!dropzone.contains(event.relatedTarget)) dropzone.classList.remove('is-dragging');
});
dropzone.addEventListener('drop', event => {
  event.preventDefault();
  dropzone.classList.remove('is-dragging');
  if ($('file').disabled || !event.dataTransfer?.files?.length) return;
  $('file').files = event.dataTransfer.files;
  $('file').dispatchEvent(new Event('change', { bubbles: true }));
});

$('upload').addEventListener('submit', async event => {
  event.preventDefault();
  const file = $('file').files[0];
  if (!file || isUploading) return;

  isUploading = true;
  session = null;
  resetResults({ resetAnswer: true });
  setSessionStatus('is-busy', 'preparingDocument');
  renderDocumentProgress('uploading', 'uploadingDocument', '', {}, `${file.name} · ${formatBytes(file.size)}`);
  updateControls();

  const body = new FormData();
  body.append('file', file);
  try {
    documentState = await call('/documents', { method: 'POST', body });
    documentState = await pollDocument(documentState);
    if (documentState.status === 'ready') {
      session = await call('/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: pretty({ document_id: documentState.document_id, version_id: documentState.version_id }),
      });
      renderDocumentReady(documentState);
      setSessionStatus('is-ready', 'documentReady');
    } else if (documentState.status === 'needs_ocr') {
      renderDocumentProblem(documentState, 'needsOcr', 'warning');
      setSessionStatus('is-waiting', 'documentUnavailable');
    } else {
      renderDocumentProblem(documentState, 'documentFailed');
      setSessionStatus('is-waiting', 'processingFailed');
    }
  } catch (error) {
    renderDocumentProblem({ filename: file.name, error_message: error.message }, 'uploadFailed');
    setSessionStatus('is-waiting', 'uploadFailed');
  } finally {
    isUploading = false;
    updateControls();
    if (session) $('message').focus();
  }
});

$('message').addEventListener('input', updateControls);
$('message').addEventListener('keydown', event => {
  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter' && !$('send').disabled) {
    event.preventDefault();
    $('send').click();
  }
});

$('type').addEventListener('change', () => {
  $('prompt-help').textContent = t($('type').value === 'report' ? 'reportHelp' : 'answerWithEvidence');
  updateControls();
});

$('model-select').addEventListener('change', () => {
  selectedModelId = $('model-select').value || null;
  try {
    if (selectedModelId) localStorage.setItem('doc-rag-model-id', selectedModelId);
    else localStorage.removeItem('doc-rag-model-id');
  } catch (_) { /* Storage may be disabled. */ }
  renderModelSelector();
  updateControls();
});

$('model-refresh').addEventListener('click', loadModels);

$('send').addEventListener('click', async () => {
  const message = $('message').value.trim();
  if (!session || !message || isRunning) return;

  isRunning = true;
  currentRun = null;
  resetResults();
  setSessionStatus('is-busy', 'analyzing');
  renderAnswerLoading($('type').value === 'report' ? 'generatingReport' : 'retrievingDocument');
  updateControls();

  try {
    const queued = await call(`/sessions/${session.session_id}/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: pretty({ message, request_type: $('type').value, model_id: selectedModelId }),
    });
    currentRun = queued.run_id;

    let run;
    do {
      await wait(500);
      run = await call(`/runs/${currentRun}`);
    } while (['queued', 'running'].includes(run.status));

    renderRunMeta(run);
    if (run.status === 'failed') {
      const errorCode = run.error_code || 'AGENT_FAILED';
      renderAnswerText(run.error_message || '', {
        error: true,
        textKey: run.error_message ? undefined : 'analysisErrorFallback',
        prefix: `${errorCode}: `,
      });
      if (run.error_message) $('answer').querySelector('.answer-content p').prepend(`${errorCode}: `);
    } else if (run.report) {
      renderReport(run.report);
    } else {
      const noEvidence = !run.citations?.length;
      renderAnswerText(run.answer || '', {
        empty: noEvidence,
        labelKey: noEvidence ? 'noEnoughEvidence' : 'analysisComplete',
        textKey: run.answer ? undefined : 'noAnalysisResult',
      });
    }
    renderCitations(run.citations || []);
    await loadTrace(currentRun);
  } catch (error) {
    renderAnswerText(error.message, { error: true });
  } finally {
    isRunning = false;
    setSessionStatus(session ? 'is-ready' : 'is-waiting', session ? 'documentReady' : 'waitingDocument');
    updateControls();
  }
});

$('language').addEventListener('change', () => {
  language = $('language').value === 'en' ? 'en' : 'zh';
  try { localStorage.setItem('doc-rag-language', language); } catch (_) { /* Storage may be disabled. */ }
  applyLanguage();
  renderModelSelector();
});

applyLanguage();
checkHealth();
loadModels();
