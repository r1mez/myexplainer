const SVG_NS = "http://www.w3.org/2000/svg";
const FRONTEND_SUPPORTED_DATASETS = [
  "mutag",
  "mutag188",
  "nci1",
  "bbbp",
  "ba2motif",
  "benzene",
  "alkane_carbonyl",
  "fluoride_carbonyl",
  "proteins",
];
const FRONTEND_SUPPORTED_SPLITS = ["training", "evaluation", "testing"];

const state = {
  datasets: [],
  datasetMap: new Map(),
  dataset: null,
  split: "testing",
  index: 0,
  graphMeta: null,
  graph: null,
  originalGraph: null,
  groundTruthMotif: null,
  originalGroundTruthMotif: null,
  originalPrediction: null,
  currentPrediction: null,
  predictionDirty: false,
  tool: "select",
  connectStartId: null,
  selectedNodeId: null,
  selectedEdgeKey: null,
  drag: {
    active: false,
    nodeId: null,
    pointerId: null,
    startPoint: null,
    origin: null,
  },
};

const refs = {};

document.addEventListener("DOMContentLoaded", () => {
  bindRefs();
  bindEvents();
  bootstrap().catch((error) => {
    console.error(error);
    setStatus(`初始化失败：${error.message}`, "error");
  });
});

function bindRefs() {
  refs.datasetSelect = document.getElementById("datasetSelect");
  refs.splitSelect = document.getElementById("splitSelect");
  refs.graphIndexInput = document.getElementById("graphIndexInput");
  refs.graphCountHint = document.getElementById("graphCountHint");
  refs.datasetAvailability = document.getElementById("datasetAvailability");
  refs.loadGraphButton = document.getElementById("loadGraphButton");
  refs.prevGraphButton = document.getElementById("prevGraphButton");
  refs.nextGraphButton = document.getElementById("nextGraphButton");
  refs.toolPalette = document.getElementById("toolPalette");
  refs.toolHint = document.getElementById("toolHint");
  refs.resetGraphButton = document.getElementById("resetGraphButton");
  refs.predictButton = document.getElementById("predictButton");
  refs.statusBox = document.getElementById("statusBox");
  refs.graphCanvas = document.getElementById("graphCanvas");
  refs.graphSubtitle = document.getElementById("graphSubtitle");
  refs.propertyPanel = document.getElementById("propertyPanel");
  refs.predictionPanel = document.getElementById("predictionPanel");
  refs.predictionHint = document.getElementById("predictionHint");
  refs.deviceBadge = document.getElementById("deviceBadge");
}

function bindEvents() {
  refs.loadGraphButton.addEventListener("click", () => runAsync(loadGraph));
  refs.prevGraphButton.addEventListener("click", () => shiftGraphIndex(-1));
  refs.nextGraphButton.addEventListener("click", () => shiftGraphIndex(1));
  refs.resetGraphButton.addEventListener("click", () => resetGraph());
  refs.predictButton.addEventListener("click", () => runAsync(predictCurrentGraph));

  refs.datasetSelect.addEventListener("change", () => {
    state.dataset = refs.datasetSelect.value;
    state.index = 0;
    updateDatasetBadge();
    syncIndexControls();
    runAsync(loadGraph);
  });

  refs.splitSelect.addEventListener("change", () => {
    state.split = refs.splitSelect.value;
    state.index = 0;
    syncIndexControls();
    runAsync(loadGraph);
  });

  refs.graphIndexInput.addEventListener("change", () => {
    const parsed = Number.parseInt(refs.graphIndexInput.value, 10);
    state.index = Number.isFinite(parsed) ? parsed : 0;
    syncIndexControls();
  });

  refs.graphIndexInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      runAsync(loadGraph);
    }
  });

  refs.toolPalette.addEventListener("click", (event) => {
    const button = event.target.closest("[data-tool]");
    if (!button) {
      return;
    }
    setTool(button.dataset.tool);
  });

  refs.graphCanvas.addEventListener("pointerdown", handleCanvasPointerDown);
  window.addEventListener("pointermove", handleGlobalPointerMove);
  window.addEventListener("pointerup", handleGlobalPointerUp);
}

async function bootstrap() {
  const summaryPayload = await refreshDatasetSummary();
  renderDatasetOptions(summaryPayload.default_dataset);
  updateDatasetBadge();
  const currentDatasetInfo = getCurrentDatasetInfo();
  if (currentDatasetInfo && currentDatasetInfo.data_available === false && currentDatasetInfo.data_error) {
    setStatus(currentDatasetInfo.data_error, "warning");
  }

  syncIndexControls();
  setTool("select");

  if (state.dataset) {
    await loadGraph();
  }
  return;

  const response = await fetch("/api/datasets");
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "无法获取数据集列表");
  }

  state.datasets = mergeDatasetOptions(payload.datasets || [], payload.supported_datasets || []);
  state.datasetMap = new Map(state.datasets.map((item) => [item.name, item]));
  refs.deviceBadge.textContent = payload.device || "unknown";

  renderDatasetOptions(payload.default_dataset);
  updateDatasetBadge();
  const datasetInfo = getCurrentDatasetInfo();
  if (datasetInfo && datasetInfo.data_available === false && datasetInfo.data_error) {
    setStatus(datasetInfo.data_error, "warning");
  }

  syncIndexControls();
  setTool("select");

  if (state.dataset) {
    await loadGraph();
  }
}

async function refreshDatasetSummary() {
  const response = await fetch(`/api/datasets?ts=${Date.now()}`);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Failed to fetch dataset summary.");
  }

  state.datasets = mergeDatasetOptions(payload.datasets || [], payload.supported_datasets || []);
  state.datasetMap = new Map(state.datasets.map((item) => [item.name, item]));
  refs.deviceBadge.textContent = payload.device || "unknown";
  return payload;
}

function renderDatasetOptions(defaultDataset) {
  refs.datasetSelect.innerHTML = "";
  for (const dataset of state.datasets) {
    const option = document.createElement("option");
    option.value = dataset.name;
    option.textContent = dataset.display_name || dataset.name;
    refs.datasetSelect.appendChild(option);
  }

  const fallback = state.datasets.length > 0 ? state.datasets[0].name : "";
  state.dataset = state.datasetMap.has(defaultDataset) ? defaultDataset : fallback;
  refs.datasetSelect.value = state.dataset;
  refs.splitSelect.value = state.split;
}

function mergeDatasetOptions(apiDatasets, apiSupportedDatasets) {
  const orderedNames = [];
  const seenNames = new Set();
  const addName = (name) => {
    const normalized = String(name || "").trim().toLowerCase();
    if (!normalized || seenNames.has(normalized)) {
      return;
    }
    seenNames.add(normalized);
    orderedNames.push(normalized);
  };

  FRONTEND_SUPPORTED_DATASETS.forEach(addName);
  apiSupportedDatasets.forEach(addName);
  apiDatasets.forEach((dataset) => addName(dataset?.name));

  const byName = new Map();
  for (const name of orderedNames) {
    byName.set(name, {
      name,
      display_name: name,
      splits: Object.fromEntries(FRONTEND_SUPPORTED_SPLITS.map((split) => [split, 0])),
      data_available: false,
      data_error: "Dataset metadata was not returned by /api/datasets.",
      model_available: false,
      model_path: null,
      default_feature_mode: "vector",
      frontend_registered: FRONTEND_SUPPORTED_DATASETS.includes(name),
    });
  }

  for (const dataset of apiDatasets) {
    if (!dataset?.name) {
      continue;
    }
    const name = String(dataset.name).trim().toLowerCase();
    byName.set(name, {
      ...(byName.get(name) || {}),
      ...dataset,
      name,
      display_name: dataset.display_name || name,
      frontend_registered: FRONTEND_SUPPORTED_DATASETS.includes(name),
    });
  }

  return orderedNames.map((name) => byName.get(name)).filter(Boolean);
}

function updateDatasetBadge() {
  const info = getCurrentDatasetInfo();
  if (!info) {
    refs.datasetAvailability.textContent = "No dataset";
    refs.datasetAvailability.title = "";
    return;
  }

  if (info.data_available) {
    refs.datasetAvailability.textContent = info.model_available ? "Data + Model" : "Data only";
  } else {
    refs.datasetAvailability.textContent = "Data unavailable";
  }
  refs.datasetAvailability.title = info.data_error || "";
}

function syncIndexControls() {
  const info = getCurrentDatasetInfo();
  const total = info?.splits?.[state.split] ?? 0;
  const maxIndex = Math.max(total - 1, 0);
  state.index = clamp(state.index, 0, maxIndex);
  refs.graphIndexInput.min = "0";
  refs.graphIndexInput.max = String(maxIndex);
  refs.graphIndexInput.value = String(state.index);
  refs.graphCountHint.textContent = total > 0 ? `0 - ${maxIndex} / 共 ${total} 张` : "当前划分没有样本";
  if (total <= 0 && info?.data_error) {
    refs.graphCountHint.textContent = info.data_error;
  }
  refs.prevGraphButton.disabled = total <= 0 || state.index <= 0;
  refs.nextGraphButton.disabled = total <= 0 || state.index >= maxIndex;
}

function shiftGraphIndex(delta) {
  state.index += delta;
  syncIndexControls();
  runAsync(loadGraph);
}

async function loadGraph() {
  if (!state.dataset) {
    setStatus("当前没有可用数据集。", "warning");
    return;
  }

  await refreshDatasetSummary();
  updateDatasetBadge();
  syncIndexControls();
  setStatus(`正在加载 ${state.dataset}/${state.split} 的第 ${state.index} 张图...`, "info");

  const url = new URL("/api/graph", window.location.origin);
  url.searchParams.set("dataset", state.dataset);
  url.searchParams.set("split", state.split);
  url.searchParams.set("index", String(state.index));

  const response = await fetch(url.toString());
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "加载图失败");
  }

  state.graphMeta = payload.graph_meta;
  state.graph = {
    nodes: deepClone(payload.nodes || []),
    edges: deepClone(payload.edges || []),
  };
  state.originalGraph = deepClone(state.graph);
  state.groundTruthMotif = deepClone(payload.ground_truth_motif || null);
  state.originalGroundTruthMotif = deepClone(payload.ground_truth_motif || null);
  state.originalPrediction = payload.original_prediction || null;
  state.currentPrediction = payload.original_prediction || null;
  state.predictionDirty = false;
  state.selectedNodeId = null;
  state.selectedEdgeKey = null;
  state.connectStartId = null;

  refs.graphSubtitle.textContent =
    `${payload.dataset}/${payload.split} · 图 ${payload.source_index} · `
    + `${state.graphMeta.num_nodes} 节点 / ${state.graphMeta.num_edges} 边`;

  refs.predictButton.disabled = !state.graphMeta.model_available;
  renderGraph();
  renderPropertyPanel();
  renderPredictionPanel();
  refreshGraphSubtitle();

  if (state.graphMeta.model_error) {
    setStatus(state.graphMeta.model_error, "warning");
  } else {
    setStatus("图已加载，可以开始编辑。", "success");
  }
}

function setTool(toolName) {
  state.tool = toolName;
  state.connectStartId = null;
  for (const button of refs.toolPalette.querySelectorAll("[data-tool]")) {
    button.classList.toggle("is-active", button.dataset.tool === toolName);
  }

  const labels = {
    select: "选择",
    "add-node": "加节点",
    "add-edge": "连边",
    "delete-node": "删节点",
    "delete-edge": "删边",
  };
  refs.toolHint.textContent = labels[toolName] || toolName;
  renderGraph();
}

function renderGraph() {
  refs.graphCanvas.innerHTML = "";

  if (!state.graph) {
    const text = createSvg("text", {
      x: 450,
      y: 310,
      "text-anchor": "middle",
      fill: "#68757c",
      "font-size": "18",
    });
    text.textContent = "请选择一个图开始编辑";
    refs.graphCanvas.appendChild(text);
    return;
  }

  const edgeLayer = createSvg("g", { class: "edge-layer" });
  const motifEdgeLayer = createSvg("g", {
    class: "motif-edge-layer",
    "pointer-events": "none",
  });
  const motifNodeLayer = createSvg("g", {
    class: "motif-node-layer",
    "pointer-events": "none",
  });
  const nodeLayer = createSvg("g", { class: "node-layer" });
  refs.graphCanvas.append(edgeLayer, motifEdgeLayer, motifNodeLayer, nodeLayer);

  for (const edge of state.graph.edges) {
    const source = getNodeById(edge.source);
    const target = getNodeById(edge.target);
    if (!source || !target) {
      continue;
    }

    const key = edgeKey(edge.source, edge.target);
    const selected = key === state.selectedEdgeKey;
    const line = createSvg("line", {
      x1: source.pos.x,
      y1: source.pos.y,
      x2: target.pos.x,
      y2: target.pos.y,
      stroke: selected ? "#be4d25" : "rgba(31, 42, 48, 0.36)",
      "stroke-width": selected ? 5 : 3,
      "stroke-linecap": "round",
      opacity: selected ? 1 : 0.9,
    });

    line.addEventListener("pointerdown", (event) => {
      event.stopPropagation();
      handleEdgePointerDown(edge);
    });
    edgeLayer.appendChild(line);

    if (isMotifEdge(edge.source, edge.target)) {
      const motifLine = createSvg("line", {
        x1: source.pos.x,
        y1: source.pos.y,
        x2: target.pos.x,
        y2: target.pos.y,
        stroke: "rgba(217, 164, 65, 0.95)",
        "stroke-width": selected ? 8 : 6,
        "stroke-linecap": "round",
        opacity: 0.9,
        "pointer-events": "none",
      });
      motifEdgeLayer.appendChild(motifLine);
    }
  }

  for (const node of state.graph.nodes) {
    const group = createSvg("g", {
      transform: `translate(${node.pos.x}, ${node.pos.y})`,
      style: "cursor: pointer;",
    });
    const isSelected = node.id === state.selectedNodeId;
    const isEdgeAnchor = node.id === state.connectStartId;
    const isMotif = isMotifNode(node.id);
    if (isMotif) {
      const haloGroup = createSvg("g", {
        transform: `translate(${node.pos.x}, ${node.pos.y})`,
      });
      const halo = createSvg("circle", {
        r: isSelected || isEdgeAnchor ? 27 : 22,
        fill: "rgba(217, 164, 65, 0.18)",
        stroke: "rgba(217, 164, 65, 0.95)",
        "stroke-width": 3,
        "pointer-events": "none",
      });
      haloGroup.appendChild(halo);
      motifNodeLayer.appendChild(haloGroup);
    }
    const circle = createSvg("circle", {
      r: isSelected || isEdgeAnchor ? 20 : 15,
      fill: isSelected ? "#be4d25" : isEdgeAnchor ? "#7b2612" : "#2f6c87",
      stroke: isSelected ? "#fff7f1" : "#f8f3eb",
      "stroke-width": isSelected ? 4 : 3,
    });
    const label = createSvg("text", { class: "node-label", y: "-2" });
    label.textContent = trimLabel(node.label);
    const nodeId = createSvg("text", { class: "node-id", y: "12" });
    nodeId.textContent = `#${node.id}`;

    group.append(circle, label, nodeId);
    group.addEventListener("pointerdown", (event) => handleNodePointerDown(event, node.id));
    nodeLayer.appendChild(group);
  }
}

function handleCanvasPointerDown(event) {
  if (!state.graph) {
    return;
  }

  if (event.target !== refs.graphCanvas) {
    return;
  }

  const point = svgPoint(event);

  if (state.tool === "add-node") {
    addNode(point);
    return;
  }

  if (state.tool === "select") {
    state.selectedNodeId = null;
    state.selectedEdgeKey = null;
    renderGraph();
    renderPropertyPanel();
  }
}

function handleNodePointerDown(event, nodeId) {
  if (!state.graph) {
    return;
  }

  event.stopPropagation();
  const node = getNodeById(nodeId);
  if (!node) {
    return;
  }

  if (state.tool === "add-edge") {
    handleAddEdgeMode(nodeId);
    return;
  }

  if (state.tool === "delete-node") {
    deleteNode(nodeId);
    return;
  }

  state.selectedNodeId = nodeId;
  state.selectedEdgeKey = null;
  renderGraph();
  renderPropertyPanel();

  if (state.tool === "select") {
    state.drag = {
      active: true,
      nodeId,
      pointerId: event.pointerId,
      startPoint: svgPoint(event),
      origin: { x: node.pos.x, y: node.pos.y },
    };
  }
}

function handleEdgePointerDown(edge) {
  if (!state.graph) {
    return;
  }

  if (state.tool === "delete-edge") {
    deleteEdge(edge.source, edge.target);
    return;
  }

  state.selectedEdgeKey = edgeKey(edge.source, edge.target);
  state.selectedNodeId = null;
  renderGraph();
  renderPropertyPanel();
}

function handleGlobalPointerMove(event) {
  if (!state.drag.active || state.tool !== "select") {
    return;
  }

  const point = svgPoint(event);
  const dx = point.x - state.drag.startPoint.x;
  const dy = point.y - state.drag.startPoint.y;
  const node = getNodeById(state.drag.nodeId);
  if (!node) {
    return;
  }

  node.pos.x = clamp(state.drag.origin.x + dx, 24, 876);
  node.pos.y = clamp(state.drag.origin.y + dy, 24, 596);
  renderGraph();
}

function handleGlobalPointerUp() {
  if (state.drag.active) {
    state.drag.active = false;
    state.drag.nodeId = null;
  }
}

function handleAddEdgeMode(nodeId) {
  if (state.connectStartId === null) {
    state.connectStartId = nodeId;
    state.selectedNodeId = nodeId;
    state.selectedEdgeKey = null;
    setStatus(`已选择节点 #${nodeId} 作为连边起点，请继续选择终点。`, "info");
    renderGraph();
    renderPropertyPanel();
    return;
  }

  if (state.connectStartId === nodeId) {
    state.connectStartId = null;
    renderGraph();
    setStatus("已取消连边起点。", "info");
    return;
  }

  if (hasEdge(state.connectStartId, nodeId)) {
    state.connectStartId = null;
    renderGraph();
    setStatus("这条边已经存在了。", "warning");
    return;
  }

  const source = Math.min(state.connectStartId, nodeId);
  const target = Math.max(state.connectStartId, nodeId);
  state.graph.edges.push({ source, target });
  state.connectStartId = null;
  state.selectedEdgeKey = edgeKey(source, target);
  state.selectedNodeId = null;
  markGraphDirty("已添加一条新边。");
}

function addNode(point) {
  const xDim = state.graphMeta?.x_dim || 1;
  const newId = state.graph.nodes.length > 0
    ? Math.max(...state.graph.nodes.map((node) => node.id)) + 1
    : 0;
  const defaultAtomType = Number.isFinite(state.graphMeta?.default_atom_type)
    ? Number(state.graphMeta.default_atom_type)
    : null;

  const baseNode = getNodeById(state.selectedNodeId) || state.graph.nodes[0] || null;
  let feature;
  if (baseNode) {
    feature = [...baseNode.feature];
  } else if (state.graphMeta?.feature_mode === "onehot") {
    feature = Array.from({ length: xDim }, (_, index) => (index === 0 ? 1 : 0));
  } else {
    feature = Array.from(
      { length: xDim },
      (_, index) => (index === 0 && defaultAtomType !== null ? defaultAtomType : 0)
    );
  }

  const label = featureToLabel(feature, newId);
  state.graph.nodes.push({
    id: newId,
    label,
    feature,
    pos: {
      x: clamp(point.x, 24, 876),
      y: clamp(point.y, 24, 596),
    },
  });
  state.selectedNodeId = newId;
  state.selectedEdgeKey = null;
  markGraphDirty(`已添加节点 #${newId}。`);
}

function deleteNode(nodeId) {
  state.graph.nodes = state.graph.nodes.filter((node) => node.id !== nodeId);
  state.graph.edges = state.graph.edges.filter(
    (edge) => edge.source !== nodeId && edge.target !== nodeId
  );
  reindexGraph();
  state.selectedNodeId = null;
  state.selectedEdgeKey = null;
  state.connectStartId = null;
  markGraphDirty(`已删除节点 #${nodeId}。`);
}

function deleteEdge(source, target) {
  const key = edgeKey(source, target);
  state.graph.edges = state.graph.edges.filter(
    (edge) => edgeKey(edge.source, edge.target) !== key
  );
  state.selectedEdgeKey = null;
  markGraphDirty(`已删除边 (${source}, ${target})。`);
}

function reindexGraph() {
  const oldToNew = new Map();
  state.graph.nodes = state.graph.nodes.map((node, index) => {
    oldToNew.set(node.id, index);
    return { ...node, id: index };
  });

  const seen = new Set();
  const nextEdges = [];
  for (const edge of state.graph.edges) {
    if (!oldToNew.has(edge.source) || !oldToNew.has(edge.target)) {
      continue;
    }
    const source = oldToNew.get(edge.source);
    const target = oldToNew.get(edge.target);
    if (source === target) {
      continue;
    }
    const key = edgeKey(source, target);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    nextEdges.push({
      source: Math.min(source, target),
      target: Math.max(source, target),
    });
  }
  state.graph.edges = nextEdges.sort((left, right) => {
    if (left.source !== right.source) {
      return left.source - right.source;
    }
    return left.target - right.target;
  });

  if (state.groundTruthMotif) {
    state.groundTruthMotif = remapMotif(state.groundTruthMotif, oldToNew);
  }
}

function markGraphDirty(message) {
  state.predictionDirty = true;
  state.currentPrediction = null;
  refreshGraphSubtitle();
  renderGraph();
  renderPropertyPanel();
  renderPredictionPanel();
  setStatus(message + " 请点击“重新预测”查看新的置信度。", "success");
}

function renderPropertyPanel() {
  if (!state.graph || !state.graphMeta) {
    refs.propertyPanel.innerHTML = `<p class="empty-state">加载图后，这里会显示节点或边的详细信息。</p>`;
    return;
  }

  const selectedNode = getNodeById(state.selectedNodeId);
  if (selectedNode) {
    renderNodeProperty(selectedNode);
    return;
  }

  const selectedEdge = getSelectedEdge();
  if (selectedEdge) {
    renderEdgeProperty(selectedEdge);
    return;
  }

  renderGraphOverview();
}

function renderGraphOverview() {
  const motifSummary = describeMotif();
  refs.propertyPanel.innerHTML = `
    <div class="stat-grid">
      <div class="stat-card">
        <span>节点数</span>
        <strong>${state.graph.nodes.length}</strong>
      </div>
      <div class="stat-card">
        <span>边数</span>
        <strong>${state.graph.edges.length}</strong>
      </div>
      <div class="stat-card">
        <span>特征维度</span>
        <strong>${state.graphMeta.x_dim}</strong>
      </div>
      <div class="stat-card">
        <span>特征模式</span>
        <strong>${state.graphMeta.feature_mode}</strong>
      </div>
    </div>
    <p class="inline-note">
      当前图名：<strong>${state.graphMeta.name}</strong><br>
      原始标签：<strong>${state.graphMeta.y ?? "unknown"}</strong><br>
      点击某个节点或边后，这里会显示具体属性。
    </p>
    ${motifSummary}
    <div class="pill-list">
      ${(state.graphMeta.feature_labels || []).map((label) => `<span>${label}</span>`).join("")}
    </div>
  `;
}

function renderNodeProperty(node) {
  const vectorText = node.feature.join(", ");
  const atomInfo = describeAtomicNodeFeature(node.feature);
  const featureRows = renderFeatureRows(node.feature);
  const atomCards = atomInfo ? `
      <div class="stat-card">
        <span>Atom</span>
        <strong>${escapeHtml(atomInfo.label)}</strong>
      </div>
      <div class="stat-card">
        <span>atomic_num</span>
        <strong>${atomInfo.atomicNumber}</strong>
      </div>
  ` : "";
  refs.propertyPanel.innerHTML = `
    <div class="stat-grid">
      <div class="stat-card">
        <span>节点 ID</span>
        <strong>#${node.id}</strong>
      </div>
      <div class="stat-card">
        <span>标签</span>
        <strong>${escapeHtml(node.label)}</strong>
      </div>
      <div class="stat-card">
        <span>X</span>
        <strong>${Math.round(node.pos.x)}</strong>
      </div>
      <div class="stat-card">
        <span>Y</span>
        <strong>${Math.round(node.pos.y)}</strong>
      </div>
      ${atomCards}
    </div>
    ${featureRows ? `<div class="feature-list">${featureRows}</div>` : ""}
    <div class="field" id="nodeFeatureField"></div>
    <p class="inline-note">Raw vector: ${escapeHtml(vectorText)}</p>
    <p class="inline-note">原始向量：${escapeHtml(vectorText)}</p>
    <div class="property-actions">
      <button id="deleteSelectedNodeButton" class="ghost-button" type="button">删除该节点</button>
    </div>
  `;

  document
    .getElementById("deleteSelectedNodeButton")
    .addEventListener("click", () => deleteNode(node.id));

  const container = document.getElementById("nodeFeatureField");
  if (state.graphMeta.feature_mode === "onehot") {
    container.innerHTML = `
      <span>节点类型</span>
      <select id="nodeTypeSelect"></select>
    `;
    const select = document.getElementById("nodeTypeSelect");
    const labels = state.graphMeta.feature_labels || [];
    const currentType = argmax(node.feature);
    labels.forEach((label, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = label === String(index) ? label : `${index}: ${label}`;
      if (index === currentType) {
        option.selected = true;
      }
      select.appendChild(option);
    });

    select.addEventListener("change", () => {
      const nextIndex = Number.parseInt(select.value, 10);
      node.feature = node.feature.map((_, index) => (index === nextIndex ? 1 : 0));
      node.label = featureToLabel(node.feature, node.id);
      renderGraph();
      renderPropertyPanel();
      markGraphDirty(`已修改节点 #${node.id} 的类型。`);
    });
  } else {
    if (hasAtomTypeOptions()) {
      container.innerHTML = `
        <span>Atom type</span>
        <select id="nodeAtomTypeSelect"></select>
        <p class="inline-note">BBBP labels are derived from feature[0] = atomic_num.</p>
        <span>Feature vector</span>
        <textarea id="nodeVectorInput">${vectorText}</textarea>
        <button id="applyVectorButton" class="primary-button" type="button">Apply vector update</button>
      `;

      const select = document.getElementById("nodeAtomTypeSelect");
      const currentAtomicNumber = atomInfo?.atomicNumber ?? Math.round(Number(node.feature[0] || 0));
      getAtomTypeOptions().forEach((option) => {
        const optionNode = document.createElement("option");
        optionNode.value = String(option.value);
        optionNode.textContent = `${option.value}: ${option.label}`;
        if (Number(option.value) === currentAtomicNumber) {
          optionNode.selected = true;
        }
        select.appendChild(optionNode);
      });

      select.addEventListener("change", () => {
        const nextAtomicNumber = Number.parseInt(select.value, 10);
        node.feature = node.feature.map((value, index) => (index === 0 ? nextAtomicNumber : value));
        node.label = featureToLabel(node.feature, node.id);
        markGraphDirty(`Updated atom type for node #${node.id}.`);
      });

      document
        .getElementById("applyVectorButton")
        .addEventListener("click", () => applyVectorUpdate(node.id));
      return;
    }
    container.innerHTML = `
      <span>特征向量</span>
      <textarea id="nodeVectorInput">${vectorText}</textarea>
      <button id="applyVectorButton" class="primary-button" type="button">应用向量修改</button>
    `;

    document
      .getElementById("applyVectorButton")
      .addEventListener("click", () => applyVectorUpdate(node.id));
  }
}

function renderEdgeProperty(edge) {
  refs.propertyPanel.innerHTML = `
    <div class="stat-grid">
      <div class="stat-card">
        <span>起点</span>
        <strong>#${edge.source}</strong>
      </div>
      <div class="stat-card">
        <span>终点</span>
        <strong>#${edge.target}</strong>
      </div>
    </div>
    <p class="inline-note">
      当前边按照无向边处理，预测时后端会自动展开成双向 <code>edge_index</code>。
    </p>
    <div class="property-actions">
      <button id="deleteSelectedEdgeButton" class="ghost-button" type="button">删除该边</button>
    </div>
  `;

  document
    .getElementById("deleteSelectedEdgeButton")
    .addEventListener("click", () => deleteEdge(edge.source, edge.target));
}

function applyVectorUpdate(nodeId) {
  const textarea = document.getElementById("nodeVectorInput");
  const raw = textarea.value.trim();
  const parts = raw.split(",").map((item) => item.trim()).filter(Boolean);
  if (parts.length !== state.graphMeta.x_dim) {
    setStatus(
      `向量长度必须等于 ${state.graphMeta.x_dim}，当前得到 ${parts.length}。`,
      "error"
    );
    return;
  }

  const values = [];
  for (const part of parts) {
    const parsed = Number.parseFloat(part);
    if (!Number.isFinite(parsed)) {
      setStatus(`无法解析特征值 '${part}'。`, "error");
      return;
    }
    values.push(Number(parsed.toFixed(6)));
  }

  const node = getNodeById(nodeId);
  if (!node) {
    return;
  }
  node.feature = values;
  node.label = featureToLabel(values, node.id);
  markGraphDirty(`已修改节点 #${node.id} 的特征向量。`);
}

function renderPredictionPanel() {
  refs.predictionPanel.innerHTML = "";

  const labelBlock = document.createElement("div");
  labelBlock.className = "prediction-block";
  labelBlock.innerHTML = `
    <h3>原始标签</h3>
    <p>原始标签：<strong>${escapeHtml(formatOriginalLabel())}</strong></p>
  `;
  refs.predictionPanel.appendChild(labelBlock);

  const currentBlock = renderPredictionBlock(
    "当前图概率",
    state.currentPrediction,
    state.predictionDirty ? "图已修改，等待重新预测。" : null
  );
  refs.predictionPanel.appendChild(currentBlock);

  const originalBlock = renderPredictionBlock(
    "原始图概率",
    state.originalPrediction,
    state.graphMeta?.model_available ? null : (state.graphMeta?.model_error || "当前数据集没有可用权重。")
  );
  refs.predictionPanel.appendChild(originalBlock);

  if (state.currentPrediction && state.originalPrediction) {
    const deltaBlock = document.createElement("div");
    deltaBlock.className = "prediction-block";
    const deltas = state.currentPrediction.probabilities.map((value, index) => {
      const delta = Number((value - state.originalPrediction.probabilities[index]).toFixed(6));
      const tone = delta > 0 ? "up" : delta < 0 ? "down" : "flat";
      const sign = delta > 0 ? "+" : "";
      return `
        <div class="prob-row">
          <span>Class ${index}</span>
          <div class="prob-bar"><div class="prob-fill" style="width:${Math.min(Math.abs(delta) * 100, 100)}%"></div></div>
          <span class="delta-chip ${tone}">${sign}${(delta * 100).toFixed(2)}%</span>
        </div>
      `;
    }).join("");

    deltaBlock.innerHTML = `
      <h3>概率变化</h3>
      <p>和原始图相比的每一类概率差值。</p>
      ${deltas}
    `;
    refs.predictionPanel.appendChild(deltaBlock);
  }

  refs.predictionHint.textContent = state.currentPrediction
    ? `Pred Class ${state.currentPrediction.predicted_class}`
    : state.graphMeta?.model_available ? "Pending" : "Model missing";
}

function renderPredictionBlock(title, prediction, note) {
  const block = document.createElement("div");
  block.className = "prediction-block";

  if (!prediction) {
    block.innerHTML = `
      <h3>${title}</h3>
      <p>${note || "当前还没有可用预测结果。"}</p>
    `;
    return block;
  }

  const rows = prediction.probabilities.map((probability, index) => `
    <div class="prob-row">
      <span>Class ${index}</span>
      <div class="prob-bar">
        <div class="prob-fill" style="width:${Math.max(0, Math.min(probability * 100, 100))}%"></div>
      </div>
      <strong>${(probability * 100).toFixed(2)}%</strong>
    </div>
  `).join("");

  block.innerHTML = `
    <h3>${title}</h3>
    <p>${note || `当前预测类别：${prediction.predicted_class}`}</p>
    ${rows}
  `;
  return block;
}

function formatOriginalLabel() {
  const label = state.graphMeta?.y;
  return label === null || label === undefined ? "unknown" : String(label);
}

async function predictCurrentGraph() {
  if (!state.graphMeta?.model_available) {
    setStatus(state.graphMeta?.model_error || "当前数据集没有可用模型。", "warning");
    return;
  }

  const payload = {
    dataset: state.dataset,
    split: state.split,
    source_index: state.index,
    nodes: deepClone(state.graph.nodes),
    edges: deepClone(state.graph.edges),
  };

  setStatus("正在对编辑后的图重新预测...", "info");
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok) {
    setStatus(result.error || "预测失败。", "error");
    return;
  }

  if (result.normalized_graph) {
    state.graph = deepClone(result.normalized_graph);
    state.selectedNodeId = null;
    state.selectedEdgeKey = null;
    state.connectStartId = null;
  }
  state.currentPrediction = result.current_prediction;
  state.predictionDirty = false;
  refreshGraphSubtitle();
  renderGraph();
  renderPropertyPanel();
  renderPredictionPanel();
  setStatus("重预测完成。你可以继续修改图并再次观察概率变化。", "success");
}

function resetGraph() {
  if (!state.originalGraph) {
    return;
  }

  state.graph = deepClone(state.originalGraph);
  state.groundTruthMotif = deepClone(state.originalGroundTruthMotif);
  state.currentPrediction = state.originalPrediction ? deepClone(state.originalPrediction) : null;
  state.predictionDirty = false;
  state.selectedNodeId = null;
  state.selectedEdgeKey = null;
  state.connectStartId = null;
  refreshGraphSubtitle();
  renderGraph();
  renderPropertyPanel();
  renderPredictionPanel();
  setStatus("已经恢复到原始图。", "success");
}

function refreshGraphSubtitle() {
  if (!state.graph || !state.graphMeta) {
    refs.graphSubtitle.textContent = "请选择一个图开始编辑。";
    return;
  }

  const editedTag = state.predictionDirty ? " · Edited" : "";
  refs.graphSubtitle.textContent =
    `${state.dataset}/${state.split} · 图 ${state.index} · `
    + `${state.graph.nodes.length} 节点 / ${state.graph.edges.length} 边${editedTag}`;
}

function getCurrentDatasetInfo() {
  return state.datasetMap.get(state.dataset) || null;
}

function getNodeById(nodeId) {
  return state.graph?.nodes.find((node) => node.id === nodeId) || null;
}

function getSelectedEdge() {
  if (!state.selectedEdgeKey || !state.graph) {
    return null;
  }
  return state.graph.edges.find(
    (edge) => edgeKey(edge.source, edge.target) === state.selectedEdgeKey
  ) || null;
}

function hasEdge(source, target) {
  const key = edgeKey(source, target);
  return state.graph.edges.some((edge) => edgeKey(edge.source, edge.target) === key);
}

function edgeKey(source, target) {
  return `${Math.min(source, target)}:${Math.max(source, target)}`;
}

function featureToLabel(feature, nodeId) {
  if (state.graphMeta?.feature_mode === "onehot") {
    const labels = state.graphMeta.feature_labels || [];
    const idx = argmax(feature);
    return labels[idx] || String(idx);
  }
  if (state.graphMeta?.node_label_mode === "atomic_num") {
    const atomicNumber = Array.isArray(feature) && feature.length > 0
      ? Math.round(Number(feature[0]))
      : Number.NaN;
    const atomLabel = atomTypeLabelFromValue(atomicNumber);
    if (atomLabel) {
      return atomLabel;
    }
    if (Number.isFinite(atomicNumber)) {
      return `Z=${atomicNumber}`;
    }
  }
  return `n${nodeId}`;
}

function getAtomTypeOptions() {
  return Array.isArray(state.graphMeta?.atom_type_options) ? state.graphMeta.atom_type_options : [];
}

function hasAtomTypeOptions() {
  return state.graphMeta?.node_label_mode === "atomic_num" && getAtomTypeOptions().length > 0;
}

function atomTypeLabelFromValue(value) {
  const atomicNumber = Math.round(Number(value));
  if (!Number.isFinite(atomicNumber)) {
    return "";
  }
  const matched = getAtomTypeOptions().find((option) => Number(option.value) === atomicNumber);
  return matched ? String(matched.label) : "";
}

function describeAtomicNodeFeature(feature) {
  if (!hasAtomTypeOptions() || !Array.isArray(feature) || feature.length === 0) {
    return null;
  }

  const atomicNumber = Math.round(Number(feature[0]));
  if (!Number.isFinite(atomicNumber)) {
    return null;
  }

  return {
    atomicNumber,
    label: atomTypeLabelFromValue(atomicNumber) || `Z=${atomicNumber}`,
  };
}

function renderFeatureRows(feature) {
  if (!Array.isArray(feature) || feature.length === 0) {
    return "";
  }

  const labels = Array.isArray(state.graphMeta?.feature_labels) ? state.graphMeta.feature_labels : [];
  return feature.map((value, index) => `
    <div class="feature-row">
      <span>${escapeHtml(labels[index] || `f${index}`)}</span>
      <strong>${escapeHtml(formatFeatureValue(value))}</strong>
    </div>
  `).join("");
}

function formatFeatureValue(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return String(value);
  }
  if (Number.isInteger(numeric)) {
    return String(numeric);
  }
  return numeric.toFixed(6).replace(/\.?0+$/, "");
}

function svgPoint(event) {
  const rect = refs.graphCanvas.getBoundingClientRect();
  const viewBox = refs.graphCanvas.viewBox.baseVal;
  const x = ((event.clientX - rect.left) / rect.width) * viewBox.width + viewBox.x;
  const y = ((event.clientY - rect.top) / rect.height) * viewBox.height + viewBox.y;
  return { x, y };
}

function createSvg(tagName, attributes = {}) {
  const element = document.createElementNS(SVG_NS, tagName);
  for (const [name, value] of Object.entries(attributes)) {
    element.setAttribute(name, String(value));
  }
  return element;
}

function deepClone(value) {
  return JSON.parse(JSON.stringify(value));
}

function isMotifNode(nodeId) {
  return Boolean(
    state.groundTruthMotif?.available
      && Array.isArray(state.groundTruthMotif.node_ids)
      && state.groundTruthMotif.node_ids.includes(nodeId)
  );
}

function isMotifEdge(source, target) {
  if (!state.groundTruthMotif?.available || !Array.isArray(state.groundTruthMotif.edges)) {
    return false;
  }
  const key = edgeKey(source, target);
  return state.groundTruthMotif.edges.some(
    (edge) => edgeKey(edge.source, edge.target) === key
  );
}

function remapMotif(motif, oldToNew) {
  if (!motif) {
    return motif;
  }

  const remappedNodeIds = (motif.node_ids || [])
    .filter((nodeId) => oldToNew.has(nodeId))
    .map((nodeId) => oldToNew.get(nodeId));

  const seen = new Set();
  const remappedEdges = [];
  for (const edge of motif.edges || []) {
    if (!oldToNew.has(edge.source) || !oldToNew.has(edge.target)) {
      continue;
    }
    const source = oldToNew.get(edge.source);
    const target = oldToNew.get(edge.target);
    if (source === target) {
      continue;
    }
    const key = edgeKey(source, target);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    remappedEdges.push({
      source: Math.min(source, target),
      target: Math.max(source, target),
    });
  }

  return {
    ...motif,
    available: Boolean(motif.available && (remappedNodeIds.length > 0 || remappedEdges.length > 0)),
    node_ids: remappedNodeIds,
    edges: remappedEdges,
  };
}

function describeMotif() {
  const motif = state.groundTruthMotif;
  if (!motif) {
    return "";
  }

  if (motif.available) {
    return `
      <div class="prediction-block">
        <h3>正类 GT Motif</h3>
        <p>${escapeHtml(motif.description || "当前图显示了正类 ground-truth motif 高亮。")}</p>
        <div class="pill-list">
          <span>节点 ${(motif.node_ids || []).length}</span>
          <span>边 ${(motif.edges || []).length}</span>
        </div>
      </div>
    `;
  }

  if (motif.reason) {
    return `
      <div class="prediction-block">
        <h3>正类 GT Motif</h3>
        <p>${escapeHtml(motif.reason)}</p>
      </div>
    `;
  }

  return "";
}

function runAsync(task) {
  Promise.resolve()
    .then(() => task())
    .catch((error) => {
      console.error(error);
      setStatus(error.message || "发生了未预期的错误。", "error");
    });
}

function setStatus(message, kind = "info") {
  refs.statusBox.textContent = message;
  refs.statusBox.className = `status-box is-${kind}`;
}

function argmax(values) {
  if (!values || values.length === 0) {
    return 0;
  }
  let bestIndex = 0;
  let bestValue = values[0];
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] > bestValue) {
      bestValue = values[index];
      bestIndex = index;
    }
  }
  return bestIndex;
}

function clamp(value, minValue, maxValue) {
  return Math.max(minValue, Math.min(value, maxValue));
}

function trimLabel(label) {
  const raw = String(label || "");
  return raw.length > 5 ? `${raw.slice(0, 5)}…` : raw;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
