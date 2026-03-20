const state = {
  manifest: null,
  task: null,
  answers: {},
  answerContext: {},
  openedAt: Date.now(),
  selectedTestType: "single_view",
  viewHistory: [],
};

function syncViewportOffsets() {
  const topbar = document.querySelector(".topbar");
  if (!topbar) return;
  const height = Math.ceil(topbar.getBoundingClientRect().height);
  document.documentElement.style.setProperty("--topbar-height", `${height}px`);
}

function slugify(text) {
  return (text || "")
    .trim()
    .replace(/[^0-9A-Za-z._-]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "") || "anonymous";
}

function parseFdr(value) {
  const trimmed = (value || "").trim();
  if (!trimmed) return null;
  if (trimmed.startsWith("[")) {
    try {
      const parsed = JSON.parse(trimmed);
      if (Array.isArray(parsed)) {
        return parsed.map((item) => String(item).trim()).filter(Boolean);
      }
    } catch (error) {
      // fall through
    }
  }
  return trimmed.split(/[,\n]+/).map((part) => part.trim()).filter(Boolean);
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`${response.status} ${body}`);
  }
  return response.json();
}

function setStatus(message) {
  document.getElementById("status").textContent = message || "";
  syncViewportOffsets();
}

function nowIso() {
  return new Date().toISOString();
}

function testTypeLabel(testType) {
  return testType === "multi_view" ? "多视角" : "单视图";
}

function currentViewDescriptor() {
  const images = currentImages();
  if (state.selectedTestType === "multi_view") {
    return {
      view_mode: state.selectedTestType,
      selected_view_id: "all_views",
      selected_view_label: "All Views",
      visible_view_ids: images.map((image) => image.view_id),
    };
  }
  const selected = images[0] || null;
  return {
    view_mode: state.selectedTestType,
    selected_view_id: selected?.view_id || null,
    selected_view_label: selected?.label || null,
    visible_view_ids: selected?.view_id ? [selected.view_id] : [],
  };
}

function appendViewHistory(eventType) {
  const descriptor = currentViewDescriptor();
  const last = state.viewHistory[state.viewHistory.length - 1];
  const next = {
    event: eventType,
    at: nowIso(),
    ...descriptor,
  };
  if (
    last &&
    last.event === next.event &&
    last.view_mode === next.view_mode &&
    last.selected_view_id === next.selected_view_id
  ) {
    return;
  }
  state.viewHistory.push(next);
}

function summarizeResponseCondition() {
  const contexts = Object.values(state.answerContext);
  const answered = contexts.filter((row) => row && row.view_mode);
  if (!answered.length) {
    return {
      response_condition: state.selectedTestType,
      modes_used: [state.selectedTestType],
      views_used: currentViewDescriptor().visible_view_ids || [],
    };
  }

  const viewSet = new Set(
    answered.flatMap((row) => row.visible_view_ids || []).filter(Boolean)
  );
  return {
    response_condition: state.selectedTestType,
    modes_used: [state.selectedTestType],
    views_used: [...viewSet],
  };
}

function renderConditionBadge() {
  const badge = document.getElementById("condition-badge");
  if (!badge) return;
  const summary = summarizeResponseCondition();
  const usedViews = summary.views_used.length ? summary.views_used.join(", ") : "none";
  badge.textContent = `当前测试: ${testTypeLabel(summary.response_condition)} · visible: ${usedViews}`;
}

function updateAnswerCount() {
  if (!state.task) {
    document.getElementById("answer-count").textContent = "";
    renderControlSummary();
    return;
  }
  const answered = Object.values(state.answers).filter((value) => {
    if (value === null || value === undefined) return false;
    return String(value).trim() !== "";
  }).length;
  document.getElementById("answer-count").textContent =
    `已填写 ${answered} / ${state.task.summary.total_questions}`;
  renderConditionBadge();
  renderControlSummary();
}

function renderSummary(task) {
  const grid = document.getElementById("summary-grid");
  grid.innerHTML = "";
  const cards = [
    ["场景", task.scene_id],
    ["物体数", task.n_objects],
    ["题目数", task.summary.total_questions],
    ["QRR / TRR / FDR", `${task.summary.qrr_questions} / ${task.summary.trr_questions} / ${task.summary.fdr_questions}`],
  ];
  for (const [label, value] of cards) {
    const card = document.createElement("div");
    card.className = "summary-card";
    card.innerHTML = `<strong>${label}</strong><span>${value}</span>`;
    grid.appendChild(card);
  }
}

function filteredScenes() {
  if (!state.manifest?.scenes) return [];
  return state.manifest.scenes.filter((scene) =>
    (scene.available_test_types || ["single_view"]).includes(state.selectedTestType)
  );
}

function renderSceneFilterSummary(scenes) {
  const root = document.getElementById("scene-filter-summary");
  if (!root) return;
  root.textContent =
    `${testTypeLabel(state.selectedTestType)} 下可选 ${scenes.length} 个场景。先定测试条件，再选场景，答题过程不再切换测试类型。`;
  syncViewportOffsets();
}

function renderControlSummary() {
  const summary = document.getElementById("control-summary");
  if (!summary) return;
  const sceneId = state.task?.scene_id || document.getElementById("scene-select")?.value || "未选场景";
  const total = state.task?.summary?.total_questions || 0;
  const answered = Object.values(state.answers).filter((value) => {
    if (value === null || value === undefined) return false;
    return String(value).trim() !== "";
  }).length;
  const progress = total ? `${answered}/${total}` : "未开始";
  summary.textContent = `测试设置 · ${testTypeLabel(state.selectedTestType)} · ${sceneId} · ${progress}`;
}

function currentImages() {
  if (!state.task || !state.task.images) return [];
  if (state.selectedTestType === "multi_view" && state.task.images.multi_view?.length) {
    return state.task.images.multi_view;
  }
  return state.task.images.single_view ? [state.task.images.single_view] : [];
}

function renderViewToolbar(task) {
  const root = document.getElementById("view-toolbar");
  root.innerHTML = "";

  const chip = document.createElement("div");
  chip.className = "view-chip";
  const multiCount = task.images?.multi_view?.length || 0;
  chip.textContent = state.selectedTestType === "multi_view"
    ? `${testTypeLabel(state.selectedTestType)} · ${multiCount} views`
    : testTypeLabel(state.selectedTestType);
  root.appendChild(chip);

  const caption = document.createElement("div");
  caption.id = "view-caption";
  caption.className = "view-caption";
  root.appendChild(caption);
}

function renderImageViewer(task) {
  const caption = document.getElementById("view-caption");
  const grid = document.getElementById("image-grid");
  const images = currentImages();

  grid.innerHTML = "";
  if (!images.length) {
    if (caption) caption.textContent = "No image";
    return;
  }

  if (caption) {
    if (state.selectedTestType === "multi_view") {
      if (task.images && task.images.multi_view_labels_reliable) {
        caption.textContent = "多视角测试时 4 张图同时展示。";
      }
    } else {
      caption.textContent = "单视角测试时只展示 1 张图。";
    }
  }

  for (const view of images) {
    const card = document.createElement("figure");
    card.className = `image-card ${state.selectedTestType === "multi_view" ? "multi" : "single"}`;
    const cameraText = view.camera
      ? `${view.camera.azimuth}° / ${view.camera.elevation}°`
      : "single image";
    card.innerHTML = `
      <div class="image-frame">
        <span class="image-badge top-left">${view.label}</span>
        <span class="image-badge bottom-left">${cameraText}</span>
        <img src="/tasks/${view.image_path}" alt="${view.label}">
      </div>
    `;
    grid.appendChild(card);
  }
}

function renderObjects(task) {
  const tbody = document.getElementById("object-body");
  tbody.innerHTML = "";
  for (const obj of task.objects) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td><code>${obj.id}</code></td><td>${obj.desc}</td>`;
    tbody.appendChild(tr);
  }
}

function renderQuestions(task) {
  const root = document.getElementById("question-root");
  root.innerHTML = "";
  if (!task.batches.length) {
    root.innerHTML = '<p class="empty">No questions found.</p>';
    return;
  }

  for (const batch of task.batches) {
    const section = document.createElement("section");
    section.className = "batch";
    section.innerHTML = `<h3>${batch.question_type.toUpperCase()} · ${batch.batch_id}</h3>`;

    for (const question of batch.questions) {
      const wrapper = document.createElement("div");
      wrapper.className = "question";

      const qid = document.createElement("div");
      qid.className = "qid";
      qid.textContent = question.qid;
      wrapper.appendChild(qid);

      const prompt = document.createElement("p");
      prompt.className = "prompt";
      prompt.textContent = question.prompt_text;
      wrapper.appendChild(prompt);

      if (question.type === "qrr" || question.type === "trr") {
        const select = document.createElement("select");
        select.innerHTML = '<option value="">请选择</option>';
        for (const optionValue of question.answer_options) {
          const option = document.createElement("option");
          option.value = optionValue;
          option.textContent = optionValue;
          select.appendChild(option);
        }
        select.value = state.answers[question.qid] || "";
        select.addEventListener("change", (event) => {
          state.answers[question.qid] = event.target.value;
          state.answerContext[question.qid] = {
            ...currentViewDescriptor(),
            answered_at: nowIso(),
          };
          updateAnswerCount();
        });
        wrapper.appendChild(select);
      } else if (question.type === "fdr") {
        const input = document.createElement("input");
        input.type = "text";
        input.placeholder = question.answer_example || "";
        input.value = state.answers[question.qid] || "";
        input.addEventListener("input", (event) => {
          state.answers[question.qid] = event.target.value;
          state.answerContext[question.qid] = {
            ...currentViewDescriptor(),
            answered_at: nowIso(),
          };
          updateAnswerCount();
        });
        wrapper.appendChild(input);

        const help = document.createElement("p");
        help.className = "help";
        help.textContent = "题干里使用中文描述；作答时仍请按参考图或物体表中的编号输入，按从近到远填写，逗号分隔，也可以直接输入 JSON list。";
        wrapper.appendChild(help);
      }

      section.appendChild(wrapper);
    }

    root.appendChild(section);
  }
}

function collectPayload() {
  const annotatorId = slugify(document.getElementById("annotator-input").value);
  const elapsedSeconds = Math.round((Date.now() - state.openedAt) / 10) / 100;
  const batches = [];
  const responses = [];
  const conditionSummary = summarizeResponseCondition();

  for (const batch of state.task.batches) {
    const batchResponses = [];
    for (const question of batch.questions) {
      const rawValue = state.answers[question.qid];
      let answer = null;
      if (question.type === "qrr") {
        answer = rawValue || null;
      } else if (question.type === "trr") {
        answer = rawValue ? Number(rawValue) : null;
      } else if (question.type === "fdr") {
        answer = parseFdr(rawValue);
      }
      if (answer !== null && answer !== "" && !(Array.isArray(answer) && answer.length === 0)) {
        const row = {
          qid: question.qid,
          answer,
          ...(state.answerContext[question.qid] || {}),
        };
        batchResponses.push(row);
        responses.push(row);
      }
    }
    batches.push({
      batch_id: batch.batch_id,
      question_type: batch.question_type,
      elapsed_seconds: elapsedSeconds,
      responses: batchResponses,
      raw_response: JSON.stringify(batchResponses),
    });
  }

  return {
    schema_version: 1,
    scene_id: state.task.scene_id,
    annotator_id: annotatorId,
    model: `human/${annotatorId}`,
    submitted_at: new Date().toISOString(),
    n_objects: state.task.n_objects,
    total_questions: state.task.summary.total_questions,
    response_condition: conditionSummary.response_condition,
    response_condition_detail: conditionSummary,
    selected_test_type: state.selectedTestType,
    view_mode_history: state.viewHistory,
    batches,
    responses,
    raw_response: JSON.stringify(responses),
  };
}

function downloadText(filename, text) {
  const blob = new Blob([text], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

async function saveToBackend() {
  const payload = collectPayload();
  const missing = state.task.summary.total_questions - payload.responses.length;
  if (missing > 0 && !window.confirm(`还有 ${missing} 题未填写，仍然保存？`)) {
    return;
  }

  const result = await fetchJson("/api/responses", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  setStatus(`已保存到 ${result.saved_to}`);
}

function downloadJson() {
  const payload = collectPayload();
  const filename = `${payload.scene_id}__${payload.annotator_id}.json`;
  downloadText(filename, JSON.stringify(payload, null, 2));
  setStatus(`已下载 ${filename}`);
}

function downloadRaw() {
  const payload = collectPayload();
  const filename = `${payload.scene_id}__${payload.annotator_id}__raw_response.json`;
  downloadText(filename, payload.raw_response);
  setStatus(`已下载 ${filename}`);
}

async function loadTask(sceneId) {
  state.task = await fetchJson(`/api/tasks/${sceneId}`);
  state.answers = {};
  state.answerContext = {};
  state.openedAt = Date.now();
  if (
    state.selectedTestType === "multi_view" &&
    (!state.task.images?.multi_view || !state.task.images.multi_view.length)
  ) {
    throw new Error(`${sceneId} does not have multi view images`);
  }
  state.viewHistory = [];
  appendViewHistory("load_task");
  renderControlSummary();
  renderViewToolbar(state.task);
  renderImageViewer(state.task);
  renderSummary(state.task);
  renderObjects(state.task);
  renderQuestions(state.task);
  updateAnswerCount();
  syncViewportOffsets();
  setStatus(`已加载 ${sceneId}`);
}

function populateSceneSelect(preferredSceneId = null) {
  const scenes = filteredScenes();
  const select = document.getElementById("scene-select");
  select.innerHTML = "";

  for (const scene of scenes) {
    const option = document.createElement("option");
    option.value = scene.scene_id;
    option.textContent = `${scene.scene_id} · ${scene.total_questions} questions`;
    select.appendChild(option);
  }

  renderSceneFilterSummary(scenes);

  if (!scenes.length) {
    state.task = null;
    document.getElementById("image-grid").innerHTML = "";
    document.getElementById("question-root").innerHTML = '<p class="empty">当前测试类型下没有场景。</p>';
    document.getElementById("object-body").innerHTML = "";
    document.getElementById("summary-grid").innerHTML = "";
    document.getElementById("view-toolbar").innerHTML = "";
    updateAnswerCount();
    renderControlSummary();
    setStatus(`没有可用于 ${testTypeLabel(state.selectedTestType)} 的场景`);
    return null;
  }

  const resolvedSceneId = scenes.some((scene) => scene.scene_id === preferredSceneId)
    ? preferredSceneId
    : scenes[0].scene_id;
  select.value = resolvedSceneId;
  return resolvedSceneId;
}

async function init() {
  try {
    state.manifest = await fetchJson("/api/manifest");
  } catch (error) {
    setStatus(`加载 manifest 失败: ${error.message}`);
    return;
  }

  const availableTypes = new Set();
  for (const scene of state.manifest.scenes) {
    for (const testType of scene.available_test_types || ["single_view"]) {
      availableTypes.add(testType);
    }
  }

  const testTypeSelect = document.getElementById("test-type-select");
  testTypeSelect.innerHTML = "";
  for (const testType of ["single_view", "multi_view"]) {
    if (!availableTypes.has(testType)) continue;
    const option = document.createElement("option");
    option.value = testType;
    option.textContent = testTypeLabel(testType);
    testTypeSelect.appendChild(option);
  }
  state.selectedTestType = testTypeSelect.value || "single_view";

  const select = document.getElementById("scene-select");
  testTypeSelect.addEventListener("change", async (event) => {
    state.selectedTestType = event.target.value;
    const nextSceneId = populateSceneSelect(select.value);
    renderConditionBadge();
    renderControlSummary();
    if (nextSceneId) {
      await loadTask(nextSceneId).catch((error) => setStatus(error.message));
    }
  });

  select.addEventListener("change", (event) => {
    renderControlSummary();
    loadTask(event.target.value).catch((error) => setStatus(error.message));
  });

  document.getElementById("save-btn").addEventListener("click", () => {
    saveToBackend().catch((error) => setStatus(`保存失败: ${error.message}`));
  });
  document.getElementById("download-btn").addEventListener("click", downloadJson);
  document.getElementById("raw-btn").addEventListener("click", downloadRaw);

  const initialSceneId = populateSceneSelect();
  renderConditionBadge();
  if (initialSceneId) {
    await loadTask(initialSceneId);
  } else {
    setStatus("manifest 中没有场景");
  }
}

function installLayoutObserver() {
  const topbar = document.querySelector(".topbar");
  if (!topbar) return;
  syncViewportOffsets();
  if ("ResizeObserver" in window) {
    const observer = new ResizeObserver(() => syncViewportOffsets());
    observer.observe(topbar);
  }
}

window.addEventListener("DOMContentLoaded", async () => {
  installLayoutObserver();
  window.addEventListener("resize", syncViewportOffsets);
  await init();
  syncViewportOffsets();
});
