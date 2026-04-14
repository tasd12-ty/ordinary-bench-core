/* ================================================================
   ORDINARY-BENCH Human Baseline v2
   ================================================================ */

const MODE_LABELS = {
  progressive: "递进式测试",
  adaptive_sort: "Adaptive Sort",
};

const state = {
  view: "welcome",
  capabilities: null,
  testMode: "progressive",
  annotatorId: null,
  roundData: null,
  answers: {},
  gradingResult: null,
  stepSummary: null,
  nextAction: null,
  progress: null,
  pageStartTime: null,
  timerInterval: null,
  focusedQuestionIndex: -1,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);
const show = (el) => {
  if (typeof el === "string") el = $(el);
  if (el) el.classList.remove("hidden");
};
const hide = (el) => {
  if (typeof el === "string") el = $(el);
  if (el) el.classList.add("hidden");
};

let toastTimeout = null;
let trrKeyBuffer = "";
let trrKeyTimer = null;

function esc(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function showToast(msg, isError) {
  const toast = $("#toast");
  toast.textContent = msg;
  toast.classList.toggle("error", !!isError);
  toast.classList.add("visible");
  if (toastTimeout) clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => toast.classList.remove("visible"), 3500);
}

async function fetchJson(url, opts) {
  try {
    const resp = await fetch(url, opts);
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
    return data;
  } catch (err) {
    showToast(err.message || "网络请求失败", true);
    throw err;
  }
}

function postJson(url, body) {
  return fetchJson(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function startTimer() {
  state.pageStartTime = new Date();
  stopTimer();
  updateTimerDisplay();
  state.timerInterval = setInterval(updateTimerDisplay, 1000);
}

function stopTimer() {
  if (state.timerInterval) {
    clearInterval(state.timerInterval);
    state.timerInterval = null;
  }
}

function getElapsedSeconds() {
  if (!state.pageStartTime) return 0;
  return (Date.now() - state.pageStartTime.getTime()) / 1000;
}

function formatTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function updateTimerDisplay() {
  const el = $("#topbar-timer");
  if (el) el.textContent = formatTime(getElapsedSeconds());
}

function switchView(name) {
  state.view = name;
  ["welcome", "quiz", "grading", "stepdone", "done"].forEach((view) => {
    const el = $(`#view-${view}`);
    if (el) el.classList.toggle("hidden", view !== name);
  });
  if (name === "quiz") startTimer();
  else stopTimer();
}

function resolveImageUrl(path) {
  if (!path) return "";
  if (/^https?:\/\//.test(path)) return path;
  if (path.startsWith("tasks/")) return "/" + path;
  return "/data-images/" + path;
}

function openLightbox(src) {
  $("#lightbox-img").src = src;
  show("#lightbox");
}

function closeLightbox() {
  hide("#lightbox");
}

function selectedMode() {
  return document.querySelector('input[name="test-mode"]:checked')?.value || "progressive";
}

function modeMeta(mode) {
  return state.capabilities?.modes?.[mode] || {
    id: mode,
    label: MODE_LABELS[mode] || mode,
    configured: mode === "progressive",
  };
}

function isModeAvailable(mode) {
  return !!modeMeta(mode).configured;
}

function applyCapabilities(payload) {
  state.capabilities = payload;
  ["progressive", "adaptive_sort"].forEach((mode) => {
    const input = document.querySelector(`input[name="test-mode"][value="${mode}"]`);
    const card = document.querySelector(`[data-mode-card="${mode}"]`);
    const status = document.querySelector(`[data-mode-status="${mode}"]`);
    const meta = modeMeta(mode);
    const configured = !!meta.configured;

    if (input) input.disabled = !configured;
    if (card) card.classList.toggle("disabled", !configured);
    if (status) {
      status.textContent = configured
        ? mode === "adaptive_sort"
          ? `已接入 ${meta.n_scenes || 0} 个测试场景`
          : "已配置"
        : "当前未配置任务数据";
    }
  });

  if (!isModeAvailable(selectedMode())) {
    const fallback = document.querySelector('input[name="test-mode"][value="progressive"]');
    if (fallback) fallback.checked = true;
    state.testMode = "progressive";
  }
  updateStartButton();
}

async function loadCapabilities() {
  try {
    const payload = await fetchJson("/api/v2/capabilities");
    applyCapabilities(payload);
  } catch (_) {
    applyCapabilities({
      default_mode: "progressive",
      modes: {
        progressive: { id: "progressive", label: MODE_LABELS.progressive, configured: true },
        adaptive_sort: { id: "adaptive_sort", label: MODE_LABELS.adaptive_sort, configured: false },
      },
    });
  }
}

function updateStartButton() {
  const input = $("#input-annotator");
  const btn = $("#btn-start");
  if (!input || !btn) return;
  const hasId = !!input.value.trim();
  const mode = selectedMode();
  btn.disabled = !hasId || !isModeAvailable(mode);
}

function initWelcome() {
  const input = $("#input-annotator");
  const btn = $("#btn-start");

  input.addEventListener("input", updateStartButton);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !btn.disabled) btn.click();
  });
  $$('input[name="test-mode"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      state.testMode = selectedMode();
      updateStartButton();
    });
  });
  btn.addEventListener("click", handleStart);
}

async function handleStart() {
  const id = $("#input-annotator").value.trim();
  if (!id) return;

  const testMode = selectedMode();
  state.annotatorId = id;
  state.testMode = testMode;
  state.gradingResult = null;
  state.stepSummary = null;
  state.nextAction = null;

  const btn = $("#btn-start");
  const errEl = $("#welcome-error");
  btn.disabled = true;
  errEl.textContent = "";

  try {
    const result = await postJson("/api/v2/session/start", {
      annotator_id: id,
      test_mode: testMode,
    });
    state.progress = result.progress;

    if (result.has_current_scene) await loadCurrentRound();
    else await allocateScene();
  } catch (err) {
    errEl.textContent = err.message || "启动失败";
    updateStartButton();
  }
}

async function loadCurrentRound() {
  const url = `/api/v2/scene/current?annotator_id=${encodeURIComponent(state.annotatorId)}&test_mode=${encodeURIComponent(state.testMode)}`;
  const data = await fetchJson(url);

  if (data.done) {
    renderDone();
    switchView("done");
    return;
  }
  if (data.needs_allocation) {
    await allocateScene();
    return;
  }

  state.roundData = data;
  state.gradingResult = null;
  state.stepSummary = null;
  state.nextAction = null;
  state.answers = {};
  state.focusedQuestionIndex = -1;

  if (state.testMode === "adaptive_sort") loadAdaptiveDraft();

  renderQuiz();
  switchView("quiz");
}

async function allocateScene() {
  const data = await postJson("/api/v2/scene/allocate", {
    annotator_id: state.annotatorId,
    test_mode: state.testMode,
  });

  if (data.done) {
    renderDone();
    switchView("done");
    return;
  }

  state.roundData = data;
  state.gradingResult = null;
  state.stepSummary = null;
  state.nextAction = null;
  state.answers = {};
  state.focusedQuestionIndex = -1;

  if (state.testMode === "adaptive_sort") loadAdaptiveDraft();

  renderQuiz();
  switchView("quiz");
}

function renderQuiz() {
  const rd = state.roundData;
  if (!rd) return;

  $("#topbar-mode").textContent = MODE_LABELS[state.testMode] || state.testMode;
  $("#topbar-scene").textContent = rd.scene_id;
  $("#topbar-round").textContent = rd.round_label;
  $("#topbar-progress").textContent = buildTopbarProgress();

  renderImages();
  renderObjects();

  if (state.testMode === "adaptive_sort") renderAdaptiveQuiz();
  else renderProgressiveQuiz();
}

function buildTopbarProgress() {
  const p = state.progress || {};
  if (state.testMode === "adaptive_sort" && state.roundData) {
    return `场景 ${p.scenes_completed || 0}/${p.total_scenes || 0} · Step ${state.roundData.step_index || 0}/${state.roundData.n_steps_total || 0}`;
  }
  return `已完成 ${p.scenes_completed || 0}/${p.total_scenes || 0} 场景`;
}

function renderImages() {
  const rd = state.roundData || {};
  const images = rd.images || {};
  const area = $("#image-area");
  area.innerHTML = "";

  if (state.testMode === "progressive" && rd.round_number) {
    if (rd.round_number === 1) {
      area.className = "image-area";
      if (images.single_view) area.appendChild(makeImage(images.single_view, "单视角"));
    } else if (rd.round_number === 2) {
      area.className = "image-area grid-2x2";
      (images.multi_view || []).forEach((path, index) => {
        area.appendChild(makeImage(path, `视角 ${index}`));
      });
    } else {
      area.className = "image-area grid-five";
      if (images.single_view) {
        const main = makeImage(images.single_view, "主视角");
        main.classList.add("image-main");
        area.appendChild(main);
      }
      const row = document.createElement("div");
      row.className = "image-grid-row";
      (images.multi_view || []).forEach((path, index) => {
        row.appendChild(makeImage(path, `视角 ${index}`));
      });
      area.appendChild(row);
    }
  } else {
    const singleView = images.single_view ? [images.single_view] : [];
    const multiView = Array.isArray(images.multi_view) ? images.multi_view : [];
    const allViews = [
      ...singleView.map((path) => ({ path, label: "主视角" })),
      ...multiView.map((path, index) => ({ path, label: `视角 ${index}` })),
    ];

    if (singleView.length && multiView.length) area.className = "image-area grid-five";
    else if (multiView.length >= 2) area.className = "image-area grid-2x2";
    else area.className = "image-area";

    if (!allViews.length) {
      area.className = "image-area";
      const empty = document.createElement("div");
      empty.className = "image-empty";
      empty.textContent = "该测试步骤未附带图片，当前可先联调交互与接口。";
      area.appendChild(empty);
      return;
    }

    if (singleView.length && multiView.length) {
      const main = makeImage(singleView[0], "主视角");
      main.classList.add("image-main");
      area.appendChild(main);
      const row = document.createElement("div");
      row.className = "image-grid-row";
      multiView.forEach((path, index) => row.appendChild(makeImage(path, `视角 ${index}`)));
      area.appendChild(row);
      return;
    }

    allViews.forEach((view) => area.appendChild(makeImage(view.path, view.label)));
  }
}

function makeImage(path, label) {
  const wrapper = document.createElement("div");
  wrapper.className = "image-wrapper";

  const img = document.createElement("img");
  img.className = "scene-image";
  img.src = resolveImageUrl(path);
  img.alt = label;
  img.loading = "eager";
  img.addEventListener("click", () => openLightbox(img.src));

  const lbl = document.createElement("span");
  lbl.className = "image-label";
  lbl.textContent = label;

  wrapper.appendChild(lbl);
  wrapper.appendChild(img);
  return wrapper;
}

function renderObjects() {
  const list = $("#object-list");
  list.innerHTML = "";
  (state.roundData?.objects || []).forEach((obj) => {
    const item = document.createElement("div");
    item.className = "object-item";
    item.innerHTML = `
      <span class="object-id">${esc(obj.id)}</span>
      <span class="object-desc">${esc(obj.desc || obj.label || obj.id)}</span>
    `;
    list.appendChild(item);
  });
}

function renderProgressiveQuiz() {
  show("#progressive-workspace");
  hide("#adaptive-workspace");

  const rd = state.roundData;
  const banner = $("#round-banner");
  if (rd.round_number > 1) {
    const prev = rd.previous_rounds[rd.previous_rounds.length - 1];
    const wrong = rd.n_this_round;
    const viewLabel = rd.round_number === 2 ? "四视角（4 张图）" : "五视角（5 张图）";
    $("#round-banner-text").textContent =
      `上一轮答对 ${prev.n_correct}/${prev.n_total}，剩余 ${wrong} 道错题。现在提供 ${viewLabel} 帮助判断。`;
    show(banner);
  } else {
    hide(banner);
  }

  renderQuestions();
  updateAnswerCount();
}

function renderQuestions() {
  const container = $("#question-list");
  container.innerHTML = "";
  const questions = state.roundData?.questions || [];

  questions.forEach((q, index) => {
    const card = document.createElement("div");
    card.className = "question-card";
    card.dataset.qid = q.qid;
    card.dataset.index = index;

    const meta = document.createElement("div");
    meta.className = "question-meta";

    const badge = document.createElement("span");
    badge.className = `badge badge-${q.type}`;
    badge.textContent = q.type.toUpperCase();
    meta.appendChild(badge);

    const idx = document.createElement("span");
    idx.className = "question-index";
    idx.textContent = `#${index + 1}`;
    meta.appendChild(idx);

    card.appendChild(meta);

    const prompt = document.createElement("p");
    prompt.className = "question-prompt";
    prompt.textContent = q.prompt_text;
    card.appendChild(prompt);

    if (q.type === "qrr") card.appendChild(buildQrrControl(q));
    else if (q.type === "trr") card.appendChild(buildTrrControl(q));
    else if (q.type === "fdr") card.appendChild(buildFdrControl(q));

    container.appendChild(card);
    markCard(q.qid);
  });
}

function buildQrrControl(q) {
  const group = document.createElement("div");
  group.className = "qrr-group";
  group.dataset.qid = q.qid;

  const opts = [
    { value: "<", label: "前者更近", key: "1" },
    { value: "~=", label: "大致相等", key: "2" },
    { value: ">", label: "后者更近", key: "3" },
  ];

  opts.forEach((opt) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "qrr-btn";
    btn.dataset.value = opt.value;
    btn.innerHTML = `${esc(opt.label)} <span class="qrr-shortcut">${opt.key}</span>`;
    btn.addEventListener("click", () => {
      if (state.answers[q.qid] === opt.value) {
        delete state.answers[q.qid];
        btn.classList.remove("active");
      } else {
        state.answers[q.qid] = opt.value;
        group.querySelectorAll(".qrr-btn").forEach((node) => node.classList.remove("active"));
        btn.classList.add("active");
      }
      markCard(q.qid);
      updateAnswerCount();
    });
    group.appendChild(btn);
  });
  return group;
}

function buildTrrControl(q) {
  const grid = document.createElement("div");
  grid.className = "trr-grid";
  grid.dataset.qid = q.qid;

  const order = [12, 1, 2, 3, 11, 0, 0, 4, 10, 0, 0, 5, 9, 8, 7, 6];
  order.forEach((hour) => {
    if (hour === 0) {
      grid.appendChild(document.createElement("div"));
      return;
    }
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "trr-btn";
    btn.dataset.value = hour;
    btn.textContent = `${hour}`;
    btn.addEventListener("click", () => {
      state.answers[q.qid] = hour;
      grid.querySelectorAll(".trr-btn").forEach((node) => node.classList.remove("active"));
      btn.classList.add("active");
      markCard(q.qid);
      updateAnswerCount();
    });
    grid.appendChild(btn);
  });

  return grid;
}

function buildFdrControl(q) {
  const container = document.createElement("div");
  container.className = "fdr-container";
  container.dataset.qid = q.qid;

  const label = document.createElement("div");
  label.className = "fdr-label";
  label.textContent = "点击物体，按从近到远排序:";
  container.appendChild(label);

  const candDiv = document.createElement("div");
  candDiv.className = "fdr-candidates";
  (q.ranking_candidates || []).forEach((c) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "fdr-candidate-btn";
    btn.dataset.objId = c.id;
    btn.textContent = `${c.id}: ${c.desc}`;
    btn.addEventListener("click", () => {
      if (!btn.classList.contains("used")) addFdrRank(q.qid, c.id, container);
    });
    candDiv.appendChild(btn);
  });
  container.appendChild(candDiv);

  const rankingLabel = document.createElement("div");
  rankingLabel.className = "fdr-label";
  rankingLabel.textContent = "当前排序 (近 -> 远):";
  container.appendChild(rankingLabel);

  const ranking = document.createElement("div");
  ranking.className = "fdr-ranking";
  container.appendChild(ranking);

  const resetBtn = document.createElement("button");
  resetBtn.type = "button";
  resetBtn.className = "fdr-reset-btn";
  resetBtn.textContent = "重置排序";
  resetBtn.addEventListener("click", () => resetFdrRank(q.qid, container));
  container.appendChild(resetBtn);

  return container;
}

function addFdrRank(qid, objId, container) {
  if (!Array.isArray(state.answers[qid])) state.answers[qid] = [];
  if (state.answers[qid].includes(objId)) return;
  state.answers[qid].push(objId);

  const btn = container.querySelector(`.fdr-candidate-btn[data-obj-id="${objId}"]`);
  if (btn) btn.classList.add("used");

  const rankList = container.querySelector(".fdr-ranking");
  const item = document.createElement("span");
  item.className = "fdr-rank-item";
  item.innerHTML = `<span class="fdr-rank-index">#${state.answers[qid].length}</span> ${esc(objId)}`;
  rankList.appendChild(item);

  markCard(qid);
  updateAnswerCount();
}

function resetFdrRank(qid, container) {
  state.answers[qid] = [];
  container.querySelectorAll(".fdr-candidate-btn").forEach((node) => node.classList.remove("used"));
  container.querySelector(".fdr-ranking").innerHTML = "";
  markCard(qid);
  updateAnswerCount();
}

function renderAdaptiveQuiz() {
  hide("#progressive-workspace");
  show("#adaptive-workspace");
  hide("#round-banner");

  const rd = state.roundData;
  $("#adaptive-step-progress").textContent = `Step ${rd.step_index}/${rd.n_steps_total} · Level ${rd.level}`;
  $("#adaptive-step-hint").textContent = rd.allow_approx
    ? "回答候选距离对相对于 pivot 的远近关系。快捷键: 1/2/3。"
    : "当前 step 禁止使用 ~=。快捷键: 1/2。";

  ensureAdaptiveFocus();
  renderAdaptiveWorkbench();
  updateAnswerCount();
}

function ensureAdaptiveFocus() {
  const questions = state.roundData?.questions || [];
  if (!questions.length) {
    state.focusedQuestionIndex = -1;
    return;
  }
  if (state.focusedQuestionIndex >= 0 && state.focusedQuestionIndex < questions.length) return;
  const firstUnanswered = questions.findIndex((q) => !isAnswered(q.qid));
  state.focusedQuestionIndex = firstUnanswered >= 0 ? firstUnanswered : 0;
}

function renderAdaptiveWorkbench() {
  const container = $("#adaptive-workbench");
  container.innerHTML = "";
  const questions = state.roundData?.questions || [];
  if (!questions.length) {
    const empty = document.createElement("div");
    empty.className = "image-empty";
    empty.textContent = "当前 step 没有可答题目。";
    container.appendChild(empty);
    return;
  }

  const lookup = buildObjectLookup();
  const currentIndex = clampAdaptiveIndex(state.focusedQuestionIndex, questions.length);
  state.focusedQuestionIndex = currentIndex;
  const currentQuestion = questions[currentIndex];

  container.appendChild(renderAdaptivePivotCard(lookup));
  container.appendChild(renderAdaptiveCurrentCard(currentQuestion, currentIndex, lookup));
  container.appendChild(renderAdaptiveQueueCard(lookup));
}

function buildObjectLookup() {
  const lookup = {};
  (state.roundData?.objects || []).forEach((obj) => {
    lookup[obj.id] = obj.desc || obj.label || obj.id;
  });
  return lookup;
}

function clampAdaptiveIndex(index, length) {
  if (length <= 0) return -1;
  if (index < 0) return 0;
  if (index >= length) return length - 1;
  return index;
}

function pairDisplay(pair, lookup) {
  const [a, b] = pair;
  return `${a} (${lookup[a] || a}) - ${b} (${lookup[b] || b})`;
}

function pairCompactDisplay(pair) {
  return `${pair[0]} - ${pair[1]}`;
}

function renderAdaptivePivotCard(lookup) {
  const rd = state.roundData;
  const card = document.createElement("section");
  card.className = "adaptive-card adaptive-pivot-card";
  card.innerHTML = `
    <div class="adaptive-card-label">Pivot 距离对</div>
    <div class="adaptive-pair-line">${esc(pairDisplay(rd.pivot_pair, lookup))}</div>
    <p class="adaptive-card-note">本 step 中所有 candidate 都相对于这组距离来判断。</p>
  `;
  return card;
}

function renderAdaptiveCurrentCard(question, currentIndex, lookup) {
  const card = document.createElement("section");
  card.className = "adaptive-card adaptive-current-card";

  const status = isAnswered(question.qid) ? "已答" : "未答";
  const statusClass = isAnswered(question.qid) ? "answered" : "pending";
  card.innerHTML = `
    <div class="adaptive-current-meta">
      <span class="badge badge-adaptive">COMPARE</span>
      <span class="question-index">当前 #${currentIndex + 1}/${state.roundData.questions.length}</span>
      <span class="adaptive-status ${statusClass}">${status}</span>
    </div>
    <div class="adaptive-card-label">Candidate 距离对</div>
    <div class="adaptive-pair-line adaptive-pair-line-strong">${esc(pairDisplay(question.candidate_pair, lookup))}</div>
    <p class="adaptive-question-copy">${esc(question.prompt_text)}</p>
  `;

  const controls = document.createElement("div");
  controls.className = "adaptive-controls";
  const options = adaptiveAnswerOptions();
  controls.style.setProperty("--adaptive-control-cols", `${options.length}`);
  options.forEach((opt) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "adaptive-answer-btn";
    if (state.answers[question.qid] === opt.value) btn.classList.add("active");
    btn.innerHTML = `${esc(opt.label)} <span class="qrr-shortcut">${opt.key}</span>`;
    btn.addEventListener("click", () => setAdaptiveAnswer(question.qid, opt.value, currentIndex));
    controls.appendChild(btn);
  });
  card.appendChild(controls);

  const footer = document.createElement("div");
  footer.className = "adaptive-current-footer";

  const hint = document.createElement("div");
  hint.className = "adaptive-shortcut-note";
  hint.textContent = state.roundData.allow_approx
    ? "快捷键: 1 = 更近, 2 = 大致相等, 3 = 更远"
    : "快捷键: 1 = 更近, 2 = 更远";
  footer.appendChild(hint);

  const clearBtn = document.createElement("button");
  clearBtn.type = "button";
  clearBtn.className = "fdr-reset-btn";
  clearBtn.textContent = "清除当前答案";
  clearBtn.disabled = !isAnswered(question.qid);
  clearBtn.addEventListener("click", () => clearAdaptiveAnswer(question.qid));
  footer.appendChild(clearBtn);

  card.appendChild(footer);
  return card;
}

function renderAdaptiveQueueCard(lookup) {
  const card = document.createElement("section");
  card.className = "adaptive-card adaptive-queue-card";

  const title = document.createElement("div");
  title.className = "adaptive-card-label";
  title.textContent = "本 step 候选队列";
  card.appendChild(title);

  const list = document.createElement("div");
  list.className = "adaptive-queue";

  (state.roundData.questions || []).forEach((question, index) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "adaptive-queue-item";
    if (index === state.focusedQuestionIndex) btn.classList.add("current");
    if (isAnswered(question.qid)) btn.classList.add("answered");
    btn.innerHTML = `
      <span class="adaptive-queue-index">#${index + 1}</span>
      <span class="adaptive-queue-pair">${esc(pairCompactDisplay(question.candidate_pair))}</span>
      <span class="adaptive-queue-answer">${esc(formatAdaptiveAnswer(state.answers[question.qid]))}</span>
    `;
    btn.addEventListener("click", () => setFocusedQuestion(index));
    list.appendChild(btn);
  });

  card.appendChild(list);
  return card;
}

function adaptiveAnswerOptions() {
  if (state.roundData?.allow_approx) {
    return [
      { value: "<", label: "候选更近", key: "1" },
      { value: "~=", label: "差不多", key: "2" },
      { value: ">", label: "候选更远", key: "3" },
    ];
  }
  return [
    { value: "<", label: "候选更近", key: "1" },
    { value: ">", label: "候选更远", key: "2" },
  ];
}

function formatAdaptiveAnswer(answer) {
  if (!answer) return "未答";
  const label = {
    "<": "更近",
    "~=": "差不多",
    ">": "更远",
  }[answer];
  return label || String(answer);
}

function setAdaptiveAnswer(qid, value, currentIndex) {
  state.answers[qid] = value;
  saveAdaptiveDraft();
  advanceAdaptiveFocus(currentIndex);
  renderAdaptiveWorkbench();
  updateAnswerCount();
}

function clearAdaptiveAnswer(qid) {
  delete state.answers[qid];
  saveAdaptiveDraft();
  renderAdaptiveWorkbench();
  updateAnswerCount();
}

function advanceAdaptiveFocus(currentIndex) {
  const questions = state.roundData?.questions || [];
  const nextUnanswered = questions.findIndex((q, index) => index > currentIndex && !isAnswered(q.qid));
  if (nextUnanswered >= 0) {
    state.focusedQuestionIndex = nextUnanswered;
    return;
  }
  const remaining = questions.findIndex((q) => !isAnswered(q.qid));
  state.focusedQuestionIndex = remaining >= 0 ? remaining : currentIndex;
}

function draftStorageKey() {
  const rd = state.roundData;
  if (!rd || state.testMode !== "adaptive_sort") return null;
  return `human-baseline-v2:${state.annotatorId}:${rd.scene_id}:${rd.step_id}`;
}

function loadAdaptiveDraft() {
  const key = draftStorageKey();
  if (!key) return;
  try {
    const raw = sessionStorage.getItem(key);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    const allowedQids = new Set((state.roundData?.questions || []).map((q) => q.qid));
    state.answers = {};
    Object.entries(parsed || {}).forEach(([qid, value]) => {
      if (allowedQids.has(qid)) state.answers[qid] = value;
    });
  } catch (_) {
    state.answers = {};
  }
}

function saveAdaptiveDraft() {
  const key = draftStorageKey();
  if (!key) return;
  sessionStorage.setItem(key, JSON.stringify(state.answers));
}

function clearAdaptiveDraft() {
  const key = draftStorageKey();
  if (!key) return;
  sessionStorage.removeItem(key);
}

function isAnswered(qid) {
  const val = state.answers[qid];
  if (val === undefined || val === null) return false;
  if (Array.isArray(val)) return val.length > 0;
  if (typeof val === "string") return val.trim() !== "";
  return true;
}

function markCard(qid) {
  const card = document.querySelector(`.question-card[data-qid="${qid}"]`);
  if (card) card.classList.toggle("answered", isAnswered(qid));
}

function countAnswered() {
  return (state.roundData?.questions || []).filter((q) => isAnswered(q.qid)).length;
}

function updateAnswerCount() {
  const total = state.roundData?.questions?.length || 0;
  const answered = countAnswered();

  if (state.testMode === "adaptive_sort") {
    const counter = $("#adaptive-answer-count");
    if (counter) counter.textContent = `已答: ${answered}/${total}`;
    const btn = $("#btn-submit-step");
    if (btn) btn.disabled = answered < total;
  } else {
    const counter = $("#answer-count");
    if (counter) counter.textContent = `已答: ${answered}/${total}`;
    const btn = $("#btn-submit-round");
    if (btn) btn.disabled = answered < total;
  }
}

async function handleSubmitRound() {
  const rd = state.roundData;
  if (!rd) return;
  if (countAnswered() < (rd.questions || []).length) {
    showToast(`还有 ${(rd.questions || []).length - countAnswered()} 道题未回答`, true);
    return;
  }

  const btn = $("#btn-submit-round");
  btn.disabled = true;
  btn.innerHTML = '<span class="loading-spinner"></span>提交中...';

  const responses = (rd.questions || []).map((q) => {
    let answer = state.answers[q.qid];
    if (q.type === "fdr" && !Array.isArray(answer)) answer = answer ? [answer] : [];
    return { qid: q.qid, answer };
  });

  try {
    const result = await postJson("/api/v2/round/submit", {
      annotator_id: state.annotatorId,
      test_mode: state.testMode,
      scene_id: rd.scene_id,
      round_number: rd.round_number,
      responses,
      elapsed_seconds: Math.round(getElapsedSeconds() * 100) / 100,
    });
    state.gradingResult = result.grading;
    state.nextAction = result.next_action;
    if (result.progress) state.progress = result.progress;
    renderGrading();
    switchView("grading");
  } catch (err) {
    console.error("handleSubmitRound failed:", err);
    btn.disabled = false;
    btn.textContent = "提交本轮";
  }
}

async function handleSubmitStep() {
  const rd = state.roundData;
  if (!rd) return;
  if (countAnswered() < (rd.questions || []).length) {
    showToast(`还有 ${(rd.questions || []).length - countAnswered()} 个 candidate 未回答`, true);
    return;
  }

  const btn = $("#btn-submit-step");
  btn.disabled = true;
  btn.innerHTML = '<span class="loading-spinner"></span>提交中...';

  const responses = (rd.questions || []).map((q) => ({
    qid: q.qid,
    answer: state.answers[q.qid],
  }));

  try {
    const result = await postJson("/api/v2/round/submit", {
      annotator_id: state.annotatorId,
      test_mode: state.testMode,
      scene_id: rd.scene_id,
      step_id: rd.step_id,
      responses,
      elapsed_seconds: Math.round(getElapsedSeconds() * 100) / 100,
    });
    clearAdaptiveDraft();
    state.stepSummary = result.submission_summary;
    state.nextAction = result.next_action;
    if (result.progress) state.progress = result.progress;
    renderStepDone();
    switchView("stepdone");
  } catch (err) {
    console.error("handleSubmitStep failed:", err);
    btn.disabled = false;
    btn.textContent = "提交本 step";
  }
}

function renderGrading() {
  const g = state.gradingResult;
  if (!g) return;

  const rd = state.roundData;
  const summaryEl = $("#grading-summary");
  summaryEl.innerHTML = `
    <div class="grading-score ${g.all_correct ? "perfect" : "partial"}">${g.n_correct}/${g.n_total}</div>
    <div class="grading-score-label">${g.all_correct ? "全部正确" : "部分正确"}</div>
    <div class="grading-round-info">场景 ${esc(rd.scene_id)} - ${esc(rd.round_label)}</div>
  `;

  const resultsEl = $("#grading-results");
  resultsEl.innerHTML = "";
  const qMap = {};
  (rd.questions || []).forEach((q) => {
    qMap[q.qid] = q;
  });

  g.results.forEach((result) => {
    const item = document.createElement("div");
    item.className = `grading-item ${result.correct ? "correct" : "incorrect"}`;
    item.innerHTML = `
      <div class="grading-icon ${result.correct ? "correct" : "incorrect"}">${result.correct ? "\u2713" : "\u2717"}</div>
      <div class="grading-detail">
        <div class="grading-prompt"><span class="badge badge-${result.type}" style="margin-right:6px">${result.type.toUpperCase()}</span>${esc(qMap[result.qid]?.prompt_text || result.qid)}</div>
        <div class="grading-answer">
          ${result.correct
            ? `您的回答: <strong>${esc(formatAnswer(result.user_answer, result.type))}</strong>`
            : `您的回答: <strong>${esc(formatAnswer(result.user_answer, result.type))}</strong> &nbsp;|&nbsp; 正确答案: <strong style="color:var(--correct)">${esc(result.correct_answer_display)}</strong>`
          }
        </div>
      </div>
    `;
    resultsEl.appendChild(item);
  });

  hide("#btn-next-round");
  hide("#btn-next-scene");
  hide("#btn-finish");

  if (state.nextAction === "next_round") {
    const btn = $("#btn-next-round");
    const nextRound = rd.round_number + 1;
    const viewLabels = { 2: "四视角", 3: "五视角" };
    btn.textContent = `进入第 ${nextRound} 轮（${viewLabels[nextRound] || ""}）`;
    show(btn);
  } else if (state.nextAction === "scene_complete") {
    show("#btn-next-scene");
  } else if (state.nextAction === "all_done") {
    show("#btn-finish");
  }
}

function renderStepDone() {
  const summary = state.stepSummary;
  if (!summary) return;

  const el = $("#stepdone-summary");
  const label = state.nextAction === "next_step"
    ? "本 step 已保存"
    : state.nextAction === "scene_complete"
      ? "该场景已完成"
      : "全部测试完成";

  el.innerHTML = `
    <div class="stepdone-kicker">${esc(MODE_LABELS[state.testMode] || state.testMode)}</div>
    <div class="stepdone-title">${esc(label)}</div>
    <div class="stepdone-stats">
      <div class="stat-row"><span class="stat-label">场景</span><span class="stat-value">${esc(summary.scene_id)}</span></div>
      <div class="stat-row"><span class="stat-label">Step</span><span class="stat-value">${summary.step_index}/${summary.n_steps_total}</span></div>
      <div class="stat-row"><span class="stat-label">Level</span><span class="stat-value">${summary.level}</span></div>
      <div class="stat-row"><span class="stat-label">比较数</span><span class="stat-value">${summary.n_responses}</span></div>
    </div>
  `;

  hide("#btn-next-step");
  hide("#btn-step-next-scene");
  hide("#btn-step-finish");

  if (state.nextAction === "next_step") show("#btn-next-step");
  else if (state.nextAction === "scene_complete") show("#btn-step-next-scene");
  else if (state.nextAction === "all_done") show("#btn-step-finish");
}

function renderDone() {
  const progress = state.progress || {};
  const rows = [
    { label: "标注员", value: state.annotatorId || "--" },
    { label: "模式", value: MODE_LABELS[state.testMode] || state.testMode },
    { label: "完成场景", value: `${progress.scenes_completed || 0}/${progress.total_scenes || 0}` },
  ];

  if (state.testMode === "adaptive_sort") {
    rows.push({ label: "完成步骤", value: `${progress.completed_steps || 0}/${progress.total_steps || 0}` });
    rows.push({ label: "总比较数", value: `${progress.answered_questions || 0}/${progress.total_questions || 0}` });
  } else {
    rows.push({ label: "总题数", value: `${progress.answered_questions || 0}/${progress.total_questions || 0}` });
  }
  rows.push({ label: "完成率", value: `${progress.progress_pct || 0}%` });

  $("#done-stats").innerHTML = rows.map((row) => (
    `<div class="stat-row"><span class="stat-label">${esc(row.label)}</span><span class="stat-value">${esc(row.value)}</span></div>`
  )).join("");
}

function formatAnswer(answer, type) {
  if (answer === null || answer === undefined) return "未作答";
  if (type === "qrr" || type === "adaptive_sort_cmp") {
    return {
      "<": "< (前者更近)",
      "~=": "~= (大致相等)",
      ">": "> (后者更近)",
    }[answer] || String(answer);
  }
  if (type === "trr") return `${answer} 点钟`;
  if (type === "fdr") return Array.isArray(answer) ? answer.join(" > ") : String(answer);
  return String(answer);
}

function handleKeyboard(event) {
  if (state.view !== "quiz") return;
  const tag = event.target.tagName.toLowerCase();
  if (["input", "textarea", "select"].includes(tag)) return;

  if (state.testMode === "adaptive_sort") {
    handleAdaptiveKeyboard(event);
  } else {
    handleProgressiveKeyboard(event);
  }
}

function handleProgressiveKeyboard(event) {
  const questions = state.roundData?.questions || [];
  if (!questions.length) return;

  const fi = state.focusedQuestionIndex;
  const focusedQ = fi >= 0 && fi < questions.length ? questions[fi] : null;

  if (focusedQ && focusedQ.type === "qrr" && ["1", "2", "3"].includes(event.key)) {
    event.preventDefault();
    const value = { "1": "<", "2": "~=", "3": ">" }[event.key];
    state.answers[focusedQ.qid] = value;
    const card = document.querySelector(`.question-card[data-qid="${focusedQ.qid}"]`);
    card?.querySelectorAll(".qrr-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.value === value);
    });
    markCard(focusedQ.qid);
    updateAnswerCount();
    if (fi < questions.length - 1) setFocusedQuestion(fi + 1);
    return;
  }

  if (focusedQ && focusedQ.type === "trr") {
    const num = parseInt(event.key, 10);
    if (num >= 1 && num <= 9) {
      event.preventDefault();
      handleTrrKeyInput(focusedQ.qid, num);
      return;
    }
  }

  if (event.key === "ArrowDown" || event.key === "j") {
    event.preventDefault();
    setFocusedQuestion(Math.min(fi + 1, questions.length - 1));
  } else if (event.key === "ArrowUp" || event.key === "k") {
    event.preventDefault();
    setFocusedQuestion(Math.max(fi - 1, 0));
  }
}

function handleAdaptiveKeyboard(event) {
  const questions = state.roundData?.questions || [];
  if (!questions.length) return;

  ensureAdaptiveFocus();
  const current = questions[state.focusedQuestionIndex];
  if (!current) return;

  if (event.key === "ArrowDown" || event.key === "j") {
    event.preventDefault();
    setFocusedQuestion(Math.min(state.focusedQuestionIndex + 1, questions.length - 1));
    return;
  }
  if (event.key === "ArrowUp" || event.key === "k") {
    event.preventDefault();
    setFocusedQuestion(Math.max(state.focusedQuestionIndex - 1, 0));
    return;
  }
  if (event.key === "Backspace" || event.key === "Delete") {
    event.preventDefault();
    clearAdaptiveAnswer(current.qid);
    return;
  }
  if (event.key === "Enter" && countAnswered() === questions.length) {
    event.preventDefault();
    handleSubmitStep();
    return;
  }

  const allowApprox = !!state.roundData?.allow_approx;
  const keyMap = allowApprox
    ? { "1": "<", "2": "~=", "3": ">" }
    : { "1": "<", "2": ">" };
  if (keyMap[event.key]) {
    event.preventDefault();
    setAdaptiveAnswer(current.qid, keyMap[event.key], state.focusedQuestionIndex);
  }
}

function handleTrrKeyInput(qid, digit) {
  trrKeyBuffer += String(digit);
  if (trrKeyTimer) clearTimeout(trrKeyTimer);

  if (trrKeyBuffer === "1") {
    trrKeyTimer = setTimeout(() => {
      applyTrrValue(qid, 1);
      trrKeyBuffer = "";
    }, 400);
    return;
  }

  const value = parseInt(trrKeyBuffer, 10);
  trrKeyBuffer = "";
  if (trrKeyTimer) clearTimeout(trrKeyTimer);
  if (value >= 1 && value <= 12) applyTrrValue(qid, value);
  else applyTrrValue(qid, digit);
}

function applyTrrValue(qid, value) {
  state.answers[qid] = value;
  const card = document.querySelector(`.question-card[data-qid="${qid}"]`);
  card?.querySelectorAll(".trr-btn").forEach((btn) => {
    btn.classList.toggle("active", parseInt(btn.dataset.value, 10) === value);
  });
  markCard(qid);
  updateAnswerCount();
}

function setFocusedQuestion(index) {
  if (state.testMode === "adaptive_sort") {
    state.focusedQuestionIndex = clampAdaptiveIndex(index, state.roundData?.questions?.length || 0);
    renderAdaptiveWorkbench();
    return;
  }

  $$(".question-card.keyboard-focus").forEach((card) => card.classList.remove("keyboard-focus"));
  state.focusedQuestionIndex = index;
  const questions = state.roundData?.questions || [];
  if (index < 0 || index >= questions.length) return;
  const qid = questions[index].qid;
  const card = document.querySelector(`.question-card[data-qid="${qid}"]`);
  if (card) {
    card.classList.add("keyboard-focus");
    card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

function wireEvents() {
  $("#btn-submit-round")?.addEventListener("click", handleSubmitRound);
  $("#btn-submit-step")?.addEventListener("click", handleSubmitStep);

  $("#btn-next-round")?.addEventListener("click", async () => {
    const btn = $("#btn-next-round");
    btn.disabled = true;
    btn.innerHTML = '<span class="loading-spinner"></span>加载中...';
    try {
      await loadCurrentRound();
    } catch (err) {
      console.error("loadCurrentRound failed:", err);
      btn.disabled = false;
      btn.textContent = "进入下一轮";
    }
  });

  $("#btn-next-scene")?.addEventListener("click", async () => {
    const btn = $("#btn-next-scene");
    btn.disabled = true;
    btn.innerHTML = '<span class="loading-spinner"></span>加载中...';
    try {
      await allocateScene();
    } catch (err) {
      console.error("allocateScene failed:", err);
      btn.disabled = false;
      btn.textContent = "下一个场景";
    }
  });

  $("#btn-finish")?.addEventListener("click", () => {
    renderDone();
    switchView("done");
  });

  $("#btn-next-step")?.addEventListener("click", async () => {
    const btn = $("#btn-next-step");
    btn.disabled = true;
    btn.innerHTML = '<span class="loading-spinner"></span>加载中...';
    try {
      await loadCurrentRound();
    } catch (err) {
      console.error("loadCurrentRound failed:", err);
      btn.disabled = false;
      btn.textContent = "进入下一步";
    }
  });

  $("#btn-step-next-scene")?.addEventListener("click", async () => {
    const btn = $("#btn-step-next-scene");
    btn.disabled = true;
    btn.innerHTML = '<span class="loading-spinner"></span>加载中...';
    try {
      await allocateScene();
    } catch (err) {
      console.error("allocateScene failed:", err);
      btn.disabled = false;
      btn.textContent = "下一个场景";
    }
  });

  $("#btn-step-finish")?.addEventListener("click", () => {
    renderDone();
    switchView("done");
  });

  $("#btn-restart")?.addEventListener("click", () => {
    state.annotatorId = null;
    state.roundData = null;
    state.progress = null;
    state.answers = {};
    state.gradingResult = null;
    state.stepSummary = null;
    state.nextAction = null;
    state.focusedQuestionIndex = -1;
    $("#input-annotator").value = "";
    const fallback = document.querySelector('input[name="test-mode"][value="progressive"]');
    if (fallback) fallback.checked = true;
    state.testMode = "progressive";
    updateStartButton();
    switchView("welcome");
  });

  document.addEventListener("keydown", handleKeyboard);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeLightbox();
  });
  document.addEventListener("click", (event) => {
    const card = event.target.closest(".question-card");
    if (card && state.view === "quiz" && state.testMode === "progressive") {
      const index = parseInt(card.dataset.index, 10);
      if (!Number.isNaN(index)) setFocusedQuestion(index);
    }
  });
}

window.addEventListener("DOMContentLoaded", async () => {
  initWelcome();
  wireEvents();
  switchView("welcome");
  await loadCapabilities();
});
