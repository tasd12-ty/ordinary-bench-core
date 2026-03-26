/* ================================================================
   ORDINARY-BENCH Human Baseline v2 — Progressive Testing Frontend
   ================================================================ */

// ---- State ----

const state = {
  view: "welcome",
  annotatorId: null,
  roundData: null,       // current round from server
  answers: {},           // { qid: answer }
  gradingResult: null,   // grading from last submission
  nextAction: null,      // "next_round" | "scene_complete" | "all_done"
  progress: null,
  pageStartTime: null,
  timerInterval: null,
  focusedQuestionIndex: -1,
};

// ---- DOM Helpers ----

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

// ---- Toast ----

let toastTimeout = null;

function showToast(msg, isError) {
  const toast = $("#toast");
  toast.textContent = msg;
  toast.classList.toggle("error", !!isError);
  toast.classList.add("visible");
  if (toastTimeout) clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => toast.classList.remove("visible"), 3500);
}

// ---- API ----

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

// ---- Timer ----

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

function formatTime(s) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

function updateTimerDisplay() {
  const el = $("#topbar-timer");
  if (el) el.textContent = formatTime(getElapsedSeconds());
}

// ---- View Switching ----

function switchView(name) {
  state.view = name;
  ["welcome", "quiz", "grading", "done"].forEach((v) => {
    const el = $(`#view-${v}`);
    if (el) el.classList.toggle("hidden", v !== name);
  });
  if (name === "quiz") startTimer();
  else stopTimer();
}

// ---- Image URL ----

function resolveImageUrl(path) {
  if (!path) return "";
  if (path.startsWith("tasks/")) return "/" + path;
  return "/data-images/" + path;
}

// ---- Lightbox ----

function openLightbox(src) {
  const lb = $("#lightbox");
  const img = $("#lightbox-img");
  img.src = src;
  show(lb);
}

function closeLightbox() {
  hide("#lightbox");
}

// ---- HTML Escape ----

function esc(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// ---- Welcome ----

function initWelcome() {
  const input = $("#input-annotator");
  const btn = $("#btn-start");

  input.addEventListener("input", () => {
    btn.disabled = !input.value.trim();
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !btn.disabled) btn.click();
  });

  btn.addEventListener("click", handleStart);
}

async function handleStart() {
  const id = $("#input-annotator").value.trim();
  if (!id) return;

  state.annotatorId = id;
  const btn = $("#btn-start");
  const errEl = $("#welcome-error");
  btn.disabled = true;
  errEl.textContent = "";

  try {
    const result = await postJson("/api/v2/session/start", { annotator_id: id });
    state.progress = result.progress;

    if (result.has_current_scene) {
      // Resume in-progress scene.
      await loadCurrentRound();
    } else {
      // Allocate first scene.
      await allocateScene();
    }
  } catch (err) {
    errEl.textContent = err.message || "启动失败";
    btn.disabled = false;
  }
}

// ---- Scene / Round Loading ----

async function loadCurrentRound() {
  const url = `/api/v2/scene/current?annotator_id=${encodeURIComponent(state.annotatorId)}`;
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
  state.answers = {};
  state.focusedQuestionIndex = -1;
  renderQuiz();
  switchView("quiz");
}

async function allocateScene() {
  const data = await postJson("/api/v2/scene/allocate", {
    annotator_id: state.annotatorId,
  });

  if (data.done) {
    renderDone();
    switchView("done");
    return;
  }

  state.roundData = data;
  state.answers = {};
  state.focusedQuestionIndex = -1;
  renderQuiz();
  switchView("quiz");
}

// ---- Render: Quiz ----

function renderQuiz() {
  const rd = state.roundData;
  if (!rd) return;

  // Topbar.
  const p = state.progress || {};
  $("#topbar-scene").textContent = rd.scene_id;
  $("#topbar-round").textContent = rd.round_label;
  $("#topbar-progress").textContent =
    `已完成 ${p.scenes_completed || 0}/${p.total_scenes || 0} 场景`;

  // Round banner.
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

  renderImages();
  renderObjects();
  renderQuestions();
  updateAnswerCount();
}

function renderImages() {
  const rd = state.roundData;
  const area = $("#image-area");
  area.innerHTML = "";

  const images = rd.images;

  if (rd.round_number === 1) {
    // Single view.
    area.className = "image-area";
    if (images.single_view) {
      area.appendChild(makeImage(images.single_view, "单视角"));
    }
  } else if (rd.round_number === 2) {
    // Four views: 2x2 grid.
    area.className = "image-area grid-2x2";
    (images.multi_view || []).forEach((path, i) => {
      area.appendChild(makeImage(path, `视角 ${i}`));
    });
  } else {
    // Five views: single on top + 2x2 grid.
    area.className = "image-area grid-five";
    if (images.single_view) {
      const main = makeImage(images.single_view, "主视角");
      main.classList.add("image-main");
      area.appendChild(main);
    }
    const row = document.createElement("div");
    row.className = "image-grid-row";
    (images.multi_view || []).forEach((path, i) => {
      row.appendChild(makeImage(path, `视角 ${i}`));
    });
    area.appendChild(row);
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
  (state.roundData.objects || []).forEach((obj) => {
    const item = document.createElement("div");
    item.className = "object-item";
    item.innerHTML = `
      <span class="object-id">${esc(obj.id)}</span>
      <span class="object-desc">${esc(obj.desc)}</span>
    `;
    list.appendChild(item);
  });
}

function renderQuestions() {
  const container = $("#question-list");
  container.innerHTML = "";
  const questions = state.roundData.questions || [];

  questions.forEach((q, index) => {
    const card = document.createElement("div");
    card.className = "question-card";
    card.dataset.qid = q.qid;
    card.dataset.index = index;

    // Meta.
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

    // Prompt.
    const prompt = document.createElement("p");
    prompt.className = "question-prompt";
    prompt.textContent = q.prompt_text;
    card.appendChild(prompt);

    // Answer control.
    if (q.type === "qrr") card.appendChild(buildQrrControl(q));
    else if (q.type === "trr") card.appendChild(buildTrrControl(q));
    else if (q.type === "fdr") card.appendChild(buildFdrControl(q));

    container.appendChild(card);
  });
}

// ---- QRR Control ----

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
        group.querySelectorAll(".qrr-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
      }
      markCard(q.qid);
      updateAnswerCount();
    });

    group.appendChild(btn);
  });

  return group;
}

// ---- TRR Control ----

function buildTrrControl(q) {
  const grid = document.createElement("div");
  grid.className = "trr-grid";
  grid.dataset.qid = q.qid;

  // Layout: 12, 1, 2, 3, 11, (empty), (empty), 4, 10, (empty), (empty), 5, 9, 8, 7, 6
  // Simpler: just 3 rows of 4 in clock order.
  const order = [12, 1, 2, 3, 11, 0, 0, 4, 10, 0, 0, 5, 9, 8, 7, 6];

  order.forEach((h) => {
    if (h === 0) {
      const spacer = document.createElement("div");
      grid.appendChild(spacer);
      return;
    }
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "trr-btn";
    btn.dataset.value = h;
    btn.textContent = `${h}`;

    btn.addEventListener("click", () => {
      state.answers[q.qid] = h;
      grid.querySelectorAll(".trr-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      markCard(q.qid);
      updateAnswerCount();
    });

    grid.appendChild(btn);
  });

  return grid;
}

// ---- FDR Control ----

function buildFdrControl(q) {
  const container = document.createElement("div");
  container.className = "fdr-container";
  container.dataset.qid = q.qid;

  const candidates = q.ranking_candidates || [];

  // Label.
  const label = document.createElement("div");
  label.className = "fdr-label";
  label.textContent = "点击物体，按从近到远排序:";
  container.appendChild(label);

  // Candidate buttons.
  const candDiv = document.createElement("div");
  candDiv.className = "fdr-candidates";
  candidates.forEach((c) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "fdr-candidate-btn";
    btn.dataset.objId = c.id;
    btn.textContent = `${c.id}: ${c.desc}`;
    btn.addEventListener("click", () => {
      if (btn.classList.contains("used")) return;
      addFdrRank(q.qid, c.id, container);
    });
    candDiv.appendChild(btn);
  });
  container.appendChild(candDiv);

  // Ranking label.
  const rlabel = document.createElement("div");
  rlabel.className = "fdr-label";
  rlabel.textContent = "当前排序 (近 -> 远):";
  container.appendChild(rlabel);

  // Ranking display.
  const rankList = document.createElement("div");
  rankList.className = "fdr-ranking";
  container.appendChild(rankList);

  // Reset.
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
  const idx = state.answers[qid].length;
  const item = document.createElement("span");
  item.className = "fdr-rank-item";
  item.innerHTML = `<span class="fdr-rank-index">#${idx}</span> ${esc(objId)}`;
  rankList.appendChild(item);

  markCard(qid);
  updateAnswerCount();
}

function resetFdrRank(qid, container) {
  state.answers[qid] = [];
  container.querySelectorAll(".fdr-candidate-btn").forEach((b) => b.classList.remove("used"));
  container.querySelector(".fdr-ranking").innerHTML = "";
  markCard(qid);
  updateAnswerCount();
}

// ---- Answer Tracking ----

function markCard(qid) {
  const card = document.querySelector(`.question-card[data-qid="${qid}"]`);
  if (card) card.classList.toggle("answered", isAnswered(qid));
}

function isAnswered(qid) {
  const val = state.answers[qid];
  if (val === undefined || val === null) return false;
  if (Array.isArray(val)) return val.length > 0;
  if (typeof val === "string") return val.trim() !== "";
  return true;
}

function countAnswered() {
  return (state.roundData?.questions || []).filter((q) => isAnswered(q.qid)).length;
}

function updateAnswerCount() {
  const total = state.roundData?.questions?.length || 0;
  const answered = countAnswered();
  const el = $("#answer-count");
  if (el) el.textContent = `已答: ${answered}/${total}`;

  const btn = $("#btn-submit-round");
  if (btn) btn.disabled = answered < total;
}

// ---- Submit Round ----

async function handleSubmitRound() {
  const rd = state.roundData;
  if (!rd) return;

  const questions = rd.questions || [];
  if (countAnswered() < questions.length) {
    showToast(`还有 ${questions.length - countAnswered()} 道题未回答`, true);
    return;
  }

  const btn = $("#btn-submit-round");
  btn.disabled = true;
  btn.innerHTML = '<span class="loading-spinner"></span>提交中...';

  const elapsed = getElapsedSeconds();

  const responses = questions.map((q) => {
    let answer = state.answers[q.qid];
    if (q.type === "fdr" && !Array.isArray(answer)) answer = answer ? [answer] : [];
    return { qid: q.qid, answer };
  });

  try {
    const result = await postJson("/api/v2/round/submit", {
      annotator_id: state.annotatorId,
      scene_id: rd.scene_id,
      round_number: rd.round_number,
      responses,
      elapsed_seconds: Math.round(elapsed * 100) / 100,
    });

    state.gradingResult = result.grading;
    state.nextAction = result.next_action;
    if (result.progress) state.progress = result.progress;

    renderGrading();
    switchView("grading");
  } catch (err) {
    btn.disabled = false;
    btn.textContent = "提交本轮";
  }
}

// ---- Render: Grading ----

function renderGrading() {
  const g = state.gradingResult;
  if (!g) return;

  const rd = state.roundData;

  // Summary.
  const summaryEl = $("#grading-summary");
  const isPerfect = g.all_correct;
  summaryEl.innerHTML = `
    <div class="grading-score ${isPerfect ? "perfect" : "partial"}">${g.n_correct}/${g.n_total}</div>
    <div class="grading-score-label">${isPerfect ? "全部正确！" : "部分正确"}</div>
    <div class="grading-round-info">场景 ${esc(rd.scene_id)} — ${esc(rd.round_label)}</div>
  `;

  // Per-question results.
  const resultsEl = $("#grading-results");
  resultsEl.innerHTML = "";

  // Build a lookup of questions by qid for prompt text.
  const qMap = {};
  (rd.questions || []).forEach((q) => (qMap[q.qid] = q));

  g.results.forEach((r) => {
    const item = document.createElement("div");
    item.className = `grading-item ${r.correct ? "correct" : "incorrect"}`;

    const icon = document.createElement("div");
    icon.className = `grading-icon ${r.correct ? "correct" : "incorrect"}`;
    icon.textContent = r.correct ? "\u2713" : "\u2717";
    item.appendChild(icon);

    const detail = document.createElement("div");
    detail.className = "grading-detail";

    const q = qMap[r.qid];
    const prompt = document.createElement("div");
    prompt.className = "grading-prompt";
    prompt.innerHTML = `<span class="badge badge-${r.type}" style="margin-right:6px">${r.type.toUpperCase()}</span>${esc(q ? q.prompt_text : r.qid)}`;
    detail.appendChild(prompt);

    const answerLine = document.createElement("div");
    answerLine.className = "grading-answer";

    const userAnswerDisplay = formatAnswer(r.user_answer, r.type);
    if (r.correct) {
      answerLine.innerHTML = `您的回答: <strong>${esc(userAnswerDisplay)}</strong>`;
    } else {
      answerLine.innerHTML = `您的回答: <strong>${esc(userAnswerDisplay)}</strong> &nbsp;|&nbsp; 正确答案: <strong style="color:var(--correct)">${esc(r.correct_answer_display)}</strong>`;
    }
    detail.appendChild(answerLine);

    item.appendChild(detail);
    resultsEl.appendChild(item);
  });

  // Buttons.
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

function formatAnswer(answer, type) {
  if (answer === null || answer === undefined) return "未作答";
  if (type === "qrr") {
    const map = { "<": "< (前者更近)", "~=": "~= (大致相等)", ">": "> (后者更近)" };
    return map[answer] || String(answer);
  }
  if (type === "trr") return `${answer} 点钟`;
  if (type === "fdr") {
    if (Array.isArray(answer)) return answer.join(" > ");
    return String(answer);
  }
  return String(answer);
}

// ---- Render: Done ----

function renderDone() {
  const p = state.progress || {};
  const el = $("#done-stats");
  el.innerHTML = `
    <div class="stat-row"><span class="stat-label">标注员</span><span class="stat-value">${esc(state.annotatorId || "--")}</span></div>
    <div class="stat-row"><span class="stat-label">完成场景</span><span class="stat-value">${p.scenes_completed || 0}/${p.total_scenes || 0}</span></div>
    <div class="stat-row"><span class="stat-label">总题数</span><span class="stat-value">${p.answered_questions || 0}/${p.total_questions || 0}</span></div>
    <div class="stat-row"><span class="stat-label">完成率</span><span class="stat-value">${p.progress_pct || 0}%</span></div>
  `;
}

// ---- Keyboard Shortcuts ----

function handleKeyboard(e) {
  if (state.view !== "quiz") return;
  const tag = e.target.tagName.toLowerCase();
  if (tag === "input" || tag === "textarea" || tag === "select") return;

  const questions = state.roundData?.questions || [];
  if (!questions.length) return;

  const fi = state.focusedQuestionIndex;
  const focusedQ = fi >= 0 && fi < questions.length ? questions[fi] : null;

  // QRR shortcuts.
  if (focusedQ && focusedQ.type === "qrr" && ["1", "2", "3"].includes(e.key)) {
    e.preventDefault();
    const valueMap = { "1": "<", "2": "~=", "3": ">" };
    const value = valueMap[e.key];
    state.answers[focusedQ.qid] = value;

    const card = document.querySelector(`.question-card[data-qid="${focusedQ.qid}"]`);
    if (card) {
      const group = card.querySelector(".qrr-group");
      if (group) {
        group.querySelectorAll(".qrr-btn").forEach((b) => {
          b.classList.toggle("active", b.dataset.value === value);
        });
      }
    }
    markCard(focusedQ.qid);
    updateAnswerCount();
    if (fi < questions.length - 1) setFocusedQuestion(fi + 1);
    return;
  }

  // TRR shortcut.
  if (focusedQ && focusedQ.type === "trr") {
    const num = parseInt(e.key, 10);
    if (num >= 1 && num <= 9) {
      e.preventDefault();
      handleTrrKeyInput(focusedQ.qid, num);
      return;
    }
  }

  // Navigation.
  if (e.key === "ArrowDown" || e.key === "j") {
    e.preventDefault();
    setFocusedQuestion(Math.min(fi + 1, questions.length - 1));
  } else if (e.key === "ArrowUp" || e.key === "k") {
    e.preventDefault();
    setFocusedQuestion(Math.max(fi - 1, 0));
  }
}

let trrKeyBuffer = "";
let trrKeyTimer = null;

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

  const val = parseInt(trrKeyBuffer, 10);
  trrKeyBuffer = "";
  if (trrKeyTimer) clearTimeout(trrKeyTimer);

  if (val >= 1 && val <= 12) applyTrrValue(qid, val);
  else applyTrrValue(qid, digit);
}

function applyTrrValue(qid, value) {
  state.answers[qid] = value;
  const card = document.querySelector(`.question-card[data-qid="${qid}"]`);
  if (card) {
    card.querySelectorAll(".trr-btn").forEach((b) => {
      b.classList.toggle("active", parseInt(b.dataset.value, 10) === value);
    });
  }
  markCard(qid);
  updateAnswerCount();
}

function setFocusedQuestion(index) {
  $$(".question-card.keyboard-focus").forEach((c) => c.classList.remove("keyboard-focus"));
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

// ---- Event Wiring ----

function wireEvents() {
  // Submit round.
  $("#btn-submit-round")?.addEventListener("click", handleSubmitRound);

  // Next round.
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

  // Next scene.
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

  // Finish.
  $("#btn-finish")?.addEventListener("click", () => {
    renderDone();
    switchView("done");
  });

  // Restart.
  $("#btn-restart")?.addEventListener("click", () => {
    state.annotatorId = null;
    state.roundData = null;
    state.progress = null;
    state.answers = {};
    const input = $("#input-annotator");
    if (input) input.value = "";
    const btn = $("#btn-start");
    if (btn) btn.disabled = true;
    switchView("welcome");
  });

  // Keyboard.
  document.addEventListener("keydown", handleKeyboard);

  // Click question card to focus.
  document.addEventListener("click", (e) => {
    const card = e.target.closest(".question-card");
    if (card && state.view === "quiz") {
      const index = parseInt(card.dataset.index, 10);
      if (!isNaN(index)) setFocusedQuestion(index);
    }
  });

  // ESC to close lightbox.
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeLightbox();
  });
}

// ---- Init ----

window.addEventListener("DOMContentLoaded", () => {
  initWelcome();
  wireEvents();
  switchView("welcome");
});
