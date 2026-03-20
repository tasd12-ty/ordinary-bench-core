/* ================================================================
   ORDINARY-BENCH Human Baseline -- Frontend Application
   ================================================================ */

// ---- State ----

const state = {
  view: "welcome",       // "welcome" | "quiz" | "submitted" | "done"
  annotatorId: null,
  testType: "single_view",
  currentPage: null,      // page allocation from server
  progress: null,         // progress summary from server
  answers: {},            // { qid: answer_value }
  pageStartTime: null,    // Date object
  timerInterval: null,
  focusedQuestionIndex: -1,
};

// ---- DOM Helpers ----

function $(selector) {
  return document.querySelector(selector);
}

function $$(selector) {
  return document.querySelectorAll(selector);
}

function show(el) {
  if (typeof el === "string") el = $(el);
  if (el) el.classList.remove("hidden");
}

function hide(el) {
  if (typeof el === "string") el = $(el);
  if (el) el.classList.add("hidden");
}

// ---- Toast Notifications ----

let toastTimeout = null;

function showToast(message, isError) {
  let toast = $(".toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.className = "toast";
    document.body.appendChild(toast);
  }
  toast.textContent = message;
  toast.classList.toggle("error", !!isError);
  toast.classList.add("visible");

  if (toastTimeout) clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => {
    toast.classList.remove("visible");
  }, 3500);
}

// ---- API Helpers ----

async function fetchJson(url, options) {
  try {
    const resp = await fetch(url, options);
    const data = await resp.json();
    if (!resp.ok) {
      const errMsg = data.error || `HTTP ${resp.status}`;
      throw new Error(errMsg);
    }
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

function formatTime(totalSeconds) {
  const mins = Math.floor(totalSeconds / 60);
  const secs = Math.floor(totalSeconds % 60);
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function updateTimerDisplay() {
  const text = formatTime(getElapsedSeconds());
  const topTimer = $("#topbar-timer");
  const bottomTimer = $("#bottom-timer");
  if (topTimer) topTimer.textContent = text;
  if (bottomTimer) bottomTimer.textContent = text;
}

// ---- View Switching ----

function switchView(viewName) {
  state.view = viewName;
  hide("#view-welcome");
  hide("#view-quiz");
  hide("#view-submitted");
  hide("#view-done");
  show(`#view-${viewName}`);

  if (viewName === "quiz") {
    startTimer();
  } else {
    stopTimer();
  }
}

// ---- Resolve Image URL ----

function resolveImageUrl(imagePath) {
  if (!imagePath) return "";
  // Paths may start with "images/" (raw from data-gen) or "tasks/" (labeled from tasks dir).
  // The server serves /data-images/* and /tasks/* routes.
  if (imagePath.startsWith("tasks/")) {
    return "/" + imagePath;
  }
  // For raw images, use /data-images/ route.
  return "/data-images/" + imagePath;
}

// ---- Render: Welcome ----

function initWelcome() {
  const input = $("#input-annotator");
  const btnStart = $("#btn-start");

  function updateStartButton() {
    btnStart.disabled = !input.value.trim();
  }

  input.addEventListener("input", updateStartButton);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !btnStart.disabled) {
      btnStart.click();
    }
  });

  btnStart.addEventListener("click", handleStart);
  updateStartButton();
}

async function handleStart() {
  const annotatorId = $("#input-annotator").value.trim();
  if (!annotatorId) return;

  const testType = document.querySelector('input[name="test-type"]:checked').value;
  state.annotatorId = annotatorId;
  state.testType = testType;

  const btnStart = $("#btn-start");
  const errorEl = $("#welcome-error");
  btnStart.disabled = true;
  errorEl.textContent = "";

  try {
    // Start session -- the server returns a progress summary.
    state.progress = await postJson("/api/session/start", {
      annotator_id: annotatorId,
      test_type: testType,
    });

    // Load first page.
    await loadNextPage();
  } catch (err) {
    errorEl.textContent = err.message || "启动失败，请重试";
    btnStart.disabled = false;
  }
}

// ---- Page Loading ----

async function loadNextPage() {
  const url = `/api/session/next-page?annotator_id=${encodeURIComponent(state.annotatorId)}&test_type=${encodeURIComponent(state.testType)}`;
  const page = await fetchJson(url);

  if (page.done) {
    // All questions exhausted.
    await refreshProgress();
    renderDone();
    switchView("done");
    return;
  }

  state.currentPage = page;
  state.answers = {};
  state.focusedQuestionIndex = -1;

  renderQuiz();
  switchView("quiz");
}

async function refreshProgress() {
  try {
    const url = `/api/session/progress?annotator_id=${encodeURIComponent(state.annotatorId)}`;
    state.progress = await fetchJson(url);
  } catch (_) {
    // Non-critical; keep existing progress.
  }
}

// ---- Render: Quiz ----

function renderQuiz() {
  const page = state.currentPage;
  if (!page) return;

  // Top bar.
  const pageNum = (state.progress?.pages_completed || 0) + 1;
  const scenesComplete = state.progress?.scenes_complete || 0;
  const totalScenes = state.progress?.total_scenes || 0;
  $("#topbar-progress").textContent = `第 ${pageNum} 页`;
  $("#topbar-scenes").textContent = `已完成 ${scenesComplete}/${totalScenes} 场景`;
  $("#topbar-scene-id").textContent = page.scene_id;
  $("#topbar-annotator").textContent = state.annotatorId;

  renderImages();
  renderObjects();
  renderQuestions();
  updateAnswerCount();
}

function renderImages() {
  const page = state.currentPage;
  const area = $("#image-area");
  area.innerHTML = "";

  if (state.testType === "multi_view") {
    area.className = "image-area multi-view";
    const mvPaths = page.images.labeled_multi_view || page.images.multi_view || [];
    mvPaths.forEach((path, i) => {
      const wrapper = document.createElement("div");
      wrapper.className = "view-image-wrapper";

      const label = document.createElement("span");
      label.className = "view-image-label";
      label.textContent = `View ${i}`;

      const img = document.createElement("img");
      img.src = resolveImageUrl(path);
      img.alt = `View ${i}`;
      img.loading = "eager";

      wrapper.appendChild(label);
      wrapper.appendChild(img);
      area.appendChild(wrapper);
    });
  } else {
    area.className = "image-area single-view";
    const svPath = page.images.labeled_single_view || page.images.single_view;
    if (svPath) {
      const img = document.createElement("img");
      img.src = resolveImageUrl(svPath);
      img.alt = "Scene view";
      img.loading = "eager";
      area.appendChild(img);
    }
  }
}

function renderObjects() {
  const list = $("#object-list");
  list.innerHTML = "";
  const objects = state.currentPage.objects || [];

  objects.forEach((obj) => {
    const item = document.createElement("div");
    item.className = "object-item";
    item.innerHTML = `
      <span class="object-id">${escapeHtml(obj.id)}</span>
      <span class="object-desc">${escapeHtml(obj.desc)}</span>
    `;
    list.appendChild(item);
  });
}

function renderQuestions() {
  const container = $("#question-list");
  container.innerHTML = "";
  const questions = state.currentPage.questions || [];

  questions.forEach((q, index) => {
    const card = document.createElement("div");
    card.className = "question-card";
    card.dataset.qid = q.qid;
    card.dataset.index = index;

    // Meta line: type badge + qid + repeat tag
    const meta = document.createElement("div");
    meta.className = "question-meta";

    const typeBadge = document.createElement("span");
    typeBadge.className = `badge badge-${q.type}`;
    typeBadge.textContent = q.type.toUpperCase();
    meta.appendChild(typeBadge);

    const qidSpan = document.createElement("span");
    qidSpan.className = "question-id";
    qidSpan.textContent = q.qid;
    meta.appendChild(qidSpan);

    if (q._repeat) {
      const repeatTag = document.createElement("span");
      repeatTag.className = "tag-repeat";
      repeatTag.textContent = "(重复)";
      meta.appendChild(repeatTag);
    }

    card.appendChild(meta);

    // Prompt text.
    const prompt = document.createElement("p");
    prompt.className = "question-prompt";
    prompt.textContent = q.prompt_text;
    card.appendChild(prompt);

    // Answer control.
    if (q.type === "qrr") {
      card.appendChild(buildQrrControl(q));
    } else if (q.type === "trr") {
      card.appendChild(buildTrrControl(q));
    } else if (q.type === "fdr") {
      card.appendChild(buildFdrControl(q));
    }

    container.appendChild(card);
  });
}

// ---- QRR Control ----

function buildQrrControl(q) {
  const group = document.createElement("div");
  group.className = "qrr-group";
  group.dataset.qid = q.qid;

  const options = [
    { value: "<", label: "前者更近", key: "1" },
    { value: "~=", label: "大致相等", key: "2" },
    { value: ">", label: "后者更近", key: "3" },
  ];

  options.forEach((opt) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "qrr-btn";
    btn.dataset.value = opt.value;
    btn.innerHTML = `${escapeHtml(opt.label)} <span class="qrr-shortcut">${opt.key}</span>`;

    btn.addEventListener("click", () => {
      // Toggle off if already selected.
      if (state.answers[q.qid] === opt.value) {
        delete state.answers[q.qid];
        btn.classList.remove("active");
      } else {
        state.answers[q.qid] = opt.value;
        // Update active state.
        group.querySelectorAll(".qrr-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
      }
      markCardAnswered(q.qid);
      updateAnswerCount();
    });

    group.appendChild(btn);
  });

  return group;
}

// ---- TRR Control ----

function buildTrrControl(q) {
  const select = document.createElement("select");
  select.className = "trr-select";
  select.dataset.qid = q.qid;

  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "请选择钟面方向";
  select.appendChild(placeholder);

  for (let h = 1; h <= 12; h++) {
    const opt = document.createElement("option");
    opt.value = h;
    opt.textContent = `${h} 点钟`;
    select.appendChild(opt);
  }

  select.addEventListener("change", () => {
    if (select.value) {
      state.answers[q.qid] = parseInt(select.value, 10);
    } else {
      delete state.answers[q.qid];
    }
    markCardAnswered(q.qid);
    updateAnswerCount();
  });

  return select;
}

// ---- FDR Control ----

function buildFdrControl(q) {
  const container = document.createElement("div");
  container.className = "fdr-container";
  container.dataset.qid = q.qid;

  // Use ranking_candidates from the server, or fallback to objects minus anchor.
  const candidates = q.ranking_candidates || [];

  // Candidates label.
  const candLabel = document.createElement("div");
  candLabel.className = "fdr-candidates-label";
  candLabel.textContent = "点击物体，按从近到远排序:";
  container.appendChild(candLabel);

  // Candidate buttons.
  const candDiv = document.createElement("div");
  candDiv.className = "fdr-candidates";

  candidates.forEach((cand) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "fdr-candidate-btn";
    btn.dataset.objId = cand.id;
    btn.textContent = `${cand.id}: ${cand.desc}`;

    btn.addEventListener("click", () => {
      if (btn.classList.contains("used")) return;
      addToFdrRanking(q.qid, cand.id, cand.desc, container);
    });

    candDiv.appendChild(btn);
  });
  container.appendChild(candDiv);

  // Ranking display.
  const rankLabel = document.createElement("div");
  rankLabel.className = "fdr-ranking-label";
  rankLabel.textContent = "当前排序 (近 -> 远):";
  container.appendChild(rankLabel);

  const rankList = document.createElement("div");
  rankList.className = "fdr-ranking-list";
  container.appendChild(rankList);

  // Reset button.
  const resetBtn = document.createElement("button");
  resetBtn.type = "button";
  resetBtn.className = "fdr-reset-btn";
  resetBtn.textContent = "重置排序";
  resetBtn.addEventListener("click", () => {
    resetFdrRanking(q.qid, container);
  });
  container.appendChild(resetBtn);

  return container;
}

function addToFdrRanking(qid, objId, desc, container) {
  // Initialize answer array if needed.
  if (!Array.isArray(state.answers[qid])) {
    state.answers[qid] = [];
  }

  // Check if already in ranking.
  if (state.answers[qid].includes(objId)) return;

  state.answers[qid].push(objId);

  // Update candidate button state.
  const btn = container.querySelector(`.fdr-candidate-btn[data-obj-id="${objId}"]`);
  if (btn) btn.classList.add("used");

  // Add to ranking list display.
  const rankList = container.querySelector(".fdr-ranking-list");
  const index = state.answers[qid].length;
  const item = document.createElement("span");
  item.className = "fdr-rank-item";
  item.innerHTML = `<span class="fdr-rank-index">#${index}</span> ${escapeHtml(objId)}`;
  rankList.appendChild(item);

  markCardAnswered(qid);
  updateAnswerCount();
}

function resetFdrRanking(qid, container) {
  state.answers[qid] = [];

  // Reset candidate buttons.
  container.querySelectorAll(".fdr-candidate-btn").forEach((btn) => {
    btn.classList.remove("used");
  });

  // Clear ranking display.
  const rankList = container.querySelector(".fdr-ranking-list");
  rankList.innerHTML = "";

  markCardAnswered(qid);
  updateAnswerCount();
}

// ---- Answer Count ----

function markCardAnswered(qid) {
  const card = document.querySelector(`.question-card[data-qid="${qid}"]`);
  if (!card) return;
  const answered = isAnswered(qid);
  card.classList.toggle("answered", answered);
}

function isAnswered(qid) {
  const val = state.answers[qid];
  if (val === undefined || val === null) return false;
  if (Array.isArray(val)) return val.length > 0;
  if (typeof val === "string") return val.trim() !== "";
  return true;
}

function countAnswered() {
  const questions = state.currentPage?.questions || [];
  return questions.filter((q) => isAnswered(q.qid)).length;
}

function updateAnswerCount() {
  const total = state.currentPage?.questions?.length || 0;
  const answered = countAnswered();
  const text = `已答: ${answered}/${total}`;

  const countEl = $("#answer-count");
  const bottomEl = $("#bottom-completion");
  if (countEl) countEl.textContent = text;
  if (bottomEl) bottomEl.textContent = text;

  // Enable/disable submit button.
  const btnSubmit = $("#btn-submit-page");
  if (btnSubmit) btnSubmit.disabled = answered < total;
}

// ---- Submit Page ----

async function handleSubmitPage() {
  const page = state.currentPage;
  if (!page) return;

  const questions = page.questions || [];
  const answered = countAnswered();

  if (answered < questions.length) {
    showToast(`还有 ${questions.length - answered} 道题未回答`, true);
    return;
  }

  const btnSubmit = $("#btn-submit-page");
  btnSubmit.disabled = true;
  btnSubmit.innerHTML = '<span class="loading-spinner"></span>提交中...';

  const elapsed = getElapsedSeconds();

  const responses = questions.map((q) => {
    let answer = state.answers[q.qid];
    // Normalize FDR: ensure it is an array.
    if (q.type === "fdr" && !Array.isArray(answer)) {
      answer = answer ? [answer] : [];
    }
    return { qid: q.qid, answer };
  });

  try {
    const result = await postJson("/api/session/submit-page", {
      page_id: page.page_id,
      annotator_id: state.annotatorId,
      scene_id: page.scene_id,
      test_type: state.testType,
      submitted_at: new Date().toISOString(),
      elapsed_seconds: Math.round(elapsed * 100) / 100,
      responses,
    });

    // Update progress from submission result.
    if (result.progress) {
      state.progress = result.progress;
    } else {
      await refreshProgress();
    }

    renderSubmitted(elapsed);
    switchView("submitted");
  } catch (err) {
    btnSubmit.disabled = false;
    btnSubmit.textContent = "提交本页";
    showToast("提交失败: " + (err.message || "未知错误"), true);
  }
}

// ---- Render: Submitted ----

function renderSubmitted(elapsedSeconds) {
  const page = state.currentPage;
  const questions = page?.questions || [];
  const nNew = page?.n_new || questions.filter((q) => !q._repeat).length;
  const nRepeat = page?.n_repeat || questions.filter((q) => q._repeat).length;

  const statsEl = $("#submitted-stats");
  statsEl.innerHTML = `
    <div class="stat-row"><span class="stat-label">本页题目</span><span class="stat-value">${questions.length} 道 (${nNew} 新 + ${nRepeat} 重复)</span></div>
    <div class="stat-row"><span class="stat-label">场景</span><span class="stat-value">${page?.scene_id || "--"}</span></div>
    <div class="stat-row"><span class="stat-label">耗时</span><span class="stat-value">${formatTime(elapsedSeconds)}</span></div>
  `;

  const p = state.progress || {};
  const pagesCompleted = p.pages_completed || 0;
  const answeredQ = p.answered_questions || 0;
  const totalQ = p.total_questions || 1;
  const pct = p.progress_pct || 0;
  const totalScenes = p.total_scenes || 0;
  const scenesComplete = p.scenes_complete || 0;

  const progressEl = $("#submitted-progress");
  progressEl.innerHTML = `
    <div class="progress-text">总体进度: ${pagesCompleted} 页完成, ${answeredQ}/${totalQ} 道题已答, ${scenesComplete}/${totalScenes} 场景完成</div>
    <div class="progress-bar-wrapper">
      <div class="progress-bar-fill" style="width: ${Math.min(pct, 100)}%"></div>
    </div>
  `;
}

// ---- Render: Done ----

function renderDone() {
  const p = state.progress || {};
  const statsEl = $("#done-stats");
  statsEl.innerHTML = `
    <div class="stat-row"><span class="stat-label">标注员</span><span class="stat-value">${escapeHtml(state.annotatorId || "--")}</span></div>
    <div class="stat-row"><span class="stat-label">总页数</span><span class="stat-value">${p.pages_completed || 0}</span></div>
    <div class="stat-row"><span class="stat-label">总题数</span><span class="stat-value">${p.answered_questions || 0}/${p.total_questions || 0}</span></div>
    <div class="stat-row"><span class="stat-label">完成场景</span><span class="stat-value">${p.scenes_complete || 0}/${p.total_scenes || 0}</span></div>
  `;
}

// ---- Keyboard Shortcuts ----

function handleKeyboard(e) {
  if (state.view !== "quiz") return;
  // Ignore if typing in an input/select.
  const tag = e.target.tagName.toLowerCase();
  if (tag === "input" || tag === "textarea" || tag === "select") return;

  const questions = state.currentPage?.questions || [];
  if (!questions.length) return;

  // Find the focused question.
  const focusIdx = state.focusedQuestionIndex;
  const focusedQ = focusIdx >= 0 && focusIdx < questions.length ? questions[focusIdx] : null;

  // QRR shortcuts: 1, 2, 3
  if (focusedQ && focusedQ.type === "qrr" && ["1", "2", "3"].includes(e.key)) {
    e.preventDefault();
    const valueMap = { "1": "<", "2": "~=", "3": ">" };
    const value = valueMap[e.key];
    state.answers[focusedQ.qid] = value;

    // Update UI.
    const card = document.querySelector(`.question-card[data-qid="${focusedQ.qid}"]`);
    if (card) {
      const group = card.querySelector(".qrr-group");
      if (group) {
        group.querySelectorAll(".qrr-btn").forEach((b) => {
          b.classList.toggle("active", b.dataset.value === value);
        });
      }
    }
    markCardAnswered(focusedQ.qid);
    updateAnswerCount();
    // Auto-advance to next question.
    if (focusIdx < questions.length - 1) {
      setFocusedQuestion(focusIdx + 1);
    }
    return;
  }

  // TRR shortcut: type number directly for focused TRR question.
  if (focusedQ && focusedQ.type === "trr") {
    const num = parseInt(e.key, 10);
    if (num >= 1 && num <= 9) {
      e.preventDefault();
      // If the user quickly types "1" then "2" for "12", handle two-digit input.
      handleTrrKeyInput(focusedQ.qid, num);
      return;
    }
  }

  // Arrow keys to navigate between questions.
  if (e.key === "ArrowDown" || e.key === "j") {
    e.preventDefault();
    setFocusedQuestion(Math.min(focusIdx + 1, questions.length - 1));
  } else if (e.key === "ArrowUp" || e.key === "k") {
    e.preventDefault();
    setFocusedQuestion(Math.max(focusIdx - 1, 0));
  }
}

// TRR two-digit number input handling.
let trrKeyBuffer = "";
let trrKeyTimer = null;

function handleTrrKeyInput(qid, digit) {
  trrKeyBuffer += String(digit);

  if (trrKeyTimer) clearTimeout(trrKeyTimer);

  // If buffer is "1", wait briefly for a possible second digit (10, 11, 12).
  if (trrKeyBuffer === "1") {
    trrKeyTimer = setTimeout(() => {
      applyTrrValue(qid, 1);
      trrKeyBuffer = "";
    }, 400);
    return;
  }

  // Two digits entered.
  const val = parseInt(trrKeyBuffer, 10);
  trrKeyBuffer = "";
  if (trrKeyTimer) clearTimeout(trrKeyTimer);

  if (val >= 1 && val <= 12) {
    applyTrrValue(qid, val);
  } else {
    // Invalid two-digit: use just the second digit.
    applyTrrValue(qid, digit);
  }
}

function applyTrrValue(qid, value) {
  state.answers[qid] = value;
  const card = document.querySelector(`.question-card[data-qid="${qid}"]`);
  if (card) {
    const sel = card.querySelector(".trr-select");
    if (sel) sel.value = value;
  }
  markCardAnswered(qid);
  updateAnswerCount();
}

function setFocusedQuestion(index) {
  // Remove old focus.
  $$(".question-card.keyboard-focus").forEach((c) => c.classList.remove("keyboard-focus"));

  state.focusedQuestionIndex = index;
  const questions = state.currentPage?.questions || [];
  if (index < 0 || index >= questions.length) return;

  const qid = questions[index].qid;
  const card = document.querySelector(`.question-card[data-qid="${qid}"]`);
  if (card) {
    card.classList.add("keyboard-focus");
    card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

// ---- HTML Escape ----

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// ---- Event Wiring ----

function wireEvents() {
  // Submit page button.
  const btnSubmit = $("#btn-submit-page");
  if (btnSubmit) {
    btnSubmit.addEventListener("click", handleSubmitPage);
  }

  // Next page button.
  const btnNext = $("#btn-next-page");
  if (btnNext) {
    btnNext.addEventListener("click", async () => {
      btnNext.disabled = true;
      btnNext.innerHTML = '<span class="loading-spinner"></span>加载中...';
      try {
        await loadNextPage();
      } catch (_) {
        btnNext.disabled = false;
        btnNext.textContent = "继续下一页";
      }
    });
  }

  // End session button.
  const btnEnd = $("#btn-end-session");
  if (btnEnd) {
    btnEnd.addEventListener("click", () => {
      if (confirm("确定要结束测试吗？你可以之后再继续。")) {
        stopTimer();
        switchView("welcome");
        // Reset state.
        state.annotatorId = null;
        state.currentPage = null;
        state.progress = null;
        state.answers = {};
        // Re-enable start button and clear input.
        const btnStart = $("#btn-start");
        const input = $("#input-annotator");
        if (input) input.value = "";
        if (btnStart) btnStart.disabled = true;
      }
    });
  }

  // Restart button (done view).
  const btnRestart = $("#btn-restart");
  if (btnRestart) {
    btnRestart.addEventListener("click", () => {
      state.annotatorId = null;
      state.currentPage = null;
      state.progress = null;
      state.answers = {};
      const input = $("#input-annotator");
      if (input) input.value = "";
      const btnStart = $("#btn-start");
      if (btnStart) btnStart.disabled = true;
      switchView("welcome");
    });
  }

  // Keyboard shortcuts.
  document.addEventListener("keydown", handleKeyboard);

  // Click on question card to set focus.
  document.addEventListener("click", (e) => {
    const card = e.target.closest(".question-card");
    if (card && state.view === "quiz") {
      const index = parseInt(card.dataset.index, 10);
      if (!isNaN(index)) {
        setFocusedQuestion(index);
      }
    }
  });
}

// ---- Init ----

window.addEventListener("DOMContentLoaded", () => {
  initWelcome();
  wireEvents();
  switchView("welcome");
});
