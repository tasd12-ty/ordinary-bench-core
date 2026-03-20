#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from common import (
    QUESTION_TYPES,
    QRR_OPTIONS,
    TRR_OPTIONS,
    build_object_catalog,
    build_object_lookup,
    format_human_question,
    iter_question_batches,
    load_question_documents,
    load_scene_doc,
    render_labeled_image,
)


def discover_scene_ids(
    questions_dir: str,
    split: Optional[str] = None,
    scene: Optional[str] = None,
) -> List[str]:
    if scene:
        return [scene]

    base = Path(questions_dir)
    scene_ids = set()
    for qtype in QUESTION_TYPES:
        type_dir = base / qtype
        if type_dir.exists():
            scene_ids.update(path.stem for path in type_dir.glob("*.json"))

    if not scene_ids:
        scene_ids.update(path.stem for path in base.glob("*.json"))

    ordered = sorted(scene_ids)
    if split:
        ordered = [scene_id for scene_id in ordered if scene_id.startswith(split)]
    return ordered


def build_image_bundle(
    scene_id: str,
    scene_doc: dict,
    single_view_images_dir: str,
    multi_view_images_dir: str,
    output_images_dir: Path,
) -> dict:
    def view_pixel_signature(view: dict) -> tuple:
        return tuple(
            (
                obj.get("id"),
                tuple(obj.get("pixel_coords", [])),
            )
            for obj in sorted(view.get("objects", []), key=lambda row: row.get("id", ""))
        )

    scene_image_dir = output_images_dir / scene_id
    scene_image_dir.mkdir(parents=True, exist_ok=True)

    single_source = Path(single_view_images_dir) / f"{scene_id}.png"
    single_raw_output = scene_image_dir / "single_view.png"
    single_output = scene_image_dir / "single_view_labels.png"
    shutil.copy2(single_source, single_raw_output)
    render_labeled_image(str(single_source), build_object_catalog(scene_doc), str(single_output))

    bundle = {
        "single_view": {
            "view_id": "single_view",
            "label": "单视图",
            "image_path": f"images/{scene_id}/{single_raw_output.name}",
        },
        "reference_view": {
            "view_id": "single_view_reference",
            "label": "编号参考图",
            "image_path": f"images/{scene_id}/{single_output.name}",
        },
        "multi_view": [],
        "multi_view_labels_reliable": True,
    }

    meta_path = Path(multi_view_images_dir) / scene_id / "metadata.json"
    if not meta_path.exists():
        return bundle

    with open(meta_path) as f:
        meta = json.load(f)

    view_signatures = {}
    for view in meta.get("views", []):
        signature = view_pixel_signature(view)
        if signature in view_signatures:
            bundle["multi_view_labels_reliable"] = False
            break
        view_signatures[signature] = view.get("view_id")

    for view in meta.get("views", []):
        view_id = view.get("view_id") or Path(view.get("image_path", "")).stem
        source_path = Path(multi_view_images_dir) / scene_id / view["image_path"]
        raw_output_path = scene_image_dir / f"{view_id}.png"
        shutil.copy2(source_path, raw_output_path)

        if bundle["multi_view_labels_reliable"]:
            labeled_output_path = scene_image_dir / f"{view_id}_labels.png"
            render_labeled_image(
                str(source_path),
                build_object_catalog({"objects": view.get("objects", [])}),
                str(labeled_output_path),
            )
            display_output_path = labeled_output_path
        else:
            display_output_path = raw_output_path

        camera = view.get("camera", {})
        bundle["multi_view"].append({
            "view_id": view_id,
            "label": view_id.replace("_", " ").title(),
            "image_path": f"images/{scene_id}/{display_output_path.name}",
            "raw_image_path": f"images/{scene_id}/{raw_output_path.name}",
            "camera": {
                "azimuth": camera.get("azimuth"),
                "elevation": camera.get("elevation"),
                "distance": camera.get("distance"),
            },
        })

    return bundle


def build_task_document(
    scene_id: str,
    scene_doc: dict,
    question_docs: Dict[str, dict],
    image_bundle: dict,
) -> dict:
    objects = build_object_catalog(scene_doc)
    object_lookup = build_object_lookup(objects)
    batches = []
    counts = {qtype: 0 for qtype in QUESTION_TYPES}

    for batch in iter_question_batches(question_docs):
        enriched_questions = []
        for question in batch["questions"]:
            question_copy = dict(question)
            question_copy["variant"] = question_copy.get("variant", "disjoint")
            question_copy["prompt_text"] = format_human_question(question_copy, object_lookup)
            if question_copy["type"] == "qrr":
                question_copy["answer_options"] = QRR_OPTIONS
            elif question_copy["type"] == "trr":
                question_copy["answer_options"] = TRR_OPTIONS
            elif question_copy["type"] == "fdr":
                candidates = [obj for obj in objects if obj["id"] != question_copy["anchor"]]
                question_copy["ranking_candidates"] = candidates
                question_copy["answer_example"] = ", ".join(obj["id"] for obj in candidates)
            enriched_questions.append(question_copy)
            counts[question_copy["type"]] += 1

        batches.append({
            "batch_id": batch["batch_id"],
            "question_type": batch["question_type"],
            "n_questions": len(enriched_questions),
            "questions": enriched_questions,
        })

    return {
        "schema_version": 1,
        "scene_id": scene_id,
        "n_objects": scene_doc.get("n_objects", len(objects)),
        "objects": objects,
        "image_path": image_bundle["single_view"]["image_path"],
        "images": image_bundle,
        "available_test_types": (
            ["single_view", "multi_view"]
            if image_bundle["multi_view"]
            else ["single_view"]
        ),
        "source": {
            "scene_file": f"data-gen/output/scenes/{scene_id}.json",
            "question_files": {
                qtype: f"VLM-test/output/questions/{qtype}/{scene_id}.json"
                for qtype in QUESTION_TYPES
                if qtype in question_docs
            },
        },
        "batches": batches,
        "summary": {
            "total_questions": sum(counts.values()),
            "qrr_questions": counts["qrr"],
            "trr_questions": counts["trr"],
            "fdr_questions": counts["fdr"],
            "n_batches": len(batches),
        },
    }


def render_scene_page(task_doc: dict, page_image_path: str) -> str:
    page_doc = dict(task_doc)
    page_doc["page_image_path"] = page_image_path
    task_json = json.dumps(page_doc, ensure_ascii=False)
    title = f"Human Baseline Task - {task_doc['scene_id']}"
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f3efe4;
      --paper: #fffaf0;
      --ink: #20201b;
      --muted: #6f6a5d;
      --line: #d4c9b1;
      --accent: #8a4f2a;
      --accent-soft: #e7d5bf;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(138, 79, 42, 0.08), transparent 30%),
        linear-gradient(180deg, #efe7d6, var(--bg));
    }}
    header {{
      padding: 28px 32px 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 250, 240, 0.9);
      position: sticky;
      top: 0;
      backdrop-filter: blur(8px);
      z-index: 10;
    }}
    header h1 {{
      margin: 0 0 10px;
      font-size: 28px;
      letter-spacing: 0.02em;
    }}
    header p {{
      margin: 6px 0;
      color: var(--muted);
      max-width: 980px;
    }}
    .toolbar {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 14px;
      align-items: center;
    }}
    .toolbar input {{
      padding: 10px 12px;
      min-width: 220px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: white;
      font: inherit;
    }}
    .toolbar button {{
      padding: 10px 16px;
      border: 0;
      border-radius: 999px;
      background: var(--accent);
      color: white;
      font: inherit;
      cursor: pointer;
    }}
    .toolbar button.secondary {{
      background: var(--accent-soft);
      color: var(--ink);
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(320px, 420px) minmax(420px, 1fr);
      gap: 22px;
      padding: 24px 32px 40px;
      align-items: start;
    }}
    .panel {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 10px 30px rgba(92, 68, 43, 0.06);
    }}
    .scene-image {{
      width: 100%;
      display: block;
      border-radius: 14px;
      border: 1px solid var(--line);
      margin-bottom: 14px;
    }}
    .object-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    .object-table th,
    .object-table td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
      text-align: left;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }}
    .meta-card {{
      padding: 10px 12px;
      border-radius: 12px;
      background: #f6f0e4;
      border: 1px solid var(--line);
    }}
    .meta-card strong {{
      display: block;
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 4px;
    }}
    .batch {{
      margin-bottom: 18px;
      padding: 14px;
      border-radius: 16px;
      background: #fcf8ef;
      border: 1px solid var(--line);
    }}
    .batch h3 {{
      margin: 0 0 8px;
      font-size: 18px;
    }}
    .question {{
      padding: 14px 0;
      border-top: 1px dashed var(--line);
    }}
    .question:first-child {{
      border-top: 0;
      padding-top: 0;
    }}
    .qid {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 6px;
    }}
    .prompt {{
      margin: 0 0 10px;
      line-height: 1.45;
    }}
    .help {{
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 13px;
    }}
    select,
    input[type="text"] {{
      width: 100%;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: white;
      font: inherit;
    }}
    .footer-note {{
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    .status {{
      margin-top: 12px;
      min-height: 20px;
      color: var(--muted);
    }}
    @media (max-width: 980px) {{
      .layout {{
        grid-template-columns: 1fr;
        padding: 20px 16px 28px;
      }}
      header {{
        padding: 20px 16px 16px;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{task_doc['scene_id']}</h1>
    <p>图中对象标签只显示偏移文字和引导线，不叠加点位。导出的结果会按当前模型回复格式组织，后续可以直接走统一评分、聚合和场景重建流程。</p>
    <p>QRR 请选择 <code>&lt;</code> / <code>~=</code> / <code>&gt;</code>；TRR 请选择 1-12；FDR 请按 “最近到最远” 输入逗号分隔的 object ID 顺序。</p>
    <div class="toolbar">
      <input id="annotator" placeholder="annotator id" />
      <button id="export-btn">导出响应 JSON</button>
      <button id="raw-btn" class="secondary">导出 raw_response 文本</button>
      <span id="answer-count"></span>
    </div>
    <div class="status" id="status"></div>
  </header>
  <main class="layout">
    <section class="panel">
      <img id="scene-image" class="scene-image" alt="scene image">
      <div class="meta-grid" id="meta-grid"></div>
      <table class="object-table">
        <thead>
          <tr><th>ID</th><th>Description</th></tr>
        </thead>
        <tbody id="object-table-body"></tbody>
      </table>
      <p class="footer-note">建议标注时只参考当前页面导出的 JSON，不要手工改字段名。分析脚本会直接消费导出的 schema。</p>
    </section>
    <section class="panel" id="question-root"></section>
  </main>
  <script id="task-data" type="application/json">{task_json}</script>
  <script>
    const task = JSON.parse(document.getElementById("task-data").textContent);
    const pageOpenedAt = Date.now();
    const state = {{}};

    function slugify(text) {{
      return (text || "")
        .trim()
        .replace(/[^0-9A-Za-z._-]+/g, "-")
        .replace(/-+/g, "-")
        .replace(/^-|-$/g, "") || "anonymous";
    }}

    function parseFdr(value) {{
      const trimmed = (value || "").trim();
      if (!trimmed) return null;
      if (trimmed.startsWith("[")) {{
        try {{
          const parsed = JSON.parse(trimmed);
          if (Array.isArray(parsed)) {{
            return parsed.map(item => String(item).trim()).filter(Boolean);
          }}
        }} catch (error) {{
          // fall through
        }}
      }}
      return trimmed.split(/[,\\n]+/).map(part => part.trim()).filter(Boolean);
    }}

    function collectResponses() {{
      const annotatorInput = document.getElementById("annotator").value;
      const annotatorId = slugify(annotatorInput);
      const elapsedSeconds = Math.round((Date.now() - pageOpenedAt) / 10) / 100;
      const batches = [];
      const allResponses = [];

      for (const batch of task.batches) {{
        const batchResponses = [];
        for (const question of batch.questions) {{
          const rawValue = state[question.qid];
          let answer = null;
          if (question.type === "qrr") {{
            answer = rawValue || null;
          }} else if (question.type === "trr") {{
            answer = rawValue ? Number(rawValue) : null;
          }} else if (question.type === "fdr") {{
            answer = parseFdr(rawValue);
          }}
          if (answer !== null && answer !== "" && !(Array.isArray(answer) && answer.length === 0)) {{
            batchResponses.push({{ qid: question.qid, answer }});
            allResponses.push({{ qid: question.qid, answer }});
          }}
        }}
        batches.push({{
          batch_id: batch.batch_id,
          question_type: batch.question_type,
          elapsed_seconds: elapsedSeconds,
          responses: batchResponses,
          raw_response: JSON.stringify(batchResponses),
        }});
      }}

      return {{
        schema_version: 1,
        scene_id: task.scene_id,
        annotator_id: annotatorId,
        model: `human/${{annotatorId}}`,
        submitted_at: new Date().toISOString(),
        n_objects: task.n_objects,
        total_questions: task.summary.total_questions,
        batches,
        responses: allResponses,
        raw_response: JSON.stringify(allResponses),
      }};
    }}

    function downloadText(filename, text) {{
      const blob = new Blob([text], {{ type: "application/json;charset=utf-8" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      link.click();
      URL.revokeObjectURL(url);
    }}

    function updateAnswerCount() {{
      const answered = Object.values(state).filter(value => {{
        if (value === null || value === undefined) return false;
        return String(value).trim() !== "";
      }}).length;
      document.getElementById("answer-count").textContent =
        `已回答 ${{answered}} / ${{task.summary.total_questions}}`;
    }}

    function exportJson() {{
      const payload = collectResponses();
      const missing = task.summary.total_questions - payload.responses.length;
      if (missing > 0 && !window.confirm(`还有 ${{missing}} 题未填写，仍然导出？`)) {{
        return;
      }}
      const filename = `${{task.scene_id}}__${{payload.annotator_id}}.json`;
      downloadText(filename, JSON.stringify(payload, null, 2));
      document.getElementById("status").textContent =
        `已导出 ${{filename}}，raw_response 和 batch 结构已对齐。`;
    }}

    function exportRaw() {{
      const payload = collectResponses();
      const filename = `${{task.scene_id}}__${{payload.annotator_id}}__raw_response.json`;
      downloadText(filename, payload.raw_response);
      document.getElementById("status").textContent =
        `已导出 ${{filename}}。`;
    }}

    function renderObjects() {{
      document.getElementById("scene-image").src = task.page_image_path;
      const metaRoot = document.getElementById("meta-grid");
      const stats = [
        ["Objects", task.n_objects],
        ["Questions", task.summary.total_questions],
        ["Batches", task.summary.n_batches],
        ["QRR / TRR / FDR", `${{task.summary.qrr_questions}} / ${{task.summary.trr_questions}} / ${{task.summary.fdr_questions}}`],
      ];
      for (const [label, value] of stats) {{
        const card = document.createElement("div");
        card.className = "meta-card";
        card.innerHTML = `<strong>${{label}}</strong><span>${{value}}</span>`;
        metaRoot.appendChild(card);
      }}

      const tbody = document.getElementById("object-table-body");
      for (const obj of task.objects) {{
        const tr = document.createElement("tr");
        tr.innerHTML = `<td><code>${{obj.id}}</code></td><td>${{obj.desc}}</td>`;
        tbody.appendChild(tr);
      }}
    }}

    function renderQuestions() {{
      const root = document.getElementById("question-root");
      for (const batch of task.batches) {{
        const section = document.createElement("section");
        section.className = "batch";
        const heading = batch.question_type.toUpperCase();
        section.innerHTML = `<h3>${{heading}} · ${{batch.batch_id}}</h3>`;

        for (const question of batch.questions) {{
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

          if (question.type === "qrr" || question.type === "trr") {{
            const select = document.createElement("select");
            select.innerHTML = `<option value="">请选择</option>`;
            const options = question.type === "qrr" ? question.answer_options : question.answer_options;
            for (const optionValue of options) {{
              const option = document.createElement("option");
              option.value = optionValue;
              option.textContent = optionValue;
              select.appendChild(option);
            }}
            select.addEventListener("change", (event) => {{
              state[question.qid] = event.target.value;
              updateAnswerCount();
            }});
            wrapper.appendChild(select);
          }} else if (question.type === "fdr") {{
            const input = document.createElement("input");
            input.type = "text";
            input.placeholder = question.answer_example;
            input.addEventListener("input", (event) => {{
              state[question.qid] = event.target.value;
              updateAnswerCount();
            }});
            wrapper.appendChild(input);

            const help = document.createElement("p");
            help.className = "help";
            help.textContent = "按最近到最远输入 object ID，逗号分隔。也可以直接输入 JSON list。";
            wrapper.appendChild(help);
          }}

          section.appendChild(wrapper);
        }}

        root.appendChild(section);
      }}
    }}

    renderObjects();
    renderQuestions();
    updateAnswerCount();
    document.getElementById("export-btn").addEventListener("click", exportJson);
    document.getElementById("raw-btn").addEventListener("click", exportRaw);
  </script>
</body>
</html>
"""


def render_index(manifest: dict) -> str:
    rows = []
    for scene in manifest["scenes"]:
        rows.append(
            "<tr>"
            f"<td><a href=\"pages/{scene['scene_id']}.html\">{scene['scene_id']}</a></td>"
            f"<td>{scene['n_objects']}</td>"
            f"<td>{scene['total_questions']}</td>"
            f"<td>{scene['qrr_questions']}</td>"
            f"<td>{scene['trr_questions']}</td>"
            f"<td>{scene['fdr_questions']}</td>"
            f"<td><a href=\"json/{scene['scene_id']}.json\">task json</a></td>"
            "</tr>"
        )
    table_rows = "\n".join(rows)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Human Baseline Tasks</title>
  <style>
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      background: linear-gradient(180deg, #f0eadb, #f6f3eb);
      color: #222018;
    }}
    main {{
      max-width: 980px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 32px;
    }}
    p {{
      color: #6f6a5d;
      line-height: 1.5;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #fffaf0;
      border-radius: 16px;
      overflow: hidden;
      margin-top: 18px;
      border: 1px solid #d4c9b1;
    }}
    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid #e6dcc8;
      text-align: left;
    }}
    th {{
      background: #f6f0e4;
    }}
    a {{
      color: #8a4f2a;
      text-decoration: none;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Human Baseline Tasks</h1>
    <p>共生成 {manifest['n_scenes']} 个场景任务。每个页面都会导出与当前模型回复对齐的响应 JSON，可直接交给分析脚本转成统一的 `raw/` 和 `scenes/` 结果目录。</p>
    <table>
      <thead>
        <tr>
          <th>Scene</th>
          <th>Objects</th>
          <th>Total</th>
          <th>QRR</th>
          <th>TRR</th>
          <th>FDR</th>
          <th>Files</th>
        </tr>
      </thead>
      <tbody>
        {table_rows}
      </tbody>
    </table>
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate human baseline task pages")
    parser.add_argument("--questions-dir", default="VLM-test/output/questions")
    parser.add_argument("--scenes-dir", default="data-gen/output/scenes")
    parser.add_argument("--images-dir", default="data-gen/output/images/single_view")
    parser.add_argument("--multi-view-images-dir", default="data-gen/output/images/multi_view")
    parser.add_argument("--output-dir", default="human-baseline/output/tasks")
    parser.add_argument("--split", default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    args = parser.parse_args()

    scene_ids = discover_scene_ids(args.questions_dir, split=args.split, scene=args.scene)
    if args.max_scenes is not None:
        scene_ids = scene_ids[:args.max_scenes]

    if not scene_ids:
        raise SystemExit("No scenes found for task generation")

    output_dir = Path(args.output_dir)
    images_out = output_dir / "images"
    json_out = output_dir / "json"
    pages_out = output_dir / "pages"
    images_out.mkdir(parents=True, exist_ok=True)
    json_out.mkdir(parents=True, exist_ok=True)
    pages_out.mkdir(parents=True, exist_ok=True)

    manifest = {"n_scenes": 0, "scenes": []}

    for index, scene_id in enumerate(scene_ids, start=1):
        question_docs = load_question_documents(scene_id, args.questions_dir)
        scene_doc = load_scene_doc(scene_id, args.scenes_dir)

        image_bundle = build_image_bundle(
            scene_id=scene_id,
            scene_doc=scene_doc,
            single_view_images_dir=args.images_dir,
            multi_view_images_dir=args.multi_view_images_dir,
            output_images_dir=images_out,
        )

        task_doc = build_task_document(
            scene_id=scene_id,
            scene_doc=scene_doc,
            question_docs=question_docs,
            image_bundle=image_bundle,
        )

        with open(json_out / f"{scene_id}.json", "w") as f:
            json.dump(task_doc, f, indent=2, ensure_ascii=False)

        html = render_scene_page(task_doc, page_image_path=f"../{task_doc['image_path']}")
        with open(pages_out / f"{scene_id}.html", "w") as f:
            f.write(html)

        manifest["scenes"].append({
            "scene_id": scene_id,
            "n_objects": task_doc["n_objects"],
            "total_questions": task_doc["summary"]["total_questions"],
            "qrr_questions": task_doc["summary"]["qrr_questions"],
            "trr_questions": task_doc["summary"]["trr_questions"],
            "fdr_questions": task_doc["summary"]["fdr_questions"],
            "available_test_types": task_doc["available_test_types"],
            "multi_view_count": len(task_doc["images"]["multi_view"]),
        })
        print(
            f"[{index}/{len(scene_ids)}] {scene_id}: "
            f"{task_doc['summary']['total_questions']} questions "
            f"(QRR {task_doc['summary']['qrr_questions']}, "
            f"TRR {task_doc['summary']['trr_questions']}, "
            f"FDR {task_doc['summary']['fdr_questions']})"
        )

    manifest["n_scenes"] = len(manifest["scenes"])
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    with open(output_dir / "index.html", "w") as f:
        f.write(render_index(manifest))

    print(f"\nSaved tasks to {output_dir}")


if __name__ == "__main__":
    main()
