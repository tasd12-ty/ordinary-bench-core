# NeurIPS Official Template Workspace

本目录是后续论文写作的唯一工作目录。

说明：

- 截至 `2026-03-15`，NeurIPS 2026 官方模板尚未公开发布。
- 本目录因此采用最新可获得的官方模板：`neurips_2025.sty`。
- 中文稿与英文稿都放在本目录下，后续写作继续在这里进行，不再依赖 `papers/` 根目录中的旧草稿。

文件结构：

- `main_zh.tex`：中文版主稿
- `main_en.tex`：英文版主稿
- `zh_abstract.tex` / `zh_introduction.tex` / `zh_related_work.tex`：中文版三节正文
- `en_abstract.tex` / `en_introduction.tex` / `en_related_work.tex`：英文版三节正文
- `neurips_2025.sty`：官方样式文件
- `neurips_2025_official_sample.tex`：官方示例
- `spatiotemporal_vlm_refs.bib`：参考文献库

编译方式：

- 中文版：`latexmk -xelatex main_zh.tex`
- 英文版：`latexmk -pdf main_en.tex`

后续建议：

- 中文版先继续打磨论证与结构。
- 英文版以中文版为依据逐步对齐。
- 后续新增章节、图表、实验结果都优先放在本目录下。
