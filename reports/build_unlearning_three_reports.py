"""Build three editable literature reports, PDFs, diagrams, and a download ZIP.

Run from repository root with the workspace Python (python-docx, PyMuPDF, Pillow).
Requires pandoc, Graphviz, and soffice. Does not train or modify any model.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

import fitz
from PIL import Image, ImageDraw, ImageOps
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from unlearning_three_content import REPORTS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/unlearning_three_methods'
QA = Path('/workspace/.cloud-setup/task-aware/unlearning-three-qa')
SOURCE = Path('/workspace/.cloud-setup/task-aware/unlearning-three-sources')
FONT = 'Noto Sans CJK SC'
BRANCH = 'reports/unlearning-three-methods'
BASE = f'https://github.com/syh-Mason/task_aware_machine_unlearning/blob/{BRANCH}'
DATE = '2026-10-22'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diagrams():
    specs = {
        '01_sisa': [
            ('data', '训练集 D\n稳定样本 ID'),
            ('shards', '独立分片\n每片独立模型'),
            ('slices', '阶段 / 切片\n保存干净前缀状态'),
            ('delete', '删除请求\n定位受影响分片'),
            ('replay', '恢复并重放\n排除全部删除 ID'),
            ('result', '固定规则聚合\n14 节点预测 → Cost')
        ],
        '02_influence': [
            ('feature', '固定 CNN 核心\n64 维表示 + 偏置'),
            ('gradient', '删除集梯度变化\n含样本数 / 正则项'),
            ('hessian', '剩余目标 Hessian\n求解 Hᵣ v = Δ'),
            ('update', '预测头更新 θ + v\n与直接重拟合对照'),
            ('forecast', '14 节点预测\n同一真实负荷'),
            ('cost', '两阶段优化\nCost + 状态 / 残差')
        ],
        '03_ssd_layers': [
            ('cl', '持续学习工具\n参数重要性估计'),
            ('importance', '全集 I_D / 删除集 I_f\n相同模型与损失'),
            ('select', '参数选择 + 衰减\nα / λ_SSD'),
            ('layers', '显式层白名单\nhead → lin1 → Conv2'),
            ('verify', '核验冻结层哈希\n匹配的重训参考'),
            ('eval', '预测与遗忘接近度\n两阶段 Cost')
        ]
    }
    (OUT / 'assets').mkdir(exist_ok=True)
    for name, nodes in specs.items():
        lines = ['digraph G {',
                 'graph [bgcolor="white",rankdir=TB,nodesep=.32,ranksep=.32,pad=.15];',
                 'node [shape=box,style="rounded,filled",fillcolor="#EDF4F6",color="#217A8A",fontname="Noto Sans CJK SC",fontsize=19,margin=".16,.12"];',
                 'edge [color="#486375",penwidth=1.7,arrowsize=.8];']
        for key, label in nodes:
            lines.append(key + ' [label=' + json.dumps(label, ensure_ascii=False) + '];')
        # Two rows, left-to-right flow on both rows, preserving a compact diagram.
        a, b, c, d, e, f = [x[0] for x in nodes]
        lines += [f'{{rank=same;{a};{b};{c};}}', f'{{rank=same;{d};{e};{f};}}',
                  f'{a}->{b}->{c};', f'{d}->{e}->{f};',
                  f'{c}->{f} [style=invis];', f'{a}->{d} [style=invis];',
                  f'{c}->{d} [constraint=false,color="#B55B43"];',
                  f'{d} [fillcolor="#F7EEE9",color="#B55B43"];',
                  '}']
        dot = OUT / 'assets' / (name + '.dot')
        dot.write_text('\n'.join(lines) + '\n')
        for fmt in ['png', 'svg', 'pdf']:
            subprocess.run(['dot', '-Gdpi=170', '-T' + fmt, str(dot), '-o',
                            str(dot.with_suffix('.' + fmt))], check=True, capture_output=True)


def reference():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.2677), Inches(11.6929)
    sec.left_margin = sec.right_margin = Inches(.73)
    sec.top_margin = sec.bottom_margin = Inches(.67)
    sec.header_distance = sec.footer_distance = Inches(.28)
    for style in doc.styles:
        if style.type in (1, 2):
            style.font.name = FONT
            style._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), FONT)
    for n in ['Normal', 'Body Text', 'First Paragraph']:
        if n not in doc.styles:
            continue
        s = doc.styles[n]
        s.font.size = Pt(10.5)
        s.paragraph_format.line_spacing = 1.16
        s.paragraph_format.space_after = Pt(6)
    for n, size in [('Heading 1', 18), ('Heading 2', 12.5), ('Heading 3', 11)]:
        s = doc.styles[n]
        s.font.size = Pt(size)
        s.font.color.rgb = RGBColor.from_string('193B50')
        s.paragraph_format.space_before = Pt(10)
        s.paragraph_format.space_after = Pt(8)
        s.paragraph_format.keep_with_next = True
    for n in ['Source Code', 'Verbatim Char']:
        if n in doc.styles:
            doc.styles[n].font.name = 'DejaVu Sans Mono'
            doc.styles[n].font.size = Pt(8)
    header = sec.header.paragraphs[0]
    header.text = '机器遗忘文献研究  |  SISA · Influence Function · 持续学习启发的 SSD'
    header.runs[0].font.size = Pt(8)
    footer = sec.footer.paragraphs[0]
    footer.alignment = 1
    r = footer.add_run('文献与方法报告 · ' + DATE + '  |  ')
    r.font.size = Pt(8)
    field = OxmlElement('w:fldSimple')
    field.set(qn('w:instr'), 'PAGE')
    footer._p.append(field)
    path = QA / 'reference.docx'
    doc.save(path)
    return path


def build_documents():
    ref = reference()
    for report in REPORTS:
        stem = report['stem']
        md = OUT / (stem + '.md')
        md.write_text('\n\n'.join(report['pages']).rstrip() + '\n')
        path = md.with_suffix('.docx')
        r = subprocess.run(['pandoc', str(md), '--from=markdown+tex_math_dollars',
                            '--to=docx', '--standalone', '--resource-path=' + str(OUT),
                            '--reference-doc=' + str(ref), '-o', str(path)],
                           capture_output=True, text=True, check=True)
        if r.stderr:
            print(r.stderr, flush=True)
        assert 'Could not convert' not in r.stderr
        doc = Document(path)
        # Writer's OMML importer can replace non-parenthesis delimiter endings
        # with ')'. Keep fractions/scripts editable, but make those delimiters
        # literal math glyphs so absolute values, norms and brackets stay exact.
        for delimiter in reversed(list(doc._element.iter(qn('m:d')))):
            properties = delimiter.find(qn('m:dPr'))
            def char(name, fallback):
                el = properties.find(qn('m:' + name)) if properties is not None else None
                return fallback if el is None else el.get(qn('m:val'), fallback)
            left, right = char('begChr', '('), char('endChr', ')')
            if (left, right) == ('(', ')'):
                continue
            parts = []
            def glyph(value):
                if not value:
                    return
                run = OxmlElement('m:r')
                text = OxmlElement('m:t')
                text.text = value
                run.append(text)
                parts.append(run)
            glyph(left)
            for index, expression in enumerate(delimiter.findall(qn('m:e'))):
                if index:
                    glyph(char('sepChr', '|'))
                parts.extend(deepcopy(child) for child in expression)
            glyph(right)
            parent, index = delimiter.getparent(), delimiter.getparent().index(delimiter)
            parent.remove(delimiter)
            for offset, node in enumerate(parts):
                parent.insert(index + offset, node)
        headings = [p for p in doc.paragraphs if p.style.name == 'Heading 1']
        assert len(headings) == len(report['pages']) == 10
        for h in headings[1:]:
            h.paragraph_format.page_break_before = True
        # Rebuild tables using Word's standard skeleton. Preserve links/math XML.
        for old in list(doc.tables):
            table = doc.add_table(rows=len(old.rows), cols=len(old.columns))
            for a, b in zip(old.rows, table.rows):
                for c, d in zip(a.cells, b.cells):
                    for p in list(d._tc.findall(qn('w:p'))):
                        d._tc.remove(p)
                    for p in c._tc.findall(qn('w:p')):
                        d._tc.append(deepcopy(p))
                    for p in d.paragraphs:
                        p.style = 'Normal'
            old._tbl.addprevious(table._tbl)
            old._tbl.getparent().remove(old._tbl)
            table.style = 'Table Grid'
            table.autofit = False
            count = len(table.columns)
            weights = {2: [.35, .65], 3: [.34, .33, .33], 4: [.27, .25, .24, .24]}[count]
            width = 6.80
            tw = table._tbl.tblPr.find(qn('w:tblW'))
            tw.set(qn('w:type'), 'dxa')
            tw.set(qn('w:w'), str(round(width * 1440)))
            for col, fraction in zip(table.columns, weights):
                col.width = Inches(width * fraction)
            for i, row in enumerate(table.rows):
                row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
                if i == 0:
                    row._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
                for cell, fraction in zip(row.cells, weights):
                    cell.width = Inches(width * fraction)
                    if i == 0:
                        shade = OxmlElement('w:shd')
                        shade.set(qn('w:fill'), 'EAF2F5')
                        cell._tc.get_or_add_tcPr().append(shade)
                    for p in cell.paragraphs:
                        p.paragraph_format.line_spacing = 1.08
                        p.paragraph_format.space_after = Pt(4)
                        for run in p.runs:
                            run.font.size = Pt(9.2)
        # Prevent overly large embedded diagrams from consuming multiple pages.
        for shape in doc.inline_shapes:
            ratio = shape.height / shape.width
            maxwidth = Inches(6.65)
            if shape.width > maxwidth:
                shape.width = maxwidth
                shape.height = int(maxwidth * ratio)
        doc.core_properties.title = stem
        doc.core_properties.subject = '机器遗忘算法、数学推导、源码核对与电网迁移设计'
        doc.core_properties.author = 'task_aware_machine_unlearning 项目研究材料'
        doc.save(path)
        result = subprocess.run(['soffice', '-env:UserInstallation=file:///tmp/unlearning-three-lo',
                                 '--headless', '--convert-to', 'pdf', '--outdir', str(OUT), str(path)],
                                capture_output=True, text=True, timeout=100,
                                env={**os.environ, 'XDG_CACHE_HOME': '/workspace/.cache'})
        print(result.stdout.strip(), flush=True)
        if result.returncode or not path.with_suffix('.pdf').exists():
            raise RuntimeError(result.stdout + '\n' + result.stderr)


def evidence():
    external = []
    sources = [
        ('SISA', 'sisa', 'https://github.com/cleverhans-lab/machine-unlearning',
         ['README.md', 'sisa.py', 'sharded.py', 'aggregation.py']),
        ('Certified Data Removal', 'certified-removal', 'https://github.com/facebookresearch/certified-removal',
         ['README.md', 'test_removal.py']),
        ('SSD', 'ssd', 'https://github.com/if-loops/selective-synaptic-dampening',
         ['README.md', 'src/ssd.py', 'src/forget_random_strategies.py'])
    ]
    for title, dirname, url, files in sources:
        repo = SOURCE / dirname
        if not repo.exists():
            continue
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
        for file in files:
            external.append(dict(title=title, url=url + '/blob/' + commit + '/' + file,
                                 commit=commit, sha256=sha(repo / file),
                                 access='read_author_code_or_README', accessed=DATE))
    if (SOURCE / 'guo_pdf.pdf').exists():
        external.append(dict(title='Guo et al., ICML 2020 formal publication',
                             url='https://raw.githubusercontent.com/mlresearch/v119/gh-pages/guo20c/guo20c.pdf',
                             sha256=sha(SOURCE / 'guo_pdf.pdf'), pages=11,
                             access='full_text_read', accessed=DATE))
    for file, url, desc in [
        ('survey_readme.txt', 'https://github.com/tamlhp/awesome-machine-unlearning', 'author-maintained survey bibliography, not survey full text'),
        ('survey_repo_readme.txt', 'https://github.com/jessecu2024/unlearningsurvey', 'independent author survey bibliography used to disambiguate Jiang CSUR'),
        ('citations_readme.txt', 'https://github.com/crayon-go/machine-unlearning-papers', 'third-party citation snapshot, self-reported counts not independently verified')
    ]:
        p = SOURCE / file
        if p.exists():
            external.append(dict(title=desc, url=url, sha256=sha(p), access='index_read_only', accessed=DATE))
    blocked = []
    for file in ['initial_access.json', 'final_access.json', 'citation_access.json']:
        p = SOURCE / file
        if p.exists():
            for r in json.loads(p.read_text()):
                if 'error' in r:
                    blocked.append({k: r[k] for k in ['key', 'url', 'error'] if k in r})
    local = []
    for file in ['utils/net.py', 'utils/funcs.py', 'utils/dataset.py', 'clean_data.py',
                 'func_operation.py', 'utils/optimization.py',
                 'research/cnn_layer_paths/validation.json',
                 'research/cnn_temperature_response/frozen_models.npz',
                 'research/cnn_temperature_response/reference_inputs.npz']:
        local.append(dict(file=file, sha256=sha(ROOT / file)))
    manifest = dict(date=DATE, timezone='Asia/Shanghai',
                    scope='Three literature reports and synthetic mathematical checks; no new grid training, deletion benchmark, or cost solves',
                    paper_access={'SISA': 'author_code_read; full_text_blocked',
                                  'Guo_ICML2020': 'formal_full_text_11_pages_and_author_code_read',
                                  'SSD': 'author_code_read; full_text_blocked',
                                  'surveys': 'bibliographic_indices_read; survey_full_texts_blocked'},
                    sources=external, local_evidence=local, blocked_access=blocked,
                    citation_note='No live citation count or ranking claimed. Any snapshot is attributed to the third-party catalog, not independently verified.')
    path = OUT / 'source_manifest.json'
    if external:
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    elif path.exists():
        previous = json.loads(path.read_text())
        for source in previous['local_evidence']:
            assert sha(ROOT / source['file']) == source['sha256']
    else:
        raise RuntimeError('Source cache or committed manifest is required for provenance')
    # Keep only a few relevant rows; do not redistribute source papers/HTML.
    p = SOURCE / 'citations_readme.txt'
    if p.exists():
        rows = [line for line in p.read_text().splitlines()
                if ('| Nguyên et al. | [A Survey of Machine Unlearning]' in line
                    or '| Bourtoule et al. | [Machine Unlearning]' in line
                    or '| Guo et al. | [Certified Data Removal from Machine Learning Models]' in line
                    or '| Foster et al. | [Fast Machine Unlearning Without Retraining' in line)]
        (OUT / 'citation_snapshot_note.md').write_text(
            '# 引用量筛选的历史线索\n\n'
            '来源：[第三方自动目录](https://github.com/crayon-go/machine-unlearning-papers)。'
            '该目录自报引文数据刷新于 2026-05-10（KST），声称来源为 Semantic Scholar；'
            '本轮读取日期为 2026-10-22（北京时间）。以下只是该目录展示值，未独立核验，'
            '不代表今天的引用量、总排名或跨平台合并计数。Nguyen 的期刊与预印本版本可能有不同归并方式。\n\n'
            '| 作者 | 题名 | 期刊/会议（目录原标签） | 类别 | 代码 | 目录展示引用数 |\n'
            '|---|---|---|---|---|---:|\n' + '\n'.join(rows) + '\n\n'
            '目录存在书目错配：其 Xu / arXiv:2306.03558 推荐项与 Jiang 的 CSUR DOI:10.1145/3603620 混用。'
            '本报告不采用该错配行，也不依靠目录的自动元数据判断算法性质。'
            '综述作者 README 直接确认了 Nguyen 综述的 2022 预印本与 2025 TIST 版本。\n')


def overview():
    lines = ['# 三条机器遗忘路线：报告与阅读顺序', '',
             '**本轮交付：三份独立研究报告，每份均有可编辑 Word、PDF 和 Markdown，附三张流程图、14 项教学数值校验、来源清单。**', '',
             f'[一次下载完整 ZIP]({BASE}/exports/unlearning_three_reports.zip?raw=true)', '',
             '| 报告 | 代表论文 | Word | PDF |', '|---|---|---|---|']
    for report in REPORTS:
        stem = report['stem']
        lines.append(f'| {report["short"]} | ' +
                     {'SISA：分组重训练': 'Bourtoule et al., S&P 2021',
                      '影响函数与认证删除': 'Guo et al., ICML 2020',
                      '持续学习启发的 SSD': 'Foster et al., AAAI 2024'}[report['short']] +
                     f' | [下载]({stem}.docx?raw=true) | [阅读]({stem}.pdf) |')
    lines += ['', '## 先读综述，再读代表论文', '',
              '1. [Nguyen 等：A Survey of Machine Unlearning](https://arxiv.org/abs/2209.02299)：2022 预印本；作者确认的 [2025 ACM TIST 版](https://doi.org/10.1145/3749987)。',
              '2. [Jiang 等：Machine Unlearning: A Survey，ACM CSUR](https://doi.org/10.1145/3603620)：作为第二份总览。不要与 Xu 等同名预印本混用。',
              '3. 按 SISA → 影响函数 → SSD 顺序读三份报告，最后比较电网迁移方案。', '',
              '目前实时学术引文接口受限，不能给出独立核实的当前排名。[历史引用线索](citation_snapshot_note.md) 只保留第三方 2026-05-10 快照的原始展示值与局限，不充当当前引用量。', '',
              '## 三条路线的关键区别', '',
              '| 路线 | 删除机制 | 参考与保证边界 | 电网项目中的第一步 |',
              '|---|---|---|---|',
              '| SISA | 受影响分片从干净检查点重训 | 同一分片训练流程；完整状态和数据依赖需要审计 | 独立整 CNN 分片或条件于固定核心的多头分片 |',
              '| Influence / Newton | 用梯度变化和曲率修正参数 | 一般局部近似；固定特征二次目标可精确；认证另需假设 | 与现有固定核心仿射头直接重拟合核对 |',
              '| SSD | 按删除集重要性选择并衰减参数 | 持续学习启发的经验近似，没有自动认证 | head-only 参数白名单与梯度平方重要性 |', '',
              '## 阅读证据与执行状态', '',
              '- Guo：已取得并读取 PMLR 正式全文 11 页及作者实现。',
              '- SISA、SSD：作者源码与 README 已读，全文访问 403；报告明确为代码核对与独立推导，未冒充全文精读。',
              '- 两篇综述：已核对公开文献索引；全文本轮未取得。',
              '- 14 项合成教学检查通过，0 失败；不是论文基准、电网实验或隐私认证。',
              '- 本轮不重新训练模型、不运行新的电网 Cost、不修改旧实验记录。电网迁移表均为待实施计划。', '',
              '## 复算与重新生成', '',
              '仓库根目录执行：', '', '```bash',
              'python experiments/verify_unlearning_reports.py',
              'python reports/build_unlearning_three_reports.py', '```', '',
              '数值检查仅需 NumPy。文档生成还需 python-docx、PyMuPDF、Pillow，以及 pandoc、Graphviz、soffice 和 Noto CJK 字体。',
              '首次环境中的来源缓存保存在工作区工具目录；其他环境可使用已提交的来源清单核对本地证据，不要求重新下载论文。', '',
              '独立分支：`reports/unlearning-three-methods`；不改 master，不自动合并。', '',
              '[来源与访问状态](source_manifest.json) · [数学检查](numerical_checks.json) · [文档检查](document_validation.json) · [SHA256](SHA256SUMS)', '',
              f'日期：{DATE}（北京时间）。']
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')


def inspect():
    records = []
    for report in REPORTS:
        stem = report['stem']
        pdf_path = OUT / (stem + '.pdf')
        docx_path = OUT / (stem + '.docx')
        with ZipFile(docx_path) as z:
            assert z.testzip() is None
            xml = z.read('word/document.xml').decode()
            rels = z.read('word/_rels/document.xml.rels').decode()
            math_count = xml.count('<m:oMath>')
            assert math_count >= 4, (stem, math_count)
            assert 'arxiv.org' in rels and 'github.com' in rels
            hyperlinks = rels.count('/hyperlink')
        pdf = fitz.open(pdf_path)
        chunks, outside, images = [], [], []
        for i, page in enumerate(pdf):
            text = page.get_text()
            chunks.append(text)
            assert len(text.strip()) > 50, (stem, i, 'almost empty page')
            for block in page.get_text('dict')['blocks']:
                if block['type'] != 0:
                    continue
                for line in block['lines']:
                    for span in line['spans']:
                        r = fitz.Rect(span['bbox'])
                        if r.x0 < -.8 or r.y0 < -.8 or r.x1 > page.rect.width + .8 or r.y1 > page.rect.height + .8:
                            outside.append(dict(page=i + 1, text=span['text'], bbox=list(r)))
            image_path = QA / f'{report["stem"]}_{i+1:02d}.png'
            page.get_pixmap(matrix=fitz.Matrix(.9, .9), alpha=False).save(image_path)
            images.append(image_path)
        all_text = '\n'.join(chunks)
        assert '\ufffd' not in all_text
        assert not outside, outside
        # Every narrative section title must survive PDF conversion.
        # CJK quote glyphs can be extracted before the rest of a heading even
        # when visually placed correctly; compare textual words without quotes.
        def normalize_heading(text):
            return re.sub(r'[\s“”‘’\"\u0027]', '', text)
        normalized = normalize_heading(all_text)
        for content in report['pages']:
            title = content.splitlines()[0][2:]
            assert normalize_heading(title) in normalized, (stem, title)
        cols, w, h = 3, 330, 465
        sheet = Image.new('RGB', (cols * w, ((len(images) + cols - 1) // cols) * h), '#DDE5E8')
        draw = ImageDraw.Draw(sheet)
        for i, image_path in enumerate(images):
            with Image.open(image_path) as im:
                thumb = ImageOps.contain(im, (w - 12, h - 28))
                x, y = (i % cols) * w + (w - thumb.width) // 2, (i // cols) * h + 5
                sheet.paste(thumb, (x, y))
                draw.text(((i % cols) * w + 10, (i // cols) * h + h - 20), f'Page {i+1}', fill='black')
        sheet_path = QA / (stem + '_contact.png')
        sheet.save(sheet_path)
        records.append(dict(file=pdf_path.name, pages=len(pdf),
                            expected_sections=10, native_office_math_objects=math_count,
                            hyperlinks=hyperlinks, page_character_counts=[len(x.strip()) for x in chunks],
                            out_of_page_text=outside, sha256=sha(pdf_path),
                            contact_sheet=str(sheet_path)))
        pdf.close()
    result = dict(date=DATE, documents=records, total_pdf_pages=sum(r['pages'] for r in records),
                  expected_reports=3, synthetic_checks_passed=14, grid_experiments_run=False)
    (OUT / 'document_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def package():
    files = sorted(p for p in OUT.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (OUT / 'SHA256SUMS').write_text(''.join(sha(p) + '  ' + p.relative_to(OUT).as_posix() + '\n' for p in files))
    export = ROOT / 'exports/unlearning_three_reports.zip'
    with ZipFile(export, 'w', ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():
                z.write(p, 'unlearning_three_reports/' + p.relative_to(OUT).as_posix())
        for file in ['experiments/verify_unlearning_reports.py', 'reports/build_unlearning_three_reports.py',
                     'reports/unlearning_three_content.py']:
            z.write(ROOT / file, 'repository_scripts/' + file)
    with ZipFile(export) as z:
        assert z.testzip() is None
        for p in OUT.rglob('*'):
            if p.is_file():
                assert z.read('unlearning_three_reports/' + p.relative_to(OUT).as_posix()) == p.read_bytes()
    export.with_suffix('.zip.sha256').write_text(sha(export) + '  ' + export.name + '\n')
    print(json.dumps(dict(zip=str(export), bytes=export.stat().st_size, sha256=sha(export)), ensure_ascii=False))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / 'experiments/verify_unlearning_reports.py'),
                    '--output', str(OUT / 'numerical_checks.json')], check=True, capture_output=True)
    evidence()
    diagrams()
    overview()
    build_documents()
    inspect()
    package()


if __name__ == '__main__':
    main()
