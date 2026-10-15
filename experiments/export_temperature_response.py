"""Build a portable, checksummed temperature study ZIP without annual raw data."""
import hashlib
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]


def main():
    files=set()
    for directory in ('utils','torch_influence','conf','tests','experiments','research/cnn_temperature_response'):
        for path in (ROOT/directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
                files.add(path.relative_to(ROOT))
    for filename in ('func_operation.py','clean_data.py','eval_unlearn.py','eval_unchange.py',
                     'run_real.py','requirements.txt','COST_REPRODUCTION_REPORT.md',
                     'scripts/analyze_cnn_sensitivity.py'):
        files.add(Path(filename))
    title='cnn_temperature_response_study'
    archive=ROOT/'exports/cnn_temperature_response.zip'
    archive.parent.mkdir(exist_ok=True)
    guide='''# CNN 温度扰动实验下载包

先打开 research/cnn_temperature_response/导师汇报.md；三张实验图和一张流程图在该目录 figures/，均有 PNG/PDF。
完整方法、实际数据、局限和命令在 research/cnn_temperature_response/README.md。

此包含四种真实冻结模型、一个真实测试样本及官方选中节点全年原温度。
不含完整年度数据、历史 influence 文件或虚拟环境；复算本次五档曲线无需重新训练或下载全年数据。
如果要从头重新生成所有权重，则仍需完整数据和原影响分数，见成本复现报告。

Python 3.10，依赖见 requirements.txt；普通运行优先 GUROBI，auto 可回退兼容求解器。
在解压的项目根目录执行：

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python experiments/input_sensitivity.py --replay-from research/cnn_temperature_response --output-dir /tmp/temperature-replay-new
.venv/bin/python experiments/report_temperature_response.py
.venv/bin/python experiments/validate_temperature_response.py --replay-dir /tmp/temperature-replay-new
.venv/bin/python -m pytest -q
```

绘制流程图需系统安装 Graphviz（dot）；现成 PNG/PDF 已包含。/tmp 输出目录须为空。
Cost 沿用原目标函数和约束。结果来自单节点、单样本、单种子干预，不作天气因果或总体推断。
SHA256SUMS 可校验每个文件。
'''
    with tempfile.TemporaryDirectory(prefix='temperature-export-',dir='/tmp') as temp:
        folder=Path(temp)/title
        for relative in sorted(files):
            target=folder/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/relative,target)
        (folder/'README.md').write_text(guide,encoding='utf-8')
        members=sorted(path for path in folder.rglob('*') if path.is_file())
        checksums=''.join(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(folder)}\n' for path in members)
        (folder/'SHA256SUMS').write_text(checksums,encoding='utf-8')
        with ZipFile(archive,'w',ZIP_DEFLATED,compresslevel=9) as bundle:
            for path in sorted(folder.rglob('*')):
                if path.is_file():
                    bundle.write(path,str(path.relative_to(Path(temp))))
    checksum=hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix('.zip.sha256').write_text(f'{checksum}  {archive.name}\n',encoding='utf-8')
    with ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        print(f'{len(bundle.namelist())} files; {archive.stat().st_size} bytes; SHA256 {checksum}')
    print(archive)


if __name__=='__main__':
    main()
