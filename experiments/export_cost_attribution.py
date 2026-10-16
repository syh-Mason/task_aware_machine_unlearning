"""Package the cost-path cohort, verified predecessor and standalone sources."""
import hashlib
from pathlib import Path
import tempfile
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]


def main():
    archive=ROOT/'exports/cnn_cost_attribution.zip'
    files=set()
    for directory in ('utils','torch_influence','conf','experiments','scripts','tests',
                      'research/cnn_temperature_response','research/cnn_cost_attribution'):
        for path in (ROOT/directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
                files.add(path.relative_to(ROOT))
    files.update(p.relative_to(ROOT) for p in ROOT.glob('*.py'))
    files.update(map(Path,['requirements.txt','COST_REPRODUCTION_REPORT.md']))
    guide='''# CNN 温度—预测—运行成本：节点诊断与多日期验证

首先打开 research/cnn_cost_attribution/导师汇报.md。
四张图（PNG/PDF）在 research/cnn_cost_attribution/figures/。
完整逐场景预测、Cost、节点诊断、Shapley边际、每次求解轨迹及校验记录均在该目录。
research/cnn_temperature_response 是上一轮参考实验，供模型一致性和验收测试核对。

包含真实28条样本、全1752条测试时刻的真实总负荷选样池、训练温度σ和四种冻结预测头。
不包含全年原始数据、虚拟环境或训练历史；本次重放无需训练或下载全年数据。
保留了上一轮的检查点、说明和测试，不将单点或分层样本结果当作总体结论。

Python 3.10，依赖见 requirements.txt。从本解压目录运行：

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python experiments/cost_attribution.py --replay-from research/cnn_cost_attribution --output-dir /tmp/cost-path-replay-new
.venv/bin/python experiments/report_cost_attribution.py
.venv/bin/python experiments/validate_cost_attribution.py --replay-dir /tmp/cost-path-replay-new
.venv/bin/python -m pytest -q
```

输出目录须为空。读取现成结果无需运行以上命令。
逐字节相同的复算验证基于本次固定版本和相同求解器；换平台/求解器可能有数值差异。
普通QP auto可选择兼容求解器，原目标和约束不变。
SHA256SUMS记录每个文件校验值。结果仅覆盖2019年前73天、一个模型种子、节点1温度。
'''
    prefix='cnn_cost_attribution_study'
    with tempfile.TemporaryDirectory(dir='/tmp',prefix='cost-attribution-export-') as temp:
        members={str(relative):(ROOT/relative).read_bytes() for relative in sorted(files)}
        members['README.md']=guide.encode()
        sums=''.join(f'{hashlib.sha256(content).hexdigest()}  {name}\n' for name,content in sorted(members.items()))
        members['SHA256SUMS']=sums.encode()
        archive.parent.mkdir(exist_ok=True)
        with ZipFile(archive,'w',ZIP_DEFLATED,compresslevel=9) as bundle:
            for name,content in sorted(members.items()):
                bundle.writestr(f'{prefix}/{name}',content)
    checksum=hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix('.zip.sha256').write_text(f'{checksum}  {archive.name}\n')
    with ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        print(f'{len(bundle.namelist())} files; {archive.stat().st_size} bytes; SHA256 {checksum}')


if __name__=='__main__':main()
