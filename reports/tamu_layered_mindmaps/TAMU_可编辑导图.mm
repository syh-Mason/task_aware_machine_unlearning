<?xml version='1.0' encoding='utf-8'?>
<map version="1.0.1">
  <node TEXT="Task-Aware Machine Unlearning｜学习分层（非原文章节）">
    <node TEXT="学习分层（非原文章节号）｜原文全文未取得；依据作者摘要、代码与标准推导整理" />
    <node TEXT="01 研究动机与问题" COLOR="#2460A6">
      <node TEXT="问题从哪里来">
        <node TEXT="隐私撤回：指定数据不再参与学习">
          <node TEXT="删除对象是历史训练样本的影响，不是删除一个温度特征，也不是要求模型无法预测该样本。" FOLDED="true" />
        </node>
        <node TEXT="数据质量：恶意或不可靠样本">
          <node TEXT="作者摘要列出恶意数据等删除动机；是否有害要依据实际数据和评价标准判断。" FOLDED="true" />
        </node>
        <node TEXT="部署之后：模型已经学习过数据">
          <node TEXT="从文件中删掉原始记录，并不会自动更新已经训练出来的模型参数。" FOLDED="true" />
        </node>
      </node>
      <node TEXT="必须比较什么">
        <node TEXT="遗忘结果：与剩余数据重训比较">
          <node TEXT="需要先明确重训算法、删除集合和保证定义，不能只以误差变化判断是否遗忘成功。" FOLDED="true" />
        </node>
        <node TEXT="计算代价：更新与从头拟合">
          <node TEXT="影响函数和 Newton 更新提供减少某些重新拟合计算的途径；实际速度仍取决于规模与数值求解。" FOLDED="true" />
        </node>
        <node TEXT="保留效用：预测指标与经济性">
          <node TEXT="删掉有帮助的样本可能损伤预测或运行表现；性能与成本是不同评价维度。" FOLDED="true" />
        </node>
      </node>
      <node TEXT="方法逐层递进">
        <node TEXT="直接遗忘：更新剩余集预测头">
          <node TEXT="先建立可核对的删除更新，再与剩余数据重新拟合比较。" FOLDED="true" />
        </node>
        <node TEXT="PAMU：按预测指标调权">
          <node TEXT="在剩余样本重加权时关注 MSE 或 MAPE 的局部影响。" FOLDED="true" />
        </node>
        <node TEXT="TAMU：按电网 Cost 调权">
          <node TEXT="将下游调度代价引入评分，讨论删除后的经济性。" FOLDED="true" />
        </node>
      </node>
    </node>
    <node TEXT="02 数据与预测模型" COLOR="#167D8D">
      <node TEXT="数据如何组织">
        <node TEXT="TX-123BT：合成测试系统数据">
          <node TEXT="官方数据用于构建负荷预测任务；不能将其称为已接入 ERCOT 的实际观测。" FOLDED="true" />
        </node>
        <node TEXT="输入与目标：天气/时间 → 负荷">
          <node TEXT="温度等天气变量与周期时间编码为输入，Load 是预测目标。标准化和数据划分的顺序影响评估解释。" FOLDED="true" />
        </node>
        <node TEXT="D = U ∪ R：原集、删除集、剩余集">
          <node TEXT="对当前神经网络头部流程，D 通常指敏感拟合集；20% 删除比例应说明其分母。" FOLDED="true" />
        </node>
      </node>
      <node TEXT="模型怎样预测">
        <node TEXT="Linear：仿射负荷预测">
          <node TEXT="线性模型通过特征的仿射组合给出负荷预测。" FOLDED="true" />
        </node>
        <node TEXT="CNN / MLP-Mixer：提取共同隐藏表示">
          <node TEXT="先在 core 数据上训练特征提取器，再冻结核心，准备敏感样本的隐藏特征。" FOLDED="true" />
        </node>
        <node TEXT="当前 CNN：140 → 64 → 14">
          <node TEXT="输入形状为 (N,1,14,10)，64 维 tanh 隐表示进入仿射预测头，输出 14 维标准化负荷。" FOLDED="true" />
        </node>
      </node>
      <node TEXT="参数与尺度">
        <node TEXT="预测头 θ：当前为 910 个参数">
          <node TEXT="14×64 个连接权重和 14 个偏置；所有模型连接参数与训练样本权重 w 是不同对象。" FOLDED="true" />
        </node>
        <node TEXT="训练目标：MSE 拟合参数">
          <node TEXT="PAMU/TAMU 以其他指标指导样本调权，但最终预测头仍使用加权 MSE 拟合。" FOLDED="true" />
        </node>
        <node TEXT="反标准化：p.u. → 调度；MW 展示">
          <node TEXT="模型输出先恢复为 p.u. 负荷；MW 展示再乘 baseMVA。64 个隐藏坐标不能直接当成物理天气特征。" FOLDED="true" />
        </node>
      </node>
    </node>
    <node TEXT="03 直接遗忘与影响函数" COLOR="#8C5A19">
      <node TEXT="影响函数起点">
        <node TEXT="经验风险最小化：训练梯度在最优点为零">
          <node TEXT="对选定训练目标和参数定义驻点条件，再对样本增权量求导。" FOLDED="true" />
        </node>
        <node TEXT="梯度 g_i：样本推动参数的方向">
          <node TEXT="梯度是该样本损失对参数的局部导数，不是温度对预测输出的 Jacobian。" FOLDED="true" />
        </node>
        <node TEXT="Hessian H：用曲率校正更新">
          <node TEXT="单样本额外增权的局部参数导数为 −H⁻¹g_i；完整权重约定还含样本数归一化。" FOLDED="true" />
        </node>
      </node>
      <node TEXT="删除更新">
        <node TEXT="剩余目标 L_R：在旧参数 θ₀ 处计算">
          <node TEXT="计算剩余集平均 MSE 的梯度 g_R 与 Hessian H_R。" FOLDED="true" />
        </node>
        <node TEXT="Newton 一步：θ_del = θ₀ − H_R⁻¹g_R">
          <node TEXT="实现通常求解线性系统，而非显式构造整个逆矩阵。" FOLDED="true" />
        </node>
        <node TEXT="重拟合对照：检查参数和预测">
          <node TEXT="二次满秩条件下可与剩余数据精确拟合一致；CG 容差、阻尼、秩亏会影响数值结果。" FOLDED="true" />
        </node>
      </node>
      <node TEXT="数值与边界">
        <node TEXT="iHVP / CG：求解 H a = v">
          <node TEXT="Hessian-vector product 支持 CG 计算逆 Hessian 向量积，检查收敛状态和残差。" FOLDED="true" />
        </node>
        <node TEXT="正负号与缩放：增权 ≠ 删除分数">
          <node TEXT="若使用删除集平均梯度，更新可写成加号并带 k/m；评分文件的符号和分母必须核对。" FOLDED="true" />
        </node>
        <node TEXT="适用范围：二次预测头及明确假设">
          <node TEXT="参数接近重拟合不足以推出认证隐私删除；核心曾使用删除样本时，仅更新头部也不能保证移除核心影响。" FOLDED="true" />
        </node>
      </node>
    </node>
    <node TEXT="04 PAMU：性能感知重加权" COLOR="#5F52A3">
      <node TEXT="定义外层指标">
        <node TEXT="J = MSE / MAPE：预测性能指导调权">
          <node TEXT="先选择需要补偿的评价标准；该标准的梯度决定哪些样本增权有利。" FOLDED="true" />
        </node>
        <node TEXT="性能梯度 ∇θJ：参数怎样影响指标">
          <node TEXT="它与训练样本 MSE 梯度不同；训练 Hessian 仍取自剩余数据 MSE。" FOLDED="true" />
        </node>
        <node TEXT="样本分数 s_i：估计增权的局部效果">
          <node TEXT="代码在原参数 θ₀ 评分并使用剩余 Hessian；理想剩余最优点的权重导数锚点不同。" FOLDED="true" />
        </node>
      </node>
      <node TEXT="权重线性规划">
        <node TEXT="完整权重 w：分数正则倾向减权">
          <node TEXT="局部评分为正意味着增权使该外层指标上升；最优方向还受总预算约束。" FOLDED="true" />
        </node>
        <node TEXT="L1 / Linf 预算：限制整体与单点偏离">
          <node TEXT="当前 Linf≤1 给出 0≤w≤2；平均 L1=0.1 是权重偏离预算，不是删除 10%。" FOLDED="true" />
        </node>
        <node TEXT="LP 代理目标：没有权重和守恒约束">
          <node TEXT="当前代码没有 Σw=m；增添这一约束会改变问题。目标中与 w 无关的常数可省略；分数共同正比例缩放不改变 LP 最优解集合。" FOLDED="true" />
        </node>
      </node>
      <node TEXT="恢复预测头">
        <node TEXT="加权剩余 MSE：重新构造梯度与 Hessian">
          <node TEXT="最终更新使用所选 w 对应的 H_R,w 与 g_R,w，而非直接复用未加权 Hessian。" FOLDED="true" />
        </node>
        <node TEXT="加权 Newton 更新：得到新预测头参数">
          <node TEXT="加权二次系统满秩且求解准确时，更新等于该加权剩余数据的精确拟合。" FOLDED="true" />
        </node>
        <node TEXT="真实评价：重新算 MSE / MAPE / Cost">
          <node TEXT="线性代理最优不等于真实指标或全部新样本最优；PAMU 也可以评价 Cost。" FOLDED="true" />
        </node>
      </node>
    </node>
    <node TEXT="05 下游任务：两阶段电网成本" COLOR="#1D7562">
      <node TEXT="阶段一 Dispatch">
        <node TEXT="输入：预测负荷：反标准化后截断为非负">
          <node TEXT="先将标准化预测恢复为 p.u.，沿原逻辑将负预测截为 0。" FOLDED="true" />
        </node>
        <node TEXT="决策：PG、LS1、相角：发电安排与削减">
          <node TEXT="满足发电范围、线路双向限额、功率平衡、LS1 非负与参考角。" FOLDED="true" />
        </node>
        <node TEXT="cost1：发电费用 + 阶段一削减">
          <node TEXT="阶段一 LS1 费用是总成本及可微表达的一部分，不能遗漏。" FOLDED="true" />
        </node>
      </node>
      <node TEXT="阶段二 Redispatch">
        <node TEXT="输入：真实负荷 + PG：同一场景保持真实值不变">
          <node TEXT="阶段二使用数据中 ground truth，不以扰动后的温度重新生成真实负荷。" FOLDED="true" />
        </node>
        <node TEXT="决策：LS2、GS、相角：处理短缺与过量发电">
          <node TEXT="当前平衡式使用 PG−GS；GS 是代码中的过量发电处理量，不是完整物理储能模型。" FOLDED="true" />
        </node>
        <node TEXT="cost2：LS2 / GS 的一次与二次费">
          <node TEXT="阶段二沿原线路、功率平衡、非负和参考角约束；没有额外自由增发变量。" FOLDED="true" />
        </node>
      </node>
      <node TEXT="任务评价">
        <node TEXT="总 Cost = cost1 + cost2：保留全部原费用项">
          <node TEXT="当前 λ=3500、κ=1250、ρ=700、η=250；目标函数单位不能未经校准解释为实际货币。" FOLDED="true" />
        </node>
        <node TEXT="空间与方向效应：同等误差未必同等代价">
          <node TEXT="高估/低估、节点位置、发电与线路约束会使预测误差的经济后果不同。" FOLDED="true" />
        </node>
        <node TEXT="数学模型不变：求解状态与可行性要检查">
          <node TEXT="兼容求解器必须求同一个问题；不能跳过 Cost，也不能为解释模型擅自加入 LS≤负荷、GS≤PG 上界。" FOLDED="true" />
        </node>
      </node>
    </node>
    <node TEXT="06 TAMU：成本梯度与调权" COLOR="#B2533B">
      <node TEXT="前向任务链">
        <node TEXT="预测 → 阶段一 → 阶段二：计算完整 C1 + C2">
          <node TEXT="阶段二显式接收阶段一 PG；两层依赖与所有费用都必须进入计算图。" FOLDED="true" />
        </node>
        <node TEXT="外层任务目标：评估样本平均 Cost">
          <node TEXT="权重到参数的训练映射再连接两阶段决策；将训练子问题展开后依赖层数更多，勿只依赖层数名称。" FOLDED="true" />
        </node>
        <node TEXT="统计误差与经济损失：可分别改善或恶化">
          <node TEXT="按 Cost 调权不是证明 MSE、MAPE、敏感性同时更好。" FOLDED="true" />
        </node>
      </node>
      <node TEXT="反向梯度链">
        <node TEXT="KKT / 隐函数：对局部最优性关系求导">
          <node TEXT="在合适正则条件下求解映射可微；活跃集变化、退化和截断边界需要分别处理。" FOLDED="true" />
        </node>
        <node TEXT="伴随法 / 链式法则：Cost → 预测 → 参数">
          <node TEXT="同时包含阶段一自身费用和阶段一决策影响阶段二费用的路径，仅对第一阶段最优值求导不完整。" FOLDED="true" />
        </node>
        <node TEXT="CVXPYLayers / DPP：连接自动微分">
          <node TEXT="DPP 是参数化建模结构规则；求解器得到数值最优解不等于该调用自动具有 PyTorch 梯度。" FOLDED="true" />
        </node>
      </node>
      <node TEXT="重加权与边界">
        <node TEXT="Cost 影响评分：一条样本增权怎样影响 J">
          <node TEXT="与 PAMU 相同框架，主要替换外层梯度；分数锚点、数据集合和归一化仍需核对。" FOLDED="true" />
        </node>
        <node TEXT="LP → 加权预测头：同样的 L1 / Linf 预算">
          <node TEXT="得到权重后重新构造加权 MSE 系统，检查秩、CG 收敛和残差。" FOLDED="true" />
        </node>
        <node TEXT="理论条件与真实复算：局部代理不是全局保证">
          <node TEXT="标准充分条件可帮助理解局部导数，但不能冒充未取得原论文的定理；最终 Cost 要重新数值评价。" FOLDED="true" />
        </node>
      </node>
    </node>
    <node TEXT="07 实验与证据分层" COLOR="#7856A0">
      <node TEXT="作者摘要 / 仓库复现">
        <node TEXT="预测器比较：Linear / CNN / MLP-Mixer">
          <node TEXT="作者摘要明确这些模型；发表版全部实验设置、结果和图号仍需原文核对。" FOLDED="true" />
        </node>
        <node TEXT="指标与对照：MSE、MAPE、Cost、重拟合">
          <node TEXT="仓库实现直接遗忘与 PAMU/TAMU 入口；当前复现保存六档遗忘比例结果。" FOLDED="true" />
        </node>
        <node TEXT="求解与复现质量：状态、约束、费用及恢复">
          <node TEXT="项目已有成本修复、状态检查和重放记录；工程检查通过不能替代原文理论核验。" FOLDED="true" />
        </node>
      </node>
      <node TEXT="你的扩展实验一">
        <node TEXT="单输入温度：1 样本 × 5 档 × 4 模型">
          <node TEXT="固定每个模型的参数、其他输入和真实负荷，改变输入节点1温度 ±1、±0.5、0 个训练标准差。" FOLDED="true" />
        </node>
        <node TEXT="完整影响路径：温度 → 14 输出 → Cost">
          <node TEXT="参考点原始 CNN 的 +1σ 总预测变化 −0.827542 MW，总 Cost 变化 −6.778363。" FOLDED="true" />
        </node>
        <node TEXT="响应对照：原始 / 直接 / 重拟合 / TAMU">
          <node TEXT="共享冻结 CNN 核心；TAMU 不是逐个温度水平重新调权。绝对成本与敏感性应分开。" FOLDED="true" />
        </node>
      </node>
      <node TEXT="你的扩展实验二">
        <node TEXT="28 日期节点诊断：隔离 / 复位 / 参考点 Shapley">
          <node TEXT="28×4=112 组原始 CNN 非零扰动对照的峰值节点一致；这是本批观察，不是数学必然。" FOLDED="true" />
        </node>
        <node TEXT="27 新日期经济性：106 / 108 低于直接遗忘">
          <node TEXT="27 个新日期各一个小时级样本，按月与负荷层预选；参考点不计入该汇总。" FOLDED="true" />
        </node>
        <node TEXT="保留反例：TAMU 曾贵 60.929210">
          <node TEXT="2019-03-12、test1686、−1σ 场景出现该差异，不能把平均改善说成逐场景保证。" FOLDED="true" />
        </node>
      </node>
    </node>
    <node TEXT="08 结论、局限与后续" COLOR="#526675">
      <node TEXT="可以讲清楚的机制">
        <node TEXT="遗忘的参数更新：条件明确才谈精确">
          <node TEXT="二次头部的数值重拟合一致性不代表整个 CNN 的任意数据都已认证删除。" FOLDED="true" />
        </node>
        <node TEXT="按任务代价调权：区别于仅按预测误差">
          <node TEXT="TAMU 的外层指标是 Cost；PAMU 则提供性能指标导向的必要比较。" FOLDED="true" />
        </node>
        <node TEXT="输入到 Cost 的响应：模型输出反事实诊断">
          <node TEXT="人为温度输入干预和混合输出向量，并不等同真实天气的物理因果作用。" FOLDED="true" />
        </node>
      </node>
      <node TEXT="当前解释边界">
        <node TEXT="协议与范围：全年标准化、测试信息参与">
          <node TEXT="原协议中先全年标准化再划分，helpful/TAMU 使用测试性能信息；新增扰动控制未消除这些限制。" FOLDED="true" />
        </node>
        <node TEXT="场景有限：前73天、seed42、单温度坐标">
          <node TEXT="当前扩展不能推出全年、夏季、其他种子或 ERCOT 的普遍结论。" FOLDED="true" />
        </node>
        <node TEXT="原文尚待核验：章节 / 公式 / 定理不能冒认">
          <node TEXT="本套图的八层是学习顺序；需要取得完整 PDF 才能逐章对应原论文。" FOLDED="true" />
        </node>
      </node>
      <node TEXT="后续怎样补齐">
        <node TEXT="增加 PAMU 对照：统一预算、删除集和评价">
          <node TEXT="当前四模型温度实验没有 PAMU，因而不能证明 TAMU 优于 PAMU。" FOLDED="true" />
        </node>
        <node TEXT="独立测试与稳健性：多特征、多种子、多场景">
          <node TEXT="预先确定样本，分别报告平均成本、最差场景及输入响应，不根据结果挑选样本。" FOLDED="true" />
        </node>
        <node TEXT="理论与跨数据集：核验原文；明确节点映射">
          <node TEXT="原文定理逐条核对、认证定义、ERCOT 节点映射与缩放均需单独证据；当前没有完成真实 ERCOT 成本实验。" FOLDED="true" />
        </node>
      </node>
    </node>
  </node>
</map>