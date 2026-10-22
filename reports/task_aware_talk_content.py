"""A source-bounded 10–13 minute algorithm talk; v2 full text is still required.

All formulas are explanatory restatements of author implementation and standard
math. No unverified equation, theorem, table or figure number is assigned.
"""

PAPER = 'https://arxiv.org/html/2308.14412v2'
AUTHOR = 'https://github.com/xuwkk/task_aware_machine_unlearning/tree/98a3e7bf177fe40d1d538e63a869578338d70c81'
MIT_SPEAK = 'https://ocw.mit.edu/courses/res-tll-005-how-to-speak-january-iap-2018/pages/how-to-speak/'
MIT_SLIDES = 'https://mitcommlab.mit.edu/nse/commkit/slideshow/'

SLIDES = [
    dict(id='S01', kind='cover', title='删除训练数据之后，如何保住电网运行经济性？',
         kicker='TASK-AWARE MACHINE UNLEARNING', section='问题', seconds=25,
         subtitle='Task-Aware Machine Unlearning and Its Application in Load Forecasting',
         lines=['Wangkun Xu · Fei Teng  |  IEEE Transactions on Power Systems, 2024',
                '本次只讲三件事：如何删、如何补偿、Cost 梯度怎样传回来'],
         takeaway='主线：直接遗忘 → PAMU → TAMU',
         source='作者论文信息与摘要；v2 全文和图表仍待核对',
         notes='这篇工作的对象是已经训练好的负荷预测器。我的汇报集中回答一个问题：移除部分训练数据的影响之后，怎样尽量保住电网运行的经济性。接下来先解释直接遗忘，再讲剩余样本重加权，最后把两阶段调度的梯度接回来。本文标题和作者由作者仓库确认；目前这份是算法工作稿，v2 全文、原图和正式定理表述尚待核对。'),
    dict(id='S02', kind='original', title='遗忘需要移除样本影响，而不只是删除数据文件',
         section='问题', seconds=45,
         lines=['删除集 D_f：从后续拟合中移除', '剩余集 D_r：承担预测能力',
                '目标：接近“未用删除数据训练”的参考模型'],
         takeaway='直接删除记录，并不会自动改变已经学到的参数。',
         source='作者仓库原始流程图 repo_src/unlearning.png；尚未核对是否与 v2 某图一致',
         notes='这里使用作者仓库提供的原始流程图，图像没有改写，也没有把我们的复现图冒充原文图。上半部分是数据删除请求，下半部分是对已经训练好的模型进行更新。需要区分删除数据文件和移除参数中的影响。我们将原数据划分成删除集和剩余集，参考对象是只在剩余数据上训练的模型。实际负荷规律在样本间共享，所以遗忘后仍然能预测删除样本，并不自动意味着遗忘失败。图号仍需原文核对。'),
    dict(id='S03', kind='direct', title='直接遗忘用剩余集曲率修正删除后的梯度失衡',
         section='直接遗忘', seconds=70,
         equations=[
             r'$R_r(\theta)=\frac{1}{m}\sum_{i\in D_r}\ell_i(\theta)$',
             r'$g_r=\nabla R_r(\theta_0),\qquad H_r=\nabla^2 R_r(\theta_0)$',
             r'$\theta_{\mathrm{MU}}=\theta_0-H_r^{-1}g_r$'],
         lines=['删除前：完整集梯度在最优点附近平衡',
                '删除后：剩余集在旧参数处通常不再驻定',
                'Newton 更新：用剩余集梯度与曲率纠偏'],
         takeaway='二次目标 + 固定特征 + 可逆 Hessian 时，一步可等于剩余集重拟合。',
         source='作者 eval_unlearn.py / eval_unchange.py；本页为数学重述，无原文公式编号',
         notes='记旧参数为 theta 零，剩余数据的平均训练损失为 R r。完整集上原本接近平衡的梯度，在删除一批样本后通常失衡。直接遗忘计算旧参数处的剩余梯度，再解一个 Hessian 线性系统，把参数向剩余集最优点移动。这个负号来自 Newton 更新，不要和单样本增权的影响函数符号混淆。对固定特征的二次回归，Hessian 不随参数变化，一步就能等于直接重拟合；对整个非凸 CNN，只能把这种更新视为局部近似，不能直接声称精确遗忘。'),
    dict(id='S04', kind='pamu', title='PAMU 通过重加权剩余样本补偿性能变化',
         section='PAMU', seconds=75,
         equations=[
             r'$\theta(w)=\arg\min_{\theta}\ \frac{1}{m}\sum_{i\in D_r}w_i\ell_i(\theta)$',
             r'$\frac{\partial\theta}{\partial w_i}=-\frac{1}{m}H(w)^{-1}\nabla\ell_i(\theta(w))$',
             r'$\frac{\partial Q}{\partial w_i}=-\frac{1}{m}\nabla Q^T H(w)^{-1}\nabla\ell_i$'],
         lines=['外层 Q：MSE 或 MAPE 等统计性能',
                '内层拟合：仍使用加权训练 MSE',
                '改变的是样本权重，不是直接指定某个网络参数'],
         takeaway='局部影响函数把“补偿哪个样本”变成可计算的方向。',
         source='标准驻点隐式求导 + 作者重加权实现；理想最优点推导与代码近似分开讲',
         notes='直接删除可能损失有用的统计信息，PAMU 因此调整剩余样本的权重。先看一个理想的加权训练问题：每个剩余样本有完整权重 w i。对它的驻点条件求导，就得到参数对权重的导数，里面是训练 Hessian 的逆乘样本梯度。再与外层性能 Q 的梯度相乘，就知道增加某个样本的权重会怎样影响性能。这里的公式在加权最优点处成立；作者实现为了高效，在旧参数处用局部量近似，下一页会区分它的实际评分。外层评价改变，基础拟合损失仍是 MSE。'),
    dict(id='S05', kind='weights', title='权重线性规划限制补偿幅度，而不要求权重和固定',
         section='PAMU', seconds=60,
         equations=[
             r'$\min_w\ s^T w$',
             r'$\|w-\mathbf{1}\|_1\le c\,m,\qquad \|w-\mathbf{1}\|_{\infty}\le 1$',
             r'$0\le w_i\le 2$'],
         lines=['s_i < 0：增权有望降低外层损失',
                '平均 L1 预算 c：允许偏离均匀权重的程度',
                '删除样本不参与剩余集拟合；其有效权重为 0'],
         takeaway='原实现没有 Σw_i=m 约束；c=0.1 也不是“遗忘 10%”。',
         source='作者 eval_unchange.py：cp.scalar_product、L1 与 Linf 约束',
         notes='有了局部分数，重加权就转成一个线性规划：最小化分数与权重的内积。分数为负的样本，增加它的权重在局部有助于降低外层损失。两个约束分别控制总体改动量和单样本改动量，所以权重落在零到二之间。这里没有额外的权重总和约束，不能为了看起来更像概率分布而自行添加。平均 L1 预算 c 控制补偿力度，与删除比例是两个独立参数。删除记录已经从拟合集移除，不能因为补偿效果好就把它们重新加回。'),
    dict(id='S06', kind='tamu', title='TAMU 将外层评价从预测误差换成实际运行 Cost',
         section='TAMU', seconds=65,
         equations=[
             r'$Q_C(w)=\frac{1}{|V|}\sum_{(x,y)\in V}\left[C_1(z_1^*)+C_2(z_2^*)\right]$',
             r'$\hat d=f_{\theta(w)}(x),\qquad z_1^*=z_1^*(\hat d),\qquad z_2^*=z_2^*(z_1^*,y)$'],
         lines=['PAMU：让剩余样本有利于 MSE / MAPE',
                'TAMU：让剩余样本有利于下游调度成本',
                '两者都可评价 Cost；区别在于重加权时用什么梯度'],
         takeaway='低 MSE 不自动保证低 Cost：误差的位置和方向也重要。',
         source='作者摘要与 criteria=mse/mape/cost 分支；V 为讲解用评估集记号',
         notes='TAMU 的关键变化不是换一个更复杂的预测网络，而是把外层评价改成真实运行任务。预测负荷先送入调度，再结合真实负荷进行补救，两部分成本相加。在有线路和机组约束的系统中，同样大小的预测误差，发生在不同节点、朝不同方向偏离，成本可能很不一样。因此 MSE 小不等于 Cost 小。PAMU 当然也能计算和报告 Cost，它只是没有用 Cost 梯度指导这一轮重加权。这里把评估集写成 V；原代码实际使用 test loader，独立评估问题放在备份页说明。'),
    dict(id='S07', kind='stages', title='预测决定首阶段调度，真实负荷决定第二阶段补救',
         section='TAMU', seconds=80,
         equations=[
             r'$z_1^*(\hat d)\in\arg\min_{p,s_1,\varphi_1}\ C_1(p,s_1)$',
             r'$z_2^*(p^*,y)\in\arg\min_{s_2,g_s,\varphi_2}\ C_2(s_2,g_s)$',
             r'$C_{\mathrm{total}}=C_1(p^*,s_1^*)+C_2(s_2^*,g_s^*)$'],
         lines=['阶段一：发电成本 + 预测下的负荷削减惩罚',
                '阶段二：真实负荷不匹配的削减 / 过剩处理惩罚',
                '约束：DC 功率平衡、线路容量、相角基准等'],
         takeaway='第二阶段使用固定真实负荷；模型改变的是预测与首阶段计划。',
         source='作者 utils/optimization.py 的 stage_one、stage_two；完整约束在备份 B3',
         notes='第一阶段根据预测负荷安排发电，目标包括发电成本和必要的负荷削减惩罚，并满足 DC 功率平衡、机组上限与线路约束。第二阶段把第一阶段的发电作为给定参数，在真实负荷下处理不匹配，通过削减与过剩处理变量形成补救成本。这里的 generation storage 是仓库中的过剩处理变量，不能当作已经建模完整电池时序。总代价要包含两个阶段，尤其不能漏掉第一阶段的削减项。比较不同模型时，真实负荷和系统约束固定，变化来自各模型的负荷预测。'),
    dict(id='S08', kind='kkt', title='优化层的梯度来自 KKT 条件的隐式求导',
         section='梯度', seconds=95,
         equations=[
             r'$F(z^*,a)=0$',
             r'$\frac{\partial z^*}{\partial a}=-F_z^{-1}F_a$',
             r'$F_z^T v=\nabla_z C,\qquad \nabla_a C=\partial_a C-F_a^T v$'],
         lines=['F：把驻点、可行性和互补条件写成方程',
                'a：预测负荷，或上一阶段传入的调度参数',
                '正则条件成立时，求导转为求解线性系统'],
         takeaway='“求出最优值”与“把最优解作为可微层”是两件事。',
         source='标准 KKT / 伴随推导；非原文定理逐字转述，v2 定理假设仍待核对',
         notes='最难的地方是最优解本身由约束优化决定，不能把它当普通显式函数。把原始变量、对偶变量和最优性条件组成 F 等于零。输入 a 变化时，对这个方程整体求导，就得到最优解对输入的导数。反向传播不必形成庞大 Jacobian，可以解一个转置线性系统，也就是伴随方程，再得到输入梯度。两阶段依次应用这个过程，就能把第二阶段成本传回第一阶段和预测器。本页只说明 KKT Jacobian 可逆等正则条件下的标准局部推导，不声称任意约束切换处都有唯一导数；原论文更具体的存在性结论仍需要 v2 全文核对。'),
    dict(id='S09', kind='scores', title='Cost 梯度与样本梯度通过一次 iHVP 相遇',
         section='梯度', seconds=70,
         equations=[
             r'$v_C=\nabla_{\theta}Q_C,\qquad H_r u=v_C$',
             r'$s_i=-\frac{1}{n}\,g_i^T u,\qquad g_i=\nabla_{\theta}\ell_i(\theta_0)$',
             r'$\text{Cost}\ \longrightarrow\ \text{forecast}\ \longrightarrow\ \theta\ \longrightarrow\ w_i$'],
         lines=['v_C：Cost 对预测器参数的梯度',
                'H_r：剩余集训练 MSE 的 Hessian，不是 Cost Hessian',
                '一次求解得到 u，再与每个样本梯度做内积'],
         takeaway='作者评分按原训练样本数 n 归一化；它是实现中的局部近似。',
         source='作者 eval_unchange.py：M=-inverse_hvp(grad_test_ave)，scores/=len(dataset_train)',
         notes='现在把两个方向接起来。可微优化给出 Cost 对预测器参数的梯度 v C；剩余集训练 MSE 提供 Hessian H r。只需求解 H r 乘 u 等于 v C，然后逐个样本做梯度内积，就得到权重方向，不需要为每个样本重训整个模型。注意，这个 Hessian 来自训练 MSE，而不是运行成本。作者实现的样本分数再除以原拟合集大小 n，和刚才理想推导中的 m 约定不同；同一固定正数缩放不改变该线性规划的最优解，但影响分数数值解释。评估点也需要按代码区分。'),
    dict(id='S10', kind='procedure', title='求出权重后，TAMU 仍执行加权 MSE 的参数更新',
         section='算法', seconds=60,
         equations=[
             r'$R_w(\theta)=\frac{1}{m}\sum_{i\in D_r}w_i\ell_i(\theta)$',
             r'$\theta_{\mathrm{TAMU}}=\theta_0-H_w^{-1}\nabla R_w(\theta_0)$'],
         lines=['① 固定删除集，构造剩余集与旧参数',
                '② 可微两阶段优化 → Cost 参数梯度',
                '③ iHVP + 样本梯度 → 分数 s',
                '④ 权重 LP → w；⑤ 重建加权目标并更新参数'],
         takeaway='最终 H_w 随权重变化，不能直接复用评分时的 H_r。',
         source='作者 eval_unchange.py 的 with_weight=True 和最后一次 stest / reconstruct_model',
         notes='把全流程压缩成五步：先确定需要删除哪些记录；计算评估场景的运行成本梯度；通过逆 Hessian 向量积得到每条剩余样本的局部分数；解权重线性规划；最后在加权训练 MSE 上更新参数。这里有一个容易写错的细节：最后一次拟合要使用加权目标的 Hessian，它一般不同于评分阶段的剩余集 Hessian。作者代码会重新构造带权 loader 和模块，而不是把评分的矩阵直接沿用。对固定特征二次预测头，这个最后步骤可与加权重拟合对应；不等于全网以 Cost 做端到端训练。'),
    dict(id='S11', kind='scope', title='局部重加权的收益需要与遗忘程度和评估协议一起读',
         section='边界', seconds=35,
         lines=['局部近似：删除比例大、非凸表示改变时更需对照重训',
                '遗忘范围：冻结核心下的预测头 ≠ 任意 CNN 全参数',
                '独立评估：原实现使用 test loader 参与评分，不能忽略这一点'],
         takeaway='作者原文实验图仍待 v2 PDF；不把已有复现曲线写成原文证据。',
         source='作者 README / eval_unchange.py / func_operation.py；本页为实现边界分析',
         notes='理解方法之后，还要说明结论边界。影响函数与权重规划是局部工具，大比例删除和特征表示变化时需要匹配重训基线。神经网络代码采用核心与敏感头分离，不能把预测头结论推广到全部层。原实现还用 test loader 计算外层梯度，后续研究应使用独立验证和测试协议。本稿尚未取得 v2 实验原图，因此没有量化复述作者结果，也没有用我们已有的复现图顶替。'),
    dict(id='S12', kind='closing', title='TAMU 的核心是让剩余样本的补偿方向服务于运行任务',
         section='结论', seconds=30,
         lines=['直接遗忘：剩余梯度 + 曲率，移除删除集影响',
                'PAMU：按统计性能，调整剩余样本的重要性',
                'TAMU：让两阶段 Cost 梯度指导同一重加权过程'],
         takeaway='记住：Cost 梯度经训练曲率修正后，与样本梯度做内积。',
         source='算法讲解总结；文献与来源状态见备份 B5',
         notes='最后记住这条递进关系。直接遗忘试图抵消删除带来的参数影响；PAMU 用剩余样本重加权补偿统计性能；TAMU 再把补偿目标换成电网运行成本。真正连接两端的是样本梯度、训练 Hessian 和 Cost 梯度形成的影响分数。它提供一种任务感知的补偿机制，但不自动等于完整隐私认证，也不保证任何场景都更便宜。'),
    dict(id='B01', kind='backup_if', title='备份 B1｜上调权重与删除样本的符号为什么相反',
         section='备份', seconds=0,
         equations=[r'$\nabla R(\theta_\epsilon)+\epsilon\nabla\ell_i(\theta_\epsilon)=0$',
                    r'$H\frac{d\theta_\epsilon}{d\epsilon}+g_i=0$',
                    r'$\frac{d\theta_\epsilon}{d\epsilon}\bigg|_{0}=-H^{-1}g_i$'],
         lines=['增权：沿负的“逆 Hessian × 样本梯度”方向局部移动',
                '删除：减少样本权重，方向相反；有限删除不等于无穷小扰动',
                '使用平均损失时，归一化与样本数变化必须同步处理'],
         takeaway='S04 的理想导数在 θ(w) 处；作者评分是在 θ₀ 的局部量上构造。',
         source='标准影响函数推导；无原文公式编号',
         notes='若被问到负号，从训练驻点条件开始，对样本权重求导即可。一般影响函数是无穷小导数；整条删除和批量删除是有限变化。作者实现同时利用剩余集 Hessian，并以旧参数计算样本和外层梯度。因此不要把理想加权最优点处的导数，与代码所有求值点完全等同。'),
    dict(id='B02', kind='backup_quadratic', title='备份 B2｜一步精确来自二次目标，而不是 CNN 的名字',
         section='备份', seconds=0,
         equations=[r'$R_w(B)=\frac{1}{2m}\|W^{1/2}(ZB-Y)\|_F^2$',
                    r'$H_w=\frac{1}{m}Z^T W Z$',
                    r'$B^*=(Z^T W Z)^{-1}Z^T W Y$'],
         lines=['固定隐藏特征 Z，偏置可通过常数列加入',
                '正定 / 满列秩时 Newton 与加权最小二乘一致',
                '改变卷积核心后 Z 也变，二次精确条件不再适用'],
         takeaway='本页为固定特征最小二乘说明；不宣称整个 CNN 可一步精确遗忘。',
         source='作者 NN-affine 结构 + 标准最小二乘；归一化常数须与实际损失一致',
         notes='这个式子用一半平方损失方便写 Hessian。实际代码还按节点维度归一化，要同步换算。若矩阵不满秩，不能静默加正则项并声称目标没变；需要声明广义逆、正则化或重训协议。若核心或预处理含有删除数据影响，仅更新头还不足以证明整个流程删除。'),
    dict(id='B03', kind='backup_constraints', title='备份 B3｜两阶段目标与约束按作者实现完整保留',
         section='备份', seconds=0,
         equations=[r'$C_1=p^T Qp+c^T p+\lambda\mathbf{1}^T s_1$',
                    r'$C_2=\lambda\mathbf{1}^T s_2+\kappa\|s_2\|_2^2+\rho\mathbf{1}^T g_s+\eta\|g_s\|_2^2$',
                    r'$B_{\rm bus}\varphi_1=C_g p-C_l(\hat d-s_1)$',
                    r'$B_{\rm bus}\varphi_2=C_g(p^*-g_s)-C_l(y-s_2)$'],
         lines=['阶段一：0≤p≤p_max，s1≥0；两阶段均有线路限额与基准相角',
                '阶段二：s2≥0，g_s≥0；p* 固定为阶段一调度',
                '不擅自增加 s≤负荷 或 g_s≤p 等原代码没有的约束'],
         takeaway='可行性与可微性是两项检查；DPP 合规也不等于处处光滑。',
         source='作者 utils/optimization.py：stage_one / stage_two；符号为讲解统一记号',
         notes='这页用于约束细节问答。第一阶段包括发电二次项、线性项和削减惩罚；第二阶段包括削减和过剩处理的线性、二次惩罚。两阶段都满足 DC 平衡、线路限额和参考相角。第一阶段机组上限不能遗漏。原实现未写出削减不超过负荷、过剩处理不超过计划这两类上界，不能在复述时随意加入。'),
    dict(id='B04', kind='backup_figures', title='备份 B4｜原文图槽位与补图需求已经分开登记',
         section='备份', seconds=0,
         lines=['原文 R1：取得 v2 后选择直接遗忘 / 比例变化的结果图',
                '原文 R2：取得 v2 后选择 PAMU / TAMU 的关键对照图',
                '补图 D1–D3：仅提供绘图方法文本，不生成实验曲线'],
         takeaway='原文图从 PDF 原样提取；新示意图必须标为“讲解示意”。',
         source='图号、子图和结果结论均待 v2 PDF 核验；不预设论文一定含某种具体图',
         notes='这是材料准备页，不计入主讲。当前只有作者仓库流程原图可用，尚不能确认其 v2 对应图号。原论文结果图不能让生成模型凭空补画。补图文档只描述可选的教学示意图：重加权对象、Cost 梯度路径、冻结核心与预测头范围。取得原文后先选最能支持主张的一到两张结果图，再决定是否需要补图。'),
    dict(id='B05', kind='backup_sources', title='备份 B5｜来源、版本与 MIT 设计原则的核对状态',
         section='备份', seconds=0,
         lines=['论文：v2 入口已定位，全文 403；当前是作者代码核对工作稿',
                'How to Speak：已读 MIT 官方课程页面源码；未观看完整视频',
                'NSE Slideshow：页面 403，设计核对为归纳适配，未逐条核实原页'],
         takeaway='正式定稿还需要论文 v2 PDF；不把工作稿标成原文精读终稿。',
         source='来源和访问错误完整记录在 source_manifest.json 与设计核对表',
         notes='这页用于诚实说明材料来源。作者代码与原始流程图固定到 98a3e7b 提交。MIT How to Speak 官方课程仓库已取得，页面包含开场承诺、工具、说服和结尾等章节，但不声称看完视频。NSE 指南页面目前没有取得，因此这里只把受众、主张、证据、视觉层次等原则作为适配设计，不声称逐字引用或获得 MIT 评分。论文 PDF 到位后需要核对公式、定理假设和所有原文图表。')
]

SUPPLEMENT = r'''# 待补图方法论文本：只补讲解示意，不补造实验曲线

对象：10–13 分钟 Task-Aware Machine Unlearning 算法汇报。

## 优先级与边界

1. **必须先补的是原论文材料，不是生成图。** 需要用户上传 arXiv:2308.14412v2 PDF。原文图必须从该版本提取，保留图例、坐标、单位和必要子图；核对后填写图号与图注。不能让生成模型模仿论文风格创造结果。
2. 目前已使用作者仓库 `repo_src/unlearning.png` 原始流程图，但它与 v2 哪张图相同尚未核对；PPT 已明确标注来源。
3. 以下 D1–D3 都是可选教学示意图，按需要选一到两张即可。**本轮只提供方法文本，没有调用图像生成，也没有重画实验曲线。**
4. 公式排版、文字框与表格属于演示排版；它们不是新实验图。所有新示意图须标注“讲解示意，非实验结果”。

## D1｜到底改变的是数据权重还是模型参数？

**目的：** 区分输入特征 x、训练样本权重 w 和模型参数 θ。用于 S04–S05，替换部分解释文字即可，不另加主讲时间。

**方法论：** 用三个并排的区域表达依赖。左侧为训练记录，每行含 x_i、y_i、w_i；删除记录用灰色标注“移除，有效权重 0”，剩余记录权重围绕 1 调整。中间为加权 MSE 的拟合过程；右侧为预测器参数 θ 与新样本输出。箭头只能是“w 改变拟合目标 → θ 更新 → 预测改变”。不要画成 w 直接乘在测试样本输入或预测输出上。

**必备标注：** `D_f`、`D_r`、`w_i∈[0,2]`、`平均L1预算c`、`θ(w)`。权重值只用符号，不填虚构数据；不得自行添加 Σw=m。

**可直接交给制图工具的说明：**

> 绘制一张 16:9 白底科研教学示意图，三栏依次是“训练记录与样本权重”“加权MSE拟合”“预测模型参数与输出”。删除记录灰显并标注“有效权重0”；剩余样本用符号w_i表示，默认值1。用蓝色箭头表现权重改变训练目标、训练目标改变θ、θ改变预测。不要画测试输入特征被w缩放，不要添加虚构数值、实验曲线、图号或归一化约束。底部标注“讲解示意，非实验结果”。中文标签，公式保留拉丁符号，元素可编辑。

**验收：** 读者能指出“调的是剩余样本的重要性”，而不是“把温度改了”或“手工直接改某个参数”。

## D2｜Cost 梯度怎样回到每个样本权重？

**目的：** 将 S07–S09 的前向与反向计算放在同一图中。

**方法论：** 前向用蓝色实线：`w → θ(w) → 预测负荷 d̂ → 阶段一调度 z₁* → 阶段二补救 z₂* → C₁+C₂`。真实负荷 y 另从上方进入阶段二；系统约束从下方进入两个优化框。反向用橙色虚线：成本经 KKT 隐式导数返回预测与 θ，得到 v_C；再经过训练 MSE 的 H_r 解线性系统 H_r u=v_C，最后与样本梯度 g_i 做内积得到 s_i。

**必须表现的细节：** C₁ 对第一阶段变量有直接梯度，不能只画 C₂ 的反向路径；H_r 来自训练 MSE，不是 Cost；y 和系统约束在模型比较时固定。优化层梯度只标注“正则条件下的局部导数”，不能画成无条件处处可微。

**可直接交给制图工具的说明：**

> 绘制一张宽幅白底算法计算图，上排蓝色前向箭头连接w、θ(w)、预测d̂、阶段一调度、阶段二补救、总Cost。真实负荷y单独进入阶段二；固定线路和机组约束分别进入两阶段。下排橙色虚线表现Cost反向传播，经KKT隐式求导得到v_C，再解H_r u=v_C，并输出s_i=−g_iᵀu/n。明确写出“H_r：剩余训练MSE的Hessian”。总Cost同时接收C₁和C₂，第一阶段的直接成本梯度不可遗漏。不要添加数值、实验曲线、原论文图号或全局光滑性保证。标注“讲解示意，非实验结果”。

**验收：** 每个箭头说明清楚其传递的是数值还是导数；没有把两阶段 Cost 的 Hessian 当成训练 Hessian。

## D3｜CNN 中哪些层实际参与遗忘更新？

**目的：** 为 S03 / S10 的固定特征二次特例划清范围，适合答疑备用。

**方法论：** 分成固定特征提取器与可更新预测头。输入为 14节点×10特征，卷积与非线性得到 64维隐藏表示，最后 64→14 的仿射头输出节点预测。固定部分浅灰，头部蓝色；样本重加权箭头进入头部拟合过程，不直接进入温度等输入。结构参数来自作者代码与此前模型记录，不冒充 v2 论文原图。

**必备标注：** `冻结核心 h_φ(x)`、`可更新头 W,b`、`64→14`、`固定特征时训练目标是二次函数`。可在脚注写“若核心也受删除数据影响，单改预测头不足以证明完整删除”。

**可直接交给制图工具的说明：**

> 绘制白底模型结构示意：14×10输入，进入灰色冻结核心（Conv、ReLU、Flatten、FC、Tanh），得到64维隐藏表示，再进入蓝色可更新仿射头64→14，输出14节点负荷。上方w箭头仅指向“头部加权MSE拟合”。强调这是作者实现的结构讲解，不是实验结果，不画注意力热力图、不把单个隐藏单元命名为温度神经元、不声称全CNN可一步精确遗忘。中文标签和数学符号分层清楚。

**验收：** 输入 140 维与隐藏 64 维不混淆；无参数层不被画成拥有独立可训练参数；没有扩展精确性保证。

## 原文图 R1 / R2 的选图操作（待 PDF）

- 先列出 v2 所有图、完整图注和所支持的结论，再决定哪一到两图能证明“删除造成性能变化”和“任务感知补偿影响 Cost”。不预设图号，也不预设 v2 必然具有某一种具体画法。
- 若图是多子图，优先保留与讲述有关的子图，但必须保留图例、坐标、单位和方法身份；注明“裁取自原图”，记录原始页码和裁取边界。
- 幻灯片标题写可由图直接支持的主张；图旁只写一个读图结论。量化结论必须从原图或原表核对，不能从当前仓库复现结果迁移过去。
- 原文图号、版本、页码、作者、链接与文件 SHA256 一并写入 `figure_manifest.json`。本轮尚未取得 PDF，所以没有编造该清单的图号和页码。
'''
