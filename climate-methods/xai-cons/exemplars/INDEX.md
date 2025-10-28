# XAI-Cons Exemplars 索引

> 本索引帮助用户快速定位相关论文和代码示例。论文按研究主题分类，代码示例提供可复用的实现参考。

---

## 📁 快速导航

### 资源位置
- **代码示例**: `code-examples/nature_2024_dry_spells/`
- **论文库**: `papers/` (43篇PDF)
- **完整方法论**: `../references/methods.md` (详细的数学推导和理论)
- **快速参考**: `core-methodology.md` (2-3页核心要点，待创建)

### 使用建议
- **需要代码参考** → 查看下方"代码示例"部分
- **需要方法论细节** → 查看 `../references/methods.md`
- **查找特定主题论文** → 使用下方论文分类索引
- **不确定从哪里开始** → 先看"推荐学习路径"

---

## 💻 代码示例

### Nature 2024 Dry Spells 案例

**路径**: `code-examples/nature_2024_dry_spells/`
**对应论文**: `papers/nature_2024_obs_constrained_dry_spells.pdf`
**作者**: Petrova I.Y. et al. (2024) Nature

**代码文件说明：**

| 文件名 | 用途 | 何时使用 |
|--------|------|----------|
| `ECpaper_Figure1.py` | 生成EC散点图（回归线+观测约束点） | 需要绘制标准EC关系图时 |
| `ECpaper_Figure2.py` | 空间分布图 | 展示区域EC结果时 |
| `ECpaper_Figure3.py` | 时间序列分析 | 分析历史和未来趋势时 |
| `ECpaper_Figure4.py` | 综合分析图 | 多面板综合展示时 |
| `EC_KL_div_Brient_adopted.py` | ⭐ Binning分析（可靠性评估） | **评估EC关系可靠性时（核心！）** |
| `EC_KL_div_tools_Brient.py` | Binning分析工具函数 | 配合上述文件使用 |
| `Map2_Corr.py` | 空间相关性分析 | 分析区域模式时 |
| `useful_functions.py` | 通用工具函数 | 数据处理、统计计算、绘图 |

**快速适配指南：**
1. **替换数据路径** - 修改数据加载部分，指向你的CMIP6文件
2. **调整区域定义** - 修改lat/lon范围为你的研究区域
3. **修改变量名** - 将 `x_hist`、`y_future` 改为你的变量
4. **保持核心逻辑** - EC计算、回归分析、binning方法无需修改

**特别推荐：** `EC_KL_div_Brient_adopted.py` 是评估EC可靠性的金标准实现，直接可用！

---

## 📚 论文分类索引

### 使用说明
- **路径格式**: 所有论文位于 `exemplars/papers/` 目录下
- **建议阅读时机**: 说明什么情况下应该参考该论文
- **核心贡献**: 1-2句话总结论文的主要方法/发现

---

### 1️⃣ 核心方法论 (Core Methodology)

#### sciadv.adt6485 (1).pdf
- **路径**: `papers/sciadv.adt6485 (1).pdf`
- **补充材料**: `papers/sciadv.adt6485_sm.pdf`
- **何时看**: 研究**观测约束后的多年代气候预测应被视为预测而非长期投影**，以及如何运用复杂方法（如KCC）处理历史记录中的复杂不确定性时。
- **核心贡献**: 讨论了将观测约束后的多年代气候预测视为可靠**预测 (predictions)** 的合理性，并通过复杂方法（如KCC）处理了跨历史和未来时期的不确定性。

#### sciadv.adr5346 (1).pdf
- **路径**: `papers/sciadv.adr5346 (1).pdf`
- **补充材料**: `papers/sciadv.adr5346_sm.pdf`
- **何时看**: 学习如何构建**约束整个地球系统投影**（包括大气、海洋、陆地和冰冻圈变量）的框架时。
- **核心贡献**: 提出了一种方法，通过观测约束全球变暖幅度，进而**约束整个地球系统的未来投影**，以确保各系统变量之间的内部一致性，支持气候适应规划。

#### npj_clim_atmos_2024_oreilly_european_climate_ec.pdf
- **路径**: `papers/npj_clim_atmos_2024_oreilly_european_climate_ec.pdf`
- **补充材料**: `papers/41612_2024_648_MOESM1_ESM.pdf`
- **何时看**: 需要比较和评估**多种观测约束方法**（如REA, CALL, ClimWIP, KCC等）在欧洲区域气候预测中的性能和准确性时。
- **核心贡献**: 比较了五种不同的方法对欧洲地区未来季节性温度和降水变化预测的准确性和稳健性，并使用了125个伪观测数据集进行验证。

#### nat_clim_Shiogama_combined_extreme_precip_ec.pdf
- **路径**: `papers/nat_clim_Shiogama_combined_extreme_precip_ec.pdf`
- **补充材料**: `papers/supp-Combined emergent constraints on future  extreme precipitation changes.pdf`
- **何时看**: 研究如何**结合使用多种EC方法**（例如结合ΔTgm-related和ΔRgm/ΔTgm-related ECs）来最大化地减少极端降水投影不确定性时。
- **核心贡献**: 通过应用**组合约束（combined EC）**，将全球平均极端降水变化（ΔRgm）的方差减少了42%，显著优于单独约束的方法。

---

### 2️⃣ 极端降水/干旱 (Extreme Precipitation/Drought)

#### nature_2024_obs_constrained_dry_spells.pdf ⭐
- **路径**: `papers/nature_2024_obs_constrained_dry_spells.pdf`
- **补充材料**: `papers/supp-nature-Observation-constrained projections reveal longer-than-expected dry spells.pdf`
- **代码示例**: ✅ 可用 (`code-examples/nature_2024_dry_spells/`)
- **何时看**: 研究全球变暖对**干燥期持续时间（LAD）**的影响，特别是模型对LAD变化预估的系统性偏差时。
- **核心贡献**: 提供了观测约束的未来投影，揭示了全球变暖导致的**干燥期持续时间增加将比原始模型集合预测的更长**。

#### nat_commun_2022_zhang_extreme_precip_ec.pdf
- **路径**: `papers/nat_commun_2022_zhang_extreme_precip_ec.pdf`
- **补充材料**: `papers/41467_2022_34006_MOESM1_ESM.pdf`
- **何时看**: 学习如何利用模型模拟的**当前气候降水变率**作为约束因子，来纠正对未来极端降水投影的偏差时。
- **核心贡献**: 利用当前气候的降水变率与未来极端降水变化的**线性关系**，对多模式集合的极端降水投影进行了校正和约束。

#### grl_2024_li_intense_precip_ec.pdf
- **路径**: `papers/grl_2024_li_intense_precip_ec.pdf`
- **何时看**: 关注如何利用全球变暖趋势来约束**罕见强降水事件（Rare Intense Precipitation Events）**的未来变化投影时。
- **核心贡献**: 通过将投影约束在观测到的全球变暖趋势的范围内，限制了CMIP6模型集合中全球陆地罕见强降水事件的未来变化不确定性。

#### Nature_2022_Shiogama_future_precipation.pdf
- **路径**: `papers/Nature_2022_Shiogama_future_precipation.pdf`
- **何时看**: 研究利用观测约束减少CMIP5和CMIP6模型对**全球和区域平均降水变化**不确定性的案例时。
- **核心贡献**: 通过应用观测约束，减少了CMIP5和CMIP6模型对未来降水变化的投影不确定性，并利用多个观测数据集评估不确定性。

---

### 3️⃣ 极端温度/热浪 (Extreme Temperature/Heatwaves)

#### cee-2025-Simolo_constrained_hot extremes.pdf
- **路径**: `papers/cee-2025-Simolo_constrained_hot extremes.pdf`
- **何时看**: 评估和约束气候变暖情景下**极端高温事件风险**的未来投影时。
- **核心贡献**: 通过应用观测约束模型投影，揭示了在未来的气候变暖中，**极端高温事件的风险将有所提高（Enhanced risk of hot extremes）**。

---

### 4️⃣ 东亚/中国气候 (East Asia/China Climate)

#### grl_2023_chen_china_warming_ec.pdf
- **路径**: `papers/grl_2023_chen_china_warming_ec.pdf`
- **补充材料**: `papers/supp-Geophysical Research Letters - 2023 - Chen - Emergent Constrained Projections of Mean and Extreme Warming in China (1).docx`
- **何时看**: 评估和约束**中国区域**的平均气温（CSAT）和极端高温（TXx）投影，尤其是应对CMIP6模型中的**"暖模型"问题**时。
- **核心贡献**: 利用EC对CMIP6模型中的中国区域平均和极端变暖进行约束，结果预测未来变暖将比原始多模式集合预测**温和**，且不确定性减少约一半。

#### erl_2024_liu_ec_EA_tas.pdf
- **路径**: `papers/erl_2024_liu_ec_EA_tas.pdf`
- **补充材料**: `papers/supp-Liu_2024_Environ._Res._Lett._19_064050 (1).docx`
- **何时看**: 研究未来**东亚冬季地表气温（SAT）**变化的约束因素和预测时。
- **核心贡献**: 应用EC来约束CMIP6模型对未来东亚冬季地表气温变化趋势的投影不确定性。

#### npj_clim_atmos_zhu_east_china_precip_ec.pdf
- **路径**: `papers/npj_clim_atmos_zhu_east_china_precip_ec.pdf`
- **补充材料**: `papers/supp-Npj-Improve the projection of East China summer precipitation with emergent constraints.pdf`
- **何时看**: 研究如何利用**西北太平洋和热带太平洋的历史降水模式**来约束**华东夏季降水（ECSP）**投影时。
- **核心贡献**: 建立了EC框架来约束华东夏季降水，发现投影结果显示降水增加幅度较小，并成功将投影不确定性减少了**约23%**。

---

### 5️⃣ 青藏高原气候 (Tibetan Plateau Climate)

#### grl_2023_jiang_ghg_contribution_ec.pdf
- **路径**: `papers/grl_2023_jiang_ghg_contribution_ec.pdf`
- **补充材料**: `papers/2023gl105427-sup-0001-supporting information si-s01.pdf`
- **何时看**: 研究**青藏高原（TP）变暖**中，需要区分**温室气体（GHG）和人为气溶胶去除（AA）**贡献的归因分析和约束时。
- **核心贡献**: 利用最优指纹法和观测约束，量化了GHG排放和AA去除对青藏高原1961-2020年变暖趋势的归因贡献。

#### grl_2024_hu_tibetan_plateau_winter_ec.pdf
- **路径**: `papers/grl_2024_hu_tibetan_plateau_winter_ec.pdf`
- **补充材料**: `papers/supp-Geophysical Research Letters - 2024 - Hu - Emergent Constraints on Future Projections of Tibetan Plateau Warming in Winter.pdf`
- **何时看**: 研究**冬季青藏高原变暖**的未来投影，以及**雪反照率反馈（SAF）**机制如何施加约束时。
- **核心贡献**: 通过地表反照率反馈过程，对CMIP6模型中青藏高原冬季地表气温（T2m）的未来变暖投影施加了突现约束。

#### jgr_atmos_2022_chen_surface_albedo_ec.pdf
- **路径**: `papers/jgr_atmos_2022_chen_surface_albedo_ec.pdf`
- **何时看**: 需要深入理解**雪反照率反馈**对青藏高原（海拔2000米以上）增温的具体贡献及其在模型中的偏差时。
- **核心贡献**: 诊断并观测约束了雪反照率反馈（SAF）对青藏高原冬季和春季表面变暖的贡献，发现在该地区**地表反照率的变化**与地表温度的部分变化（ΔT_SAF）之间存在显著关系。

---

### 6️⃣ 季风系统 (Monsoon Systems & ENSO)

#### grl_2025_wang_enso_east_asian_ec.pdf
- **路径**: `papers/grl_2025_wang_enso_east_asian_ec.pdf`
- **补充材料**: `papers/2025gl116648-sup-0001-supporting information si-s01.pdf`
- **何时看**: 分析ENSO（厄尔尼诺-南方涛动）对**东亚-西北太平洋（WNPAC）跨季节影响**在未来气候变暖下的变化和不确定性时。
- **核心贡献**: 利用多大集合（multi-SMILEs）和EC，发现**ENSO对东亚-西北太平洋气候的跨季节影响**在未来高排放情景下将**稳健加强**，且将投影不确定性降低了67%。

#### grl_2024_cheng_summer_monsoon_ec.pdf
- **路径**: `papers/grl_2024_cheng_summer_monsoon_ec.pdf`
- **何时看**: 研究气候变暖对**孟加拉湾和南海夏季风爆发时间（Onset）**影响的预测和约束时。
- **核心贡献**: 观测约束后的投影表明，孟加拉湾和南海夏季风爆发时间的**延迟程度将小于**原始模型集合的平均预测。

#### grl_2025_cheng_indian_monsoon_ec.pdf
- **路径**: `papers/grl_2025_cheng_indian_monsoon_ec.pdf`
- **补充材料**: `papers/supp-Geophysical Research Letters - 2025 - Cheng - A Shorter Duration of the Indian Summer Monsoon in Constrained Projections.pdf`
- **何时看**: 研究气候变暖情景下**印度夏季风持续时间**变化的约束预测时。
- **核心贡献**: 使用分层统计框架对印度夏季风持续时间进行约束，通过使用日气温数据来定义持续时间。

#### nat_commun_2022_chen_afroasian_monsoon_ec.pdf
- **路径**: `papers/nat_commun_2022_chen_afroasian_monsoon_ec.pdf`
- **何时看**: 研究**非洲-亚洲季风区域**未来降水总量预测的不确定性及其对洪水和水资源管理的影响时。
- **核心贡献**: 对非洲-亚洲季风降水的未来增加幅度进行了观测约束，预测其**增加幅度小于**原始模型投影，这意味着未来洪水风险可能较低。

---

### 7️⃣ 海冰/极地 (Sea Ice/Polar)

#### esd_Vogt_antarctic_sea_ice_ec.pdf
- **路径**: `papers/esd_Vogt_antarctic_sea_ice_ec.pdf`
- **何时看**: 研究**南大洋海洋热量吸收（OHU）**及其与**南极海冰范围**之间关系，并利用海冰作为约束因子时。
- **核心贡献**: 利用观测到的南极海冰范围作为突现约束，成功地**限制了未来南大洋的海洋热量吸收**的预测，这与云反馈等因素有关。

#### nat_commun_2023_kim_ice-free Arctic.pdf
- **路径**: `papers/nat_commun_2023_kim_ice-free Arctic.pdf`
- **何时看**: 评估北极在不同排放情景下**无冰状态（Ice-free Arctic）**出现的时间和概率，特别是涉及检测和归因分析时。
- **核心贡献**: 该研究使用**最优指纹法（optimal fingerprinting）**和EC方法，分析了北极海冰面积（SIA）异常，并约束了未来北极无冰状态的投影。

---

### 8️⃣ 其他应用 (Carbon Cycle, Wildfire, Global Change)

#### nat_commun-2024-cox-obs_constrained-carbon budgets.pdf
- **路径**: `papers/nat_commun-2024-cox-obs_constrained-carbon budgets.pdf`
- **何时看**: 研究如何利用观测约束减少对**瞬态气候响应（TCRE）**的不确定性，从而确定**全球剩余碳预算**时。
- **核心贡献**: 通过观测约束，减少了CMIP6模型在TCRE和全球剩余碳预算预测中的不确定性，并提供了2°C全球变暖情景下的碳预算估计。

#### nat_commun_2022_yu_wildfire_ml_ec.pdf
- **路径**: `papers/nat_commun_2022_yu_wildfire_ml_ec.pdf`
- **补充材料**: `papers/41467_2022_28853_MOESM1_ESM.pdf`
- **何时看**: 研究如何将**机器学习（ML）**与EC方法相结合，以评估**全球野火及其社会经济风险**时。
- **核心贡献**: 使用基于ML的观测约束投影，发现未来全球野火导致的**社会经济风险将升高**，尤其是在西非和中非地区。

---

### 9️⃣ 补充材料 (Supplementary Materials)

补充材料通常包含：
- 详细的模型列表和参数
- 额外的诊断分析图表
- 方法论的数学推导
- 完美模型测试结果
- 敏感性分析

| 补充材料文件 | 对应主文 |
|------------|---------|
| `2021jd036085-sup-0001-supporting information si-s01.docx` | [待确认主文] |
| `2023gl105427-sup-0001-supporting information si-s01.pdf` | `grl_2023_jiang_ghg_contribution_ec.pdf` |
| `2023gl105605-sup-0001-supporting information si-s01.docx` | [待确认主文] |
| `2024gl110994-sup-0001-supporting information si-s01.docx` | [待确认主文] |
| `2025gl116648-sup-0001-supporting information si-s01.pdf` | `grl_2025_wang_enso_east_asian_ec.pdf` |
| `41467_2022_28853_MOESM1_ESM.pdf` | `nat_commun_2022_yu_wildfire_ml_ec.pdf` |
| `41467_2022_30106_MOESM1_ESM.docx` | [待确认主文] |
| `41467_2022_34006_MOESM1_ESM.pdf` | `nat_commun_2022_zhang_extreme_precip_ec.pdf` |
| `41612_2024_648_MOESM1_ESM.pdf` | `npj_clim_atmos_2024_oreilly_european_climate_ec.pdf` |
| `43247_2025_2133_MOESM2_ESM.pdf` | [待确认主文] |
| `supp-Combined emergent constraints...pdf` | `nat_clim_Shiogama_combined_extreme_precip_ec.pdf` |
| `supp-Geophysical Research Letters - 2023 - Chen...docx` | `grl_2023_chen_china_warming_ec.pdf` |
| `supp-Geophysical Research Letters - 2024 - Hu...pdf` | `grl_2024_hu_tibetan_plateau_winter_ec.pdf` |
| `supp-Geophysical Research Letters - 2025 - Cheng...pdf` | `grl_2025_cheng_indian_monsoon_ec.pdf` |
| `supp-Liu_2024_Environ._Res._Lett...docx` | `erl_2024_liu_ec_EA_tas.pdf` |
| `supp-Npj-Improve the projection...pdf` | `npj_clim_atmos_zhu_east_china_precip_ec.pdf` |
| `supp-nature-Observation-constrained...pdf` | `nature_2024_obs_constrained_dry_spells.pdf` |
| `sciadv.adt6485_sm.pdf` | `sciadv.adt6485 (1).pdf` |
| `sciadv.adr5346_sm.pdf` | `sciadv.adr5346 (1).pdf` |

---

## 🎯 使用指南（For Claude Code）

### Claude 应该何时推荐用户阅读论文

#### 场景 1：用户问方法论问题
**用户问**："如何评估我的EC是否可靠？"
**Claude 回答**：
```
建议参考以下资源：

1. **代码实现**: `exemplars/code-examples/nature_2024_dry_spells/EC_KL_div_Brient_adopted.py`
   - 这是 binning analysis 的完整实现，可直接使用

2. **方法论论文**: `exemplars/papers/nat_clim_Shiogama_combined_extreme_precip_ec.pdf`
   - 第2-3页详细描述了binning方法
   - 补充材料包含完整的统计检验

3. **理论基础**: `references/methods.md` 第4章（Binning Analysis）
```

#### 场景 2：用户需要特定主题案例
**用户问**："有没有东亚降水的EC分析案例？"
**Claude 回答**：
```
推荐以下论文：

1. `exemplars/papers/npj_clim_atmos_zhu_east_china_precip_ec.pdf` ⭐
   - 专门研究华东夏季降水
   - 将不确定性减少了23%
   - 补充材料包含34个CMIP6模型列表和留一法验证

2. `exemplars/papers/nat_commun_2022_chen_afroasian_monsoon_ec.pdf`
   - 涵盖更广的亚洲季风区域
   - 包含降水变化的物理机制分析
```

#### 场景 3：用户遇到技术问题
**用户问**："我的EC关系r=0.35，p=0.02，这可靠吗？"
**Claude 回答**：
```
你的相关系数r=0.35处于"可接受"范围（见 `references/methods.md` 表格）。

建议进行以下验证：

1. **运行 binning analysis**:
   使用 `exemplars/code-examples/nature_2024_dry_spells/EC_KL_div_Brient_adopted.py`

2. **参考类似强度的案例**:
   - `exemplars/papers/npj_clim_atmos_zhu_east_china_precip_ec.pdf` (r≈0.4)
   - 查看他们如何进行可靠性评估

3. **检查物理机制**: 确保你的EC关系有合理的物理解释
```

#### 场景 4：用户需要完整工作流程
**用户问**："我第一次做EC分析，从哪开始？"
**Claude 回答**：
```
推荐学习路径：

**第1步 - 理解概念** (1-2小时)
- 阅读 `references/methods.md` 第I部分（核心三步框架）
- 重点理解：建立关系 → 物理机制 → 质量评估

**第2步 - 学习完整案例** (3-4小时)
- 阅读 `exemplars/papers/nature_2024_obs_constrained_dry_spells.pdf`
- 查看配套代码 `code-examples/nature_2024_dry_spells/`
- 运行 `ECpaper_Figure1.py` 理解EC散点图

**第3步 - 改写代码适配你的数据** (1-2天)
- 复制 `useful_functions.py` 作为起点
- 修改数据路径、区域定义、变量名
- 保持核心EC计算逻辑不变

**第4步 - 评估可靠性** (1天)
- 运行 `EC_KL_div_Brient_adopted.py`
- 对照 `references/methods.md` 的可靠性标准

**遇到问题时**：
- 技术细节 → 查看论文补充材料
- 物理机制 → 查看相关主题的论文（见上方分类）
```

---

## 🔍 快速查找指南

### 按研究问题查找

| 你的研究问题 | 推荐论文 |
|------------|---------|
| 干旱/干燥期 | `nature_2024_obs_constrained_dry_spells.pdf` ⭐代码可用 |
| 极端高温 | `cee-2025-Simolo_constrained_hot extremes.pdf` |
| 极端降水 | `nat_clim_Shiogama_combined_extreme_precip_ec.pdf`, `nat_commun_2022_zhang_extreme_precip_ec.pdf` |
| 中国区域变暖 | `grl_2023_chen_china_warming_ec.pdf` |
| 东亚降水 | `npj_clim_atmos_zhu_east_china_precip_ec.pdf` |
| 青藏高原 | `grl_2024_hu_tibetan_plateau_winter_ec.pdf`, `jgr_atmos_2022_chen_surface_albedo_ec.pdf` |
| 季风系统 | `grl_2025_cheng_indian_monsoon_ec.pdf`, `nat_commun_2022_chen_afroasian_monsoon_ec.pdf` |
| ENSO影响 | `grl_2025_wang_enso_east_asian_ec.pdf` |
| 极地/海冰 | `nat_commun_2023_kim_ice-free Arctic.pdf`, `esd_Vogt_antarctic_sea_ice_ec.pdf` |
| 碳预算 | `nat_commun-2024-cox-obs_constrained-carbon budgets.pdf` |
| 方法比较 | `npj_clim_atmos_2024_oreilly_european_climate_ec.pdf` |

### 按技术方法查找

| 技术方法 | 推荐论文 |
|---------|---------|
| Binning Analysis | `nat_clim_Shiogama_combined_extreme_precip_ec.pdf` + 代码示例 ⭐ |
| 组合约束 (Combined EC) | `nat_clim_Shiogama_combined_extreme_precip_ec.pdf` |
| 最优指纹法 | `nat_commun_2023_kim_ice-free Arctic.pdf`, `grl_2023_jiang_ghg_contribution_ec.pdf` |
| 机器学习+EC | `nat_commun_2022_yu_wildfire_ml_ec.pdf` |
| 多方法比较 | `npj_clim_atmos_2024_oreilly_european_climate_ec.pdf` |
| 留一法验证 | 大多数论文的补充材料 |
| 分层统计框架 | `grl_2025_cheng_indian_monsoon_ec.pdf` |

### 按模型代次查找

| 模型版本 | 主要使用的论文 |
|---------|--------------|
| CMIP6 | 大部分2023年后的论文 |
| CMIP5+CMIP6 | `Nature_2022_Shiogama_future_precipation.pdf`, `nat_clim_Shiogama_combined_extreme_precip_ec.pdf` |
| 大集合 (SMILEs) | `grl_2025_wang_enso_east_asian_ec.pdf` |

---

## 📝 推荐学习路径

### 新手路径（第一次接触EC）
1. ✅ 读 `references/methods.md` 核心部分（1-2小时）
2. ✅ 读 `nature_2024_obs_constrained_dry_spells.pdf` 主文（2-3小时）
3. ✅ 运行代码示例 `code-examples/nature_2024_dry_spells/ECpaper_Figure1.py`（1小时）
4. ✅ 改写代码适配自己的数据（1-2天）

### 进阶路径（已有基础，需要特定技术）
1. 🔍 在上方"快速查找指南"找到相关主题论文
2. 📖 阅读论文主文 + 补充材料的方法部分
3. 💻 查看是否有配套代码（目前只有Nature 2024案例）
4. 🔬 实施并验证

### 专家路径（准备发表论文）
1. 📚 阅读3-5篇相关主题论文（对比不同方法）
2. 🧪 实施多种可靠性评估（binning + 留一法 + 敏感性测试）
3. 🔍 阅读 `npj_clim_atmos_2024_oreilly_european_climate_ec.pdf` 了解方法比较
4. ✍️ 参考 `nature_2024_obs_constrained_dry_spells.pdf` 的论文结构

---

## 📊 统计信息

- **总论文数**: 43篇（含补充材料）
- **主文论文**: ~25篇
- **补充材料**: ~18个文件
- **代码示例**: 1个完整案例（Nature 2024）
- **覆盖期刊**: Nature, Nature Communications, Nature Climate Change, GRL, Science Advances, NPJ Climate, ERL, ESD, JGR等
- **时间跨度**: 2021-2025年（最新研究）

---

**最后更新**: 2025-10-28
**维护者**: Climate-AI Research Team
**版本**: 1.0

**Note**: 如发现论文分类错误或补充材料对应关系有误，请更新此文件。
