#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
戴维南定理（Thévenin's Theorem）验证 —— PySpice 直流工作点仿真
================================================================

原电路（分压器，输出端从 R2 两端取）::

                  R1 = 1 kΩ            vout
        ┌────────[=====]──────┬──────────────┬──────────┐
        │                     │              │          │
      ┌─┴─┐                 ┌─┴─┐        （负载 RL） （测量支路）
      │V1 │ 10 V          R2 │   │ 2 kΩ       │          │
      └─┬─┘                 └─┬─┘            │          │
        │                     │              │          │
       gnd                   gnd            gnd        gnd

理论值（手算）：
    开路电压  Vth = V1 · R2/(R1+R2) = 10 × 2/3 = 6.6667 V
    短路电流  Isc = V1 / R1 = 10 V / 1 kΩ = 10 mA
              （输出短路后 R2 被短路，电流只受 R1 限制）
    等效电阻  Rth = Vth / Isc = R1 ∥ R2 = 666.667 Ω
    带载电压  VL  = Vth · RL/(RL+Rth)

仿真流程：
    ① 开路    ：输出端什么都不接，直接测 vout 对地电压 → Vth
    ② 短路    ：输出端接一个 0 V 电压源当“电流表”（即理想短路），
                读该支路电流 → Isc，再用 Rth = Vth / Isc 求等效电阻
    ③ 带载    ：输出端分别接 1 kΩ / 2 kΩ / 5 kΩ / 10 kΩ 负载，测输出电压
    ④ 等效电路：用 Vth 串联 Rth 重新搭一个电路，重复 ①②③，
                对比“原电路”与“等效电路”的端电压是否一致 → 验证戴维南定理
    ⑤ 附加验证：把 10 V 电源置零（换成 0 V 源 = 短路），从输出端注入 1 A
                测试电流，测得的端电压在数值上就等于 Rth
    ⑥ 用 matplotlib 画“负载电压 vs 负载电阻”对比图

运行环境：
    pip install PySpice matplotlib
    另外需要系统安装 ngspice 动态库（PySpice 只是封装，求解由 ngspice 完成）：
      * Windows ：把含 ngspice.dll 的目录加入 PATH，或设置
                  NgSpiceShared.LIBRARY_PATH = r"C:\path\to\ngspice.dll"
      * Linux   ：sudo apt install ngspice
      * macOS   ：brew install ngspice

输出：
    * 控制台：Vth / Isc / Rth、不同负载下的电压、以及仿真与理论/模型的对比表
    * 图片  ：thevenin_load_voltage.png
"""

import numpy as np
import matplotlib.pyplot as plt

from PySpice.Spice.Netlist import Circuit                  # 电路 / 网表描述
from PySpice.Unit import u_A, u_Ohm, u_V, u_kOhm                # 带单位的物理量
from PySpice.Spice.NgSpice.Shared import NgSpiceShared

NgSpiceShared.LIBRARY_PATH = r"D:\ngspice-47_dll_64\Spice64_dll\dll-vs\ngspice.dll"
# ============================================================================
# 0. 绘图全局设置
# ============================================================================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                   'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False      # 负号正常显示，避免出现方框

# ============================================================================
# 1. 电路参数（集中在此，便于修改）
# ============================================================================
V1_VALUE = 10 @ u_V          # 直流电压源 10 V
R1_VALUE = 1 @ u_kOhm        # 上臂电阻 R1 = 1 kΩ
R2_VALUE = 2 @ u_kOhm        # 下臂电阻 R2 = 2 kΩ（输出取自它两端）

# 测试用的负载电阻
R_LOADS = [1 @ u_kOhm, 2 @ u_kOhm, 5 @ u_kOhm, 10 @ u_kOhm]

# 附加验证用的测试电流（电流源注入 1 A，测得端电压即等效电阻）
I_TEST = 1 @ u_A

OUTPUT_PNG = 'thevenin_load_voltage.png'

# ============================================================================
# 2. 网表构建
# ============================================================================
def _add_divider(circuit):
    """把原电路的固定部分（V1 + R1 + R2）加进网表：

        V1  vin  0    10V     ← 10 V 直流电源
        R1  vin  vout 1kOhm   ← 上臂电阻
        R2  vout 0    2kOhm   ← 下臂电阻，输出 vout 就取自它两端
    """
    circuit.V('1', 'vin', circuit.gnd, V1_VALUE)
    circuit.R(1, 'vin', 'vout', R1_VALUE)
    circuit.R(2, 'vout', circuit.gnd, R2_VALUE)

def _add_terminal(circuit, load=None, ammeter=False):
    """在输出端（vout 对地）接上测量用的元件：

    ammeter=True → 接一个 0 V 电压源。这是 SPICE 里测电流的标准技巧：
                   0 V 源相当于理想短路，同时它所在支路的电流可以被读出来，
                   正好用来测“短路电流”。
    load 不为 None → 接一个阻值为 load 的负载电阻（RL 接在 vout 与地之间）。
    两者都不给 → 输出端开路。
    """
    if ammeter:
        circuit.V('sc', 'vout', circuit.gnd, 0 @ u_V)
    elif load is not None:
        circuit.R('load', 'vout', circuit.gnd, load)

def build_original_circuit(load=None, ammeter=False):
    """原电路：10 V 电源 + R1 + R2 分压，输出取自 R2 两端。"""
    circuit = Circuit('原电路：10V 电源 + R1(1kΩ) + R2(2kΩ)')
    _add_divider(circuit)
    _add_terminal(circuit, load=load, ammeter=ammeter)
    return circuit

def build_thevenin_circuit(vth, rth, load=None, ammeter=False):
    """戴维南等效电路：电压源 Vth 串联等效电阻 Rth，输出端同样接负载 / 电流表。

        Vth  vsrc  0    <vth>   ← 戴维南等效电压源
        Rth  vsrc  vout <rth>   ← 戴维南等效电阻
    """
    circuit = Circuit('戴维南等效电路：Vth + Rth')
    circuit.V('th', 'vsrc', circuit.gnd, vth)
    circuit.R('th', 'vsrc', 'vout', rth)
    _add_terminal(circuit, load=load, ammeter=ammeter)
    return circuit

def build_rth_test_circuit(i_test=I_TEST):
    """附加验证：用“外加测试源法”直接测等效电阻。

    做法：把原电路里的独立电压源“置零”（电压源置零 = 短路，这里用一个 0 V
          源代替它，通路保持不变），再从输出端注入 1 A 测试电流。此时从输出
          端看进去只剩 R1 ∥ R2，端电压 V(vout) = Rth × 1 A，数值上等于 Rth。
    """
    circuit = Circuit('Rth 测试电路：电源置零 + 外加 1A 测试电流')
    circuit.V('1', 'vin', circuit.gnd, 0 @ u_V)        # 10 V 电源置零（短路）
    circuit.R(1, 'vin', 'vout', R1_VALUE)
    circuit.R(2, 'vout', circuit.gnd, R2_VALUE)
    # 电流源 Itest：正节点是地、负节点是 vout，即向 vout 注入 1 A
    circuit.I('test', circuit.gnd, 'vout', i_test)
    return circuit

# ============================================================================
# 3. 仿真与取值工具
# ============================================================================
def _to_float(waveform):
    """工作点分析里每个变量只有一个数据点，把它取出来转成普通浮点数。

    注意：PySpice 给结果数据带了单位（u_V / u_A），直接 np.asarray 可能拿到
    带单位的对象，所以这里统一用 float() 转换，它会做单位换算得到基本单位值。
    """
    values = np.asarray(waveform).ravel()
    return float(values[0])

def run_operating_point(circuit):
    """执行直流工作点分析（对应 SPICE 指令 .op），返回两个字典：

        nodes    : {节点名: 电压 [V]}
        branches : {电压源名: 流过它的电流 [A]}
    """
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.operating_point()

    nodes = {name: _to_float(waveform) for name, waveform in analysis.nodes.items()}
    branches = {name: _to_float(waveform) for name, waveform in analysis.branches.items()}
    return nodes, branches

def _find_branch(branches, source_name):
    """在支路电流字典里按名字（忽略大小写）找出某个电压源的电流。

    ngspice 把支路电流命名为 “vsc#branch”，PySpice 去掉后缀后以 'vsc' 为键。
    """
    for name, value in branches.items():
        if name.lower() == source_name.lower():
            return value
    raise KeyError('未找到支路 {}，可用支路：{}'.format(source_name, list(branches)))

def measure_open_circuit_voltage(circuit):
    """① 开路：输出端不接负载，测 vout 对地电压（即戴维南电压 Vth）"""
    nodes, _ = run_operating_point(circuit)
    return nodes['vout']

def measure_short_circuit_current(circuit):
    """② 短路：输出端由 0 V 电流表短接到地，读该支路电流（即短路电流 Isc）

    SPICE 中支路电流的正方向是“从电压源正节点流入、由负节点流出”，
    这里电流由 vout 流经电流表到地，实测为正值；取绝对值可避免依赖符号约定。
    """
    _, branches = run_operating_point(circuit)
    return abs(_find_branch(branches, 'vsc'))

def measure_load_voltage(circuit):
    """③ 带载：测负载两端的电压（负载直接并在输出端，所以仍是 vout 对地电压）"""
    nodes, _ = run_operating_point(circuit)
    return nodes['vout']

# ============================================================================
# 4. 绘图：负载电压 vs 负载电阻
# ============================================================================
def plot_load_voltage(vth, rth, r_loads, v_original, v_thevenin, save_path=OUTPUT_PNG):
    """画对比图：戴维南模型曲线 + 两个电路的 SPICE 仿真点 + 开路电压/最大功率点标注"""
    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    # ---- (1) 戴维南等效模型曲线：VL = Vth·RL/(RL+Rth) ----
    rl_curve = np.logspace(np.log10(100.0), np.log10(100e3), 400)   # 100 Ω ~ 100 kΩ
    ax.semilogx(rl_curve, vth * rl_curve / (rl_curve + rth),
                color='#1f77b4', linewidth=2.2,
                label='戴维南模型 $V_L=V_{th}\\,R_L/(R_L+R_{th})$')

    # ---- (2) 原电路的 SPICE 仿真结果（散点）----
    ax.semilogx(r_loads, v_original, 'o', markersize=10, markerfacecolor='none',
                markeredgewidth=2, color='#d62728', label='原电路 SPICE 仿真')

    # ---- (3) 戴维南等效电路的 SPICE 仿真结果（散点，应与原电路完全重合）----
    ax.semilogx(r_loads, v_thevenin, 'x', markersize=9, markeredgewidth=2,
                color='#2ca02c', label='戴维南等效电路 SPICE 仿真')

    # ---- (4) 开路电压 Vth 参考线 ----
    ax.axhline(vth, color='gray', linestyle='--', linewidth=1.2,
               # 注意：LaTeX 里的 {th} 必须写成 {{th}}，否则会被 str.format 当成占位符
               label='开路电压 $V_{{th}}$ = {:.3f} V'.format(vth))

    # ---- (5) RL = Rth 处（最大功率传输点，输出电压恰为 Vth/2）----
    ax.axvline(rth, color='gray', linestyle=':', linewidth=1.2)
    ax.annotate('$R_L=R_{{th}}$ = {:.1f} Ω 时\n$V_L=V_{{th}}/2$ = {:.3f} V'.format(rth, vth / 2),
                xy=(rth, vth / 2), xycoords='data',
                xytext=(0.60, 0.30), textcoords='axes fraction',
                fontsize=10, color='dimgray',
                arrowprops=dict(arrowstyle='->', color='gray', linewidth=1.2))

    # ---- (6) 给每个仿真点标注电压值 ----
    for rl, vl in zip(r_loads, v_original):
        ax.annotate('{:.3f} V'.format(vl), xy=(rl, vl), xycoords='data',
                    xytext=(0, 11), textcoords='offset points',
                    ha='center', fontsize=9, color='#d62728')

    ax.set_xlabel('负载电阻 $R_L$ (Ω)', fontsize=12)
    ax.set_ylabel('负载电压 $V_L$ (V)', fontsize=12)
    ax.set_title('戴维南定理验证：负载电压 vs 负载电阻\n'
                 'V1 = 10 V, R1 = 1 kΩ, R2 = 2 kΩ  →  '
                 '$V_{{th}}$ = {:.3f} V, $R_{{th}}$ = {:.1f} Ω'.format(vth, rth),
                 fontsize=12)
    ax.set_xlim(100, 100e3)
    ax.set_ylim(0, vth * 1.15)
    ax.grid(True, which='both', linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print('\n[图片] 对比图已保存到: {}'.format(save_path))
    return fig

# ============================================================================
# 5. 主流程
# ============================================================================
def main():
    print('=' * 74)
    print('戴维南定理验证 —— PySpice 直流工作点仿真')
    print('=' * 74)

    v1 = float(V1_VALUE)
    r1 = float(R1_VALUE)
    r2 = float(R2_VALUE)

    # ---- 理论值（手算，用于对照）----
    vth_theory = v1 * r2 / (r1 + r2)          # 分压公式
    isc_theory = v1 / r1                      # 短路时 R2 被旁路
    rth_theory = r1 * r2 / (r1 + r2)          # 从输出端看进去的等效电阻

    print('电路参数：V1 = {:g} V, R1 = {:g} Ω, R2 = {:g} Ω'.format(v1, r1, r2))
    print('理论值  ：Vth = {:.6f} V, Isc = {:.6f} A, Rth = {:.6f} Ω'.format(
        vth_theory, isc_theory, rth_theory))

    # ======================================================================
    # ① 开路电压 → Vth
    # ======================================================================
    circuit_open = build_original_circuit()
    print('\n---- 原电路网表（开路）----')
    print(circuit_open)

    vth = measure_open_circuit_voltage(circuit_open)
    print('---------- ① 开路电压 Vth ----------')
    print('仿真值 : Vth = {:.6f} V'.format(vth))
    print('理论值 : {:.6f} V      相对误差 {:.3e} %'.format(
        vth_theory, abs(vth - vth_theory) / vth_theory * 100))

    # ======================================================================
    # ② 短路电流 → Isc
    # ======================================================================
    circuit_short = build_original_circuit(ammeter=True)
    print('\n---- 原电路网表（输出端加 0V 电流表，即短路）----')
    print(circuit_short)

    isc = measure_short_circuit_current(circuit_short)
    print('---------- ② 短路电流 Isc ----------')
    print('仿真值 : Isc = {:.6f} A = {:.6f} mA'.format(isc, isc * 1000))
    print('理论值 : {:.6f} mA     相对误差 {:.3e} %'.format(
        isc_theory * 1000, abs(isc - isc_theory) / isc_theory * 100))

    # ======================================================================
    # ③ 等效电阻 Rth = Vth / Isc
    # ======================================================================
    rth = vth / isc
    rth_parallel = r1 * r2 / (r1 + r2)
    print('---------- ③ 等效电阻 Rth = Vth / Isc ----------')
    print('仿真值 : Rth = {:.6f} Ω'.format(rth))
    print('校验值 : R1∥R2 = {:.6f} Ω     相对误差 {:.3e} %'.format(
        rth_parallel, abs(rth - rth_parallel) / rth_parallel * 100))

    # ======================================================================
    # ④ 接不同负载时的输出电压：原电路 vs 戴维南等效电路
    # ======================================================================
    print('---------- ④ 不同负载下的输出电压 ----------')
    r_loads, v_original, v_thevenin = [], [], []
    for load in R_LOADS:
        rl = float(load)                                     # 负载阻值 [Ω]
        # 原电路带该负载
        vo = measure_load_voltage(build_original_circuit(load=load))
        # 戴维南等效电路带同一负载（Vth、Rth 都用上一步的仿真值）
        vt = measure_load_voltage(build_thevenin_circuit(vth @ u_V, rth @ u_Ohm, load=load))
        r_loads.append(rl)
        v_original.append(vo)
        v_thevenin.append(vt)

    print('{:>10} {:>16} {:>16} {:>16} {:>14}'.format(
        '负载 RL', '原电路 VL (V)', '等效电路 VL (V)', '模型 Vth·RL/(RL+Rth)', '误差 (%)'))
    for rl, vo, vt in zip(r_loads, v_original, v_thevenin):
        v_model = vth * rl / (rl + rth)                      # 戴维南模型预测值
        error = abs(vo - v_model) / v_model * 100            # 原电路 vs 模型
        print('{:>9.0f}Ω {:>16.9f} {:>16.9f} {:>16.9f} {:>14.2e}'.format(
            rl, vo, vt, v_model, error))

    # 两电路之间的最大偏差：这是戴维南定理成立与否的直接证据
    max_diff = max(abs(a - b) for a, b in zip(v_original, v_thevenin))
    print('原电路与等效电路输出电压的最大差值：{:.3e} V（≈ 0，说明两者外特性一致）'.format(max_diff))

    # ======================================================================
    # ⑤ 附加验证：电源置零 + 外加 1 A 测试电流，直接测 Rth
    # ======================================================================
    circuit_test = build_rth_test_circuit()
    print('\n---- Rth 测试电路网表（电源置零，外加 1A 测试电流）----')
    print(circuit_test)

    nodes_test, _ = run_operating_point(circuit_test)
    rth_test = abs(nodes_test['vout']) / float(I_TEST)       # R = V / I
    print('---------- ⑤ 外加测试电流法测 Rth ----------')
    print('仿真值 : V(vout) = {:.6f} mV（注入 1 A）→ Rth = {:.6f} Ω'.format(
        nodes_test['vout'] * 1000, rth_test))
    print('与 ③ 的 Rth 相对误差：{:.3e} %'.format(abs(rth_test - rth) / rth * 100))

    # ======================================================================
    # ⑥ 绘图
    # ======================================================================
    plot_load_voltage(vth, rth, r_loads, v_original, v_thevenin)

    plt.show()

if __name__ == '__main__':
    main()