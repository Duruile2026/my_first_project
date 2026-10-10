#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
RC 一阶低通滤波器 —— 交流小信号扫频（AC）仿真
================================================================

电路拓扑（无源一阶 RC 低通滤波器）::

              R = 1 kΩ
    vin ─────[=====]─────┬───── vout
                         │
                      ───┴───  C = 100 nF
                         │
                        gnd

    输入 vin 为幅值 1 V 的交流正弦信号（AC 分析使用它的相量幅值）。

理论公式：
    传递函数      H(jω) = Vout/Vin = 1 / (1 + j·ωRC) ,  ω = 2πf
    幅频特性      |H| = 1 / √(1 + (f/fc)²)
    相频特性      φ   = -arctan(f/fc)
    截止频率      fc  = 1 / (2πRC) ≈ 1591.55 Hz
    在 fc 处：|H| = 1/√2 ≈ 0.7071（即 -3.0103 dB），φ = -45°

仿真内容：
    1. 用 PySpice 搭建 RC 电路（R = 1 kΩ，C = 100 nF）
    2. 输入 1 V 正弦电压源，做 AC 扫频分析：10 Hz ~ 1 MHz，每十倍频程 100 点
    3. 计算幅频特性（dB）与相频特性（度）
    4. 用 matplotlib 绘制伯德图，并在图上标出截止频率（-3 dB 点）
    5. 把仿真结果与理论值逐点对比，验证正确性

运行环境：
    pip install PySpice matplotlib
    还需要系统安装 ngspice 动态库（PySpice 只是 Python 封装，实际求解由 ngspice 完成）：
      * Windows ：下载 ngspice，把含 ngspice.dll 的目录加入 PATH，
                  或在脚本中指定：
                      from PySpice.Spice.NgSpice.Shared import NgSpiceShared
                      NgSpiceShared.LIBRARY_PATH = r"C:\path\to\ngspice.dll"
      * Linux   ：sudo apt install ngspice        （或 libngspice0 / ngspice-dev）
      * macOS   ：brew install ngspice

输出：
    * 控制台：仿真值与理论值对比表
    * 图片  ：rc_lowpass_bode.png（上：幅频特性，下：相频特性）
"""

import numpy as np
import matplotlib.pyplot as plt

from PySpice.Spice.Netlist import Circuit                    # 电路 / 网表描述
from PySpice.Unit import u_Hz, u_V, u_kHz, u_kOhm, u_nF       # 带单位的物理量
from PySpice.Spice.NgSpice.Shared import NgSpiceShared

NgSpiceShared.LIBRARY_PATH = r"D:\ngspice-47_dll_64\Spice64_dll\dll-vs\ngspice.dll"
# ============================================================================
# 0. 绘图全局设置
# ============================================================================
# 指定中文字体，避免图中的中文显示成方框（按系统已有字体依次回退）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                   'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False      # 让负号“−”正常显示而不是方框

# ============================================================================
# 1. 电路参数与仿真设置（都集中在这里，便于修改）
# ============================================================================
R = 1 @ u_kOhm         # 电阻 R = 1 kΩ
C = 100 @ u_nF         # 电容 C = 100 nF
V_AC = 1 @ u_V         # 输入正弦波（相量）幅值 = 1 V

# 换算成基本单位（Ω、F）的纯浮点数，供 numpy 做理论计算
R_OHM = float(R)       # 1000.0 Ω
C_FARAD = float(C)     # 1e-7 F

F_START = 10.0           # 扫频起始频率 10 Hz
F_STOP = 1e6             # 扫频终止频率 1 MHz
POINTS_PER_DECADE = 100  # 每个十倍频程取 100 个点（对数均匀分布）

# |H| = 1/√2 对应的分贝值，即严格意义上的“-3 dB”点：≈ -3.0103 dB
DB_MINUS_3 = 20.0 * np.log10(1.0 / np.sqrt(2.0))

OUTPUT_PNG = 'rc_lowpass_bode.png'      # 图片输出文件名

# ============================================================================
# 2. 理论计算（作为仿真结果的对照基准）
# ============================================================================
def theoretical_cutoff_frequency(r=R_OHM, c=C_FARAD):
    """理论截止频率 fc = 1 / (2πRC)  [Hz]"""
    return 1.0 / (2.0 * np.pi * r * c)

def theoretical_transfer_function(frequency):
    """理论传递函数 H(jω) = 1 / (1 + j·2πfRC)，返回复数结果"""
    omega_rc = 2.0 * np.pi * np.asarray(frequency, dtype=float) * R_OHM * C_FARAD
    return 1.0 / (1.0 + 1j * omega_rc)

def theoretical_gain_phase(frequency):
    """理论幅频特性 [dB] 与相频特性 [°]"""
    h = theoretical_transfer_function(frequency)
    return 20.0 * np.log10(np.abs(h)), np.degrees(np.angle(h))

# ============================================================================
# 3. 用 PySpice 搭建电路
# ============================================================================
def build_rc_circuit():
    """
    构建 RC 低通滤波器的 SPICE 网表::

        Vinput  vin  0    DC 0V AC 1V SIN(0V 1V 1kHz)
        R1      vin  vout 1kOhm
        C1      vout 0    100nF

    说明：AC 分析只使用电压源的“交流幅值” ac_magnitude，
          与 SIN(...) 中的瞬态幅值 / 频率无关，这里两者都取 1 V 以保持一致。
    """
    circuit = Circuit('RC 一阶低通滤波器')

    # 交流电压源：正端接 vin，负端接参考地 circuit.gnd（即节点 0）
    #   ac_magnitude : AC 分析所用的相量幅值 = 1 V（初相位默认 0°）
    #   amplitude / frequency : 瞬态分析用的正弦参数，本脚本做 AC 分析用不到
    circuit.SinusoidalVoltageSource('input', 'vin', circuit.gnd,
                                    amplitude=V_AC,
                                    frequency=1 @ u_kHz,
                                    ac_magnitude=V_AC)

    # 串联电阻 R1：vin -> vout
    circuit.R(1, 'vin', 'vout', R)

    # 并联电容 C1：vout -> 地（与 R 一起构成低通网络）
    circuit.C(1, 'vout', circuit.gnd, C)

    return circuit

# ============================================================================
# 4. 执行 AC 扫频分析（调用 ngspice 求解）
# ============================================================================
def run_ac_analysis(circuit):
    """
    做交流小信号分析（对应 SPICE 指令 .ac dec ...）。

    返回
    ----
    frequency : ndarray            扫频频率点 [Hz]
    vout      : ndarray(complex)   输出节点 vout 的电压相量 [V]（含幅值与相位）
    """
    # 创建仿真器（默认后端即 ngspice 共享库）
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)

    # AC 分析参数：
    #   variation='dec'       -> 频率按“十倍频程”对数分布
    #   number_of_points=100  -> 每个十倍频程取 100 个频点
    analysis = simulator.ac(start_frequency=F_START @ u_Hz,
                            stop_frequency=F_STOP @ u_Hz,
                            number_of_points=POINTS_PER_DECADE,
                            variation='dec')

    frequency = np.asarray(analysis.frequency, dtype=float)   # 频率轴 [Hz]
    vout = np.asarray(analysis['vout'])                       # 复数相量
    return frequency, vout

# ============================================================================
# 5. 由复数相量得到幅频 / 相频特性
# ============================================================================
def compute_bode(frequency, vout):
    """
    把复数输出电压换算成画伯德图需要的数据：

        幅频特性 gain_db   = 20·log10(|Vout|)     [dB]
                             （Vin = 1 V，所以 |Vout/Vin| 就等于 |Vout|）
        相频特性 phase_deg = ∠Vout                [°]
    """
    gain_db = 20.0 * np.log10(np.abs(vout))              # 幅度用 20log，单位 dB
    phase_deg = np.degrees(np.unwrap(np.angle(vout)))    # 弧度转角度，unwrap 防止 ±180° 跳变
    return gain_db, phase_deg

def find_cutoff_frequency_from_data(frequency, gain_db):
    """
    从仿真曲线上反查截止频率：幅值随频率单调下降，
    用线性插值求出幅值等于 -3.0103 dB（即 |H| = 1/√2）处的频率。
    """
    return float(np.interp(DB_MINUS_3, gain_db[::-1], frequency[::-1]))

# ============================================================================
# 6. 绘制伯德图并标注截止频率
# ============================================================================
def plot_bode(frequency, gain_db, phase_deg, f_c_theory, f_c_sim,
              save_path=OUTPUT_PNG,
              title='RC 一阶低通滤波器（R = 1 kΩ, C = 100 nF）频率响应'):
    """绘制上下两栏的伯德图：上栏幅频特性，下栏相频特性。"""
    fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # ------------------------------------------------------------------
    # 6.1 幅频特性（dB）—— x 轴取对数
    # ------------------------------------------------------------------
    ax_mag.semilogx(frequency, gain_db, color='#1f77b4', linewidth=2.0,
                    label='仿真：$20\\log_{10}|V_{out}/V_{in}|$')

    # 理想伯德渐近线：fc 之前为 0 dB，fc 之后按 -20 dB/十倍频程下降
    ax_mag.semilogx([F_START, f_c_theory], [0.0, 0.0],
                    color='gray', linestyle=(0, (6, 4)), linewidth=1.0,
                    label='理想渐近线（0 dB / −20 dB·dec$^{-1}$）')
    ax_mag.semilogx([f_c_theory, F_STOP],
                    [0.0, -20.0 * np.log10(F_STOP / f_c_theory)],
                    color='gray', linestyle=(0, (6, 4)), linewidth=1.0)

    # -3 dB 参考水平线 + 截止频率垂直线
    ax_mag.axhline(DB_MINUS_3, color='red', linestyle='--', linewidth=1.2,
                   label='−3 dB 参考线（|H| = 1/√2 = 0.7071）')
    ax_mag.axvline(f_c_theory, color='red', linestyle=':', linewidth=1.2)

    # 在 -3 dB 与 fc 的交点处画标记点并加文字标注（本图的关键标注）
    ax_mag.plot([f_c_sim], [DB_MINUS_3], marker='o', markersize=8,
                color='red', zorder=5)
    ax_mag.annotate('截止频率 $f_c$ = {:.1f} Hz\n'
                    '幅值 −3.01 dB（|H| = 0.7071）\n'
                    '相位 −45°'.format(f_c_theory),
                    xy=(f_c_sim, DB_MINUS_3), xycoords='data',
                    xytext=(0.06, 0.30), textcoords='axes fraction',
                    verticalalignment='bottom',      # 让整段文字浮在锚点上方，避免压住图例
                    fontsize=10, color='red',
                    arrowprops=dict(arrowstyle='->', color='red', linewidth=1.2))

    ax_mag.set_ylabel('幅值 (dB)', fontsize=11)
    ax_mag.set_title(title + '\nAC 扫频：{:.0f} Hz ~ {:.0f} MHz，每十倍频程 {} 点'.format(
        F_START, F_STOP / 1e6, POINTS_PER_DECADE), fontsize=12)
    ax_mag.set_ylim(-60, 5)
    ax_mag.grid(True, which='both', linestyle=':', alpha=0.6)
    ax_mag.legend(loc='lower left', fontsize=9, framealpha=0.9)

    # ------------------------------------------------------------------
    # 6.2 相频特性（度）
    # ------------------------------------------------------------------
    ax_ph.semilogx(frequency, phase_deg, color='#d62728', linewidth=2.0,
                   label='仿真：$\\angle V_{out}$')
    ax_ph.axhline(-45.0, color='red', linestyle='--', linewidth=1.2,
                  label='−45° 参考线（该处 f = $f_c$）')
    ax_ph.axvline(f_c_theory, color='red', linestyle=':', linewidth=1.2)
    ax_ph.plot([f_c_sim], [-45.0], marker='o', markersize=8, color='red', zorder=5)
    ax_ph.annotate('$f = f_c$ 时相位恰为 −45°',
                   xy=(f_c_sim, -45.0), xycoords='data',
                   xytext=(0.07, 0.60), textcoords='axes fraction',
                   fontsize=10, color='red',
                   arrowprops=dict(arrowstyle='->', color='red', linewidth=1.2))

    ax_ph.set_xlabel('频率 f (Hz)', fontsize=11)
    ax_ph.set_ylabel('相位 (°)', fontsize=11)
    ax_ph.set_xlim(F_START, F_STOP)
    ax_ph.set_ylim(-100, 10)
    ax_ph.grid(True, which='both', linestyle=':', alpha=0.6)
    ax_ph.legend(loc='lower left', fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')    # 保存图片
    print('\n[图片] 伯德图已保存到: {}'.format(save_path))
    return fig

# ============================================================================
# 7. 主流程
# ============================================================================
def main():
    print('=' * 68)
    print('RC 一阶低通滤波器 —— AC 扫频仿真 (PySpice + ngspice + matplotlib)')
    print('=' * 68)

    # ---- 理论值 ----
    f_c_theory = theoretical_cutoff_frequency()
    print('电路参数 : R = 1 kΩ = {:g} Ω,  C = 100 nF = {:g} F'.format(R_OHM, C_FARAD))
    print('理论截止频率 fc = 1/(2πRC) = {:.4f} Hz'.format(f_c_theory))
    print('  该频率处：|H| = 1/√2 = {:.4f}（{:.4f} dB），相位 = -45°'.format(
        1 / np.sqrt(2), DB_MINUS_3))

    # ---- 搭建电路并打印网表，便于核对 ----
    circuit = build_rc_circuit()
    print('\n---- SPICE 网表 ----')
    print(circuit)

    # ---- 运行 AC 分析 ----
    try:
        frequency, vout = run_ac_analysis(circuit)
    except OSError as exc:      # 通常是找不到 / 无法加载 ngspice 动态库
        raise SystemExit(
            '\n[错误] 无法加载 ngspice 动态库，PySpice 无法完成仿真。\n'
            '       请先安装 ngspice，并在脚本中指定库路径，例如：\n'
            '           from PySpice.Spice.NgSpice.Shared import NgSpiceShared\n'
            '           NgSpiceShared.LIBRARY_PATH = r"C:\\path\\to\\ngspice.dll"\n'
            '       原始错误：{}'.format(exc))

    print('仿真完成：共 {} 个频点，{:.1f} Hz ~ {:.6g} Hz'.format(
        frequency.size, frequency[0], frequency[-1]))

    # ---- 换算成幅频 / 相频特性 ----
    gain_db, phase_deg = compute_bode(frequency, vout)

    # ---- 从仿真数据中提取 -3 dB 频率，并与理论值比较 ----
    f_c_sim = find_cutoff_frequency_from_data(frequency, gain_db)
    idx = int(np.argmin(np.abs(frequency - f_c_sim)))
    print('\n截止频率（由仿真曲线插值）: {:.4f} Hz'.format(f_c_sim))
    print('截止频率（理论公式）       : {:.4f} Hz'.format(f_c_theory))
    print('相对误差                   : {:.3e} %'.format(
        abs(f_c_sim - f_c_theory) / f_c_theory * 100.0))
    print('fc 附近频点处的仿真值      : {:.4f} dB, {:.4f}°'.format(
        gain_db[idx], phase_deg[idx]))

    # ---- 仿真值 vs 理论值 对比表 ----
    print('\n---- 仿真值 vs 理论值 ----')
    print('{:>12} {:>13} {:>13} {:>14} {:>14}'.format(
        '频率 (Hz)', '仿真|H|(dB)', '理论|H|(dB)', '仿真相位(°)', '理论相位(°)'))
    for f_test in [F_START, 100.0, f_c_theory / 10, f_c_theory, f_c_theory * 10, F_STOP]:
        db_sim = float(np.interp(f_test, frequency, gain_db))    # 仿真值（插值到该频点）
        ph_sim = float(np.interp(f_test, frequency, phase_deg))
        db_theory, ph_theory = theoretical_gain_phase(f_test)    # 理论值
        print('{:12.1f} {:13.3f} {:13.3f} {:14.3f} {:14.3f}'.format(
            f_test, db_sim, float(db_theory), ph_sim, float(ph_theory)))

    # ---- 绘图（含截止频率标注）----
    plot_bode(frequency, gain_db, phase_deg, f_c_theory, f_c_sim)

    plt.show()      # 弹出窗口显示（在无界面后端下会被忽略）

if __name__ == '__main__':
    main()