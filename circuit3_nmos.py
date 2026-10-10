#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
NMOS 共源级放大电路仿真（PySpice + ngspice）
================================================================

电路结构::

                 VDD = 5 V
                    │
        ┌───────────┼───────────────┐
        │           │               │
      [Rg1]       [Rd] 2 kΩ         │
       60 kΩ        │               │
        │           ├──── vout ─────┘   (漏极输出)
        ├───────────┤
        │         ┌─┴─┐
      [Rg2]       │ M1│  NMOS
       40 kΩ      │   │  (源极接地)
        │         └─┬─┘
       gnd          │
                   gnd

    vin ──┤├── vg         （Cb1 = 10 µF 输入耦合电容，隔直流、通交流）
          Cb1

偏置与模型:
    Vg  = VDD · Rg2/(Rg1+Rg2) = 5 × 40/100 = 2.0 V      （栅极无电流，静态 Vgs = Vg）
    NMOS（SPICE level-1 / Shichman-Hodges 模型）:
        饱和区  Id = (KP/2)(W/L)(Vgs-Vth)²(1+λ·Vds)
        本脚本取 W/L = 1，KP = 0.8 mA/V², Vth = 1 V, λ = 0.02 /V
        即 Id = 0.4 mA/V² · (Vgs-1)²·(1+0.02·Vds)
    注意：如果按某些教材把 K 定义为 Id = K(Vgs-Vth)²，则把 KP 改成 1.6 mA/V²
          （或把 W/L 改成 2）即可，其余代码不用动。

仿真内容:
    ① 直流工作点分析（.op）：打印 Vgs、Id、Vds，并判断是否工作在饱和区
    ② 小信号参数估算：gm = 2Id/(Vgs-Vth)、ro ≈ 1/(λId)、Av = -gm(Rd∥ro)
    ③ 瞬态分析（.tran）：输入 10 mV / 1 kHz 正弦，观察输出波形
    ④ 测量增益 Av = Vout/Vin（用稳态段的峰峰值之比和最小二乘斜率两种方法）
    ⑤ matplotlib 画输入 / 输出波形对比图

运行环境:
    pip install PySpice matplotlib
    另外需要系统安装 ngspice 动态库（PySpice 只是封装，求解由 ngspice 完成）：
      * Windows ：把含 ngspice.dll 的目录加入 PATH，或设置
                  NgSpiceShared.LIBRARY_PATH = r"C:\path\to\ngspice.dll"
      * Linux   ：sudo apt install ngspice
      * macOS   ：brew install ngspice

输出:
    * 控制台：直流工作点、小信号参数与增益测量结果
    * 图片  ：nmos_cs_amplifier.png
"""

import numpy as np
import matplotlib.pyplot as plt

from PySpice.Spice.Netlist import Circuit                      # 电路 / 网表描述
from PySpice.Unit import u_kOhm, u_kHz, u_mV, u_ms, u_uF, u_um, u_us, u_V
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
VDD_VALUE = 5 @ u_V            # 电源电压 5 V
RG1_VALUE = 60 @ u_kOhm        # 上偏置电阻 Rg1 = 60 kΩ
RG2_VALUE = 40 @ u_kOhm        # 下偏置电阻 Rg2 = 40 kΩ
RD_VALUE = 2 @ u_kOhm          # 漏极负载电阻 Rd = 2 kΩ
CB1_VALUE = 10 @ u_uF          # 输入耦合电容 Cb1 = 10 µF

VI_AMPLITUDE = 10 @ u_mV       # 输入正弦幅值 10 mV
VI_FREQUENCY = 1 @ u_kHz       # 输入正弦频率 1 kHz

# ---- NMOS 模型参数（SPICE level-1）----
MODEL_NAME = 'NMOS_CS'         # 模型名，网表中写 .model NMOS_CS NMOS(...)
VTH = 1.0                      # 开启电压 Vth = 1 V   → SPICE 参数 VTO
KP = 0.8e-3                    # 跨导参数 K = 0.8 mA/V² → SPICE 参数 KP [A/V²]
LAMBDA = 0.02                  # 沟道长度调制系数 λ = 0.02 /V → SPICE 参数 LAMBDA
W_UM = 100.0                   # 沟道宽度 100 µm
L_UM = 100.0                   # 沟道长度 100 µm（W/L = 1，与给定 K 对应）

# ---- 瞬态仿真设置 ----
T_STEP = 1 @ u_us              # 打印步长 1 µs（1 kHz 一个周期 1 ms，即每周期 1000 点）
T_END = 5 @ u_ms               # 仿真总时长 5 ms（5 个周期）
T_SETTLE = 2 @ u_ms            # 前 2 ms 不作为测量窗口（避开起始暂态，取稳态段）

OUTPUT_PNG = 'nmos_cs_amplifier.png'

# ============================================================================
# 2. 构建电路
# ============================================================================
def build_amplifier():
    """搭建 NMOS 共源放大电路并返回 Circuit 对象。

    生成的网表（关键部分）::

        VDD   vdd 0   5V
        Vin   vin 0   DC 0V AC 1V SIN(0V 10mV 1kHz 0s 0Hz)
        Cb1   vin vg  10uF                 ← 输入耦合电容
        Rg1   vdd vg  60kOhm               ← 分压偏置上臂
        Rg2   vg  0   40kOhm               ← 分压偏置下臂
        Rd    vdd vout 2kOhm               ← 漏极负载
        M1    vout vg 0 0 NMOS_CS l=100um w=100um   ← NMOS，衬底接源极(地)
        .model NMOS_CS NMOS(LEVEL=1 VTO=1 KP=0.0008 LAMBDA=0.02)
    """
    circuit = Circuit('NMOS 共源级放大电路')

    # ---- NMOS 模型：SPICE level-1（Shichman-Hodges）----
    circuit.model(MODEL_NAME, 'NMOS',
                  LEVEL=1,          # 模型级别 1，即经典的平方律模型
                  VTO=VTH,          # 开启电压 Vth [V]
                  KP=KP,            # 跨导参数 K [A/V²]
                  LAMBDA=LAMBDA)    # 沟道长度调制系数 λ [1/V]

    # ---- 直流电源 ----
    circuit.V('DD', 'vdd', circuit.gnd, VDD_VALUE)

    # ---- 输入正弦信号源（DC 分量为 0，交流经 Cb1 耦合到栅极）----
    circuit.SinusoidalVoltageSource('in', 'vin', circuit.gnd,
                                    amplitude=VI_AMPLITUDE,
                                    frequency=VI_FREQUENCY)

    # ---- 输入耦合电容：隔断直流，把交流信号送到栅极 ----
    circuit.C('b1', 'vin', 'vg', CB1_VALUE)

    # ---- 分压式偏置：Vg = VDD·Rg2/(Rg1+Rg2) = 2 V ----
    circuit.R('g1', 'vdd', 'vg', RG1_VALUE)
    circuit.R('g2', 'vg', circuit.gnd, RG2_VALUE)

    # ---- 漏极负载电阻：把漏极电流的变化转换成电压输出 ----
    circuit.R('d', 'vdd', 'vout', RD_VALUE)

    # ---- NMOS 管 ----
    # 四个端子：漏极 vout、栅极 vg、源极 0、衬底 0（衬底与源极短接 → 无体效应）
    circuit.MOSFET('1',
                   drain='vout', gate='vg',
                   source=circuit.gnd, bulk=circuit.gnd,
                   model=MODEL_NAME,
                   w=W_UM @ u_um, l=L_UM @ u_um)

    return circuit

# ============================================================================
# 3. 仿真与数据提取工具
# ============================================================================
def _to_float(waveform):
    """把工作点结果里的单个数据点取出并转成普通浮点数（自动处理 PySpice 的单位）"""
    return float(np.asarray(waveform).ravel()[0])

def run_operating_point(circuit):
    """① 直流工作点分析（.op），返回 {节点名: 电压[V]}"""
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.operating_point()
    return {name: _to_float(waveform) for name, waveform in analysis.nodes.items()}

def run_transient(circuit):
    """③ 瞬态分析（.tran），返回 (时间[秒], {节点名: 电压[V]})"""
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.transient(step_time=T_STEP, end_time=T_END)

    time = np.asarray(analysis.time, dtype=float)          # 时间轴 [s]
    waves = {name: np.asarray(analysis[name], dtype=float)  # 各节点电压 [V]
             for name in ('vin', 'vg', 'vout')}
    return time, waves

# ============================================================================
# 4. 小信号参数估算
# ============================================================================
def small_signal_analysis(vgs, id_dc, vds):
    """由工作点估算小信号参数与理论电压增益。

        gm = 2·Id/(Vgs-Vth)          （对平方律器件是精确式）
        ro ≈ 1/(λ·Id)                （常用的近似式；精确值为 1/(λ·KP/2·(W/L)·Vov²)）
        Av = -gm·(Rd ∥ ro)           （共源级的电压增益，负号表示反相）
    """
    vov = vgs - VTH                       # 过驱动电压 Vgs - Vth
    gm = 2.0 * id_dc / vov                # 跨导 [A/V]
    ro_approx = 1.0 / (LAMBDA * id_dc)    # 输出电阻近似值 [Ω]
    # 精确的输出电阻：gds = ∂Id/∂Vds = λ·(KP/2)(W/L)Vov²
    gds = LAMBDA * (KP / 2.0) * (W_UM / L_UM) * vov ** 2
    ro_exact = 1.0 / gds
    rd = float(RD_VALUE)
    av_approx = -gm * (rd * ro_approx) / (rd + ro_approx)
    av_exact = -gm * (rd * ro_exact) / (rd + ro_exact)
    return dict(vov=vov, gm=gm, ro_approx=ro_approx, ro_exact=ro_exact,
                av_approx=av_approx, av_exact=av_exact)

def model_drain_current(vgs, vds):
    """按 SPICE level-1 模型公式计算漏极电流（用于和仿真结果互相校验）。

        截止区 vgs < Vth            : Id = 0
        饱和区 vds > vgs-Vth        : Id = (KP/2)(W/L)(vgs-Vth)²(1+λ·vds)
        可变电阻区 vds ≤ vgs-Vth    : Id = (KP/2)(W/L)[2(vgs-Vth)vds - vds²](1+λ·vds)
    """
    beta = KP * (W_UM / L_UM)          # β = KP·(W/L)
    vov = vgs - VTH
    if vov <= 0:
        return 0.0
    if vds > vov:                      # 饱和区
        return (beta / 2.0) * vov ** 2 * (1.0 + LAMBDA * vds)
    return (beta / 2.0) * (2.0 * vov * vds - vds ** 2) * (1.0 + LAMBDA * vds)

# ============================================================================
# 5. 增益测量
# ============================================================================
def measure_gain(time, vin, vout, t_settle=float(T_SETTLE)):
    """在稳态段内测量电压增益 Av = Vout/Vin（交流量，剔除直流分量）。

    返回字典：
        av_amp  : 由峰峰值之比得到（已知反相，前面补负号）
        av_fit  : 把 Vout 对 Vin 做一次最小二乘直线拟合，斜率即小信号增益（自带符号）
        vin_amp / vout_amp : 输入、输出的单边幅值
        t_window : 测量所用时间窗
    """
    steady = time >= t_settle                          # 只取进入稳态后的数据
    t_win, vin_win, vout_win = time[steady], vin[steady], vout[steady]

    vin_ac = vin_win - vin_win.mean()                  # 去掉直流，只留交流分量
    vout_ac = vout_win - vout_win.mean()

    vin_amp = (vin_ac.max() - vin_ac.min()) / 2.0      # 单边幅值 = 峰峰值/2
    vout_amp = (vout_ac.max() - vout_ac.min()) / 2.0
    av_amp = -vout_amp / vin_amp                       # 共源级反相，故增益为负

    av_fit = float(np.polyfit(vin_ac, vout_ac, 1)[0])  # 斜率 = ΔVout/ΔVin

    return dict(av_amp=av_amp, av_fit=av_fit,
                vin_amp=vin_amp, vout_amp=vout_amp,
                vin_dc=float(vin_win.mean()), vout_dc=float(vout_win.mean()),
                t_window=(float(t_win[0]), float(t_win[-1])))

# ============================================================================
# 6. 绘图：输入 / 输出波形对比
# ============================================================================
def plot_waveforms(time, vin, vout, gain, op, save_path=OUTPUT_PNG):
    """画输入输出波形对比图（两条曲线都扣掉直流，单位换算成 mV，便于同轴对比）"""
    t_ms = time * 1e3                                  # 时间换算成 ms
    vin_mv = (vin - vin.mean()) * 1e3                  # 扣直流 → mV
    vout_mv = (vout - vout.mean()) * 1e3

    fig, ax = plt.subplots(figsize=(10, 5.6))

    # 注意：LaTeX 的 {in}/{out} 必须写成 {{in}}/{{out}}，否则会被 str.format 当占位符
    ax.plot(t_ms, vin_mv, color='#1f77b4', linewidth=1.8,
            label='输入 $v_{{in}}$ = {:.0f} mV / 1 kHz'.format(gain['vin_amp'] * 1e3))
    ax.plot(t_ms, vout_mv, color='#d62728', linewidth=1.8,
            label='输出 $v_{{out}}$（已扣除直流 {:.3f} V）'.format(gain['vout_dc']))

    ax.axhline(0, color='gray', linewidth=0.8)
    # 用浅色标出用于测量增益的稳态时间窗
    ax.axvspan(float(T_SETTLE) * 1e3, t_ms[-1], color='orange', alpha=0.08)
    ax.text(0.60, 0.955, '增益测量窗口（稳态段）', transform=ax.transAxes,
            ha='center', fontsize=9, color='darkorange')

    # ---- 左上角信息框：工作点 + 增益 ----
    info = ('直流工作点：$V_{{GS}}$ = {vgs:.3f} V,  $I_D$ = {idc:.4f} mA,  '
            '$V_{{DS}}$ = {vds:.3f} V\n'
            '小信号参数：$g_m$ = {gm:.3f} mS,  $r_o$ = {ro:.1f} kΩ\n'
            '$A_v$ = $v_{{out}}/v_{{in}}$ = {av:.3f} （精确理论值 {avt:.3f}）').format(
        vgs=op['vgs'], idc=op['id'] * 1e3, vds=op['vds'],
        gm=op['gm'] * 1e3, ro=op['ro_approx'] / 1e3,
        av=gain['av_fit'], avt=op['av_exact'])
    ax.text(0.015, 0.02, info, transform=ax.transAxes, fontsize=10,
            va='bottom', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    ax.set_xlabel('时间 (ms)', fontsize=12)
    ax.set_ylabel('电压（交流分量, mV）', fontsize=12)
    ax.set_title('NMOS 共源级放大电路：输入 / 输出波形对比\n'
                 '$V_{{DD}}$ = 5 V, $R_{{g1}}$ = 60 kΩ, $R_{{g2}}$ = 40 kΩ, '
                 '$R_d$ = 2 kΩ, $C_{{b1}}$ = 10 µF',
                 fontsize=12)
    ax.set_xlim(0, t_ms[-1])
    ax.set_ylim(-27, 23)          # 上下留白，避免信息框 / 图例压住波形
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print('\n[图片] 波形对比图已保存到: {}'.format(save_path))
    return fig

# ============================================================================
# 7. 主流程
# ============================================================================
def main():
    print('=' * 72)
    print('NMOS 共源级放大电路仿真（PySpice + ngspice）')
    print('=' * 72)

    circuit = build_amplifier()
    print('---- SPICE 网表 ----')
    print(circuit)

    # ======================================================================
    # ① 直流工作点分析
    # ======================================================================
    nodes = run_operating_point(circuit)

    vg = nodes['vg']                      # 栅极直流电压
    vs = 0.0                              # 源极接地
    vd = nodes['vout']                    # 漏极直流电压
    vgs = vg - vs                         # 栅源电压
    vds = vd - vs                         # 漏源电压
    rd = float(RD_VALUE)
    id_dc = (float(VDD_VALUE) - vd) / rd  # 流过 Rd 的电流就是 Id（栅极不取电流）

    print('---------- ① 直流工作点（.op）----------')
    print('Vgs = {:.6f} V      （分压偏置理论值 = {:.6f} V）'.format(
        vgs, float(VDD_VALUE) * float(RG2_VALUE) /
        (float(RG1_VALUE) + float(RG2_VALUE))))
    print('Id  = {:.6f} mA   （由 Rd 上压降算出：({:.1f}V - {:.4f}V)/{:.0f}Ω）'.format(
        id_dc * 1e3, float(VDD_VALUE), vd, rd))
    print('Vds = {:.6f} V'.format(vds))

    # ---- 用模型公式反算 Id，与仿真结果互相校验 ----
    id_model = model_drain_current(vgs, vds)
    print('校验：模型公式 Id = {:.6f} mA，与仿真值相对误差 {:.3e} %'.format(
        id_model * 1e3, abs(id_model - id_dc) / id_dc * 100))

    # ---- 工作区判断 ----
    vov = vgs - VTH
    if vov <= 0:
        region = '截止区（Vgs < Vth，管子未导通）'
    elif vds > vov:
        region = '饱和区（恒流区）—— 放大器的正常工作区 ✓'
    else:
        region = '可变电阻区（线性区，此时增益很小）✗'
    print('工作区判断：Vov = Vgs-Vth = {:.3f} V, Vds = {:.3f} V → {}'.format(vov, vds, region))

    # ======================================================================
    # ② 小信号参数与理论增益
    # ======================================================================
    op = small_signal_analysis(vgs, id_dc, vds)
    op.update(vgs=vgs, vds=vds, id=id_dc)

    print('\n---------- ② 小信号参数与理论增益 ----------')
    print('gm = 2·Id/(Vgs-Vth) = 2 × {:.4f} mA / {:.3f} V = {:.6f} mS'.format(
        id_dc * 1e3, vov, op['gm'] * 1e3))
    print('ro ≈ 1/(λ·Id) = {:.3f} kΩ   （精确值 1/gds = {:.3f} kΩ）'.format(
        op['ro_approx'] / 1e3, op['ro_exact'] / 1e3))
    print('Av = -gm·(Rd∥ro) = {:.4f}   （用精确 ro 计算：{:.4f}）'.format(
        op['av_approx'], op['av_exact']))

    # ======================================================================
    # ③ 瞬态分析
    # ======================================================================
    time, waves = run_transient(circuit)
    vin, vout = waves['vin'], waves['vout']
    print('\n---------- ③ 瞬态分析（.tran {} µs ~ {} ms）----------'.format(
        int(T_STEP * 1e6), int(T_END * 1e3)))
    print('仿真点数：{}，时间范围 {:.3f} ms ~ {:.3f} ms'.format(
        time.size, time[0] * 1e3, time[-1] * 1e3))
    print('输入信号：幅值 {:.1f} mV，频率 {:.1f} kHz'.format(
        float(VI_AMPLITUDE) * 1e3, float(VI_FREQUENCY) / 1e3))

    # ======================================================================
    # ④ 增益测量
    # ======================================================================
    gain = measure_gain(time, vin, vout)

    print('\n---------- ④ 增益测量（稳态段 {:.0f}~{:.0f} ms）----------'.format(
        gain['t_window'][0] * 1e3, gain['t_window'][1] * 1e3))
    print('输入幅值 ：{:8.4f} mV  （直流分量 {:.6f} V）'.format(
        gain['vin_amp'] * 1e3, gain['vin_dc']))
    print('输出幅值 ：{:8.4f} mV  （直流分量 {:.6f} V）'.format(
        gain['vout_amp'] * 1e3, gain['vout_dc']))
    print('Av（峰峰值之比）  = {:.4f}'.format(gain['av_amp']))
    print('Av（最小二乘斜率）= {:.4f}  ← 推荐，符号自动包含反相信息'.format(gain['av_fit']))
    print('理论值 -gm·(Rd∥ro) = {:.4f}（用近似 ro）/ {:.4f}（用精确 ro）'.format(
        op['av_approx'], op['av_exact']))
    print('与精确理论值的相对误差：{:.3f} %'.format(
        abs(gain['av_fit'] - op['av_exact']) / abs(op['av_exact']) * 100))
    print('输出与输入相位相反（共源级的倒相作用），即 Av < 0')

    # ======================================================================
    # ⑤ 绘图
    # ======================================================================
    plot_waveforms(time, vin, vout, gain, op)

    plt.show()

if __name__ == '__main__':
    main()