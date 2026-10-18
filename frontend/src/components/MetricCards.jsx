import React from 'react';
import { Gauge, Clock, Cpu, HardDrive, ArrowRightLeft, Timer, Layers, Info } from 'lucide-react';

export default function MetricCards({ summary }) {
  if (!summary) return null;

  const {
    total_tasks = 0,
    cpu_tasks = 0,
    gpu_tasks = 0,
    cpu_ratio = 0,
    gpu_ratio = 0,
    total_wall_time_ms = 0,
    throughput_tasks_per_sec = 0,
    avg_decision_us = 0,
    avg_exec_us = 0,
    avg_transfer_us_gpu = 0,
    avg_submission_latency_us = 0,
    p95_exec_us = 0,
  } = summary;

  const cards = [
    {
      title: 'Throughput',
      value: `${throughput_tasks_per_sec.toLocaleString()}`,
      unit: 'tasks/sec',
      subtitle: `${total_tasks} tasks scheduled`,
      icon: Gauge,
      color: '#38bdf8',
      bgGlow: 'rgba(56, 189, 248, 0.1)',
    },
    {
      title: 'Scheduler Wall Time',
      value: `${total_wall_time_ms.toFixed(2)}`,
      unit: 'ms',
      subtitle: 'Pure C++ runtime scheduling & execution',
      icon: Clock,
      color: '#10b981',
      bgGlow: 'rgba(16, 185, 129, 0.1)',
    },
    {
      title: 'Avg Decision Overhead',
      value: `${avg_decision_us.toFixed(3)}`,
      unit: 'μs',
      subtitle: 'Two-tier controller decision time',
      icon: Timer,
      color: '#f59e0b',
      bgGlow: 'rgba(245, 158, 11, 0.1)',
    },
    {
      title: 'Device Split',
      value: `${gpu_tasks} GPU / ${cpu_tasks} CPU`,
      unit: `${gpu_ratio}% GPU`,
      subtitle: `${cpu_ratio}% allocated to host cores`,
      icon: Layers,
      color: '#a855f7',
      bgGlow: 'rgba(168, 85, 247, 0.1)',
    },
    {
      title: 'Avg Kernel Execution',
      value: `${avg_exec_us.toFixed(1)}`,
      unit: 'μs',
      subtitle: `P95: ${p95_exec_us.toFixed(1)} μs`,
      icon: Cpu,
      color: '#3b82f6',
      bgGlow: 'rgba(59, 130, 246, 0.1)',
    },
    {
      title: 'Avg GPU PCIe Transfer',
      value: `${avg_transfer_us_gpu.toFixed(1)}`,
      unit: 'μs',
      subtitle: 'Offload memory copy duration',
      icon: ArrowRightLeft,
      color: '#ec4899',
      bgGlow: 'rgba(236, 72, 153, 0.1)',
    },
  ];

  return (
    <div style={{ marginBottom: '1.5rem' }}>
      {/* 6 Core Metric Cards */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))',
        gap: '1rem',
        marginBottom: '1rem',
      }}>
        {cards.map((c, idx) => {
          const Icon = c.icon;
          return (
            <div
              key={idx}
              className="glass-panel"
              style={{
                padding: '1.25rem',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                position: 'relative',
                overflow: 'hidden',
              }}
            >
              <div style={{
                position: 'absolute',
                top: 0,
                right: 0,
                width: '70px',
                height: '70px',
                background: c.bgGlow,
                borderRadius: '50%',
                filter: 'blur(20px)',
                pointerEvents: 'none',
              }} />

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.65rem' }}>
                <span style={{ fontSize: '0.8rem', fontWeight: '600', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                  {c.title}
                </span>
                <div style={{
                  padding: '0.4rem',
                  borderRadius: '8px',
                  background: 'rgba(255, 255, 255, 0.05)',
                  display: 'flex',
                  alignItems: 'center',
                }}>
                  <Icon size={16} color={c.color} />
                </div>
              </div>

              <div>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.4rem' }}>
                  <span className="font-mono" style={{ fontSize: '1.65rem', fontWeight: '700', color: '#f8fafc' }}>
                    {c.value}
                  </span>
                  <span style={{ fontSize: '0.825rem', color: c.color, fontWeight: '600' }}>
                    {c.unit}
                  </span>
                </div>
                <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '0.35rem' }}>
                  {c.subtitle}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Latency Semantics Notice */}
      <div style={{
        background: 'rgba(30, 41, 59, 0.4)',
        border: '1px solid rgba(148, 163, 184, 0.12)',
        borderRadius: '8px',
        padding: '0.65rem 1rem',
        fontSize: '0.775rem',
        color: '#94a3b8',
        display: 'flex',
        alignItems: 'center',
        gap: '0.65rem',
      }}>
        <Info size={16} color="#38bdf8" style={{ flexShrink: 0 }} />
        <div>
          <strong style={{ color: '#e2e8f0' }}>Timing Metric Note: </strong>
          The benchmark column <code className="font-mono" style={{ color: '#38bdf8' }}>t_total_latency_us</code> (avg: {avg_submission_latency_us.toFixed(2)} μs)
          records the time elapsed during <code className="font-mono" style={{ color: '#cbd5e1' }}>runtime.SubmitTask()</code> (dispatch/enqueue latency).
          Because GPU streams and CPU workers execute asynchronously, kernel compute occurs concurrently with execution timing measured separately in <code className="font-mono" style={{ color: '#38bdf8' }}>t_exec_us</code>.
        </div>
      </div>
    </div>
  );
}
