import React from 'react';
import { BarChart3, PieChart, Layers, HardDrive, Cpu, Activity } from 'lucide-react';

export default function Visualizations({ summary }) {
  if (!summary) return null;

  const {
    total_tasks = 0,
    cpu_tasks = 0,
    gpu_tasks = 0,
    cpu_ratio = 0,
    gpu_ratio = 0,
    task_type_stats = {},
    stream_distribution = {},
    cpu_core_distribution = {},
    avg_decision_us = 0,
    avg_transfer_us_gpu = 0,
    avg_exec_us = 0,
  } = summary;

  // Normalize stream distribution (0-3)
  const streams = ['0', '1', '2', '3'];
  const cores = ['0', '1', '2', '3'];

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))',
      gap: '1.25rem',
      marginBottom: '1.5rem',
    }}>
      {/* Panel 1: Device Allocation & Kernel Placement */}
      <div className="glass-panel" style={{ padding: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.25rem' }}>
          <Layers size={18} color="#38bdf8" />
          <h2 style={{ fontSize: '0.95rem', fontWeight: '600', color: '#f8fafc', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Workload Distribution by Kernel
          </h2>
        </div>

        {/* Global Split Bar */}
        <div style={{ marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '0.45rem' }}>
            <span style={{ color: '#10b981', fontWeight: '600' }}>GPU: {gpu_tasks} ({gpu_ratio}%)</span>
            <span style={{ color: '#60a5fa', fontWeight: '600' }}>CPU: {cpu_tasks} ({cpu_ratio}%)</span>
          </div>
          <div style={{ height: '10px', background: 'rgba(255, 255, 255, 0.08)', borderRadius: '9999px', overflow: 'hidden', display: 'flex' }}>
            <div style={{ width: `${gpu_ratio}%`, background: 'linear-gradient(90deg, #10b981, #06b6d4)', transition: 'width 0.5s ease' }} />
            <div style={{ width: `${cpu_ratio}%`, background: 'linear-gradient(90deg, #3b82f6, #8b5cf6)', transition: 'width 0.5s ease' }} />
          </div>
        </div>

        {/* Kernel Rows */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
          {Object.entries(task_type_stats).map(([ktype, stats]) => {
            const kGpuPercent = stats.total > 0 ? Math.round((stats.gpu / stats.total) * 100) : 0;
            const kCpuPercent = 100 - kGpuPercent;
            return (
              <div
                key={ktype}
                style={{
                  background: 'rgba(15, 23, 42, 0.6)',
                  border: '1px solid rgba(148, 163, 184, 0.1)',
                  borderRadius: '8px',
                  padding: '0.75rem 1rem',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span className="font-mono" style={{ fontWeight: '600', color: '#f8fafc', fontSize: '0.85rem' }}>
                      {ktype}
                    </span>
                    <span style={{ fontSize: '0.75rem', color: '#64748b' }}>
                      ({stats.total} tasks)
                    </span>
                  </div>
                  <span className="font-mono" style={{ fontSize: '0.775rem', color: '#38bdf8' }}>
                    avg: {stats.avg_exec_us.toLocaleString()} μs
                  </span>
                </div>

                {/* Kernel Split Bar */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.75rem' }}>
                  <div style={{ flex: 1, height: '6px', background: 'rgba(255, 255, 255, 0.06)', borderRadius: '9999px', overflow: 'hidden', display: 'flex' }}>
                    <div style={{ width: `${kGpuPercent}%`, background: '#10b981' }} />
                    <div style={{ width: `${kCpuPercent}%`, background: '#3b82f6' }} />
                  </div>
                  <span style={{ color: '#10b981', minWidth: '45px' }}>{stats.gpu} GPU</span>
                  <span style={{ color: '#60a5fa', minWidth: '45px' }}>{stats.cpu} CPU</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Panel 2: Execution Concurrency & Streams */}
      <div className="glass-panel" style={{ padding: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.25rem' }}>
          <Activity size={18} color="#10b981" />
          <h2 style={{ fontSize: '0.95rem', fontWeight: '600', color: '#f8fafc', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Hardware Concurrency Allocations
          </h2>
        </div>

        {/* CUDA Stream Pool */}
        <div style={{ marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.75rem' }}>
            <HardDrive size={15} color="#10b981" />
            <span style={{ fontSize: '0.825rem', fontWeight: '600', color: '#e2e8f0' }}>
              CUDA Multi-Stream Pool (4 Streams)
            </span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.65rem' }}>
            {streams.map((s) => {
              const count = stream_distribution[s] || 0;
              const pct = gpu_tasks > 0 ? Math.round((count / gpu_tasks) * 100) : 0;
              return (
                <div
                  key={s}
                  style={{
                    background: 'rgba(16, 185, 129, 0.08)',
                    border: '1px solid rgba(16, 185, 129, 0.2)',
                    borderRadius: '8px',
                    padding: '0.65rem',
                    textAlign: 'center',
                  }}
                >
                  <div style={{ fontSize: '0.7rem', color: '#64748b', textTransform: 'uppercase' }}>Stream {s}</div>
                  <div className="font-mono" style={{ fontSize: '1.15rem', fontWeight: '700', color: '#34d399', margin: '0.2rem 0' }}>
                    {count}
                  </div>
                  <div style={{ fontSize: '0.7rem', color: '#94a3b8' }}>{pct}%</div>
                </div>
              );
            })}
          </div>
        </div>

        {/* CPU Worker Pool */}
        <div style={{ marginBottom: '1.25rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.75rem' }}>
            <Cpu size={15} color="#60a5fa" />
            <span style={{ fontSize: '0.825rem', fontWeight: '600', color: '#e2e8f0' }}>
              CPU Worker Runqueues (4 Threads)
            </span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.65rem' }}>
            {cores.map((c) => {
              const count = cpu_core_distribution[c] || 0;
              const pct = cpu_tasks > 0 ? Math.round((count / cpu_tasks) * 100) : 0;
              return (
                <div
                  key={c}
                  style={{
                    background: 'rgba(59, 130, 246, 0.08)',
                    border: '1px solid rgba(59, 130, 246, 0.2)',
                    borderRadius: '8px',
                    padding: '0.65rem',
                    textAlign: 'center',
                  }}
                >
                  <div style={{ fontSize: '0.7rem', color: '#64748b', textTransform: 'uppercase' }}>Core {c}</div>
                  <div className="font-mono" style={{ fontSize: '1.15rem', fontWeight: '700', color: '#60a5fa', margin: '0.2rem 0' }}>
                    {count}
                  </div>
                  <div style={{ fontSize: '0.7rem', color: '#94a3b8' }}>{pct}%</div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Controller Overhead Profile */}
        <div style={{
          background: 'rgba(15, 23, 42, 0.7)',
          border: '1px solid rgba(245, 158, 11, 0.25)',
          borderRadius: '8px',
          padding: '0.75rem 1rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}>
          <div>
            <div style={{ fontSize: '0.75rem', color: '#f59e0b', fontWeight: '600', textTransform: 'uppercase' }}>
              Decision Engine Overhead Ratio
            </div>
            <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.2rem' }}>
              Average controller scheduling time vs compute duration
            </div>
          </div>
          <div className="font-mono" style={{ fontSize: '1.1rem', fontWeight: '700', color: '#fbbf24' }}>
            {avg_exec_us > 0 ? ((avg_decision_us / avg_exec_us) * 100).toFixed(4) : 0}%
          </div>
        </div>
      </div>
    </div>
  );
}
