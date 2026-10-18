import React, { useState, useMemo } from 'react';
import { Search, Download, ArrowUpDown, Filter, ChevronLeft, ChevronRight, FileText, Info } from 'lucide-react';
import { getDownloadUrl } from '../api/client';

export default function TaskTable({ tasks = [] }) {
  const [searchTerm, setSearchTerm] = useState('');
  const [deviceFilter, setDeviceFilter] = useState('ALL');
  const [typeFilter, setTypeFilter] = useState('ALL');
  const [sortField, setSortField] = useState('task_id');
  const [sortAsc, setSortAsc] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);

  // Extract unique kernel types for dropdown
  const kernelTypes = useMemo(() => {
    return Array.from(new Set(tasks.map((t) => t.type))).sort();
  }, [tasks]);

  // Filtering and sorting
  const filteredTasks = useMemo(() => {
    return tasks.filter((t) => {
      // Device filter
      if (deviceFilter !== 'ALL' && t.device !== deviceFilter) return false;
      // Type filter
      if (typeFilter !== 'ALL' && t.type !== typeFilter) return false;
      // Search term
      if (searchTerm) {
        const query = searchTerm.toLowerCase();
        const matchesId = String(t.task_id).includes(query);
        const matchesType = t.type.toLowerCase().includes(query);
        const matchesDevice = t.device.toLowerCase().includes(query);
        if (!matchesId && !matchesType && !matchesDevice) return false;
      }
      return true;
    });
  }, [tasks, deviceFilter, typeFilter, searchTerm]);

  const sortedTasks = useMemo(() => {
    return [...filteredTasks].sort((a, b) => {
      let valA = a[sortField];
      let valB = b[sortField];
      if (typeof valA === 'string') {
        return sortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
      }
      return sortAsc ? valA - valB : valB - valA;
    });
  }, [filteredTasks, sortField, sortAsc]);

  // Pagination
  const totalPages = Math.ceil(sortedTasks.length / pageSize) || 1;
  const paginatedTasks = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return sortedTasks.slice(start, start + pageSize);
  }, [sortedTasks, currentPage, pageSize]);

  const handleSort = (field) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(true);
    }
  };

  return (
    <div className="glass-panel" style={{ padding: '1.5rem', marginBottom: '2rem' }}>
      {/* Header & CSV Download Controls */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '1rem',
        marginBottom: '1.25rem',
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <h2 style={{ fontSize: '1.1rem', fontWeight: '700', color: '#f8fafc' }}>
              Task-Level Telemetry Data
            </h2>
            <span className="badge" style={{ background: 'rgba(56, 189, 248, 0.12)', color: '#38bdf8' }}>
              {sortedTasks.length} {sortedTasks.length === 1 ? 'task' : 'tasks'}
            </span>
          </div>
          <p style={{ fontSize: '0.775rem', color: '#94a3b8', marginTop: '0.2rem' }}>
            Inspect per-task scheduling decisions, compute duration, transfer overheads, and hardware mapping
          </p>
        </div>

        {/* Action Buttons: Clean CSV vs Raw CSV */}
        <div style={{ display: 'flex', gap: '0.65rem' }}>
          <a
            href={getDownloadUrl('clean')}
            download
            className="btn-primary"
            style={{ textDecoration: 'none', padding: '0.55rem 1rem', fontSize: '0.8125rem' }}
          >
            <Download size={15} />
            <span>Export Clean CSV</span>
          </a>
          <a
            href={getDownloadUrl('raw')}
            download
            className="btn-secondary"
            title="Download original results.csv including trailing summary comment line"
            style={{ textDecoration: 'none' }}
          >
            <FileText size={15} color="#94a3b8" />
            <span>Raw CSV</span>
          </a>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div style={{
        display: 'flex',
        gap: '0.75rem',
        flexWrap: 'wrap',
        marginBottom: '1.25rem',
        alignItems: 'center',
      }}>
        {/* Search input */}
        <div style={{
          position: 'relative',
          flex: '1 1 240px',
        }}>
          <Search size={16} color="#64748b" style={{ position: 'absolute', left: '0.85rem', top: '50%', transform: 'translateY(-50%)' }} />
          <input
            type="text"
            placeholder="Search by Task ID, Kernel, or Device..."
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              setCurrentPage(1);
            }}
            style={{
              width: '100%',
              background: 'rgba(15, 23, 42, 0.8)',
              border: '1px solid rgba(148, 163, 184, 0.2)',
              borderRadius: '8px',
              padding: '0.55rem 0.85rem 0.55rem 2.4rem',
              color: '#f8fafc',
              fontSize: '0.85rem',
              outline: 'none',
            }}
          />
        </div>

        {/* Device Filter */}
        <select
          value={deviceFilter}
          onChange={(e) => {
            setDeviceFilter(e.target.value);
            setCurrentPage(1);
          }}
          style={{
            background: 'rgba(15, 23, 42, 0.8)',
            border: '1px solid rgba(148, 163, 184, 0.2)',
            borderRadius: '8px',
            padding: '0.55rem 0.85rem',
            color: '#f8fafc',
            fontSize: '0.825rem',
            outline: 'none',
            cursor: 'pointer',
          }}
        >
          <option value="ALL">All Devices</option>
          <option value="GPU">GPU Only</option>
          <option value="CPU">CPU Only</option>
        </select>

        {/* Kernel Type Filter */}
        <select
          value={typeFilter}
          onChange={(e) => {
            setTypeFilter(e.target.value);
            setCurrentPage(1);
          }}
          style={{
            background: 'rgba(15, 23, 42, 0.8)',
            border: '1px solid rgba(148, 163, 184, 0.2)',
            borderRadius: '8px',
            padding: '0.55rem 0.85rem',
            color: '#f8fafc',
            fontSize: '0.825rem',
            outline: 'none',
            cursor: 'pointer',
          }}
        >
          <option value="ALL">All Kernels</option>
          {kernelTypes.map((kt) => (
            <option key={kt} value={kt}>{kt}</option>
          ))}
        </select>

        {/* Page size */}
        <select
          value={pageSize}
          onChange={(e) => {
            setPageSize(Number(e.target.value));
            setCurrentPage(1);
          }}
          style={{
            background: 'rgba(15, 23, 42, 0.8)',
            border: '1px solid rgba(148, 163, 184, 0.2)',
            borderRadius: '8px',
            padding: '0.55rem 0.85rem',
            color: '#f8fafc',
            fontSize: '0.825rem',
            outline: 'none',
            cursor: 'pointer',
          }}
        >
          <option value={15}>15 per page</option>
          <option value={25}>25 per page</option>
          <option value={50}>50 per page</option>
          <option value={100}>100 per page</option>
        </select>
      </div>

      {/* Table Container */}
      <div style={{
        overflowX: 'auto',
        borderRadius: '8px',
        border: '1px solid rgba(148, 163, 184, 0.12)',
        background: 'rgba(11, 17, 32, 0.5)',
      }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.8125rem' }}>
          <thead>
            <tr style={{ background: 'rgba(15, 23, 42, 0.95)', borderBottom: '1px solid rgba(148, 163, 184, 0.15)' }}>
              {[
                { field: 'task_id', label: 'ID' },
                { field: 'type', label: 'Kernel' },
                { field: 'priority', label: 'Priority' },
                { field: 'dim_m', label: 'Dimensions (M×N×K)' },
                { field: 'device', label: 'Device' },
                { field: 't_decision_us', label: 'Decision (μs)' },
                { field: 't_transfer_us', label: 'PCIe Transfer (μs)' },
                { field: 't_exec_us', label: 'Exec Time (μs)' },
                { field: 't_submission_latency_us', label: 'Submission Latency (μs)', tooltip: 'Measured during runtime.SubmitTask; asynchronous task dispatch enqueue time.' },
                { field: 'stream_id', label: 'Assigned Stream/Core' },
              ].map((col) => (
                <th
                  key={col.field}
                  onClick={() => handleSort(col.field)}
                  style={{
                    padding: '0.75rem 0.9rem',
                    color: sortField === col.field ? '#38bdf8' : '#94a3b8',
                    fontWeight: '600',
                    cursor: 'pointer',
                    userSelect: 'none',
                    whiteSpace: 'nowrap',
                  }}
                >
                  <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
                    <span>{col.label}</span>
                    <ArrowUpDown size={12} color={sortField === col.field ? '#38bdf8' : '#475569'} />
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {paginatedTasks.length === 0 ? (
              <tr>
                <td colSpan={10} style={{ padding: '2.5rem', textAlign: 'center', color: '#64748b' }}>
                  No tasks matched your current filter criteria.
                </td>
              </tr>
            ) : (
              paginatedTasks.map((t) => {
                const isGpu = t.device === 'GPU';
                return (
                  <tr
                    key={t.task_id}
                    style={{
                      borderBottom: '1px solid rgba(148, 163, 184, 0.07)',
                      transition: 'background 0.15s ease',
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.background = 'rgba(30, 41, 59, 0.4)')}
                    onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                  >
                    <td className="font-mono" style={{ padding: '0.65rem 0.9rem', color: '#94a3b8' }}>
                      #{t.task_id}
                    </td>
                    <td style={{ padding: '0.65rem 0.9rem', fontWeight: '600', color: '#f8fafc' }}>
                      {t.type}
                    </td>
                    <td style={{ padding: '0.65rem 0.9rem' }}>
                      <span style={{
                        padding: '0.15rem 0.45rem',
                        borderRadius: '4px',
                        fontSize: '0.7rem',
                        fontWeight: '600',
                        background: t.priority === 2 ? 'rgba(244, 63, 94, 0.15)' : t.priority === 1 ? 'rgba(245, 158, 11, 0.15)' : 'rgba(100, 116, 139, 0.15)',
                        color: t.priority === 2 ? '#f87171' : t.priority === 1 ? '#fbbf24' : '#94a3b8',
                      }}>
                        P{t.priority}
                      </span>
                    </td>
                    <td className="font-mono" style={{ padding: '0.65rem 0.9rem', color: '#cbd5e1' }}>
                      {t.dim_m} × {t.dim_n} × {t.dim_k}
                    </td>
                    <td style={{ padding: '0.65rem 0.9rem' }}>
                      <span className={`badge ${isGpu ? 'badge-gpu' : 'badge-cpu'}`}>
                        {t.device}
                      </span>
                    </td>
                    <td className="font-mono" style={{ padding: '0.65rem 0.9rem', color: '#f59e0b' }}>
                      {t.t_decision_us.toFixed(3)}
                    </td>
                    <td className="font-mono" style={{ padding: '0.65rem 0.9rem', color: isGpu ? '#ec4899' : '#64748b' }}>
                      {isGpu ? t.t_transfer_us.toFixed(1) : '—'}
                    </td>
                    <td className="font-mono" style={{ padding: '0.65rem 0.9rem', color: '#38bdf8', fontWeight: '500' }}>
                      {t.t_exec_us.toFixed(1)}
                    </td>
                    <td className="font-mono" style={{ padding: '0.65rem 0.9rem', color: '#cbd5e1' }}>
                      {t.t_submission_latency_us.toFixed(1)}
                    </td>
                    <td style={{ padding: '0.65rem 0.9rem' }}>
                      {isGpu ? (
                        <span style={{ color: '#10b981', fontSize: '0.75rem' }}>Stream {t.stream_id}</span>
                      ) : (
                        <span style={{ color: '#60a5fa', fontSize: '0.75rem' }}>Core {t.cpu_core}</span>
                      )}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination Footer */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginTop: '1rem',
        fontSize: '0.8rem',
        color: '#94a3b8',
        flexWrap: 'wrap',
        gap: '0.5rem',
      }}>
        <div>
          Showing page <span style={{ color: '#f8fafc', fontWeight: '600' }}>{currentPage}</span> of{' '}
          <span style={{ color: '#f8fafc', fontWeight: '600' }}>{totalPages}</span> ({filteredTasks.length} total tasks)
        </div>

        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            className="btn-secondary"
            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            disabled={currentPage === 1}
            style={{ opacity: currentPage === 1 ? 0.4 : 1, cursor: currentPage === 1 ? 'not-allowed' : 'pointer' }}
          >
            <ChevronLeft size={16} /> Previous
          </button>
          <button
            className="btn-secondary"
            onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
            disabled={currentPage === totalPages}
            style={{ opacity: currentPage === totalPages ? 0.4 : 1, cursor: currentPage === totalPages ? 'not-allowed' : 'pointer' }}
          >
            Next <ChevronRight size={16} />
          </button>
        </div>
      </div>
    </div>
  );
}
