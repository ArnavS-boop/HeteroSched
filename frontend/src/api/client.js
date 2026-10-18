/**
 * API client for interacting with the HeteroSched local Flask backend.
 * Provides sensible timeouts, in-flight request deduplication, and AbortSignal support.
 */

const BASE_URL = '/api';
const DEFAULT_TIMEOUT_MS = 8000;
const BENCHMARK_TIMEOUT_MS = 90000;

/**
 * Fetch wrapper with timeout support.
 */
async function fetchWithTimeout(url, options = {}, timeoutMs = DEFAULT_TIMEOUT_MS) {
  const controller = new AbortController();
  const { signal: externalSignal, ...fetchOptions } = options;

  let timeoutId = null;
  if (timeoutMs > 0) {
    timeoutId = setTimeout(() => {
      controller.abort(new Error(`Request to ${url} timed out after ${timeoutMs}ms`));
    }, timeoutMs);
  }

  try {
    const res = await fetch(url, { ...fetchOptions, signal: controller.signal });
    return res;
  } finally {
    if (timeoutId) clearTimeout(timeoutId);
  }
}

// In-flight promise caches to avoid duplicate simultaneous requests (e.g., React StrictMode double mounts)
let inFlightSystemInfo = null;
let inFlightTraces = null;
let inFlightLatestBenchmark = null;

export async function fetchSystemInfo(options = {}) {
  const { signal } = options;
  if (signal?.aborted) {
    throw new DOMException('The user aborted a request.', 'AbortError');
  }

  if (!inFlightSystemInfo) {
    inFlightSystemInfo = fetchWithTimeout(`${BASE_URL}/system/info`, {}, DEFAULT_TIMEOUT_MS)
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(`Failed to fetch system info: ${res.statusText}`);
        }
        return await res.json();
      })
      .finally(() => {
        inFlightSystemInfo = null;
      });
  }

  if (!signal) {
    return inFlightSystemInfo;
  }

  return new Promise((resolve, reject) => {
    const onAbort = () => {
      signal.removeEventListener('abort', onAbort);
      reject(new DOMException('The user aborted a request.', 'AbortError'));
    };
    signal.addEventListener('abort', onAbort, { once: true });

    inFlightSystemInfo
      .then((data) => {
        signal.removeEventListener('abort', onAbort);
        resolve(data);
      })
      .catch((err) => {
        signal.removeEventListener('abort', onAbort);
        reject(err);
      });
  });
}

export async function fetchTraces(options = {}) {
  const { signal } = options;
  if (signal?.aborted) {
    throw new DOMException('The user aborted a request.', 'AbortError');
  }

  if (!inFlightTraces) {
    inFlightTraces = fetchWithTimeout(`${BASE_URL}/traces`, {}, DEFAULT_TIMEOUT_MS)
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(`Failed to fetch approved traces: ${res.statusText}`);
        }
        return await res.json();
      })
      .finally(() => {
        inFlightTraces = null;
      });
  }

  if (!signal) {
    return inFlightTraces;
  }

  return new Promise((resolve, reject) => {
    const onAbort = () => {
      signal.removeEventListener('abort', onAbort);
      reject(new DOMException('The user aborted a request.', 'AbortError'));
    };
    signal.addEventListener('abort', onAbort, { once: true });

    inFlightTraces
      .then((data) => {
        signal.removeEventListener('abort', onAbort);
        resolve(data);
      })
      .catch((err) => {
        signal.removeEventListener('abort', onAbort);
        reject(err);
      });
  });
}

export async function fetchLatestBenchmark(options = {}) {
  const { signal } = options;
  if (signal?.aborted) {
    throw new DOMException('The user aborted a request.', 'AbortError');
  }

  if (!inFlightLatestBenchmark) {
    inFlightLatestBenchmark = fetchWithTimeout(`${BASE_URL}/benchmark/latest`, {}, 12000)
      .then(async (res) => {
        if (!res.ok) {
          if (res.status === 404) return null;
          throw new Error(`Failed to fetch latest benchmark: ${res.statusText}`);
        }
        return await res.json();
      })
      .finally(() => {
        inFlightLatestBenchmark = null;
      });
  }

  if (!signal) {
    return inFlightLatestBenchmark;
  }

  return new Promise((resolve, reject) => {
    const onAbort = () => {
      signal.removeEventListener('abort', onAbort);
      reject(new DOMException('The user aborted a request.', 'AbortError'));
    };
    signal.addEventListener('abort', onAbort, { once: true });

    inFlightLatestBenchmark
      .then((data) => {
        signal.removeEventListener('abort', onAbort);
        resolve(data);
      })
      .catch((err) => {
        signal.removeEventListener('abort', onAbort);
        reject(err);
      });
  });
}

export async function runBenchmark(traceKey, options = {}) {
  const res = await fetchWithTimeout(
    `${BASE_URL}/benchmark/run`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ trace: traceKey }),
      ...options,
    },
    BENCHMARK_TIMEOUT_MS
  );

  const data = await res.json();
  if (!res.ok) {
    const errorMsg = data.error || `Benchmark run failed with status ${res.status}`;
    throw new Error(errorMsg);
  }
  return data.run;
}

export function getDownloadUrl(mode = 'clean') {
  return `${BASE_URL}/benchmark/download?mode=${mode}`;
}
