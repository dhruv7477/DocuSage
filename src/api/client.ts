import axios, { AxiosError, AxiosInstance, InternalAxiosRequestConfig } from 'axios'

const BASE = '/api/v1'

/**
 * Pull the most specific human-readable message out of an error returned by
 * the API. Backend errors land as `{ error_code, message, detail }`; network
 * failures only have `error.message`. Falls back to a generic string so the
 * caller never has to deal with `undefined`.
 */
export function getApiErrorMessage(err: unknown, fallback = 'Request failed'): string {
  const ax = err as AxiosError<{ message?: string; detail?: string; error_code?: string }>
  const data = ax?.response?.data
  if (data) {
    // Return user-facing message only. `detail` carries technical context
    // (stack traces, SQL, remediation hints) that belongs in server logs,
    // not in the UI.
    if (data.message) return data.message
  }
  if (ax?.code === 'ERR_NETWORK') return 'Cannot reach the backend (is the API server running?)'
  if (ax?.message) return ax.message
  return fallback
}

export const api: AxiosInstance = axios.create({
  baseURL: BASE,
  headers: { 'Content-Type': 'application/json' },
})

// ── Request: attach JWT + log ─────────────────────────────────────
api.interceptors.request.use((config: InternalAxiosRequestConfig) => {
  const token = localStorage.getItem('access_token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  console.log(`[DocuSage API] → ${config.method?.toUpperCase()} ${config.baseURL}${config.url}`, {
    params: config.params,
    hasBody: !!config.data,
  })
  return config
})

// ── Response: log ─────────────────────────────────────────────────
api.interceptors.response.use(
  (res) => {
    console.log(`[DocuSage API] ← ${res.status} ${res.config.url}`, res.data)
    return res
  },
  (err) => {
    console.error(`[DocuSage API] ✗ ${err?.response?.status ?? 'NETWORK'} ${err?.config?.url}`, {
      status: err?.response?.status,
      data: err?.response?.data,
      message: err?.message,
    })
    return Promise.reject(err)
  }
)

// ── Response: auto-refresh on 401 ────────────────────────────────
let _refreshing = false
let _queue: Array<(token: string) => void> = []

api.interceptors.response.use(
  (res) => res,
  async (err) => {
    const original = err.config
    if (err.response?.status !== 401 || original._retry) {
      return Promise.reject(err)
    }
    original._retry = true

    if (_refreshing) {
      return new Promise((resolve) => {
        _queue.push((token) => {
          original.headers.Authorization = `Bearer ${token}`
          resolve(api(original))
        })
      })
    }

    _refreshing = true
    try {
      const refresh = localStorage.getItem('refresh_token')
      if (!refresh) throw new Error('No refresh token')
      const { data } = await axios.post(`${BASE}/auth/refresh`, { refresh_token: refresh })
      localStorage.setItem('access_token', data.access_token)
      localStorage.setItem('refresh_token', data.refresh_token)
      _queue.forEach((cb) => cb(data.access_token))
      _queue = []
      original.headers.Authorization = `Bearer ${data.access_token}`
      return api(original)
    } catch {
      localStorage.clear()
      window.location.href = '/login'
      return Promise.reject(err)
    } finally {
      _refreshing = false
    }
  }
)
