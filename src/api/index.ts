import { api } from './client'

// ── Types ─────────────────────────────────────────────────────────

export interface User {
  id: string
  email: string
  full_name: string
  role: string
  is_active: boolean
  is_email_verified: boolean
}

export interface TokenResponse {
  access_token: string
  refresh_token: string
  token_type: string
}

export interface Workspace {
  id: string
  name: string
  description: string | null
  owner_id: string
  created_at: string
  updated_at: string
  member_count?: number
}

export interface WorkspaceMember {
  id: string
  user_id: string
  workspace_id: string
  role: string
  user_email: string
  user_full_name: string
  joined_at: string
}

export interface WorkspaceDetail extends Workspace {
  members: WorkspaceMember[]
}

export type DocumentStatus =
  | 'pending'
  | 'processing'
  | 'parsing'
  | 'chunking'
  | 'embedding'
  | 'indexing'
  | 'completed'
  | 'failed'

export const IN_FLIGHT_STATUSES: ReadonlySet<DocumentStatus> = new Set([
  'pending', 'processing', 'parsing', 'chunking', 'embedding', 'indexing',
])

export interface Document {
  id: string
  workspace_id: string
  user_id: string
  filename: string
  original_filename: string
  file_type: string
  file_size_bytes: number
  status: DocumentStatus
  /** 0–100 progress through the ingestion pipeline (0 if pending/failed). */
  progress: number
  chunk_count: number | null
  error_message: string | null
  created_at: string
  updated_at: string
}

export interface Chat {
  id: string
  workspace_id: string
  user_id: string
  title: string | null
  message_count: number
  created_at: string
  updated_at: string
}

export interface Message {
  id: string
  chat_id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  token_count: number | null
  /** Citations backing an assistant message. The API serializes this as
   *  `sources` (matching QueryResponse + the SSE `sources` event). */
  sources?: CitedSource[]
  /** 'not_from_documents' on an answer that cites none of the documents. */
  label?: AnswerLabel | null
  created_at: string
}

export type AnswerLabel = 'not_from_documents'

export interface CitedSource {
  document_id: string
  /** Stable chunk id; useful for "view exact passage" affordances. */
  chunk_id?: string
  source: string
  page: number | null
  score: number
  /** Heading hierarchy under which this chunk lives, e.g. ["Section 2", "2.1 Acceptable Documents"]. */
  heading_path?: string[]
  /** Character offset of this chunk in the original document text. */
  char_start?: number
  char_end?: number
  /** Chunk text — used by the hover lens to show the source passage with query terms highlighted. */
  text?: string
  /** false only on answers saved before every shown citation was checked
   *  against its page text: the page is shown without a highlight. */
  highlight?: boolean
}

export interface QueryResponse {
  answer: string
  sources: CitedSource[]
  chat_id: string
  message_id: string
}

// ── Auth ──────────────────────────────────────────────────────────

export const auth = {
  register: (email: string, password: string, full_name: string) =>
    api.post<{ user: User }>('/auth/register', { email, password, full_name }),

  login: (email: string, password: string) =>
    api.post<TokenResponse>('/auth/login', { email, password }),

  me: () => api.get<User>('/auth/me'),

  changePassword: (current_password: string, new_password: string) =>
    api.put('/auth/me/password', { current_password, new_password }),

  googleLogin: (credential: string) =>
    api.post<TokenResponse>('/auth/google', { credential }),
}

// ── Demo ──────────────────────────────────────────────────────────

/** GET /api/v1/demo: limits and data handling of a public demo deployment. */
export interface DemoInfo {
  enabled: boolean
  max_documents: number // 0 = unlimited
  max_upload_mb: number // 0 = unlimited
  questions_per_day: number // per user; 0 = unlimited
  site_questions_per_day: number // all users together; 0 = unlimited
  wipe_hour_utc: number | null
  llm_host: string | null
}

export const demo = {
  info: () => api.get<DemoInfo>('/demo'),
}

// ── Workspaces ────────────────────────────────────────────────────

export const workspaces = {
  list: () => api.get<Workspace[]>('/workspaces'),

  create: (name: string, description?: string) =>
    api.post<Workspace>('/workspaces', { name, description }),

  get: (id: string) => api.get<WorkspaceDetail>(`/workspaces/${id}`),

  update: (id: string, data: Partial<{ name: string; description: string }>) =>
    api.patch<Workspace>(`/workspaces/${id}`, data),

  delete: (id: string) => api.delete(`/workspaces/${id}`),

  addMember: (id: string, email: string, role = 'member') =>
    api.post(`/workspaces/${id}/members`, { email, role }),

  removeMember: (workspaceId: string, userId: string) =>
    api.delete(`/workspaces/${workspaceId}/members/${userId}`),
}

// ── Documents ─────────────────────────────────────────────────────

export const documents = {
  list: (workspace_id: string) =>
    api.get<{ documents: Document[]; total: number }>('/documents', {
      params: { workspace_id },
    }),

  upload: (workspace_id: string, file: File, onProgress?: (pct: number) => void) => {
    const form = new FormData()
    form.append('workspace_id', workspace_id)
    form.append('file', file)
    return api.post<Document>('/documents/upload', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e) => {
        if (onProgress && e.total) onProgress(Math.round((e.loaded * 100) / e.total))
      },
    })
  },

  get: (id: string) => api.get<Document>(`/documents/${id}`),

  /** Fetch the original file bytes (download / non-PDF preview). */
  file: (id: string) =>
    api.get<ArrayBuffer>(`/documents/${id}/file`, { responseType: 'arraybuffer' }),

  /** Server-rendered PNG of one PDF page with the cited chunk highlighted.
   *  `crop` clips the image to just the highlighted passage. */
  pageImage: (id: string, page: number, chunkId?: string, scale = 2, crop = false) =>
    api.get<Blob>(`/documents/${id}/page-image`, {
      params: { page, ...(chunkId ? { chunk_id: chunkId } : {}), scale, ...(crop ? { crop: true } : {}) },
      responseType: 'blob',
    }),

  delete: (id: string) => api.delete(`/documents/${id}`),
}

// ── Chat ──────────────────────────────────────────────────────────

export const chat = {
  list: (workspace_id: string) =>
    api.get<Chat[]>('/chat', { params: { workspace_id } }),

  create: (workspace_id: string, title?: string) =>
    api.post<Chat>('/chat', { workspace_id, title }),

  get: (id: string) => api.get<Chat>(`/chat/${id}`),

  messages: (id: string, limit = 50) =>
    api.get<Message[]>(`/chat/${id}/messages`, { params: { limit } }),

  query: (chat_id: string, query: string, document_ids?: string[]) =>
    api.post<QueryResponse>(`/chat/${chat_id}/query`, { query, document_ids }),

  rename: (id: string, title: string) =>
    api.patch<Chat>(`/chat/${id}/title`, { title }),

  delete: (id: string) => api.delete(`/chat/${id}`),

  // Returns an SSE URL — token is sent via Authorization header, not query param
  streamUrl: (chat_id: string, query: string) => {
    const params = new URLSearchParams({ query })
    return `/api/v1/chat/${chat_id}/stream?${params}`
  },
}
