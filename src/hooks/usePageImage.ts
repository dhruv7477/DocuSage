import { useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { documents as docsApi } from '@/api'

interface PageBlob {
  blob: Blob
  pageCount: number
  highlightFound: boolean
}

interface PageImage {
  url: string
  pageCount: number
  highlightFound: boolean
}

/**
 * Fetch a server-rendered page screenshot (PNG with highlight marks).
 *
 * The PNG needs the Authorization header, so a plain <img src> won't
 * work — we fetch the blob via axios. React Query caches the *blob*
 * (a stable, non-revocable value safe to share across components);
 * each component instance then derives its OWN object URL and revokes
 * only that one on unmount. Creating the URL inside the cached queryFn
 * is unsafe: under StrictMode the first effect-cleanup would revoke the
 * shared URL and break every other consumer's <img> (broken-image icon).
 */
export function usePageImage(
  documentId: string | undefined,
  page: number,
  chunkId?: string,
  scale = 2,
  crop = false,
) {
  const query = useQuery<PageBlob>({
    queryKey: ['page-image', documentId, page, chunkId ?? null, scale, crop],
    enabled: !!documentId && page >= 1,
    staleTime: 5 * 60_000,
    gcTime: 5 * 60_000,
    queryFn: async () => {
      const res = await docsApi.pageImage(documentId!, page, chunkId, scale, crop)
      return {
        blob: res.data,
        pageCount: Number(res.headers['x-page-count'] ?? 0),
        highlightFound: res.headers['x-highlight-found'] === '1',
      }
    },
  })

  // Own object URL per consumer, recreated cleanly on a StrictMode remount.
  const blob = query.data?.blob
  const [url, setUrl] = useState<string>()
  useEffect(() => {
    if (!blob) {
      setUrl(undefined)
      return
    }
    const objectUrl = URL.createObjectURL(blob)
    setUrl(objectUrl)
    return () => URL.revokeObjectURL(objectUrl)
  }, [blob])

  const data: PageImage | undefined =
    query.data && url
      ? { url, pageCount: query.data.pageCount, highlightFound: query.data.highlightFound }
      : undefined

  return { data, isLoading: query.isLoading, isError: query.isError }
}
