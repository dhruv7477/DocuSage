import { useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { documents as docsApi } from '@/api'

interface PageImage {
  url: string
  pageCount: number
  highlightFound: boolean
}

/**
 * Fetch a server-rendered page screenshot (PNG with highlight marks)
 * as an object URL. The image needs the Authorization header, so a
 * plain <img src> won't work — we fetch the blob via axios and hand
 * the component a revocable object URL.
 */
export function usePageImage(
  documentId: string | undefined,
  page: number,
  chunkId?: string,
  scale = 2,
) {
  const query = useQuery<PageImage>({
    queryKey: ['page-image', documentId, page, chunkId ?? null, scale],
    enabled: !!documentId && page >= 1,
    staleTime: 5 * 60_000,
    gcTime: 5 * 60_000,
    queryFn: async () => {
      const res = await docsApi.pageImage(documentId!, page, chunkId, scale)
      return {
        url: URL.createObjectURL(res.data),
        pageCount: Number(res.headers['x-page-count'] ?? 0),
        highlightFound: res.headers['x-highlight-found'] === '1',
      }
    },
  })

  // Revoke the object URL when it's replaced or the component unmounts.
  const url = query.data?.url
  useEffect(() => {
    return () => {
      if (url) URL.revokeObjectURL(url)
    }
  }, [url])

  return query
}
