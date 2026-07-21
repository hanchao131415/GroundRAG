import { render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { getDocuments, getIndexStatus } from '../../api/client'
import { LangProvider } from '../../i18n'
import { KnowledgeBaseView } from '../KnowledgeBaseView'

vi.mock('../../api/client', () => ({
  deleteDocument: vi.fn(),
  getDocuments: vi.fn(),
  getIndexStatus: vi.fn(),
  reindexDocuments: vi.fn(),
  uploadDocument: vi.fn(),
}))

describe('KnowledgeBaseView index status', () => {
  beforeEach(() => {
    localStorage.setItem('groundrag_lang', 'en')
    vi.mocked(getDocuments).mockResolvedValue([])
  })

  it('shows a degraded rebuild error and keeps admin retry enabled', async () => {
    vi.mocked(getIndexStatus).mockResolvedValue({ status: 'degraded', error: 'rebuild failed' })

    render(
      <LangProvider>
        <KnowledgeBaseView
          demoUsers={[]}
          currentUser={{ user_id: 'admin', name: 'Admin', departments: ['*'], role: 'admin' }}
          loginAs={vi.fn()}
        />
      </LangProvider>,
    )

    expect(await screen.findByText('rebuild failed')).toBeInTheDocument()
    expect(screen.getByText('Index degraded')).toBeInTheDocument()
    await waitFor(() => expect(screen.getByRole('button', { name: 'Reindex' })).toBeEnabled())
  })
})
