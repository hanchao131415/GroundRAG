import { useCallback, useEffect, useState } from 'react'
import { deleteDocument, getDocuments, getIndexStatus, reindexDocuments, uploadDocument } from '../api/client'
import type { DemoUser, DocumentRecord } from '../types'
import { useLang } from '../i18n'
import { UserSwitcher } from '../components/UserSwitcher'

export function KnowledgeBaseView({ demoUsers, currentUser, loginAs }: { demoUsers: DemoUser[]; currentUser: DemoUser | null; loginAs: (id: string | null) => Promise<void> }) {
  const { t } = useLang()
  const [documents, setDocuments] = useState<DocumentRecord[]>([])
  const [status, setStatus] = useState('ready')
  const [error, setError] = useState('')
  const [department, setDepartment] = useState('')
  const isAdmin = currentUser?.departments.includes('*') ?? false
  const isBusy = status === 'reindexing' || status === 'initializing'

  useEffect(() => {
    setDepartment(isAdmin ? '公共' : (currentUser?.departments[0] ?? ''))
  }, [currentUser, isAdmin])

  const refresh = useCallback(async () => {
    try {
      setDocuments(await getDocuments())
      const state = await getIndexStatus()
      setStatus(state.status)
      setError(state.error ?? '')
    } catch (err) {
      setError((err as Error).message)
    }
  }, [])

  useEffect(() => {
    if (currentUser) void refresh()
    else { setDocuments([]); setError('') }
  }, [currentUser, refresh])
  useEffect(() => {
    if (status === 'reindexing' || status === 'initializing') {
      const timer = setInterval(() => { void refresh() }, 1500)
      return () => clearInterval(timer)
    }
    return undefined
  }, [status, refresh])

  const onUpload = async (file: File | undefined) => {
    if (!file || !department) return
    setError('')
    try {
      await uploadDocument(file, department)
      setStatus('reindexing')
      await refresh()
    } catch (err) {
      setError((err as Error).message)
    }
  }

  const onDelete = async (doc: DocumentRecord) => {
    if (!window.confirm(`${t('confirm_delete')} ${doc.source}?`)) return
    try {
      await deleteDocument(doc.id)
      setStatus('reindexing')
      await refresh()
    } catch (err) {
      setError((err as Error).message)
    }
  }

  const onReindex = async () => {
    try {
      await reindexDocuments()
      setStatus('reindexing')
    } catch (err) {
      setError((err as Error).message)
    }
  }

  return (
    <div className="p-4 space-y-4 max-w-5xl mx-auto">
      <div className="border-b border-slate-200 pb-3"><UserSwitcher users={demoUsers} current={currentUser} onSelect={loginAs} /></div>
      <div className="flex flex-wrap items-center gap-3 border-b border-slate-200 pb-3">
        <div>
          <h2 className="font-semibold text-slate-800">{t('kb_title')}</h2>
          <p className="text-xs text-slate-500">{t('kb_subtitle')}</p>
        </div>
        <span className={`ml-auto text-xs px-2 py-1 rounded ${status === 'ready' ? 'bg-emerald-50 text-emerald-700' : 'bg-amber-50 text-amber-700'}`}>
          {t(`index_${status}`)}
        </span>
        {isAdmin && <button onClick={() => void onReindex()} disabled={isBusy} className="border rounded px-3 py-1.5 text-sm disabled:opacity-50">{t('reindex')}</button>}
      </div>
      {error && <div className="border border-red-200 bg-red-50 text-red-700 rounded p-3 text-sm">{error}</div>}
      <div className="flex flex-wrap items-center gap-2">
        {isAdmin ? (
          <input value={department} onChange={(e) => setDepartment(e.target.value)} className="border rounded px-2 py-2 text-sm" placeholder={t('department')} />
        ) : (
          <select value={department} onChange={(e) => setDepartment(e.target.value)} className="border rounded px-2 py-2 text-sm">
            {(currentUser?.departments ?? []).map((d) => <option key={d} value={d}>{d}</option>)}
          </select>
        )}
        <label className={`bg-blue-600 text-white rounded px-3 py-2 text-sm ${isBusy ? 'cursor-not-allowed opacity-50' : 'cursor-pointer hover:bg-blue-700'}`}>
          {t('upload')}
          <input type="file" className="hidden" disabled={isBusy} accept=".pdf,.docx,.md,.txt,.xlsx" onChange={(e) => { void onUpload(e.target.files?.[0]); e.currentTarget.value = '' }} />
        </label>
        <span className="text-xs text-slate-400">{t('upload_hint')}</span>
      </div>
      <div className="overflow-x-auto border border-slate-200 rounded bg-white">
        <table className="w-full text-sm">
          <thead className="bg-slate-50 text-left text-xs text-slate-500"><tr><th className="p-3">{t('document')}</th><th className="p-3">{t('department')}</th><th className="p-3">{t('size')}</th><th className="p-3">{t('actions')}</th></tr></thead>
          <tbody>{documents.map((doc) => <tr key={doc.id} className="border-t border-slate-100"><td className="p-3 font-medium">{doc.source}</td><td className="p-3">{doc.department}</td><td className="p-3 text-slate-500">{formatBytes(doc.size)}</td><td className="p-3"><button onClick={() => void onDelete(doc)} disabled={isBusy} className="text-red-600 hover:underline disabled:cursor-not-allowed disabled:opacity-50">{t('delete')}</button></td></tr>)}</tbody>
        </table>
        {documents.length === 0 && <div className="p-8 text-center text-sm text-slate-400">{t('no_documents')}</div>}
      </div>
    </div>
  )
}

function formatBytes(size: number) {
  if (size < 1024) return `${size} B`
  if (size < 1024 * 1024) return `${Math.round(size / 1024)} KB`
  return `${(size / 1024 / 1024).toFixed(1)} MB`
}
