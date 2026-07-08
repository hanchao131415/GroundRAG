import { useLang } from '../i18n'
import type { DemoUser } from '../types'

export function UserSwitcher({
  users, current, onSelect,
}: {
  users: DemoUser[]
  current: DemoUser | null
  onSelect: (userId: string | null) => void
}) {
  const { t } = useLang()
  return (
    <div className="flex items-center gap-2 text-sm">
      <span className="text-slate-500">{t('current_user')}</span>
      <select
        value={current?.user_id ?? ''}
        onChange={(e) => onSelect(e.target.value || null)}
        className="border border-slate-300 rounded px-2 py-1 bg-white"
      >
        <option value="">{t('not_logged_in')}</option>
        {users.map((u) => (
          <option key={u.user_id} value={u.user_id}>
            {u.name}（{u.departments.join('/')}）{u.role}
          </option>
        ))}
      </select>
    </div>
  )
}
