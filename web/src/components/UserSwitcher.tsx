import type { DemoUser } from '../types'

export function UserSwitcher({
  users, current, onSelect,
}: {
  users: DemoUser[]
  current: DemoUser | null
  onSelect: (userId: string | null) => void
}) {
  return (
    <div className="flex items-center gap-2 text-sm">
      <span className="text-slate-500">当前用户:</span>
      <select
        value={current?.user_id ?? ''}
        onChange={(e) => onSelect(e.target.value || null)}
        className="border border-slate-300 rounded px-2 py-1 bg-white"
      >
        <option value="">未登录（全库）</option>
        {users.map((u) => (
          <option key={u.user_id} value={u.user_id}>
            {u.name}（{u.departments.join('/')}）{u.role}
          </option>
        ))}
      </select>
    </div>
  )
}
