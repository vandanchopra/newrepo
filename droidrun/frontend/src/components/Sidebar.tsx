import { Plus, MessageSquare, History, Brain, Search } from 'lucide-react'
import clsx from 'clsx'

interface SidebarProps {
  onNewChat: () => void
  memoryContext: string | null
}

export function Sidebar({ onNewChat, memoryContext }: SidebarProps) {
  return (
    <aside className="w-64 border-r border-border bg-background flex flex-col">
      {/* New Chat Button */}
      <div className="p-4">
        <button
          onClick={onNewChat}
          className={clsx(
            'w-full flex items-center gap-2 px-4 py-2 rounded-lg',
            'bg-primary text-primary-foreground',
            'hover:bg-primary/90 transition-colors'
          )}
        >
          <Plus className="w-5 h-5" />
          New Chat
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-2 space-y-1">
        <SidebarItem icon={MessageSquare} label="Conversations" active />
        <SidebarItem icon={History} label="Task History" />
        <SidebarItem icon={Search} label="Research" />
        <SidebarItem icon={Brain} label="Memory" />
      </nav>

      {/* Memory Context */}
      {memoryContext && (
        <div className="p-4 border-t border-border">
          <div className="flex items-center gap-2 text-sm text-muted-foreground mb-2">
            <Brain className="w-4 h-4" />
            Active Memory Context
          </div>
          <div className="text-xs bg-muted p-2 rounded-lg max-h-24 overflow-y-auto">
            {memoryContext.slice(0, 200)}
            {memoryContext.length > 200 && '...'}
          </div>
        </div>
      )}

      {/* Status */}
      <div className="p-4 border-t border-border">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-green-500" />
          <span className="text-sm text-muted-foreground">Connected</span>
        </div>
      </div>
    </aside>
  )
}

interface SidebarItemProps {
  icon: React.ElementType
  label: string
  active?: boolean
}

function SidebarItem({ icon: Icon, label, active }: SidebarItemProps) {
  return (
    <button
      className={clsx(
        'w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors',
        active
          ? 'bg-muted text-foreground'
          : 'text-muted-foreground hover:bg-muted hover:text-foreground'
      )}
    >
      <Icon className="w-5 h-5" />
      {label}
    </button>
  )
}
