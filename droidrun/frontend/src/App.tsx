import { useState } from 'react'
import { ChatInterface } from './components/ChatInterface'
import { Sidebar } from './components/Sidebar'
import { Header } from './components/Header'
import { useChat } from './hooks/useChat'
import { useTheme } from './hooks/useTheme'

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const { theme, toggleTheme } = useTheme()
  const {
    messages,
    isStreaming,
    error,
    sendMessage,
    clearMessages,
    memoryContext,
  } = useChat()

  return (
    <div className={`min-h-screen flex ${theme === 'dark' ? 'dark' : ''}`}>
      <div className="flex-1 flex flex-col bg-background">
        <Header
          onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
          onToggleTheme={toggleTheme}
          theme={theme}
        />

        <div className="flex-1 flex overflow-hidden">
          {sidebarOpen && (
            <Sidebar
              onNewChat={clearMessages}
              memoryContext={memoryContext}
            />
          )}

          <main className="flex-1 flex flex-col">
            <ChatInterface
              messages={messages}
              isStreaming={isStreaming}
              error={error}
              onSendMessage={sendMessage}
            />
          </main>
        </div>
      </div>
    </div>
  )
}

export default App
