import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { User, Bot, ChevronDown, ChevronUp, Wrench, Loader2 } from 'lucide-react'
import { Message } from '../hooks/useChat'
import clsx from 'clsx'

interface MessageBubbleProps {
  message: Message
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const [showThinking, setShowThinking] = useState(false)
  const [showToolCalls, setShowToolCalls] = useState(false)
  const isUser = message.role === 'user'

  return (
    <div
      className={clsx(
        'flex gap-3 message-enter',
        isUser ? 'flex-row-reverse' : 'flex-row'
      )}
    >
      {/* Avatar */}
      <div
        className={clsx(
          'flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center',
          isUser ? 'bg-primary' : 'bg-secondary'
        )}
      >
        {isUser ? (
          <User className="w-5 h-5 text-primary-foreground" />
        ) : (
          <Bot className="w-5 h-5 text-secondary-foreground" />
        )}
      </div>

      {/* Content */}
      <div
        className={clsx(
          'flex-1 max-w-[80%] rounded-2xl px-4 py-3',
          isUser
            ? 'bg-primary text-primary-foreground'
            : 'bg-secondary text-secondary-foreground'
        )}
      >
        {/* Thinking section (collapsible) */}
        {message.thinking && (
          <div className="mb-3 border-b border-border/50 pb-3">
            <button
              onClick={() => setShowThinking(!showThinking)}
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              {showThinking ? (
                <ChevronUp className="w-4 h-4" />
              ) : (
                <ChevronDown className="w-4 h-4" />
              )}
              Thinking
            </button>
            {showThinking && (
              <div className="mt-2 text-sm opacity-70 italic">
                {message.thinking}
              </div>
            )}
          </div>
        )}

        {/* Main content */}
        <div className="prose prose-sm max-w-none dark:prose-invert">
          {message.isStreaming && !message.content ? (
            <div className="flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="text-sm">Thinking...</span>
            </div>
          ) : (
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                // Custom code block rendering
                code({ className, children, ...props }) {
                  const match = /language-(\w+)/.exec(className || '')
                  const isInline = !match

                  return isInline ? (
                    <code
                      className="bg-muted px-1 py-0.5 rounded text-sm"
                      {...props}
                    >
                      {children}
                    </code>
                  ) : (
                    <pre className="bg-muted rounded-lg p-3 overflow-x-auto">
                      <code className={className} {...props}>
                        {children}
                      </code>
                    </pre>
                  )
                },
                // Custom link rendering
                a({ href, children }) {
                  return (
                    <a
                      href={href}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary hover:underline"
                    >
                      {children}
                    </a>
                  )
                },
              }}
            >
              {message.content}
            </ReactMarkdown>
          )}
        </div>

        {/* Tool calls section (collapsible) */}
        {message.toolCalls && message.toolCalls.length > 0 && (
          <div className="mt-3 border-t border-border/50 pt-3">
            <button
              onClick={() => setShowToolCalls(!showToolCalls)}
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              <Wrench className="w-4 h-4" />
              {showToolCalls ? (
                <ChevronUp className="w-4 h-4" />
              ) : (
                <ChevronDown className="w-4 h-4" />
              )}
              {message.toolCalls.length} tool call
              {message.toolCalls.length > 1 ? 's' : ''}
            </button>
            {showToolCalls && (
              <div className="mt-2 space-y-2">
                {message.toolCalls.map((tool, index) => (
                  <div
                    key={index}
                    className="bg-muted/50 rounded-lg p-2 text-sm"
                  >
                    <div className="font-medium">{tool.name}</div>
                    <pre className="text-xs mt-1 overflow-x-auto">
                      {JSON.stringify(tool.input, null, 2)}
                    </pre>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Streaming indicator */}
        {message.isStreaming && message.content && (
          <span className="inline-block w-2 h-4 bg-current animate-pulse ml-1" />
        )}

        {/* Timestamp */}
        <div
          className={clsx(
            'text-xs mt-2 opacity-50',
            isUser ? 'text-right' : 'text-left'
          )}
        >
          {message.timestamp.toLocaleTimeString()}
        </div>
      </div>
    </div>
  )
}
