"use client";

import "crypto"

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  MessageSquare,
  Send,
  Bot,
  User,
  Loader2,
  Sparkles,
  Database,
  Search,
  TrendingUp
} from "lucide-react";
import { useState, useRef, useEffect, useCallback } from "react";
import { PredictionResults, PlanetTypeClassification, PreTrainedModel } from "../types";

interface PlanetChatbotProps {
  planetData?: PlanetRecord[];
  predictionResults?: PredictionResults;
  planetTypeClassifications?: PlanetTypeClassification[];
  modelInfo?: PreTrainedModel[];
}

type ChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  // Future: tool metadata placeholder
  toolInvocations?: Array<{ toolName: string }>;
};

type PlanetRecord = {
  toi?: string;
  toipfx?: string;
  pl_rade?: number;
  pl_orbper?: number;
  pl_eqt?: number;
  type_name?: string;
  type_confidence?: number;
  // Allow additional dynamic fields without using any
  [key: string]: unknown;
};

// ID generator without any casts
const generateId = (): string => {
  const c = (globalThis as unknown as { crypto?: { randomUUID?: () => string } }).crypto;
  if (c && typeof c.randomUUID === 'function') {
    return c.randomUUID();
  }
  return 'id_' + Math.random().toString(36).slice(2, 10);
};

export function PlanetChatbot({
  planetData = [],
  predictionResults,
  planetTypeClassifications = [],
  modelInfo = []
}: PlanetChatbotProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [isExpanded, setIsExpanded] = useState(false);

  // Local chat state (quick unblock replacement for useChat)
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const quickPrompts = [
    {
      icon: Search,
      text: "Search for planets larger than 2 Earth radii",
      prompt: "Show me all planets with radius larger than 2 Earth radii and their classifications"
    },
    {
      icon: TrendingUp,
      text: "Analyze detection confidence",
      prompt: "What's the average confidence of our predictions and which planets have the highest confidence?"
    },
    {
      icon: Database,
      text: "Dataset statistics",
      prompt: "Give me comprehensive statistics about the current dataset including type distributions"
    },
    {
      icon: Sparkles,
      text: "Find habitability candidates",
      prompt: "Which planets in our dataset might be potentially habitable based on their characteristics?"
    }
  ];

  const formatMessage = (content: string) => {
    return content
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`(.*?)`/g, '<code class="bg-gray-800 px-1 py-0.5 rounded text-sm">$1</code>')
      .replace(/\n/g, '<br/>');
  };

  const appendMessage = useCallback((msg: ChatMessage) => {
    setMessages(prev => [...prev, msg]);
  }, []);

  const updateAssistantMessage = useCallback((id: string, delta: string) => {
    setMessages(prev =>
      prev.map(m => m.id === id ? { ...m, content: m.content + delta } : m)
    );
  }, []);

  // Record a tool invocation (for displaying badges) without altering existing content
  const addToolInvocation = useCallback((id: string, toolName: string) => {
    setMessages(prev =>
      prev.map(m =>
        m.id === id
          ? {
            ...m,
            toolInvocations: [
              ...(m.toolInvocations || []),
              { toolName }
            ]
          }
          : m
      )
    );
  }, []);

  const sendToApi = useCallback(async (userContent: string) => {
    setIsLoading(true);
    setError(null);

    // Prepare OpenAI-style messages for API route
    const apiMessages = [
      ...messages.map(m => ({ role: m.role, content: m.content })),
      { role: 'user', content: userContent }
    ];

    const assistantId = generateId();
    appendMessage({ id: assistantId, role: 'assistant', content: '' });

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: apiMessages,
          // Provide the contextual datasets
          planetData,
          predictionResults,
          planetTypeClassifications,
          modelInfo
        })
      });

      if (!res.ok) {
        throw new Error(`Request failed (${res.status})`);
      }

      if (!res.body) {
        const text = await res.text();
        updateAssistantMessage(assistantId, text);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Attempt to parse Server-Sent Event style "data:" lines
        const lines = buffer.split(/\r?\n/);
        // Keep last partial line in buffer
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          // Typical AI stream events often prefixed by data:
          if (!trimmed.startsWith('data:')) continue;
          const data = trimmed.slice(5).trim();
          console.log('[PlanetChatbot] Received data chunk:', data);
          if (data === '[DONE]') {
            buffer = '';
            break;
          }
          if (!data) continue;

          // Append raw chunk (you could parse JSON if needed)
          try {
            // Some frameworks send JSON per line
            if (data.startsWith('{') || data.startsWith('[')) {
              // Try JSON parse for structured content
              const parsed = JSON.parse(data);
              if (typeof parsed === 'object' && parsed !== null) {
                // New event-based protocol from backend
                if (parsed.event === 'tool_result') {
                  // Record tool badge
                  addToolInvocation(assistantId, String(parsed.tool || 'tool'));
                  // Optionally append a short note (commented out to avoid clutter)
                  // updateAssistantMessage(assistantId, `\\n[Tool ${parsed.tool} ready]\\n`);
                } else if (parsed.event === 'final_answer') {
                  if (typeof parsed.chunk === 'string') {
                    updateAssistantMessage(assistantId, parsed.chunk);
                  }
                } else if (parsed.event === 'error') {
                  updateAssistantMessage(assistantId, `Error: ${parsed.message || 'Unknown tool error'}`);
                } else if (typeof parsed.text === 'string') {
                  // Legacy shape
                  updateAssistantMessage(assistantId, parsed.text);
                } else if (typeof parsed.delta === 'string') {
                  updateAssistantMessage(assistantId, parsed.delta);
                } else if (typeof parsed.content === 'string') {
                  updateAssistantMessage(assistantId, parsed.content);
                } else {
                  updateAssistantMessage(assistantId, JSON.stringify(parsed));
                }
              } else {
                updateAssistantMessage(assistantId, data);
              }
            } else {
              // Plain string chunk (server may send JSON.stringify(text) when only a tool answered)
              let chunkText = data;
              if (
                (chunkText.startsWith('"') && chunkText.endsWith('"')) ||
                (chunkText.startsWith("\\") && chunkText.endsWith("\\"))
              ) {
                try {
                  chunkText = JSON.parse(chunkText);
                } catch {
                  chunkText = chunkText.slice(1, -1);
                }
              }
              updateAssistantMessage(assistantId, chunkText);
            }
          } catch {
            // Fallback on parse errors: still try to strip quotes
            let chunkText = data;
            if (
              (chunkText.startsWith('"') && chunkText.endsWith('"')) ||
              (chunkText.startsWith("\\") && chunkText.endsWith("\\"))
            ) {
              try {
                chunkText = JSON.parse(chunkText);
              } catch {
                chunkText = chunkText.slice(1, -1);
              }
            }
            updateAssistantMessage(assistantId, chunkText);
          }
        }
      }

      // Flush any remaining buffer
      if (buffer.trim()) {
        updateAssistantMessage(assistantId, buffer.trim());
      }

    } catch (e: unknown) {
      const err = e instanceof Error ? e : new Error('Unknown error');
      setError(err);
      updateAssistantMessage(assistantId, `Error: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  }, [messages, appendMessage, planetData, predictionResults, planetTypeClassifications, modelInfo, updateAssistantMessage]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    const userId = generateId();
    appendMessage({ id: userId, role: 'user', content: input.trim() });
    const content = input.trim();
    setInput('');
    void sendToApi(content);
  };

  const handleQuickPrompt = (prompt: string) => {
    if (!isExpanded) {
      setIsExpanded(true);
      setTimeout(() => handleQuickPrompt(prompt), 50);
      return;
    }
    const userId = generateId();
    appendMessage({ id: userId, role: 'user', content: prompt });
    void sendToApi(prompt);
  };

  if (!isExpanded) {
    return (
      <Card className="w-full">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <MessageSquare className="w-5 h-5" />
              AI Planet Assistant
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsExpanded(true)}
              className="text-xs"
              disabled={isLoading}
            >
              Open Chat
            </Button>
          </CardTitle>
          <CardDescription>
            Ask questions about your planet data, get insights, and explore predictions
          </CardDescription>
        </CardHeader>
        <CardContent className="overflow-y-auto max-h-[180px]">
          <div className="grid grid-cols-2 gap-2">
            {quickPrompts.slice(0, 4).map((p, idx) => (
              <Button
                key={idx}
                variant="ghost"
                size="sm"
                className="justify-start h-auto p-3 text-xs"
                onClick={() => {
                  setIsExpanded(true);
                  setTimeout(() => handleQuickPrompt(p.prompt), 80);
                }}
                disabled={isLoading}
              >
                <p.icon className="w-3 h-3 mr-2 flex-shrink-0" />
                <span className="truncate">{p.text}</span>
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="w-full h-[700px] flex flex-col">
      <CardHeader className="pb-3 border-b flex-shrink-0">
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-gradient-to-r from-purple-600 to-purple-700 rounded-lg flex items-center justify-center">
              <MessageSquare className="w-4 h-4 text-white" />
            </div>
            <div>
              <h3 className="font-semibold">AI Planet Assistant</h3>
              <p className="text-xs text-muted-foreground">
                {planetData.length} planets • {planetTypeClassifications.length} classified
              </p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsExpanded(false)}
            className="text-xs"
            disabled={isLoading}
          >
            Minimize
          </Button>
        </CardTitle>
      </CardHeader>

      <CardContent className="flex-1 flex flex-col p-0 min-h-0 overflow-hidden">
        <div className="flex-1 p-4 overflow-y-auto overflow-x-hidden" ref={scrollRef}>
          <div className="space-y-4 max-w-full">
            {messages.length === 0 && (
              <div className="text-center py-8">
                <Bot className="w-12 h-12 mx-auto mb-4 text-purple-500" />
                <h3 className="text-lg font-semibold mb-2">Ready to help!</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  I can analyze your planet data, explain predictions, and provide insights.
                </p>
                <div className="grid grid-cols-1 gap-2">
                  {quickPrompts.map((p, idx) => (
                    <Button
                      key={idx}
                      variant="outline"
                      size="sm"
                      className="justify-start text-xs"
                      onClick={() => handleQuickPrompt(p.prompt)}
                      disabled={isLoading}
                    >
                      <p.icon className="w-3 h-3 mr-2" />
                      {p.text}
                    </Button>
                  ))}
                </div>
              </div>
            )}

            {messages.map(message => (
              <div
                key={message.id}
                className={`flex gap-3 min-w-0 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {message.role === 'assistant' && (
                  <div className="w-8 h-8 bg-purple-600 rounded-lg flex items-center justify-center flex-shrink-0">
                    <Bot className="w-4 h-4 text-white" />
                  </div>
                )}

                <div className={`max-w-[80%] min-w-0 rounded-lg p-3 ${message.role === 'user' ? 'bg-purple-600 text-white' : 'bg-muted'
                  }`}>
                  {message.role === 'assistant' && (message as { toolInvocations?: Array<{ toolName?: string }> }).toolInvocations && (
                    <div className="mb-2 space-y-1">
                      {(message as { toolInvocations?: Array<{ toolName?: string }> }).toolInvocations!.map((tool: { toolName?: string }, idx: number) => (
                        <Badge key={idx} variant="secondary" className="text-xs">
                          <Database className="w-3 h-3 mr-1" />
                          {tool.toolName || 'Tool'}
                        </Badge>
                      ))}
                    </div>
                  )}
                  <div
                    className="text-sm leading-relaxed break-words overflow-wrap-anywhere [word-break:break-word]"
                    dangerouslySetInnerHTML={{ __html: formatMessage(message.content) }}
                  />
                </div>

                {message.role === 'user' && (
                  <div className="w-8 h-8 bg-gray-600 rounded-lg flex items-center justify-center flex-shrink-0">
                    <User className="w-4 h-4 text-white" />
                  </div>
                )}
              </div>
            ))}

            {isLoading && (
              <div className="flex gap-3 justify-start">
                <div className="w-8 h-8 bg-purple-600 rounded-lg flex items-center justify-center flex-shrink-0">
                  <Bot className="w-4 h-4 text-white" />
                </div>
                <div className="bg-muted rounded-lg p-3">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Analyzing data...
                  </div>
                </div>
              </div>
            )}

            {error && (
              <div className="text-center py-4">
                <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-3">
                  <p className="text-sm text-red-600 dark:text-red-400">
                    Error: {error.message}
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="border-t p-4 flex-shrink-0 bg-background">
          <form onSubmit={handleSubmit} className="flex gap-2">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about planets, predictions, or data insights..."
              className="flex-1"
              disabled={isLoading}
            />
            <Button type="submit" size="sm" disabled={isLoading || !input.trim()}>
              {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            </Button>
          </form>

          {planetData.length > 0 && (
            <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
              <Database className="w-3 h-3" />
              Connected to {planetData.length} planets
              {predictionResults && `, ${predictionResults.predictions.length} predictions`}
              {modelInfo.length > 0 && `, ${modelInfo.length} models`}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
