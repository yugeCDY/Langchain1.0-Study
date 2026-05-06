<script setup lang="ts">
import { fetchEventSource } from '@microsoft/fetch-event-source'
import { computed, nextTick, reactive, ref } from 'vue'

type SourceItem = {
  source?: string
  page?: number | string
  [key: string]: unknown
}

type UploadResponse = {
  domain: string
  collection_name: string
  inserted_chunks: number
  collection_total_documents: number
  errors: string[]
}

type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  status?: 'streaming' | 'done'
  score?: number | null
  needsFallback?: boolean | null
  sources?: SourceItem[]
  rewrittenQueries?: string[]
  processLogs?: string[]
}

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8087'
const mode = ref<'fast' | 'full'>('fast')

const domain = ref('resume_kb')
const rolesInput = ref('hr')
const collectionName = ref('')
const question = ref('')
const files = ref<File[]>([])
const chatMessages = ref<ChatMessage[]>([])

const isAsking = ref(false)
const isUploading = ref(false)
const uploadMessage = ref('')
const uploadError = ref('')
const askError = ref('')
const chatPanelRef = ref<HTMLElement | null>(null)
const streamQueue = ref<string[]>([])
const isDrainingQueue = ref(false)

const TYPEWRITER_INTERVAL_MS = 22

const resolvedCollectionName = computed(() => {
  const value = collectionName.value.trim()
  return value || null
})

const roleList = computed(() =>
  rolesInput.value
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean),
)

const streamEndpoint = computed(() =>
  mode.value === 'fast'
    ? `${API_BASE}/api/v1/retrieval/full-pipeline/fast-stream`
    : `${API_BASE}/api/v1/retrieval/full-pipeline/stream`,
)

function createMessageId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function onFileChange(event: Event) {
  const input = event.target as HTMLInputElement
  files.value = Array.from(input.files ?? [])
}

async function scrollChatToBottom() {
  await nextTick()
  const el = chatPanelRef.value
  if (!el) return
  el.scrollTop = el.scrollHeight
}

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms))
}

async function drainStreamQueue(targetMessage: ChatMessage) {
  if (isDrainingQueue.value) {
    return
  }

  isDrainingQueue.value = true

  try {
    while (streamQueue.value.length) {
      const chunk = streamQueue.value.shift()
      if (!chunk) {
        continue
      }

      for (const char of chunk) {
        targetMessage.content += char
        await scrollChatToBottom()
        await sleep(TYPEWRITER_INTERVAL_MS)
      }
    }
  } finally {
    isDrainingQueue.value = false
  }
}

async function uploadDocuments() {
  uploadError.value = ''
  uploadMessage.value = ''

  if (!files.value.length) {
    uploadError.value = '请先选择至少一个 PDF、DOCX 或 CSV 文件。'
    return
  }

  isUploading.value = true

  try {
    const formData = new FormData()
    formData.append('domain', domain.value.trim())
    formData.append('allowed_roles', roleList.value.join(','))
    if (resolvedCollectionName.value) {
      formData.append('collection_name', resolvedCollectionName.value)
    }
    for (const file of files.value) {
      formData.append('files', file)
    }

    const response = await fetch(`${API_BASE}/api/v1/ingestion/multimodal`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(errorText || '上传失败')
    }

    const data = (await response.json()) as UploadResponse
    uploadMessage.value = `已入库 ${data.inserted_chunks} 个切片，当前集合共 ${data.collection_total_documents} 条。`
    if (!collectionName.value.trim()) {
      collectionName.value = data.collection_name
    }
  } catch (error) {
    uploadError.value = error instanceof Error ? error.message : '上传失败'
  } finally {
    isUploading.value = false
  }
}

async function askQuestion() {
  askError.value = ''
  streamQueue.value = []

  const prompt = question.value.trim()
  if (!prompt) {
    askError.value = '请输入一个查询问题。'
    return
  }

  const userMessage = reactive<ChatMessage>({
    id: createMessageId(),
    role: 'user',
    content: prompt,
    status: 'done',
  })
  const assistantMessage = reactive<ChatMessage>({
    id: createMessageId(),
    role: 'assistant',
    content: '',
    status: 'streaming',
    score: null,
    needsFallback: null,
    sources: [],
    rewrittenQueries: [],
    processLogs: [],
  })

  chatMessages.value.push(userMessage, assistantMessage)
  question.value = ''
  isAsking.value = true
  await scrollChatToBottom()

  try {
    const history = chatMessages.value
      .filter((message) => message.id !== assistantMessage.id)
      .map((message) => ({
        role: message.role,
        content: message.content,
      }))

    await fetchEventSource(streamEndpoint.value, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: prompt,
        roles: roleList.value.length ? roleList.value : ['hr'],
        collection_name: resolvedCollectionName.value,
        domain: domain.value.trim() || null,
        active_only: true,
        chat_history: history,
      }),
      async onopen(response) {
        if (!response.ok) {
          const errorText = await response.text()
          throw new Error(errorText || '问答请求失败')
        }
        console.log('[sse] onopen', new Date().toISOString())
      },
      async onmessage(event) {
        console.log('[sse] onmessage', new Date().toISOString(), event.event, event.data)
        const payload = JSON.parse(event.data) as {
          type: string
          content?: string
          answer?: string
          stage?: string
          detail?: string
          rewritten_queries?: string[]
          relevance_score?: number
          needs_fallback?: boolean
          sources?: SourceItem[]
        }

        if (payload.type === 'stage' && payload.detail) {
          assistantMessage.processLogs = [
            ...(assistantMessage.processLogs ?? []),
            payload.detail,
          ]
        }

        if (payload.type === 'meta') {
          assistantMessage.rewrittenQueries = payload.rewritten_queries ?? []
        }

        if (payload.type === 'status') {
          assistantMessage.score = payload.relevance_score ?? null
          assistantMessage.needsFallback = payload.needs_fallback ?? null
        }

        if (payload.type === 'token' && payload.content) {
          streamQueue.value.push(payload.content)
          void drainStreamQueue(assistantMessage)
        }

        if (payload.type === 'done') {
          while (streamQueue.value.length || isDrainingQueue.value) {
            await sleep(10)
          }
          assistantMessage.content = payload.answer ?? assistantMessage.content
          assistantMessage.sources = payload.sources ?? []
          assistantMessage.score = payload.relevance_score ?? assistantMessage.score ?? null
          assistantMessage.needsFallback = payload.needs_fallback ?? assistantMessage.needsFallback ?? null
          assistantMessage.status = 'done'
          await scrollChatToBottom()
        }
      },
      onerror(error) {
        throw error
      },
    })

    assistantMessage.status = 'done'
  } catch (error) {
    streamQueue.value = []
    assistantMessage.status = 'done'
    assistantMessage.content = '当前请求失败，请检查后端服务、模型配置或网络连接。'
    askError.value = error instanceof Error ? error.message : '问答请求失败'
  } finally {
    isAsking.value = false
  }
}
</script>

<template>
  <div class="shell">
    <header class="masthead">
      <div class="masthead-copy">
        <p class="eyebrow">Enterprise Knowledge Console</p>
        <h1>企业知识库问答台</h1>
        <p class="lede">
          这是一块面向内部知识问答的工作台：左侧做文档入库，右侧保留完整多轮上下文，
          模型一吐出第一个字，界面就立即开始显示。
        </p>
      </div>
      <div class="masthead-badge">
        <span>API Base</span>
        <strong>{{ API_BASE }}</strong>
      </div>
    </header>

    <main class="workspace">
      <aside class="panel panel-upload">
        <div class="panel-heading">
          <p class="panel-kicker">Ingestion</p>
          <h2>上传知识文件</h2>
        </div>

        <label class="field">
          <span>业务域</span>
          <input v-model="domain" type="text" placeholder="resume_kb" />
        </label>

        <label class="field">
          <span>角色</span>
          <input v-model="rolesInput" type="text" placeholder="hr,ops" />
        </label>

        <label class="field">
          <span>集合名</span>
          <input
            v-model="collectionName"
            type="text"
            placeholder="为空时默认跟随 domain"
          />
        </label>

        <label class="uploader">
          <input
            class="uploader-input"
            type="file"
            multiple
            accept=".pdf,.docx,.csv"
            @change="onFileChange"
          />
          <span class="uploader-title">拖入或选择 PDF / DOCX / CSV</span>
          <span class="uploader-subtitle">
            当前已选 {{ files.length }} 个文件
          </span>
        </label>

        <ul class="file-list" v-if="files.length">
          <li v-for="file in files" :key="`${file.name}-${file.size}`">
            <span>{{ file.name }}</span>
            <small>{{ Math.ceil(file.size / 1024) }} KB</small>
          </li>
        </ul>

        <button class="action-button" :disabled="isUploading" @click="uploadDocuments">
          {{ isUploading ? '正在入库...' : '上传并入库' }}
        </button>

        <p class="feedback success" v-if="uploadMessage">{{ uploadMessage }}</p>
        <p class="feedback error" v-if="uploadError">{{ uploadError }}</p>
      </aside>

      <section class="panel panel-chat">
        <div class="panel-heading">
          <p class="panel-kicker">Conversation</p>
          <h2>多轮问答</h2>
        </div>

        <div class="mode-switch" role="tablist" aria-label="问答模式切换">
          <button
            class="mode-chip"
            :class="{ active: mode === 'fast' }"
            type="button"
            @click="mode = 'fast'"
          >
            极速模式
          </button>
          <button
            class="mode-chip"
            :class="{ active: mode === 'full' }"
            type="button"
            @click="mode = 'full'"
          >
            完整模式
          </button>
        </div>

        <div class="chat-panel" ref="chatPanelRef">
          <div v-if="!chatMessages.length" class="empty-chat">
            <p>还没有对话。</p>
            <span>先上传文档，再在下方输入你的第一个问题。</span>
          </div>

          <article
            v-for="message in chatMessages"
            :key="message.id"
            class="chat-bubble"
            :class="message.role"
          >
            <div class="bubble-head">
              <span>{{ message.role === 'user' ? '你' : '知识库助手' }}</span>
              <small v-if="message.role === 'assistant' && message.score !== null">
                相关度 {{ message.score?.toFixed(2) }}
              </small>
            </div>

            <p class="bubble-content">
              {{ message.content }}
              <span
                v-if="message.role === 'assistant' && message.status === 'streaming'"
                class="caret"
              />
            </p>

            <div
              v-if="message.role === 'assistant' && message.rewrittenQueries?.length"
              class="bubble-meta"
            >
              <span>改写查询</span>
              <ul>
                <li v-for="item in message.rewrittenQueries" :key="item">{{ item }}</li>
              </ul>
            </div>

            <div
              v-if="message.role === 'assistant' && message.processLogs?.length"
              class="bubble-meta process"
            >
              <span>过程</span>
              <ul>
                <li v-for="(item, index) in message.processLogs" :key="`${message.id}-process-${index}`">
                  {{ item }}
                </li>
              </ul>
            </div>

            <div
              v-if="message.role === 'assistant' && message.sources?.length"
              class="bubble-meta sources"
            >
              <span>引用来源</span>
              <ul>
                <li v-for="(item, index) in message.sources" :key="index">
                  <strong>{{ item.source ?? 'unknown' }}</strong>
                  <em>第 {{ item.page ?? '?' }} 页</em>
                </li>
              </ul>
            </div>
          </article>
        </div>

        <div class="composer">
          <label class="field">
            <span>输入问题</span>
            <textarea
              v-model="question"
              rows="4"
              placeholder="例如：继续展开上一个回答里提到的学校经历"
            />
          </label>

          <div class="toolbar">
            <button class="action-button primary" :disabled="isAsking" @click="askQuestion">
              {{ isAsking ? '正在生成...' : '发送问题' }}
            </button>
            <p class="toolbar-tip">
              {{
                mode === 'fast'
                  ? '当前接入极速流式接口，优先压低首字等待时间。'
                  : '当前接入完整流式接口，包含更完整的检索链路。'
              }}
            </p>
          </div>

          <p class="feedback error" v-if="askError">{{ askError }}</p>
        </div>
      </section>
    </main>
  </div>
</template>
