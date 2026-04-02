/**
 * Gemini LLM Adapter
 *
 * Wraps the Google Generative AI SDK to expose an Anthropic-compatible interface.
 * This allows the rest of the codebase to call `client.beta.messages.create(...)`
 * as usual while the request is transparently forwarded to the Gemini API.
 *
 * Environment variables:
 *   GEMINI_API_KEY   — Google AI Studio API key (required)
 *   GOOGLE_API_KEY   — Alternative name for the same key
 */

import type {
  BetaContentBlockParam,
  BetaMessageStreamParams,
  BetaRawMessageStreamEvent,
  BetaToolUnion,
} from '@anthropic-ai/sdk/resources/beta/messages/messages.mjs'
import { randomUUID } from 'crypto'

// ---------------------------------------------------------------------------
// Types re-exported from @google/generative-ai (imported dynamically)
// ---------------------------------------------------------------------------

type GeminiPart =
  | { text: string }
  | { inlineData: { mimeType: string; data: string } }
  | { functionCall: { name: string; args: Record<string, unknown> } }
  | { functionResponse: { name: string; response: Record<string, unknown> } }

type GeminiContent = {
  role: 'user' | 'model'
  parts: GeminiPart[]
}

type GeminiFunctionDeclaration = {
  name: string
  description?: string
  parameters?: Record<string, unknown>
}

type GeminiTool = {
  functionDeclarations: GeminiFunctionDeclaration[]
}

// ---------------------------------------------------------------------------
// Message translation: Anthropic → Gemini
// ---------------------------------------------------------------------------

/**
 * Translate an array of Anthropic BetaContentBlockParam blocks into Gemini parts,
 * accumulating tool_use id→name mappings for later tool_result resolution.
 */
function contentBlocksToGeminiParts(
  blocks: BetaContentBlockParam | BetaContentBlockParam[] | string,
  toolUseIdToName: Map<string, string>,
): GeminiPart[] {
  if (typeof blocks === 'string') {
    return [{ text: blocks }]
  }
  const arr = Array.isArray(blocks) ? blocks : [blocks]
  const parts: GeminiPart[] = []

  for (const block of arr) {
    if (block.type === 'text') {
      if (block.text) parts.push({ text: block.text })
    } else if (block.type === 'tool_use') {
      toolUseIdToName.set(block.id, block.name)
      parts.push({
        functionCall: {
          name: block.name,
          args: (block.input ?? {}) as Record<string, unknown>,
        },
      })
    } else if (block.type === 'tool_result') {
      const toolName = toolUseIdToName.get(block.tool_use_id) ?? block.tool_use_id
      let resultText = ''
      if (typeof block.content === 'string') {
        resultText = block.content
      } else if (Array.isArray(block.content)) {
        resultText = block.content
          .map(c => (c.type === 'text' ? c.text : ''))
          .join('')
      }
      parts.push({
        functionResponse: {
          name: toolName,
          response: { output: resultText },
        },
      })
    } else if (block.type === 'image') {
      const src = block.source
      if (src.type === 'base64') {
        parts.push({ inlineData: { mimeType: src.media_type, data: src.data } })
      }
      // url-type images are not supported by the Gemini inlineData format
    }
    // document, thinking, redacted_thinking — skip unsupported blocks
  }
  return parts
}

/**
 * Convert Anthropic messages to Gemini `contents` array.
 */
function toGeminiContents(
  messages: BetaMessageStreamParams['messages'],
): GeminiContent[] {
  const toolUseIdToName = new Map<string, string>()
  const contents: GeminiContent[] = []

  for (const msg of messages) {
    const role: 'user' | 'model' = msg.role === 'assistant' ? 'model' : 'user'
    const parts = contentBlocksToGeminiParts(
      msg.content as BetaContentBlockParam | BetaContentBlockParam[] | string,
      toolUseIdToName,
    )
    if (parts.length > 0) {
      contents.push({ role, parts })
    }
  }

  return contents
}

/**
 * Convert Anthropic tool definitions to Gemini FunctionDeclarations.
 */
function toGeminiTools(tools: BetaToolUnion[] | undefined): GeminiTool[] {
  if (!tools || tools.length === 0) return []
  const declarations: GeminiFunctionDeclaration[] = []
  for (const tool of tools) {
    if (tool.type === 'custom' || !('type' in tool) || tool.type === undefined) {
      // standard user-defined tool
      declarations.push({
        name: (tool as { name: string }).name,
        description: (tool as { description?: string }).description,
        parameters: (tool as { input_schema?: Record<string, unknown> })
          .input_schema as Record<string, unknown> | undefined,
      })
    }
    // built-in / computer_use / bash tools — skip, Gemini has no equivalent
  }
  return declarations.length > 0 ? [{ functionDeclarations: declarations }] : []
}

// ---------------------------------------------------------------------------
// Stream translation: Gemini → BetaRawMessageStreamEvent
// ---------------------------------------------------------------------------

/**
 * Async generator that converts a Gemini streaming response into the sequence
 * of `BetaRawMessageStreamEvent` events expected by the rest of the codebase.
 */
async function* geminiToAnthropicStream(
  streamResult: {
    stream: AsyncIterable<{
      candidates?: Array<{
        content?: { parts: Array<Record<string, unknown>> }
        finishReason?: string
      }>
    }>
    response: Promise<{
      candidates?: Array<{
        content?: { parts: Array<Record<string, unknown>> }
        finishReason?: string
      }>
      usageMetadata?: { promptTokenCount?: number; candidatesTokenCount?: number }
    }>
  },
  msgId: string,
  modelName: string,
): AsyncGenerator<BetaRawMessageStreamEvent> {
  // ── message_start ──────────────────────────────────────────────────────────
  yield {
    type: 'message_start',
    message: {
      id: msgId,
      type: 'message',
      role: 'assistant',
      model: modelName,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: {
        input_tokens: 0,
        output_tokens: 0,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
      },
    },
  } as unknown as BetaRawMessageStreamEvent

  let blockIndex = 0
  let textBlockOpen = false
  let hasFunctionCall = false

  for await (const chunk of streamResult.stream) {
    if (!chunk.candidates?.length) continue
    const candidate = chunk.candidates[0]
    if (!candidate.content?.parts) continue

    for (const part of candidate.content.parts) {
      if (typeof part.text === 'string' && part.text) {
        // ── open text block if needed ─────────────────────────────────────
        if (!textBlockOpen) {
          yield {
            type: 'content_block_start',
            index: blockIndex,
            content_block: { type: 'text', text: '' },
          } as unknown as BetaRawMessageStreamEvent
          textBlockOpen = true
        }
        yield {
          type: 'content_block_delta',
          index: blockIndex,
          delta: { type: 'text_delta', text: part.text },
        } as unknown as BetaRawMessageStreamEvent
      } else if (
        part.functionCall &&
        typeof (part.functionCall as Record<string, unknown>).name === 'string'
      ) {
        hasFunctionCall = true
        // ── close any open text block ─────────────────────────────────────
        if (textBlockOpen) {
          yield {
            type: 'content_block_stop',
            index: blockIndex,
          } as unknown as BetaRawMessageStreamEvent
          blockIndex++
          textBlockOpen = false
        }
        const fc = part.functionCall as { name: string; args?: Record<string, unknown> }
        const toolId = `toolu_gemini_${randomUUID().replace(/-/g, '').slice(0, 20)}`
        // ── tool_use block ────────────────────────────────────────────────
        yield {
          type: 'content_block_start',
          index: blockIndex,
          content_block: { type: 'tool_use', id: toolId, name: fc.name, input: {} },
        } as unknown as BetaRawMessageStreamEvent
        yield {
          type: 'content_block_delta',
          index: blockIndex,
          delta: { type: 'input_json_delta', partial_json: JSON.stringify(fc.args ?? {}) },
        } as unknown as BetaRawMessageStreamEvent
        yield {
          type: 'content_block_stop',
          index: blockIndex,
        } as unknown as BetaRawMessageStreamEvent
        blockIndex++
      }
    }
  }

  // ── close any remaining text block ─────────────────────────────────────────
  if (textBlockOpen) {
    yield {
      type: 'content_block_stop',
      index: blockIndex,
    } as unknown as BetaRawMessageStreamEvent
  }

  // ── gather final usage / stop reason ───────────────────────────────────────
  let inputTokens = 0
  let outputTokens = 0
  let stopReason: string = hasFunctionCall ? 'tool_use' : 'end_turn'

  try {
    const finalResp = await streamResult.response
    inputTokens = finalResp.usageMetadata?.promptTokenCount ?? 0
    outputTokens = finalResp.usageMetadata?.candidatesTokenCount ?? 0
    const finishReason = finalResp.candidates?.[0]?.finishReason
    if (finishReason === 'MAX_TOKENS') stopReason = 'max_tokens'
    else if (finishReason === 'SAFETY') stopReason = 'stop_sequence'
    else if (hasFunctionCall) stopReason = 'tool_use'
    else stopReason = 'end_turn'
  } catch {
    // ignore usage errors
  }

  // ── message_delta ──────────────────────────────────────────────────────────
  yield {
    type: 'message_delta',
    delta: { stop_reason: stopReason, stop_sequence: null },
    usage: { output_tokens: outputTokens },
  } as unknown as BetaRawMessageStreamEvent

  // Patch input_tokens back via a second message_start-like event is not
  // supported; instead surface them in a synthetic message_start update.
  // The codebase reads input tokens from the initial message_start event,
  // so re-emit with the real counts now that we have them.
  if (inputTokens > 0) {
    yield {
      type: 'message_start',
      message: {
        id: msgId,
        type: 'message',
        role: 'assistant',
        model: modelName,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: {
          input_tokens: inputTokens,
          output_tokens: 0,
          cache_creation_input_tokens: 0,
          cache_read_input_tokens: 0,
        },
      },
    } as unknown as BetaRawMessageStreamEvent
  }

  // ── message_stop ───────────────────────────────────────────────────────────
  yield {
    type: 'message_stop',
  } as unknown as BetaRawMessageStreamEvent
}

// ---------------------------------------------------------------------------
// GeminiStream — minimal Stream<BetaRawMessageStreamEvent> shim
// ---------------------------------------------------------------------------

/**
 * A minimal shim that satisfies the `Stream<BetaRawMessageStreamEvent>` interface
 * used by `claude.ts`:
 *   - async iterable (`for await`)
 *   - `controller` property for `cleanupStream()`
 */
class GeminiStream {
  readonly controller: AbortController
  private readonly _gen: AsyncGenerator<BetaRawMessageStreamEvent>

  constructor(gen: AsyncGenerator<BetaRawMessageStreamEvent>) {
    this.controller = new AbortController()
    this._gen = gen
  }

  [Symbol.asyncIterator](): AsyncGenerator<BetaRawMessageStreamEvent> {
    return this._gen
  }
}

// ---------------------------------------------------------------------------
// GeminiClient — minimal Anthropic client shim
// ---------------------------------------------------------------------------

type StreamWithResponse = GeminiStream & {
  withResponse: () => Promise<{
    data: GeminiStream
    response: Response
    request_id: string
  }>
}

/**
 * Returns a minimal object that looks enough like `Anthropic` for the parts of
 * the codebase that use it (primarily `claude.ts`):
 *
 *   anthropic.beta.messages.create({ ...params, stream: true }).withResponse()
 *   anthropic.beta.messages.create({ ...params })            // non-streaming
 */
export async function createGeminiClient(): Promise<{
  beta: {
    messages: {
      create: (
        params: BetaMessageStreamParams & { stream?: boolean },
        options?: { signal?: AbortSignal; headers?: Record<string, string> },
      ) => StreamWithResponse | Promise<unknown>
    }
  }
}> {
  const apiKey =
    process.env.GEMINI_API_KEY ?? process.env.GOOGLE_API_KEY ?? ''
  if (!apiKey) {
    throw new Error(
      'GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable is required when CLAUDE_CODE_USE_GEMINI=1',
    )
  }

  const { GoogleGenerativeAI } = await import('@google/generative-ai')
  const genAI = new GoogleGenerativeAI(apiKey)

  function buildCreateFn(
    params: BetaMessageStreamParams & { stream?: boolean },
    _options?: { signal?: AbortSignal; headers?: Record<string, string> },
  ): StreamWithResponse | Promise<unknown> {
    const modelName = (params.model as string) || 'gemini-2.5-pro'
    const geminiModel = genAI.getGenerativeModel({ model: modelName })

    const contents = toGeminiContents(params.messages)
    const geminiTools = toGeminiTools(
      params.tools as BetaToolUnion[] | undefined,
    )

    // System prompt
    const systemText =
      typeof params.system === 'string'
        ? params.system
        : Array.isArray(params.system)
          ? (params.system as Array<{ type: string; text?: string }>)
              .filter(b => b.type === 'text')
              .map(b => b.text ?? '')
              .join('\n')
          : undefined

    const generationConfig: Record<string, unknown> = {
      maxOutputTokens: params.max_tokens ?? 8192,
    }
    if (params.temperature != null) {
      generationConfig.temperature = params.temperature
    }

    if (params.stream) {
      // ── Streaming path ────────────────────────────────────────────────────
      const msgId = `msg_gemini_${randomUUID().replace(/-/g, '').slice(0, 24)}`

      const streamPromise = geminiModel.generateContentStream({
        contents,
        ...(geminiTools.length > 0 ? { tools: geminiTools } : {}),
        ...(systemText
          ? { systemInstruction: { role: 'system', parts: [{ text: systemText }] } }
          : {}),
        generationConfig,
      })

      const gen = (async function* (): AsyncGenerator<BetaRawMessageStreamEvent> {
        const streamResult = await streamPromise
        yield* geminiToAnthropicStream(streamResult, msgId, modelName)
      })()

      const geminiStream = new GeminiStream(gen)

      // Fake HTTP response
      const fakeResponse = new Response(null, {
        status: 200,
        headers: { 'content-type': 'text/event-stream' },
      })

      const streamWithResponse = Object.assign(geminiStream, {
        withResponse: () =>
          Promise.resolve({
            data: geminiStream,
            response: fakeResponse,
            request_id: msgId,
          }),
      })

      return streamWithResponse
    }

    // ── Non-streaming path ──────────────────────────────────────────────────
    return (async () => {
      const result = await geminiModel.generateContent({
        contents,
        ...(geminiTools.length > 0 ? { tools: geminiTools } : {}),
        ...(systemText
          ? { systemInstruction: { role: 'system', parts: [{ text: systemText }] } }
          : {}),
        generationConfig,
      })

      const candidate = result.response.candidates?.[0]
      const parts = candidate?.content?.parts ?? []

      // Convert Gemini response back to Anthropic BetaMessage format
      type ContentBlock =
        | { type: 'text'; text: string }
        | { type: 'tool_use'; id: string; name: string; input: unknown }

      const content: ContentBlock[] = []
      for (const part of parts) {
        if (typeof (part as Record<string, unknown>).text === 'string') {
          content.push({ type: 'text', text: (part as { text: string }).text })
        } else if ((part as Record<string, unknown>).functionCall) {
          const fc = (part as { functionCall: { name: string; args?: unknown } })
            .functionCall
          content.push({
            type: 'tool_use',
            id: `toolu_gemini_${randomUUID().replace(/-/g, '').slice(0, 20)}`,
            name: fc.name,
            input: fc.args ?? {},
          })
        }
      }

      const usageMeta = result.response.usageMetadata
      const finishReason = candidate?.finishReason ?? 'STOP'
      let stopReason = 'end_turn'
      if (finishReason === 'MAX_TOKENS') stopReason = 'max_tokens'
      else if (content.some(b => b.type === 'tool_use')) stopReason = 'tool_use'

      return {
        id: `msg_gemini_${randomUUID().replace(/-/g, '').slice(0, 24)}`,
        type: 'message',
        role: 'assistant',
        model: modelName,
        content,
        stop_reason: stopReason,
        stop_sequence: null,
        usage: {
          input_tokens: usageMeta?.promptTokenCount ?? 0,
          output_tokens: usageMeta?.candidatesTokenCount ?? 0,
          cache_creation_input_tokens: 0,
          cache_read_input_tokens: 0,
        },
      }
    })()
  }

  return {
    beta: {
      messages: {
        create: buildCreateFn,
      },
    },
  }
}
