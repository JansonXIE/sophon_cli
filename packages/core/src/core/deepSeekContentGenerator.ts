import type {
  CountTokensResponse,
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
  Content,
  Part,
} from '@google/genai';
import type { ContentGenerator } from './contentGenerator.js';
import type { UserTierId, GeminiUserTier } from '../code_assist/types.js';
import type { LlmRole } from '../telemetry/llmRole.js';

export class DeepSeekContentGenerator implements ContentGenerator {
  userTier?: UserTierId;
  userTierName?: string;
  paidTier?: GeminiUserTier;

  constructor(private readonly apiKey: string) {}

  private extractSystemInstruction(
    request: GenerateContentParameters,
  ): Content[] {
    if (!request.config?.systemInstruction) return [];

    // Some structures allow parts, others just text, handle multiple formats of systemInstruction
    const sysInst = request.config.systemInstruction;
    let parts: Part[] = [];
    if (typeof sysInst === 'string') {
      parts = [{ text: sysInst }];
    } else if (
      typeof sysInst === 'object' &&
      'parts' in sysInst &&
      Array.isArray((sysInst as any).parts)
    ) {
      parts = (sysInst as any).parts;
    } else {
      parts = [{ text: JSON.stringify(sysInst) }];
    }

    return [{ role: 'system', parts }];
  }

  private convertContentsToMessages(contents: Content[]): any[] {
    return contents.flatMap((c): any[] => {
      let role = 'user';
      if (c.role === 'model') role = 'assistant';
      if (c.role === 'system') role = 'system';

      const parts = c.parts || [];

      const functionResponses = parts.filter((p) => 'functionResponse' in p);
      if (functionResponses.length > 0) {
        return functionResponses.map((p) => {
          const fr = (p as any).functionResponse;
          return {
            role: 'tool',
            tool_call_id: fr.id || fr.name,
            content: JSON.stringify(fr.response),
            name: fr.name,
          };
        });
      }

      const functionCalls = parts.filter((p) => 'functionCall' in p);
      if (functionCalls.length > 0) {
        return [
          {
            role: 'assistant',
            tool_calls: functionCalls.map((p) => {
              const fc = (p as any).functionCall;
              return {
                id: fc.id || fc.name,
                type: 'function',
                function: {
                  name: fc.name,
                  arguments:
                    typeof fc.args === 'string'
                      ? fc.args
                      : JSON.stringify(fc.args || {}),
                },
              };
            }),
          },
        ];
      }

      const textParts = parts
        .filter((p: Part) => typeof p.text === 'string')
        .map((p: Part) => p.text);
      if (textParts.length > 0) {
        const content = textParts.join('\n');
        return [{ role, content }];
      }
      return [];
    });
  }

  async generateContent(
    request: GenerateContentParameters,
    userPromptId: string,
    role: LlmRole,
  ): Promise<GenerateContentResponse> {
    const messages = [];

    const systemContents = this.extractSystemInstruction(request);
    if (systemContents.length > 0) {
      messages.push(...this.convertContentsToMessages(systemContents));
    }

    if (request.contents) {
      let contentsObj = request.contents;
      if (!Array.isArray(contentsObj)) {
        contentsObj = [contentsObj as any];
      }

      const isPartUnion = (contentsObj as any[]).some(
        (c) => typeof c === 'string' || (!('role' in c) && 'text' in c),
      );
      if (isPartUnion) {
        contentsObj = [
          {
            role: 'user',
            parts: contentsObj.map((c) =>
              typeof c === 'string' ? { text: c } : c,
            ) as any,
          },
        ];
      }

      messages.push(
        ...this.convertContentsToMessages(contentsObj as Content[]),
      );
    }

    let modelName =
      typeof request.model === 'string' ? request.model : 'deepseek-chat';
    if (modelName.startsWith('models/')) {
      modelName = modelName.replace('models/', '');
    }

    if (modelName.includes('gemini')) {
      modelName = 'deepseek-chat';
    }

    let tools: any[] | undefined = undefined;
    if (request.config?.tools && request.config.tools.length > 0) {
      tools = request.config.tools.flatMap((t: any) =>
        (t.functionDeclarations || []).map((fd: any) => ({
          type: 'function',
          function: {
            name: fd.name,
            description: fd.description,
            parameters: fd.parameters,
          },
        })),
      );
    }

    const body = {
      model: modelName,
      messages,
      tools,
      temperature: request.config?.temperature,
      top_p: request.config?.topP,
      max_tokens: request.config?.maxOutputTokens,
      stream: false,
    };

    const response = await fetch('https://api.deepseek.com/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`DeepSeek API error (${response.status}): ${errorText}`);
    }

    const data = await response.json();
    const message = data.choices?.[0]?.message;
    const messageContent = message?.content || '';
    const toolCalls = message?.tool_calls || [];

    const parts: Part[] = [];
    const extractedFunctionCalls: any[] = [];
    if (messageContent) {
      parts.push({ text: messageContent });
    }
    for (const tc of toolCalls) {
      if (tc.type === 'function') {
        const functionCall: any = {
          name: tc.function.name,
          args: tc.function.arguments ? JSON.parse(tc.function.arguments) : {},
        };
        if (tc.id) {
          functionCall.id = tc.id;
        }

        parts.push({ functionCall });
        extractedFunctionCalls.push(functionCall);
      }
    }

    return {
      text: messageContent,
      functionCalls:
        extractedFunctionCalls.length > 0
          ? extractedFunctionCalls
          : undefined,
      candidates: [
        {
          content: { parts, role: 'model' },
          finishReason: 'STOP',
        },
      ],
      usageMetadata: {
        promptTokenCount: data.usage?.prompt_tokens || 0,
        candidatesTokenCount: data.usage?.completion_tokens || 0,
        totalTokenCount: data.usage?.total_tokens || 0,
      },
    } as any;
  }

  async generateContentStream(
    request: GenerateContentParameters,
    userPromptId: string,
    role: LlmRole,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const parent = this;
    return (async function* () {
      const messages = [];

      const systemContents = parent.extractSystemInstruction(request);
      if (systemContents.length > 0) {
        messages.push(...parent.convertContentsToMessages(systemContents));
      }

      if (request.contents) {
        let contentsObj = request.contents;
        if (!Array.isArray(contentsObj)) {
          contentsObj = [contentsObj as any];
        }

        const isPartUnion = (contentsObj as any[]).some(
          (c) => typeof c === 'string' || (!('role' in c) && 'text' in c),
        );
        if (isPartUnion) {
          contentsObj = [
            {
              role: 'user',
              parts: contentsObj.map((c) =>
                typeof c === 'string' ? { text: c } : c,
              ) as any,
            },
          ];
        }

        messages.push(
          ...parent.convertContentsToMessages(contentsObj as Content[]),
        );
      }

      let modelName =
        typeof request.model === 'string' ? request.model : 'deepseek-chat';
      if (modelName.startsWith('models/')) {
        modelName = modelName.replace('models/', '');
      }

      if (modelName.includes('gemini')) {
        modelName = 'deepseek-chat';
      }

      let tools: any[] | undefined = undefined;
      if (request.config?.tools && request.config.tools.length > 0) {
        tools = request.config.tools.flatMap((t: any) =>
          (t.functionDeclarations || []).map((fd: any) => ({
            type: 'function',
            function: {
              name: fd.name,
              description: fd.description,
              parameters: fd.parameters,
            },
          })),
        );
      }

      const body = {
        model: modelName,
        messages,
        tools,
        temperature: request.config?.temperature,
        top_p: request.config?.topP,
        max_tokens: request.config?.maxOutputTokens,
        stream: true,
      };

      const response = await fetch(
        'https://api.deepseek.com/chat/completions',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${parent.apiKey}`,
          },
          body: JSON.stringify(body),
        },
      );

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(
          `DeepSeek stream error (${response.status}): ${errorText}`,
        );
      }

      if (!response.body) {
        throw new Error('No response body from DeepSeek');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      const toolCallsBuffer: any[] = [];
      let lastUsage: any = null;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.trim() === '') continue;
          if (line.trim() === 'data: [DONE]') {
            if (toolCallsBuffer.length > 0) {
              for (const tc of toolCallsBuffer) {
                if (tc && tc.name) {
                  const fc = {
                    id: tc.id,
                    name: tc.name,
                    args: tc.args ? JSON.parse(tc.args) : {},
                  };
                  yield {
                    functionCalls: [fc],
                    candidates: [
                      {
                        content: {
                          parts: [
                            {
                              functionCall: fc as any,
                            },
                          ],
                          role: 'model',
                        },
                        finishReason: 'STOP',
                      },
                    ],
                    usageMetadata: lastUsage
                      ? {
                          promptTokenCount:
                            lastUsage.prompt_tokens || 0,
                          candidatesTokenCount:
                            lastUsage.completion_tokens || 0,
                          totalTokenCount:
                            lastUsage.total_tokens || 0,
                        }
                      : undefined,
                  } as any;
                }
              }
            } else {
              yield {
                candidates: [
                  {
                    content: { parts: [], role: 'model' },
                    finishReason: 'STOP',
                  },
                ],
                usageMetadata: lastUsage
                  ? {
                      promptTokenCount: lastUsage.prompt_tokens || 0,
                      candidatesTokenCount:
                        lastUsage.completion_tokens || 0,
                      totalTokenCount: lastUsage.total_tokens || 0,
                    }
                  : undefined,
              } as any;
            }
            return;
          }

          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              const delta = data.choices?.[0]?.delta;
              if (data.usage) lastUsage = data.usage;
              if (!delta) continue;

              const content = delta.content || '';
              const toolCalls = delta.tool_calls;

              if (content) {
                yield {
                  text: content,
                  candidates: [
                    {
                      content: { parts: [{ text: content }], role: 'model' },
                    },
                  ],
                } as any;
              }

              if (toolCalls && toolCalls.length > 0) {
                for (const tc of toolCalls) {
                  const idx = tc.index;
                  if (!toolCallsBuffer[idx])
                    toolCallsBuffer[idx] = { id: '', name: '', args: '' };
                  if (tc.id) toolCallsBuffer[idx].id += tc.id;
                  if (tc.function?.name)
                    toolCallsBuffer[idx].name += tc.function.name;
                  if (tc.function?.arguments)
                    toolCallsBuffer[idx].args += tc.function.arguments;
                }
              }
            } catch (e) {
              console.error('Error parsing streaming JSON:', e, line);
            }
          }
        }
      }
    })();
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    return { totalTokens: 0 } as any;
  }

  async embedContent(
    request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    throw new Error('embedContent not supported by DeepSeekContentGenerator');
  }
}
