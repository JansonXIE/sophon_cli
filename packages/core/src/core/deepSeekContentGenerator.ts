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

  private extractSystemInstruction(request: GenerateContentParameters): Content[] {
    if (!request.config?.systemInstruction) return [];
    
    // Some structures allow parts, others just text, handle multiple formats of systemInstruction
    const sysInst = request.config.systemInstruction;
    let parts: Part[] = [];
    if (typeof sysInst === 'string') {
      parts = [{ text: sysInst }];
    } else if (typeof sysInst === 'object' && 'parts' in sysInst && Array.isArray((sysInst as any).parts)) {
      parts = (sysInst as any).parts;
    } else {
      parts = [{ text: JSON.stringify(sysInst) }];
    }
    
    return [{ role: 'system', parts }];
  }

  private convertContentsToMessages(contents: Content[]): any[] {
    return contents.map((c) => {
      let role = 'user';
      if (c.role === 'model') role = 'assistant';
      if (c.role === 'system') role = 'system';

      const parts = c.parts || [];
      const textParts = parts
        .filter((p: Part) => typeof p.text === 'string')
        .map((p: Part) => p.text);
      const content = textParts.join('\n');
      return { role, content };
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
      
      const isPartUnion = (contentsObj as any[]).some(c => typeof c === 'string' || (!('role' in c) && ('text' in c)));
      if (isPartUnion) {
        contentsObj = [{ role: 'user', parts: contentsObj.map(c => typeof c === 'string' ? { text: c } : c) as any }];
      }

      messages.push(...this.convertContentsToMessages(contentsObj as Content[]));
    }

    let modelName = typeof request.model === 'string' ? request.model : 'deepseek-chat';
    if (modelName.startsWith('models/')) {
        modelName = modelName.replace('models/', '');
    }

    if (modelName.includes('gemini')) {
        modelName = 'deepseek-chat';
    }

    const body = {
      model: modelName,
      messages,
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
    const messageContent = data.choices?.[0]?.message?.content || '';

    return {
      text: messageContent,
      candidates: [
        {
          content: { parts: [{ text: messageContent }], role: 'model' },
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
        
        const isPartUnion = (contentsObj as any[]).some(c => typeof c === 'string' || (!('role' in c) && ('text' in c)));
        if (isPartUnion) {
          contentsObj = [{ role: 'user', parts: contentsObj.map(c => typeof c === 'string' ? { text: c } : c) as any }];
        }

        messages.push(...parent.convertContentsToMessages(contentsObj as Content[]));
      }

      let modelName = typeof request.model === 'string' ? request.model : 'deepseek-chat';
      if (modelName.startsWith('models/')) {
          modelName = modelName.replace('models/', '');
      }

      if (modelName.includes('gemini')) {
          modelName = 'deepseek-chat';
      }

      const body = {
        model: modelName,
        messages,
        temperature: request.config?.temperature,
        top_p: request.config?.topP,
        max_tokens: request.config?.maxOutputTokens,
        stream: true,
      };

      const response = await fetch('https://api.deepseek.com/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${parent.apiKey}`,
        },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`DeepSeek stream error (${response.status}): ${errorText}`);
      }

      if (!response.body) {
        throw new Error('No response body from DeepSeek');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.trim() === '') continue;
          if (line.trim() === 'data: [DONE]') return;

          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              const content = data.choices?.[0]?.delta?.content || '';
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
