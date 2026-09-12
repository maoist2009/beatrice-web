export type ModelFormat = 'beatrice-rc0' | 'beatrice-beta2';

// beta.2 Trainer deliberately exports PARAPHERNALIA_VERSION="2.0.0-beta.1".
// Do not guess a model's architecture from its filename, byte count, or a marketing version.
export function resolveModelFormat(version: string): ModelFormat {
  if (version === '2.0.0-rc.0') return 'beatrice-rc0';
  if (version === '2.0.0-beta.1' || version === '2.0.0-beta.2') return 'beatrice-beta2';
  throw new Error(`Unsupported paraphernalia version: ${version}. Expected Beatrice beta.1/beta.2 format or rc.0. RVC .pth/.index models use a different engine.`);
}

export const formatLabel = (format: ModelFormat) => format === 'beatrice-beta2' ? 'Beatrice 2 beta.2' : 'Beatrice 2 rc.0';