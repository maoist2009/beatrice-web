// Optional cleanup of this project's reproducible, ignored development downloads. Never called by src/.
import assert from 'node:assert/strict';
import { existsSync, readFileSync, rmSync } from 'node:fs';
if (existsSync('.research')) {
  const provenance = JSON.parse(readFileSync('.research/provenance.json', 'utf8'));
  assert.equal(provenance.trainer.revision, '5ddb63ea854832a914a846a22830feb9ff7e691b');
  assert(existsSync('docs/beta2-validation.json'), 'Preserve the validation evidence before deleting fixtures');
  rmSync('.research', { recursive: true, force: true });
  console.log('[cleanup] Removed research downloads, generated ONNX, Python packages, test audio and browsers. Source/tests and validation report retained.');
}